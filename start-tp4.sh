#!/usr/bin/env bash
# ============================================================================
# start-tp4.sh — EXPERIMENTAL TP=4 GLM-5.3-Flash-NVFP4 on 4× DGX Spark
# ============================================================================
#
# OPTIONAL sibling of start.sh. Does not modify start.sh, stop.sh, or the
# 2-node containers (glm53-flash-head / glm53-flash-worker).
#
# THIS PATH IS UNTESTED ON THIS KIT. We only have 2 Sparks. A 4-Spark owner
# with a RoCE switch can try it and report back. `/health` is the only
# success signal — container "Up" is not serving.
#
# Fabric contract (read this):
#   Supported : 4× GB10, one GPU each, TP=4 via Ray, every CX7 on the SAME
#               RoCE L2 (a switch). One HCA / netdev per node, pinned.
#   NOT this  : switchless QSFP ring / daisy-chain. Stock NCCL + this script
#               will hang in ncclCommInitRank. Use Alex Ellis's recipe:
#               https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless
#   NOT this  : 1 GbE management LAN as the NCCL path.
#
# HEAD_IP and WORKER*_IP must be the IPs on the CX7 NICs.
#
# Setup:
#   1. cp env.tp4.example .env.tp4
#   2. fill in the four nodes (SSH, RoCE IPs, CX7 IF + IB HCA)
#   3. passwordless SSH from the head to all three workers
#   4. ./start-tp4.sh
#
# Usage:
#   ./start-tp4.sh                    start (download / sync×3 / launch)
#   ./start-tp4.sh stop               stop the four TP4 containers only
#   ./start-tp4.sh restart            stop + start
#   ./start-tp4.sh status             four containers, API, Ray
#   ./start-tp4.sh logs               follow head
#   ./start-tp4.sh logs worker [1-3]  follow a worker (default 1)
#
# Same env knobs as start.sh (MOE_BACKEND, MAX_MODEL_LEN, GPU_MEM_UTIL, …)
# plus the WORKER{1,2,3}_* fabric vars. GPU_MEM_UTIL=0.84 was tuned for TP=2;
# TP=4 holds fewer weights per rank — raise it only if you know you need KV.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
if [ -f "$SCRIPT_DIR/.env.tp4" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env.tp4"
    set +a
fi

# ----------------------------- configuration -------------------------------
MODEL="LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_CACHE_NAME="models--LibertAIDAI--GLM-5.3-Flash-NVFP4"
IMAGE="${IMAGE:-mia/glm53-flash-spark:mm-ray-v1}"
RAY_VERSION="${RAY_VERSION:-2.58.0}"

HEAD_IP="${HEAD_IP:-}"
HEAD_CX7_IF="${HEAD_CX7_IF:-}"
HEAD_CX7_IB="${HEAD_CX7_IB:-}"

NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
USE_HOST_NCCL="${USE_HOST_NCCL:-1}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"  # 4 GiB

TP=4
RAY_PORT="${RAY_PORT:-6379}"
PORT="${PORT:-8888}"

MTP_TOKENS="${MTP_TOKENS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.84}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
LIMIT_MM="${LIMIT_MM:-{\"image\":4,\"video\":1}}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"
BLOCK_SIZE="${BLOCK_SIZE:-2304}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"
CLUSTER_WAIT_ITERS="${CLUSTER_WAIT_ITERS:-120}"

# Distinct from the 2-node recipe so ./start.sh and ./start-tp4.sh cannot
# clobber each other.
CONTAINER_HEAD="glm53-flash-tp4-head"
CONTAINER_2N_HEAD="glm53-flash-head"
CONTAINER_2N_WORKER="glm53-flash-worker"
CACHE_VOLUME="${CACHE_VOLUME:-glm53-flash-cache-sm121}"

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"

LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-tp4-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-tp4-worker.inner.sh"
KERNEL_ERR_PAT='NoKernelImageForDevice|no kernel image is available'

W_SSH=() W_IP=() W_IF=() W_IB=() W_HOME=() W_NCCL=() W_NAME=()

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53-tp4]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53-tp4]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53-tp4]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

banner() {
    warn "EXPERIMENTAL TP=4 — untested on this kit. 4 Sparks + RoCE switch only."
    warn "2-node path is untouched: ./start.sh  (containers glm53-flash-head / -worker)"
}

usage() { sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ssh_user_of() {
    local spec="$1"
    case "$spec" in
        *@*) printf '%s' "${spec%%@*}" ;;
        *)   printf '%s' "${USER:-spark}" ;;
    esac
}

worker_ssh() {
    local spec="$1"
    shift
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$spec" "$@"
}

load_workers() {
    W_SSH=() W_IP=() W_IF=() W_IB=() W_HOME=() W_NCCL=() W_NAME=()
    local n ssh_v ip_v if_v ib_v home_v nccl_v
    local -a missing=()
    for n in 1 2 3; do
        ssh_v="WORKER${n}_SSH"; ip_v="WORKER${n}_IP"
        if_v="WORKER${n}_CX7_IF"; ib_v="WORKER${n}_CX7_IB"
        home_v="WORKER${n}_HOME"; nccl_v="WORKER${n}_NCCL_HOST_DIR"
        [ -n "${!ssh_v:-}" ] || missing+=("$ssh_v")
        [ -n "${!ip_v:-}" ]  || missing+=("$ip_v")
        [ -n "${!if_v:-}" ]  || missing+=("$if_v")
        [ -n "${!ib_v:-}" ]  || missing+=("$ib_v")
        local home="${!home_v:-/home/$(ssh_user_of "${!ssh_v:-x}")}"
        local nccl="${!nccl_v:-$home/nccl-2.30.7}"
        W_SSH+=("${!ssh_v:-}")
        W_IP+=("${!ip_v:-}")
        W_IF+=("${!if_v:-}")
        W_IB+=("${!ib_v:-}")
        W_HOME+=("$home")
        W_NCCL+=("$nccl")
        W_NAME+=("glm53-flash-tp4-w${n}")
    done
    [ -n "$HEAD_IP" ]     || missing+=("HEAD_IP")
    [ -n "$HEAD_CX7_IF" ] || missing+=("HEAD_CX7_IF")
    [ -n "$HEAD_CX7_IB" ] || missing+=("HEAD_CX7_IB")
    if [ "${#missing[@]}" -gt 0 ]; then
        die "set these in .env.tp4 (see env.tp4.example) or the environment: ${missing[*]}"
    fi
}

resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run without SKIP_DOWNLOAD first"
    hash="$(<"$ref")"
    [ -n "$hash" ] || die "empty refs/main at $ref"
    local dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/processor_config.json" ] \
        || die "processor_config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start-tp4.sh restart' or './start-tp4.sh stop' first"
        fi
        die "port ${port} is already in use — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — TP4 containers keep running ('"'"'./start-tp4.sh logs'"'"' / '"'"'./start-tp4.sh stop'"'"')"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    command -v python3 >/dev/null 2>&1 || die "python3 not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — it must be the CX7 / RoCE address"

    local i others
    for i in 0 1 2; do
        log "checking worker $((i+1)) ${W_SSH[$i]} ..."
        worker_ssh "${W_SSH[$i]}" true 2>/dev/null \
            || die "cannot ssh (key-based) to ${W_SSH[$i]}"
        worker_ssh "${W_SSH[$i]}" "docker info >/dev/null 2>&1" \
            || die "${W_SSH[$i]} cannot talk to its docker daemon"
        worker_ssh "${W_SSH[$i]}" "nvidia-smi -L 2>/dev/null | grep -q GB10" \
            || warn "no GB10 GPU visible on ${W_SSH[$i]}"
    done

    if docker inspect -f '{{.State.Running}}' "$CONTAINER_2N_HEAD" 2>/dev/null | grep -q true; then
        die "2-node ${CONTAINER_2N_HEAD} is running — ./start.sh stop first (same GPUs / ports)"
    fi

    others=$(docker ps --format '  {{.Names}}  ({{.Image}})' | grep -v "^  ${CONTAINER_HEAD}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the head:"
        echo "$others" >&2
        warn "GLM-5.3-Flash wants the GB10 to itself — stop GPU containers first"
    fi
    for i in 0 1 2; do
        others=$(worker_ssh "${W_SSH[$i]}" "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null \
            | grep -v "^  ${W_NAME[$i]}" || true)
        if [ -n "$others" ]; then
            warn "other containers on ${W_SSH[$i]}:"
            echo "$others" >&2
        fi
    done

    check_port_free "$PORT" PORT
    check_port_free "$RAY_PORT" RAY_PORT

    local need_kb=$((190 * 1024 * 1024)) avail
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~181 GiB model"
    for i in 0 1 2; do
        avail=$(worker_ssh "${W_SSH[$i]}" "df -Pk '${W_HOME[$i]}' 2>/dev/null" | awk 'NR==2{print $4}' || true)
        [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on ${W_SSH[$i]}"
    done

    log "preflight OK (head=$(hostname) ${HEAD_IP} if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB})"
    for i in 0 1 2; do
        log "  w$((i+1)) ${W_SSH[$i]} ip=${W_IP[$i]} if=${W_IF[$i]} hca=${W_IB[$i]}"
    done
}

# ------------------------------ image --------------------------------------
image_is_local() {
    case "$IMAGE" in
        mia/glm53-flash-spark:*|glm53-flash-sm121:*|localhost/*) return 0 ;;
        */*) return 1 ;;
        *) return 0 ;;
    esac
}

ship_image_to_workers() {
    mkdir -p "$LOGDIR"
    local tar="$LOGDIR/glm53-tp4-image.tar"
    log "docker save ${IMAGE} → $tar (then load on 3 workers) ..."
    docker save -o "$tar" "$IMAGE"
    local i
    for i in 0 1 2; do
        log "  loading on ${W_SSH[$i]} ..."
        worker_ssh "${W_SSH[$i]}" docker load <"$tar"
    done
    rm -f "$tar"
}

ensure_local_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 i
    docker image inspect "$IMAGE" >/dev/null 2>&1 && head_ok=1
    local need_ship=0
    if [ "$head_ok" = "0" ] || [ "${PULL:-0}" = "1" ]; then
        log "building ${IMAGE} (log: $LOGDIR/build-sm121-tp4.log) ..."
        SKIP_WORKER_LOAD=1 IMAGE="$IMAGE" \
            "$SCRIPT_DIR/files/build.sh" \
            >"$LOGDIR/build-sm121-tp4.log" 2>&1 \
            || { tail -n 40 "$LOGDIR/build-sm121-tp4.log" >&2; die "docker build of $IMAGE failed"; }
        need_ship=1
    fi
    for i in 0 1 2; do
        if ! worker_ssh "${W_SSH[$i]}" "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
            need_ship=1
        fi
    done
    if [ "$need_ship" = "1" ] || [ "${PULL:-0}" = "1" ]; then
        ship_image_to_workers
    else
        log "image $IMAGE present on head + 3 workers"
    fi
}

pull_images() {
    mkdir -p "$LOGDIR"
    if image_is_local; then
        ensure_local_image
        return
    fi
    local i need=0
    docker image inspect "$IMAGE" >/dev/null 2>&1 || need=1
    for i in 0 1 2; do
        worker_ssh "${W_SSH[$i]}" "docker image inspect '$IMAGE' >/dev/null 2>&1" || need=1
    done
    if [ "$need" = "0" ] && [ "${PULL:-0}" != "1" ]; then
        log "image $IMAGE present on all 4 nodes (PULL=1 to refresh)"
        return
    fi
    log "pulling ${IMAGE} on head + 3 workers ..."
    docker pull "$IMAGE" >"$LOGDIR/pull-tp4-head.log" 2>&1 &
    local pids=($!)
    for i in 0 1 2; do
        worker_ssh "${W_SSH[$i]}" "docker pull '$IMAGE'" >"$LOGDIR/pull-tp4-w$((i+1)).log" 2>&1 &
        pids+=($!)
    done
    local fail=0 pid
    for pid in "${pids[@]}"; do
        wait "$pid" || fail=1
    done
    [ "$fail" = "0" ] || die "image pull failed — see $LOGDIR/pull-tp4-*.log"
    log "image ready on all 4 nodes"
}

# ---------------------------- weights --------------------------------------
download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    local need=0
    if [ ! -d "$MODEL_PATH" ]; then
        need=1
    elif [ -z "$(find "$MODEL_PATH/snapshots" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
        need=1
    elif [ "${REFRESH_WEIGHTS:-0}" = "1" ]; then
        need=1
    fi
    [ "$need" = "0" ] && { log "weights already present: $MODEL_PATH"; return; }

    local hf
    hf="$(command -v hf || command -v huggingface-cli || true)"
    [ -n "$hf" ] || die "neither 'hf' nor 'huggingface-cli' found — pip install --user -U 'huggingface_hub[cli]'"

    log "downloading ${MODEL} (~181 GiB / 120 shards) into ${HF_CACHE_DIR} ..."
    "$hf" download "$MODEL"
    log "download complete"
}

CHAT_TEMPLATE_URL="${CHAT_TEMPLATE_URL:-https://huggingface.co/${MODEL}/resolve/main/chat_template.jinja}"

refresh_chat_template() {
    local ref="$MODEL_PATH/refs/main" hash dest tmp i
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run download first"
    hash="$(<"$ref")"
    dest="$MODEL_PATH/snapshots/$hash/chat_template.jinja"
    tmp="$(mktemp)"
    log "fetching chat_template.jinja from Hugging Face ..."
    curl -fsSL "$CHAT_TEMPLATE_URL" -o "$tmp" \
        || { rm -f "$tmp"; die "failed to download $CHAT_TEMPLATE_URL"; }
    grep -q 'emit_image' "$tmp" \
        || { rm -f "$tmp"; die "downloaded chat_template.jinja is missing emit_image"; }
    if [ -L "$dest" ]; then
        cat "$tmp" > "$(readlink -f "$dest")"
    else
        mkdir -p "$(dirname "$dest")"
        cat "$tmp" > "$dest"
    fi
    rm -f "$tmp"
    log "chat_template.jinja updated"
    local real
    real="$(readlink -f "$dest")"
    for i in 0 1 2; do
        worker_ssh "${W_SSH[$i]}" "mkdir -p '${W_HOME[$i]}/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$hash'"
        rsync -a "$real" "${W_SSH[$i]}:${W_HOME[$i]}/.cache/huggingface/hub/$MODEL_CACHE_NAME/snapshots/$hash/chat_template.jinja"
    done
    log "chat_template.jinja synced to 3 workers"
}

sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to workers"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    local i
    for i in 0 1 2; do
        log "syncing weights to ${W_SSH[$i]} (~181 GiB first run) ..."
        worker_ssh "${W_SSH[$i]}" "mkdir -p '${W_HOME[$i]}/.cache/huggingface/hub'"
        rsync -a --partial --info=progress2 \
            "$MODEL_PATH/" "${W_SSH[$i]}:${W_HOME[$i]}/.cache/huggingface/hub/${MODEL_CACHE_NAME}/"
        log "  ${W_SSH[$i]} weights in sync"
    done
}

# ------------------------ inner container scripts --------------------------
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start-tp4.sh — runs INSIDE the head container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-tp4-head] $*"; }

if ! command -v ray >/dev/null 2>&1; then
    say "ray not present in image — pip install ray[default]==${RAY_VERSION}"
    pip install -q --no-cache-dir "ray[default]==${RAY_VERSION}"
fi
say "ray $(ray --version 2>&1 | head -1)"

say "starting Ray head on ${HEAD_IP}:${RAY_PORT}"
ray start --head --port "${RAY_PORT}" --node-ip-address "${HEAD_IP}" \
    --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-4294967296}" \
    --dashboard-host 127.0.0.1 --disable-usage-stats

say "waiting for ${CLUSTER_SIZE} Ray node(s) to join ..."
n=0
for i in $(seq 1 "${CLUSTER_WAIT_ITERS}"); do
    n=$(python3 -c 'import ray; ray.init(logging_level="ERROR"); print(sum(nd["Alive"] for nd in ray.nodes()))' 2>/dev/null || echo 0)
    [ "${n:-0}" -ge "${CLUSTER_SIZE}" ] && break
    sleep 5
done
if [ "${n:-0}" -lt "${CLUSTER_SIZE}" ]; then
    say "FATAL: Ray cluster stuck at ${n:-0}/${CLUSTER_SIZE} node(s)"
    ray status || true
    exit 1
fi
say "Ray cluster ready (${n} node(s))"
say "SM121 arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset} FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-unset}"

ARGS=(
    --tensor-parallel-size "${TP}"
    --distributed-executor-backend ray
    --tool-call-parser glm47
    --enable-auto-tool-choice
    --reasoning-parser glm45
    --host 0.0.0.0
    --port "${PORT}"
)
if [ "${TRUST_REMOTE_CODE:-1}" = "1" ]; then
    ARGS+=(--trust-remote-code)
fi
if [ "${MTP_TOKENS:-0}" != "0" ]; then
    ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS}}")
fi
[ -n "${MAX_MODEL_LEN:-}" ] && ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
[ -n "${GPU_MEM_UTIL:-}" ]  && ARGS+=(--gpu-memory-utilization "${GPU_MEM_UTIL}")
[ -n "${BLOCK_SIZE:-}" ]    && ARGS+=(--block-size "${BLOCK_SIZE}")
[ -n "${MAX_NUM_SEQS:-}" ]  && ARGS+=(--max-num-seqs "${MAX_NUM_SEQS}")
if [ -n "${KV_CACHE_DTYPE:-}" ]; then
    ARGS+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    say "kv-cache-dtype=${KV_CACHE_DTYPE}"
fi
if [ -n "${KV_CACHE_MEMORY:-}" ]; then
    ARGS+=(--kv-cache-memory "${KV_CACHE_MEMORY}")
    say "kv-cache-memory=${KV_CACHE_MEMORY}"
fi
[ -n "${LIMIT_MM:-}" ]      && ARGS+=(--limit-mm-per-prompt "${LIMIT_MM}")
if [ "${SKIP_MM_PROFILING:-1}" = "1" ]; then
    ARGS+=(--skip-mm-profiling)
    say "skip-mm-profiling: image+video serving on, no max-size MM dummy forward at init"
fi
ARGS+=(--chat-template "${MODEL_DIR}/chat_template.jinja")
say "chat-template: ${MODEL_DIR}/chat_template.jinja"
if [ "${ENFORCE_EAGER:-1}" = "1" ]; then
    ARGS+=(--enforce-eager)
    say "enforce-eager: no CUDA graph capture at init"
fi

if [ "${MOE_MODE:-native}" = "marlin" ]; then
    ARGS+=(--moe-backend marlin --enforce-eager)
    say "MoE backend: marlin (enforce-eager)"
else
    say "MoE backend: native NVFP4 kernels"
fi

if [ -n "${EXTRA_ARGS:-}" ]; then
    # shellcheck disable=SC2206
    EXTRA=(${EXTRA_ARGS})
    ARGS+=("${EXTRA[@]}")
fi

if [ ! -f "${MODEL_DIR}/processor_config.json" ]; then
    say "FATAL: ${MODEL_DIR}/processor_config.json missing — Glm5NextProcessor.from_pretrained() opens this as a local file (repo ids fail)."
    ls -la "${MODEL_DIR}" 2>/dev/null | head -n 30 || true
    exit 1
fi

say "launching: vllm serve ${MODEL_DIR} ${ARGS[*]} (served-model-name=${MODEL})"
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}" --served-model-name "${MODEL}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start-tp4.sh — runs INSIDE a worker container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-tp4-worker] $*"; }

if ! command -v ray >/dev/null 2>&1; then
    say "ray not present in image — pip install ray[default]==${RAY_VERSION}"
    pip install -q --no-cache-dir "ray[default]==${RAY_VERSION}"
fi
say "ray $(ray --version 2>&1 | head -1)"

say "joining Ray cluster at ${HEAD_IP}:${RAY_PORT} as ${WORKER_IP}"
for i in $(seq 1 "${CLUSTER_WAIT_ITERS}"); do
    if ray start --address "${HEAD_IP}:${RAY_PORT}" --node-ip-address "${WORKER_IP}" \
        --object-store-memory "${RAY_OBJECT_STORE_MEMORY:-4294967296}" \
        --disable-usage-stats --block; then
        exit 0
    fi
    say "head not reachable yet (${HEAD_IP}:${RAY_PORT}), retrying in 5s ..."
    sleep 5
done
say "FATAL: could not join Ray cluster at ${HEAD_IP}:${RAY_PORT}"
exit 1
EOF
    chmod +x "$HEAD_SCRIPT" "$WORKER_SCRIPT"
}

# ------------------------------- launch ------------------------------------
stop_tp4_containers() {
    local i
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    for i in 0 1 2; do
        worker_ssh "${W_SSH[$i]}" "docker rm -f '${W_NAME[$i]}'" >/dev/null 2>&1 || true
    done
}

launch_cluster() {
    local moe_mode="$1"
    local i e v val

    stop_tp4_containers

    local -a nccl_common=(
        -e NCCL_IB_DISABLE=0
        -e NCCL_IB_ROCE_VERSION_NUM=2
        -e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX"
        -e NCCL_NET=IB
        -e NCCL_NET_PLUGIN=none
        -e NCCL_NVLS_ENABLE=0
        -e NCCL_CUMEM_ENABLE=0
        -e NCCL_IB_MERGE_NICS=0
        -e "NCCL_CROSS_NIC=$NCCL_CROSS_NIC"
        -e NCCL_IGNORE_CPU_AFFINITY=1
        -e "NCCL_DEBUG=$NCCL_DEBUG"
        -e HF_HUB_OFFLINE=1
        -e TRANSFORMERS_OFFLINE=1
        -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
        -e "FLASHINFER_CUDA_ARCH_LIST=$FLASHINFER_CUDA_ARCH_LIST"
        -e FLASHINFER_DISABLE_VERSION_CHECK=1
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    )
    local worker_nccl=""
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=()
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    local -a head_env=()
    local worker_passthru=""
    for v in HF_HUB_OFFLINE; do
        val="${!v:-}"
        if [ -n "$val" ]; then
            head_env+=(-e "$v=$val")
            worker_passthru+=" -e $v='$val'"
        fi
    done

    for i in 0 1 2; do
        scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${W_SSH[$i]}:/tmp/${W_NAME[$i]}.sh"
        local worker_preload=""
        if [ "$USE_HOST_NCCL" = "1" ]; then
            if worker_ssh "${W_SSH[$i]}" "test -f '${W_NCCL[$i]}/$NCCL_SO_NAME'"; then
                worker_preload="-v '${W_NCCL[$i]}:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
                log "w$((i+1)): LD_PRELOAD $NCCL_SO_NAME"
            else
                warn "w$((i+1)): ${W_NCCL[$i]}/$NCCL_SO_NAME missing — using image NCCL"
            fi
        fi
        log "starting ${W_NAME[$i]} on ${W_SSH[$i]} (MoE ${moe_mode}; if=${W_IF[$i]} hca=${W_IB[$i]}) ..."
        worker_ssh "${W_SSH[$i]}" "docker run -d --name '${W_NAME[$i]}' \
            --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
            --device /dev/infiniband --cap-add IPC_LOCK \
            --ulimit memlock=-1 --ulimit stack=67108864 \
            -v '${W_HOME[$i]}/.cache/huggingface:/root/.cache/huggingface' \
            -v '$CACHE_VOLUME:/root/.cache' \
            -v '/tmp/${W_NAME[$i]}.sh:/start.sh:ro' \
            ${worker_preload} \
            ${worker_nccl} \
            -e NCCL_SOCKET_IFNAME='${W_IF[$i]}' \
            -e GLOO_SOCKET_IFNAME='${W_IF[$i]}' \
            -e NCCL_IB_HCA='${W_IB[$i]}' \
            -e HEAD_IP='$HEAD_IP' -e RAY_PORT='$RAY_PORT' -e WORKER_IP='${W_IP[$i]}' \
            -e RAY_VERSION='$RAY_VERSION' \
            -e CLUSTER_WAIT_ITERS=$CLUSTER_WAIT_ITERS \
            -e VLLM_HOST_IP='${W_IP[$i]}' \
            -e RAY_OBJECT_STORE_MEMORY='$RAY_OBJECT_STORE_MEMORY' \
            -e VLLM_ENGINE_READY_TIMEOUT_S='$READY_TIMEOUT' \
            ${worker_passthru} \
            --entrypoint bash '$IMAGE' /start.sh" >/dev/null
    done

    log "starting head ${CONTAINER_HEAD} (Ray head + vLLM; if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
    docker run -d --name "$CONTAINER_HEAD" \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
        -v "$CACHE_VOLUME:/root/.cache" \
        -v "$HEAD_SCRIPT:/start.sh:ro" \
        "${head_preload[@]}" \
        "${nccl_common[@]}" \
        -e NCCL_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e GLOO_SOCKET_IFNAME="$HEAD_CX7_IF" \
        -e NCCL_IB_HCA="$HEAD_CX7_IB" \
        -e HEAD_IP="$HEAD_IP" -e RAY_PORT="$RAY_PORT" \
        -e RAY_VERSION="$RAY_VERSION" \
        -e CLUSTER_SIZE="$TP" -e CLUSTER_WAIT_ITERS="$CLUSTER_WAIT_ITERS" \
        -e MODEL="$MODEL" -e MODEL_DIR="$MODEL_DIR" -e TP="$TP" -e PORT="$PORT" -e MTP_TOKENS="$MTP_TOKENS" \
        -e MAX_MODEL_LEN="$MAX_MODEL_LEN" -e GPU_MEM_UTIL="$GPU_MEM_UTIL" \
        -e BLOCK_SIZE="$BLOCK_SIZE" -e MAX_NUM_SEQS="$MAX_NUM_SEQS" \
        -e KV_CACHE_DTYPE="$KV_CACHE_DTYPE" -e KV_CACHE_MEMORY="$KV_CACHE_MEMORY" \
        -e TRUST_REMOTE_CODE="$TRUST_REMOTE_CODE" \
        -e LIMIT_MM="$LIMIT_MM" -e SKIP_MM_PROFILING="$SKIP_MM_PROFILING" \
        -e ENFORCE_EAGER="$ENFORCE_EAGER" \
        -e MOE_MODE="$moe_mode" -e EXTRA_ARGS="${EXTRA_ARGS:-}" \
        -e VLLM_HOST_IP="$HEAD_IP" \
        -e RAY_OBJECT_STORE_MEMORY="$RAY_OBJECT_STORE_MEMORY" \
        -e VLLM_ENGINE_READY_TIMEOUT_S="$READY_TIMEOUT" \
        "${head_env[@]}" \
        --entrypoint bash "$IMAGE" /start.sh >/dev/null

    log "containers up — head=${CONTAINER_HEAD}, workers=${W_NAME[*]}"
}

# ---------------------------- health wait ----------------------------------
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (320B MoE init is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    trap '_stop_logtail; warn "interrupted — TP4 containers keep running ('"'"'./start-tp4.sh logs'"'"' / '"'"'./start-tp4.sh stop'"'"')"; exit 130' INT
    docker logs -f --tail 0 "$CONTAINER_HEAD" 2>&1 &
    logpid=$!

    local elapsed=0 healthy=0 exited=0
    while [ "$elapsed" -lt "$READY_TIMEOUT" ]; do
        if curl -fsS -m 5 "$url" >/dev/null 2>&1; then healthy=1; break; fi
        if ! docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            log "head container exited during startup"
            exited=1; break
        fi
        sleep 10; elapsed=$((elapsed + 10))
    done

    _stop_logtail
    trap 'warn "interrupted — TP4 containers keep running ('"'"'./start-tp4.sh logs'"'"' / '"'"'./start-tp4.sh stop'"'"')"; exit 130' INT

    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "head container exited after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

collect_failure_logs() {
    local tag="$1" i
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/tp4-head-${tag}.log" 2>&1 || true
    for i in 0 1 2; do
        {
            echo "### docker logs ${W_NAME[$i]} on ${W_SSH[$i]}"
            worker_ssh "${W_SSH[$i]}" "docker logs '${W_NAME[$i]}' 2>&1" || true
            echo
            echo "### Ray session logs (kernel-image filter)"
            worker_ssh "${W_SSH[$i]}" "docker exec '${W_NAME[$i]}' sh -c 'grep -rhE \"$KERNEL_ERR_PAT\" /tmp/ray/session_latest/logs/ 2>/dev/null | head -n 40'" || true
        } >"$LOGDIR/tp4-w$((i+1))-${tag}.log" 2>&1 || true
    done
}

on_ready() {
    local mode="$1" how="$2"
    log "======================================================================"
    log "EXPERIMENTAL TP=4 GLM-5.3-Flash-NVFP4 is UP (${how}; MoE backend: ${mode})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN ips: $(hostname -I))"
    log "  model name : ${MODEL}"
    log "  layout     : TP=4 Ray  (head + 3 workers)"
    log "  features   : tools=glm47+auto, reasoning=glm45, MTP spec-decode (${MTP_TOKENS} tokens)"
    [ "$mode" = "marlin" ] && \
    log "  NOTE       : marlin dequant-to-FP16 fallback"
    log "  manage     : ./start-tp4.sh status | logs | logs worker 2 | stop"
    log "======================================================================"
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — TP4 containers keep running"; exit 130' INT
        log "detached from logs; server still running"
    fi
}

# ------------------------------- start -------------------------------------
start() {
    banner
    load_workers
    preflight
    pull_images
    download_weights
    refresh_chat_template
    sync_weights
    write_inner_scripts

    MODEL_DIR="$(resolve_model_dir)"
    log "model load path (in-container): ${MODEL_DIR}"

    local mode="native"
    case "$MOE_BACKEND" in
        native) mode="native" ;;
        marlin) mode="marlin" ;;
        auto)   mode="native" ;;
        *) die "MOE_BACKEND must be auto | native | marlin (got: ${MOE_BACKEND})" ;;
    esac

    log "config: image=${IMAGE} tp=${TP} first-attempt=${mode} mtp=${MTP_TOKENS}" \
        "max-len=${MAX_MODEL_LEN:-<model default>} gpu-util=${GPU_MEM_UTIL} block=${BLOCK_SIZE} kv=${KV_CACHE_DTYPE} port=${PORT}"

    launch_cluster "$mode"
    if wait_for_health; then
        on_ready "$mode" "first attempt"
        return
    fi

    collect_failure_logs "$mode"
    echo "---- last 60 lines of head log ($LOGDIR/tp4-head-${mode}.log) ----"
    tail -n 60 "$LOGDIR/tp4-head-${mode}.log" || true

    if [ "$MOE_BACKEND" = "auto" ] && [ "$mode" = "native" ] \
       && grep -qE "$KERNEL_ERR_PAT" "$LOGDIR"/tp4-*-native.log 2>/dev/null; then
        warn "cudaErrorNoKernelImageForDevice from the native FP4 MoE kernels —"
        warn "falling back to marlin MoE backend"
        launch_cluster "marlin"
        if wait_for_health; then
            on_ready "marlin" "after sm_121 fallback"
            return
        fi
        collect_failure_logs "marlin"
        echo "---- last 60 lines of head log ($LOGDIR/tp4-head-marlin.log) ----"
        tail -n 60 "$LOGDIR/tp4-head-marlin.log" || true
        die "server failed in marlin mode too — full logs in $LOGDIR/"
    fi
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    banner
    load_workers
    log "stopping ${CONTAINER_HEAD} ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    local i
    for i in 0 1 2; do
        log "stopping ${W_NAME[$i]} on ${W_SSH[$i]} ..."
        worker_ssh "${W_SSH[$i]}" "docker rm -f '${W_NAME[$i]}'" >/dev/null 2>&1 \
            || log "  (no worker container on ${W_SSH[$i]})"
    done
    log "TP4 stopped. 2-node ./start.sh containers were not touched."
}

# ------------------------------ status -------------------------------------
status() {
    banner
    load_workers
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
        docker exec "$CONTAINER_HEAD" ray status 2>/dev/null | sed 's/^/  /' | head -n 30 || true
    fi
    local i
    for i in 0 1 2; do
        log "worker $((i+1)) (${W_NAME[$i]} on ${W_SSH[$i]}):"
        worker_ssh "${W_SSH[$i]}" "docker ps -a --filter name=${W_NAME[$i]} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
            || log "  (worker unreachable)"
    done
}

# ------------------------------- logs --------------------------------------
logs() {
    load_workers
    case "${1:-head}" in
        worker)
            local n="${2:-1}"
            [[ "$n" =~ ^[123]$ ]] || die "logs worker takes 1, 2, or 3 (got: $n)"
            local i=$((n-1))
            log "following ${W_NAME[$i]} on ${W_SSH[$i]} ..."
            trap '' INT
            worker_ssh "${W_SSH[$i]}" "docker logs -f --tail 100 '${W_NAME[$i]}'" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
        head|*)
            log "following head logs (driver + API server) ..."
            trap '' INT
            docker logs -f --tail 100 "$CONTAINER_HEAD" || true
            trap 'warn "interrupted"; exit 130' INT
            ;;
    esac
}

# ------------------------------- main --------------------------------------
main() {
    local cmd="${1:-start}"
    case "$cmd" in
        start)   shift || true; start ;;
        stop)    stop ;;
        restart) stop; start ;;
        status)  status ;;
        logs)    shift || true; logs "$@" ;;
        -h|--help|help) usage ;;
        *) usage; exit 1 ;;
    esac
}

main "$@"
