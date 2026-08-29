#!/usr/bin/env bash
# ============================================================================
# start.sh — Mia's Spark runtime for GLM-5.3-Flash-NVFP4
# ============================================================================
#
# We serve LibertAIDAI/GLM-5.3-Flash-NVFP4 (the checkpoint) on this 2× DGX
# Spark (GB10 / SM121) kit: multimodal, Ray TP=2, OpenAI API on :8888.
#
#   head   : this machine (spark1, 10.0.0.1) — Ray head + vLLM API :8888
#   worker : spark2 (10.0.0.2, zurih)        — Ray worker, 1× GB10
#   layout : tensor-parallel-size 2 (one GPU per node, Ray executor)
#
# What we do:
#   1. preflight  — docker/ssh/disk on both nodes
#   2. image      — mia/glm53-flash-spark:mm-ray-v1 (Ray + MM on our
#                   glm53-flash-sm121:v8 kernel layer). files/build.sh if missing.
#   3. download   — weights into local HF cache if missing (~181 GiB)
#   4. sync       — rsync that cache to zurih@10.0.0.2 (each rank loads local disk)
#   5. launch     — Ray worker on spark2, Ray head + `vllm serve` on spark1
#                   (both --network host --ipc=host)
#   6. wait       — poll /health up to READY_TIMEOUT (320B MoE init is slow;
#                   VLLM_ENGINE_READY_TIMEOUT_S=3600)
#   7. sm_121 net — if native NVFP4 MoE dies with cudaErrorNoKernelImageForDevice,
#                   relaunch with --moe-backend marlin --enforce-eager
#
# Model-card flags we honor:
#   --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
#   --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
#   VLLM_ENGINE_READY_TIMEOUT_S=3600
#
# Usage:
#   ./start.sh                    start (download/sync/launch) — default
#   ./start.sh stop               stop both nodes
#   ./start.sh restart            stop + start
#   ./start.sh status             containers, API health, Ray cluster
#   ./start.sh logs               follow head logs (driver + API server)
#   ./start.sh logs worker        follow worker container logs
#
# Handy env overrides (all optional):
#   MOE_BACKEND=marlin            go straight to the marlin fallback
#   MOE_BACKEND=native            never fall back to marlin
#   MAX_MODEL_LEN=1048576         model is 1M-native; default here is 262144
#   GPU_MEM_UTIL=0.84             vLLM memory budget (GB10 UMA; 0.90 fails free-mem check)
#                                 profiles KV unless KV_CACHE_MEMORY is set
#   KV_CACHE_MEMORY=<bytes>       optional --kv-cache-memory pin (UMA OOM only)
#   MTP_TOKENS=4                  MTP speculative tokens
#   PORT=8888                     API port on this machine
#   EXTRA_ARGS='--max-num-seqs 64'   extra flags appended to `vllm serve`
#   NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=...   passed through to both containers
#   HF_HUB_OFFLINE=1              skip HF etag checks at startup
#   REFRESH_WEIGHTS=1  SKIP_SYNC=1  SKIP_DOWNLOAD=1  PULL=1  TAIL=0
# ============================================================================
set -euo pipefail

# ----------------------------- configuration -------------------------------
MODEL="LibertAIDAI/GLM-5.3-Flash-NVFP4"
MODEL_CACHE_NAME="models--LibertAIDAI--GLM-5.3-Flash-NVFP4"
IMAGE="${IMAGE:-mia/glm53-flash-spark:mm-ray-v1}"
# IMAGE is this kit's serving tag (Ray TP2 + MM + :8888). Kernel layer
# glm53-flash-sm121:v8 is our local FROM of glm53-flash-arm64-cu130 after
# files/glm53-flash_SM121.py sm90. Ray in the image makes the inner pip a no-op.
# Do not bind-mount glm53-flash_SM121.py at runtime (including unused sm120).
RAY_VERSION="${RAY_VERSION:-2.58.0}"

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER_SSH="${WORKER_SSH:-zurih@10.0.0.2}"
WORKER_IP="${WORKER_IP:-10.0.0.2}"
WORKER_HOME="${WORKER_HOME:-/home/zurih}"

# Direct CX7 QSFP link (spark1 rocep1s0f1 ↔ spark2 rocep1s0f0). Ray can use the
# 10.0.0.1/32 loopback aliases; NCCL cannot — without these pins it busy-waits
# forever in ncclCommInitRank (~100% CPU, ~0.7 GiB GPU, no /health).
HEAD_CX7_IF="${HEAD_CX7_IF:-enp1s0f1np1}"
WORKER_CX7_IF="${WORKER_CX7_IF:-enp1s0f0np0}"
HEAD_CX7_IB="${HEAD_CX7_IB:-rocep1s0f1}"
WORKER_CX7_IB="${WORKER_CX7_IB:-rocep1s0f0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-0}"
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
WORKER_NCCL_HOST_DIR="${WORKER_NCCL_HOST_DIR:-$WORKER_HOME/nccl-2.30.7}"
NCCL_SO_NAME="${NCCL_SO_NAME:-libnccl.so.2.30.7}"
USE_HOST_NCCL="${USE_HOST_NCCL:-1}"
# GB10 is UMA: Ray's default object store (~30% of RAM) steals from the GPU
# budget and vLLM then refuses to start (free < gpu-memory-utilization).
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-4294967296}"  # 4 GiB

TP="${TP:-2}"                            # 1 GB10 per node x 2 nodes
RAY_PORT="${RAY_PORT:-6379}"
PORT="${PORT:-8888}"

MTP_TOKENS="${MTP_TOKENS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
# 0.86 with the pin unset let TP0 reserve 11.37 GiB KV (TP1 7.53 GiB /
# 924k tokens). Engine init finished, then RayWorkerProc rank 0 died
# during API MM warmup (NVRM NV_ERR_NO_MEMORY on the head UMA). 0.84
# leaves ~2.4 GiB headroom for that extra alloc without re-pinning KV.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.84}"
# This image prefers FLASHINFER_MLA_SPARSE_SM90 + FA2 on GB10. Our
# checkpoint is NoPE (pe_dim=0). Stock glm53-flash only lists SM120
# packed fp8_ds_mla — the unused path, not what we bake.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.1a}"
FLASHINFER_CUDA_ARCH_LIST="${FLASHINFER_CUDA_ARCH_LIST:-12.1a}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"      # skip CUDA-graph capture during init (GB10 UMA)
# Keep image+video enabled. Do NOT profile a max-size video at init — that
# dummy forward OOMs the leftover UMA after ~91 GiB of weights (rank 1 dies).
# --skip-mm-profiling still serves images/videos; KV budget is estimated from
# the language backbone. Override LIMIT_MM if you need more items per prompt.
LIMIT_MM="${LIMIT_MM:-{\"image\":4,\"video\":1}}"
SKIP_MM_PROFILING="${SKIP_MM_PROFILING:-1}"
MOE_BACKEND="${MOE_BACKEND:-marlin}"      # auto | native | marlin
BLOCK_SIZE="${BLOCK_SIZE:-2304}"         # DeepGEMM arch-12 fp8 paged-MQA: 64-entry pool pages
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}"
KV_CACHE_MEMORY="${KV_CACHE_MEMORY:-}"  # empty: omit --kv-cache-memory so GPU_MEM_UTIL profiles KV; set bytes if UMA OOM
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"

READY_TIMEOUT="${READY_TIMEOUT:-3600}"          # = VLLM_ENGINE_READY_TIMEOUT_S
CLUSTER_WAIT_ITERS="${CLUSTER_WAIT_ITERS:-120}" # x5s = 10 min to join Ray

CONTAINER_HEAD="glm53-flash-head"
CONTAINER_WORKER="glm53-flash-worker"
CACHE_VOLUME="${CACHE_VOLUME:-glm53-flash-cache-sm121}"  # FA2 JIT; do not reuse SM120 cubin cache

HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
MODEL_PATH="$HF_CACHE_DIR/hub/$MODEL_CACHE_NAME"
WORKER_CACHE_DIR="$WORKER_HOME/.cache/huggingface"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
LOGDIR="$SCRIPT_DIR/logs"
HEAD_SCRIPT="$SCRIPT_DIR/.glm53-head.inner.sh"
WORKER_SCRIPT="$SCRIPT_DIR/.glm53-worker.inner.sh"
KERNEL_ERR_PAT='NoKernelImageForDevice|no kernel image is available'

# ------------------------------- helpers -----------------------------------
log()  { printf '\033[1;36m[glm53]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[glm53]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[glm53]\033[0m ERROR: %s\n' "$*" >&2; exit 1; }

worker_ssh() { ssh -o BatchMode=yes -o ConnectTimeout=15 "$WORKER_SSH" "$@"; }

usage() { sed -n '2,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

# Resolve the HF-cache snapshot directory into the IN-CONTAINER path that
# vllm serve should load from. We pass this instead of the repo id because
# vllm's custom Glm5NextProcessor.from_pretrained() does a raw
# `open(os.path.join(model_path, "processor_config.json"))` and does NOT
# resolve HF repo ids through the cache — so a repo-id string makes it look
# for `./LibertAIDAI/GLM-5.3-Flash-NVFP4/processor_config.json` (CWD-relative)
# and crash with FileNotFoundError. The snapshot dir has processor_config.json.
# Container mount is -v "$HF_CACHE_DIR:/root/.cache/huggingface", so the
# in-container root is /root/.cache/huggingface.
resolve_model_dir() {
    local ref="$MODEL_PATH/refs/main" hash
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run download first"
    hash="$(<"$ref")"
    [ -n "$hash" ] || die "empty refs/main at $ref"
    local dir="$MODEL_PATH/snapshots/$hash"
    [ -f "$dir/processor_config.json" ] \
        || die "processor_config.json missing in $dir — re-run with REFRESH_WEIGHTS=1"
    printf '/root/.cache/huggingface/hub/%s/snapshots/%s' "$MODEL_CACHE_NAME" "$hash"
}

# die early if a port we need is already bound (vLLM would fail to bind ~40 min in)
check_port_free() {
    local port="$1" envname="$2"
    command -v ss >/dev/null 2>&1 || return 0
    if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]${port}\$"; then
        if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
            die "port ${port} is held by ${CONTAINER_HEAD} — use './start.sh restart' or './start.sh stop' first"
        fi
        die "port ${port} is already in use by another service — stop it or rerun with ${envname}=<free-port>"
    fi
}

trap 'warn "interrupted — containers keep running in the background ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

# ------------------------------ preflight ----------------------------------
preflight() {
    command -v docker  >/dev/null 2>&1 || die "docker not found on head"
    command -v curl    >/dev/null 2>&1 || die "curl not found on head"
    command -v rsync   >/dev/null 2>&1 || die "rsync not found on head"
    command -v python3 >/dev/null 2>&1 || die "python3 not found on head"
    docker info >/dev/null 2>&1 || die "cannot talk to docker daemon on head"

    ip -4 addr show 2>/dev/null | grep -q "inet ${HEAD_IP}/" \
        || die "HEAD_IP=${HEAD_IP} is not assigned on this host — set HEAD_IP=<ip the worker can reach>"

    log "checking worker ${WORKER_SSH} ..."
    worker_ssh true 2>/dev/null \
        || die "cannot ssh (key-based) to ${WORKER_SSH} — set up passwordless ssh first"
    worker_ssh "docker info >/dev/null 2>&1" \
        || die "worker cannot talk to its docker daemon (docker group?)"
    worker_ssh "nvidia-smi -L 2>/dev/null | grep -q GB10" \
        || warn "no GB10 GPU visible on worker"

    [ "$TP" = "2" ] || warn "TP=${TP} on a 2x1-GPU cluster — expected TP=2"

    # GLM-5.3-Flash-NVFP4 needs ~90 GiB of the 128 GiB on each GB10 — it cannot
    # coexist with another served model on either node.
    local others
    others=$(docker ps --format '  {{.Names}}  ({{.Image}})' | grep -v "^  ${CONTAINER_HEAD}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the head:"
        echo "$others" >&2
        warn "GLM-5.3-Flash needs ~90 GiB of each GB10 — stop GPU containers first"
    fi
    others=$(worker_ssh "docker ps --format '  {{.Names}}  ({{.Image}})'" 2>/dev/null | grep -v "^  ${CONTAINER_WORKER}" || true)
    if [ -n "$others" ]; then
        warn "other containers are running on the worker:"
        echo "$others" >&2
        warn "GLM-5.3-Flash needs ~90 GiB of each GB10 — stop GPU containers first"
    fi

    # hard port checks — failing to bind would only surface ~40 min later
    check_port_free "$PORT" PORT
    check_port_free "$RAY_PORT" RAY_PORT

    # disk space: model is ~181 GiB
    local need_kb=$((190 * 1024 * 1024))
    local avail
    avail=$(df -Pk "$HF_CACHE_DIR" 2>/dev/null | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on head for a ~181 GiB model"
    avail=$(worker_ssh "df -Pk '$WORKER_HOME' 2>/dev/null" | awk 'NR==2{print $4}' || true)
    [ "${avail:-0}" -ge "$need_kb" ] || warn "only $((avail/1024/1024)) GiB free on worker for a ~181 GiB model"

    log "preflight OK (head=$(hostname) ${HEAD_IP}, worker=${WORKER_SSH})"
}

# ------------------------------ image pull ---------------------------------
# mia/glm53-flash-spark:* and glm53-flash-sm121:* are local (files/build.sh
# + save|load). Other org/name tags still docker-pull from a registry.
image_is_local() {
    case "$IMAGE" in
        mia/glm53-flash-spark:*|glm53-flash-sm121:*|localhost/*) return 0 ;;
        */*) return 1 ;;
        *) return 0 ;;
    esac
}

ship_image_to_worker() {
    log "shipping ${IMAGE} to worker via docker save | ssh docker load ..."
    docker save "$IMAGE" | worker_ssh docker load >/dev/null
}

ensure_local_image() {
    mkdir -p "$LOGDIR"
    local head_ok=0 worker_ok=0
    docker image inspect "$IMAGE" >/dev/null 2>&1 && head_ok=1
    worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1" && worker_ok=1
    if [ "$head_ok" = "1" ] && [ "$worker_ok" = "1" ] && [ "${PULL:-0}" != "1" ]; then
        log "image $IMAGE present on both nodes"
        return
    fi
    if [ "$head_ok" = "0" ] || [ "${PULL:-0}" = "1" ]; then
        # Kernel: docker build -f files/Dockerfile -t glm53-flash-sm121:v8 "$SCRIPT_DIR/files"
        # mm-ray:  docker build -f files/Dockerfile.mm-ray -t ... "$SCRIPT_DIR/files"
        log "building ${IMAGE} (kernel v8 if needed + mm-ray layer; log: $LOGDIR/build-sm121.log) ..."
        SKIP_WORKER_LOAD=1 IMAGE="$IMAGE" \
            "$SCRIPT_DIR/files/build.sh" \
            >"$LOGDIR/build-sm121.log" 2>&1 \
            || { tail -n 40 "$LOGDIR/build-sm121.log" >&2; die "docker build of $IMAGE failed"; }
        worker_ok=0
    fi
    if [ "$worker_ok" = "0" ] || [ "${PULL:-0}" = "1" ]; then
        ship_image_to_worker
    fi
    log "image ready on both nodes"
}

pull_images() {
    mkdir -p "$LOGDIR"
    if image_is_local; then
        ensure_local_image
        return
    fi
    if [ "${PULL:-0}" != "1" ] \
       && docker image inspect "$IMAGE" >/dev/null 2>&1 \
       && worker_ssh "docker image inspect '$IMAGE' >/dev/null 2>&1"; then
        log "image $IMAGE present on both nodes (PULL=1 to refresh)"
        return
    fi
    log "pulling ${IMAGE} on head and worker in parallel (logs: $LOGDIR/pull-*.log) ..."
    docker pull "$IMAGE" >"$LOGDIR/pull-head.log" 2>&1 &
    local p1=$!
    worker_ssh "docker pull '$IMAGE'" >"$LOGDIR/pull-worker.log" 2>&1 &
    local p2=$!
    local fail=0
    wait "$p1" || { warn "docker pull failed on head:"; tail -n 20 "$LOGDIR/pull-head.log" >&2; fail=1; }
    wait "$p2" || { warn "docker pull failed on worker:"; tail -n 20 "$LOGDIR/pull-worker.log" >&2; fail=1; }
    [ "$fail" = "0" ] || die "image pull failed"
    log "image ready on both nodes"
}

# ---------------------------- weight download ------------------------------
download_weights() {
    [ "${SKIP_DOWNLOAD:-0}" = "1" ] && { log "SKIP_DOWNLOAD=1 — skipping download check"; return; }
    local need=0
    if [ ! -d "$MODEL_PATH" ]; then
        need=1
    elif [ -z "$(find "$MODEL_PATH/snapshots" -name '*.safetensors' -print -quit 2>/dev/null)" ]; then
        # NOTE: no '-type f' — HF cache snapshot entries are symlinks into blobs/
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

# Latest HF chat_template.jinja (multimodal emit_image/emit_video). The cached
# snapshot still has the older file that emits a "no multi-modal input" reminder.
CHAT_TEMPLATE_URL="${CHAT_TEMPLATE_URL:-https://huggingface.co/${MODEL}/resolve/main/chat_template.jinja}"

refresh_chat_template() {
    local ref="$MODEL_PATH/refs/main" hash dest tmp
    [ -f "$ref" ] || die "no refs/main under $MODEL_PATH — run download first"
    hash="$(<"$ref")"
    dest="$MODEL_PATH/snapshots/$hash/chat_template.jinja"
    tmp="$(mktemp)"
    log "fetching chat_template.jinja from Hugging Face ..."
    curl -fsSL "$CHAT_TEMPLATE_URL" -o "$tmp" \
        || { rm -f "$tmp"; die "failed to download $CHAT_TEMPLATE_URL"; }
    grep -q 'emit_image' "$tmp" \
        || { rm -f "$tmp"; die "downloaded chat_template.jinja is missing emit_image — unexpected file"; }
    if [ -L "$dest" ]; then
        cat "$tmp" > "$(readlink -f "$dest")"
    else
        mkdir -p "$(dirname "$dest")"
        cat "$tmp" > "$dest"
    fi
    rm -f "$tmp"
    log "chat_template.jinja updated ($(wc -c < "$dest" | tr -d ' ') bytes, emit_image+emit_video)"
    local real
    real="$(readlink -f "$dest")"
    worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub/$MODEL_CACHE_NAME/snapshots/$hash'"
    rsync -a "$real" "${WORKER_SSH}:$WORKER_CACHE_DIR/hub/$MODEL_CACHE_NAME/snapshots/$hash/chat_template.jinja"
    log "chat_template.jinja synced to worker"
}

# ------------------------------ weight sync --------------------------------
sync_weights() {
    [ "${SKIP_SYNC:-0}" = "1" ] && { log "SKIP_SYNC=1 — not syncing to worker"; return; }
    [ -d "$MODEL_PATH" ] || die "weights missing at $MODEL_PATH — run without SKIP_DOWNLOAD first"
    log "syncing weights to worker (first run moves ~181 GiB over the p2p link) ..."
    worker_ssh "mkdir -p '$WORKER_CACHE_DIR/hub'"
    rsync -a --partial --info=progress2 \
        "$MODEL_PATH/" "${WORKER_SSH}:${WORKER_CACHE_DIR}/hub/${MODEL_CACHE_NAME}/"
    log "worker weights in sync"
}

# ------------------------ inner container scripts --------------------------
# Written to disk and mounted into the containers; config comes in via -e env.
write_inner_scripts() {
    cat > "$HEAD_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start.sh — runs INSIDE the head container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-head] $*"; }

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
say "SM121 arch: TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-unset} FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST:-unset} (patched image: SM90 FA2 NoPE, not SM120 fp8_ds_mla)"

# ---- model-card recipe ----
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
    # GB10/sm_121 fallback: dequant-to-FP16 marlin MoE kernels (known-good)
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
# MODEL_DIR must be the snapshot filesystem path. Glm5NextProcessor.from_pretrained()
# does open(os.path.join(model_config.model, "processor_config.json")) and does not
# resolve HF repo ids. --served-model-name keeps the public API name as the repo id.
exec vllm serve "${MODEL_DIR}" "${ARGS[@]}" --served-model-name "${MODEL}"
EOF

    cat > "$WORKER_SCRIPT" <<'EOF'
#!/bin/bash
# generated by start.sh — runs INSIDE the worker container as: bash /start.sh
set -euo pipefail
say() { echo "[glm53-worker] $*"; }

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
launch_cluster() {
    local moe_mode="$1"

    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || true
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 || true

    scp -q -o BatchMode=yes "$WORKER_SCRIPT" "${WORKER_SSH}:/tmp/${CONTAINER_WORKER}.sh"

    # Shared NCCL/RoCE pins (same as the working 2× Spark Qwen launch on this kit).
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
    local worker_nccl="" e
    for e in "${nccl_common[@]}"; do
        [ "$e" = "-e" ] && continue
        worker_nccl+=" -e $e"
    done

    local -a head_preload=() worker_preload=""
    if [ "$USE_HOST_NCCL" = "1" ]; then
        if [ -f "$NCCL_HOST_DIR/$NCCL_SO_NAME" ]; then
            head_preload=(-v "$NCCL_HOST_DIR:/nccl:ro" -e "LD_PRELOAD=/nccl/$NCCL_SO_NAME")
            log "head: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "head: $NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
        if worker_ssh "test -f '$WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME'"; then
            worker_preload="-v '$WORKER_NCCL_HOST_DIR:/nccl:ro' -e LD_PRELOAD='/nccl/$NCCL_SO_NAME'"
            log "worker: LD_PRELOAD $NCCL_SO_NAME"
        else
            warn "worker: $WORKER_NCCL_HOST_DIR/$NCCL_SO_NAME missing — using image NCCL"
        fi
    fi

    # optional extra passthrough
    local worker_passthru="" v val
    local -a head_env=()
    for v in HF_HUB_OFFLINE VLLM_API_KEY; do
        val="${!v:-}"
        if [ -n "$val" ]; then
            head_env+=(-e "$v=$val")
            worker_passthru+=" -e $v='$val'"
        fi
    done

    log "starting worker container on ${WORKER_SSH} (MoE mode: ${moe_mode}; NCCL if=${WORKER_CX7_IF} hca=${WORKER_CX7_IB}) ..."
    worker_ssh "docker run -d --name '$CONTAINER_WORKER' \
        --gpus all --network host --ipc=host --shm-size 32g --stop-timeout 60 \
        --device /dev/infiniband --cap-add IPC_LOCK \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        -v '$WORKER_CACHE_DIR:/root/.cache/huggingface' \
        -v '$CACHE_VOLUME:/root/.cache' \
        -v '/tmp/${CONTAINER_WORKER}.sh:/start.sh:ro' \
        ${worker_preload} \
        ${worker_nccl} \
        -e NCCL_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e GLOO_SOCKET_IFNAME='$WORKER_CX7_IF' \
        -e NCCL_IB_HCA='$WORKER_CX7_IB' \
        -e HEAD_IP='$HEAD_IP' -e RAY_PORT='$RAY_PORT' -e WORKER_IP='$WORKER_IP' \
        -e RAY_VERSION='$RAY_VERSION' \
        -e CLUSTER_WAIT_ITERS=$CLUSTER_WAIT_ITERS \
        -e VLLM_HOST_IP='$WORKER_IP' \
        -e RAY_OBJECT_STORE_MEMORY='$RAY_OBJECT_STORE_MEMORY' \
        -e VLLM_ENGINE_READY_TIMEOUT_S='$READY_TIMEOUT' \
        ${worker_passthru} \
        --entrypoint bash '$IMAGE' /start.sh" >/dev/null

    log "starting head container (Ray head + vLLM API server; NCCL if=${HEAD_CX7_IF} hca=${HEAD_CX7_IB}) ..."
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

    log "containers up — head=${CONTAINER_HEAD}, worker=${CONTAINER_WORKER}"
}

# ---------------------------- health wait ----------------------------------
# Stream the head container logs live while we poll /health, so the user sees
# real vLLM progress (weight load, kernel compile, warmup) instead of a silent
# timer. The background `docker logs -f` is killed on every exit path.
wait_for_health() {
    local url="http://127.0.0.1:${PORT}/health"
    log "waiting for ${url} (weight load + warmup on a 320B MoE is slow; timeout ${READY_TIMEOUT}s) ..."
    log "streaming head logs live — Ctrl-C detaches, the server keeps running"

    local logpid=""
    _stop_logtail() {
        [ -n "$logpid" ] && kill "$logpid" 2>/dev/null || true
        wait "$logpid" 2>/dev/null || true
        logpid=""
    }
    # Ensure Ctrl-C during the wait kills the background `docker logs -f`, not
    # just the polling loop (the global INT trap exits without reaping it).
    trap '_stop_logtail; warn "interrupted — containers keep running in the background ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT
    # `docker logs -f --tail 0` follows from now; prints without buffering so
    # the user sees each vLLM line as it lands.
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

    # Restore the default top-level INT trap for the rest of the run.
    trap 'warn "interrupted — containers keep running in the background ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT

    # After the stream stops, print a one-shot status line so the tail of the
    # run is unambiguous (the live log may not end with a clear "ready").
    if [ "$healthy" = "1" ]; then
        log "health check passed after ${elapsed}s — server is up"
    elif [ "$exited" = "1" ]; then
        warn "head container exited after ${elapsed}s"
    else
        warn "timed out after ${elapsed}s without becoming healthy"
    fi
    [ "$healthy" = "1" ]
}

# --------------------------- failure logs ----------------------------------
collect_failure_logs() {
    local tag="$1"
    mkdir -p "$LOGDIR"
    docker logs "$CONTAINER_HEAD" >"$LOGDIR/head-${tag}.log" 2>&1 || true
    {
        echo "### docker logs ${CONTAINER_WORKER}"
        worker_ssh "docker logs '$CONTAINER_WORKER' 2>&1" || true
        echo
        echo "### worker Ray session logs (filtered for CUDA kernel-image errors)"
        worker_ssh "docker exec '$CONTAINER_WORKER' sh -c 'grep -rhE \"$KERNEL_ERR_PAT\" /tmp/ray/session_latest/logs/ 2>/dev/null | head -n 40'" || true
    } >"$LOGDIR/worker-${tag}.log" 2>&1 || true
}

# ------------------------------ on ready -----------------------------------
on_ready() {
    local mode="$1" how="$2"
    log "======================================================================"
    log "GLM-5.3-Flash-NVFP4 is UP (${how}; MoE backend: ${mode})"
    log "  endpoints  : http://127.0.0.1:${PORT}/v1   (LAN ips: $(hostname -I))"
    log "  model name : ${MODEL}"
    log "  features   : tools=glm47+auto, reasoning=glm45, MTP spec-decode (${MTP_TOKENS} tokens), chat_template.jinja (image+video)"
    [ "$mode" = "marlin" ] && \
    log "  NOTE       : running the marlin dequant-to-FP16 fallback — set MOE_BACKEND=marlin"
    [ "$mode" = "marlin" ] && \
    log "               to skip the failed native attempt next time."
    log "  quick test :"
    log "    curl -s http://127.0.0.1:${PORT}/v1/chat/completions \\"
    log "      -H 'Content-Type: application/json' \\"
    log "      -d '{\"model\": \"${MODEL}\", \"messages\": [{\"role\": \"user\", \"content\": \"hello!\"}]}'"
    log "  manage     : ./start.sh status | ./start.sh logs | ./start.sh logs worker | ./start.sh stop"
    log "======================================================================"
    # By default, return to the shell once the server is up — the live logs
    # were already streamed during wait_for_health(). Set TAIL=1 to keep
    # following the head logs after readiness instead.
    if [ "${TAIL:-0}" = "1" ]; then
        log "tailing head logs — Ctrl-C just detaches, the server keeps running"
        trap '' INT
        docker logs -f --tail 20 "$CONTAINER_HEAD" || true
        trap 'warn "interrupted — containers keep running in the background ('"'"'./start.sh logs'"'"' to watch, '"'"'./start.sh stop'"'"' to stop)"; exit 130' INT
        log "detached from logs; server still running"
    fi
}

# ------------------------------- start -------------------------------------
start() {
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
        "max-len=${MAX_MODEL_LEN:-<model default>} gpu-util=${GPU_MEM_UTIL} block=${BLOCK_SIZE} kv=${KV_CACHE_DTYPE} arch=${TORCH_CUDA_ARCH_LIST} port=${PORT}"

    launch_cluster "$mode"
    if wait_for_health; then
        on_ready "$mode" "first attempt"
        return
    fi

    collect_failure_logs "$mode"
    echo "---- last 60 lines of head log ($LOGDIR/head-${mode}.log) ----"
    tail -n 60 "$LOGDIR/head-${mode}.log" || true

    if [ "$MOE_BACKEND" = "auto" ] && [ "$mode" = "native" ] \
       && grep -qE "$KERNEL_ERR_PAT" "$LOGDIR/head-native.log" "$LOGDIR/worker-native.log" 2>/dev/null; then
        warn "cudaErrorNoKernelImageForDevice from the native FP4 MoE kernels (expected on sm_121) —"
        warn "falling back to marlin MoE backend: --moe-backend marlin --enforce-eager"
        launch_cluster "marlin"
        if wait_for_health; then
            on_ready "marlin" "after sm_121 fallback"
            return
        fi
        collect_failure_logs "marlin"
        echo "---- last 60 lines of head log ($LOGDIR/head-marlin.log) ----"
        tail -n 60 "$LOGDIR/head-marlin.log" || true
        die "server failed in marlin mode too — full logs in $LOGDIR/"
    fi
    die "server did not become healthy — full logs in $LOGDIR/"
}

# ------------------------------- stop --------------------------------------
stop() {
    log "stopping head container ..."
    docker rm -f "$CONTAINER_HEAD" >/dev/null 2>&1 || log "  (no head container was running)"
    log "stopping worker container on ${WORKER_SSH} ..."
    worker_ssh "docker rm -f '$CONTAINER_WORKER'" >/dev/null 2>&1 \
        || log "  (no worker container was running)"
    log "stopped."
}

# ------------------------------ status -------------------------------------
status() {
    log "head (${CONTAINER_HEAD} on $(hostname)):"
    docker ps -a --filter "name=${CONTAINER_HEAD}" --format '  {{.Names}}  {{.Status}}' || true
    if curl -fsS -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        log "  API: healthy — http://127.0.0.1:${PORT}/v1"
    else
        log "  API: not responding"
    fi
    if docker inspect -f '{{.State.Running}}' "$CONTAINER_HEAD" 2>/dev/null | grep -q true; then
        docker exec "$CONTAINER_HEAD" ray status 2>/dev/null | sed 's/^/  /' | head -n 25 || true
    fi
    log "worker (${CONTAINER_WORKER} on ${WORKER_SSH}):"
    worker_ssh "docker ps -a --filter name=${CONTAINER_WORKER} --format '  {{.Names}}  {{.Status}}'" 2>/dev/null \
        || log "  (worker unreachable)"
}

# ------------------------------- logs --------------------------------------
logs() {
    case "${1:-head}" in
        worker)
            log "following worker container logs on ${WORKER_SSH} ..."
            log "(per-rank engine stdout lives inside the container: /tmp/ray/session_latest/logs/worker-*.out)"
            trap '' INT
            worker_ssh "docker logs -f --tail 100 '$CONTAINER_WORKER'" || true
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
