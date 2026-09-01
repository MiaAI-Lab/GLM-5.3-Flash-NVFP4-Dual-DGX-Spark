<h1 align="center">GLM-5.3-Flash-NVFP4-Dual-DGX-Spark</h1>

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://github.com/sponsors/MiaAI-Lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Sponsor%20me%20on%20GitHub-181717?style=for-the-badge&logo=githubsponsors&logoColor=white" alt="Sponsor me on GitHub" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

Serve the **[LibertAIDAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4)** checkpoint on a **2× DGX Spark** kit: multimodal image+video, Ray tensor-parallel 2, OpenAI API on **:8888**. One command (`./start.sh`) builds the local serving image if needed, syncs weights, stands up the Ray cluster over CX7, and waits for `/health`.

## Hardware / topology

Two GB10 Sparks (SM121, 128 GiB UMA each), one GPU per node, **TP=2** via Ray. This branch also has experimental [`./start-tp4.sh`](start-tp4.sh) for **4× Spark / TP=4** on a RoCE switch — see [Experimental TP=4](#experimental-tp4-4-spark-this-branch-only).

```
spark1 (head)                         spark2 (worker)
10.0.0.1                              zurih@10.0.0.2
Ray head + vLLM API :8888             Ray worker
HF cache (local disk)                 HF cache (rsync’d once)
        CX7 QSFP  (NCCL / RoCE)
  enp1s0f1np1 / rocep1s0f1  ↔  enp1s0f0np0 / rocep1s0f0
```

Both containers run `--network host --ipc=host`. Ray may use the `10.0.0.1`/`10.0.0.2` aliases; **NCCL cannot** — `start.sh` pins the CX7 NICs and IB HCAs so `ncclCommInitRank` does not hang.

**Needs:** Docker (no sudo) on both nodes, passwordless SSH from head → `zurih@10.0.0.2`, `hf`/`huggingface-cli` + `curl` + `rsync` + `python3` on the head, ~200 GiB free per node (~181 GiB weights).

## What you get

| Capability | Default on this kit |
|---|---|
| OpenAI-compatible API | `http://127.0.0.1:8888/v1` (LAN: head IP) |
| Tool calling | `--tool-call-parser glm47 --enable-auto-tool-choice` |
| Reasoning | `--reasoning-parser glm45` |
| Speculative decode | MTP, **4** tokens |
| Multimodal | image + video (`LIMIT_MM={"image":4,"video":1}`) |
| Context | `--max-model-len 262144` (checkpoint is 1M-native) |
| Scheduler | `--max-num-seqs 8`, `--block-size 2304` |
| KV | `--kv-cache-dtype fp8_e4m3` |
| MoE | **marlin** + `--enforce-eager` (GB10-known-good) |

Model name on the wire: `LibertAIDAI/GLM-5.3-Flash-NVFP4`.

## Structural decode tok/s

On this recipe (SM90 NoPE + FA2, marlin, MTP-4, fp8_e4m3 KV, 262k, eager). Concurrent streams = `--max-num-seqs` class.

| Concurrency | agg tok/s | tok/s/stream | TTFT |
|---|---|---|---|
| ×1 | 23–30 | 23–30 | 6.54s |
| ×2 | 31–37 | 16–19 | 6.40s |
| ×4 | 43 | 13 | 8.35s |
| ×6 | 64 | 11 | 1.03s † |
| ×8 | 72 | 10 | 6.17s |

† warm

## Quickstart

```bash
./start.sh
```

First run: build/ship the local image if missing → download ~181 GiB into the HF cache → refresh `chat_template.jinja` from Hugging Face (`emit_image` / `emit_video`) → rsync weights to the worker → Ray worker on spark2, Ray head + `vllm serve` on spark1 → poll `/health` up to **3600s** (320B MoE init is slow).

Weights already on disk:

```bash
SKIP_DOWNLOAD=1 ./start.sh
```

```bash
./start.sh status          # containers, API health, Ray cluster
./start.sh logs            # follow head (driver + API)
./start.sh logs worker     # follow worker container
./start.sh stop            # tear down both ranks
./start.sh restart         # stop + start
```

Ctrl-C during the wait detaches; the cluster keeps running. Default is not to tail after ready (`TAIL=1` to keep following).

```bash
curl -s http://127.0.0.1:8888/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "LibertAIDAI/GLM-5.3-Flash-NVFP4",
    "messages": [{"role": "user", "content": "hello!"}]
  }'
```

Image + video use standard OpenAI multimodal content parts (`image_url` / `video_url`). Turn thinking off per request with `chat_template_kwargs` (see [Notes](#notes)).

## Experimental TP=4 (4× Spark, this branch only)

`experimental/tp4` adds an **untested** sibling launcher for **4× DGX Spark** on a **RoCE switch**. It does not change `./start.sh`. We cannot verify it here (2 Sparks only); a 4-node owner can.

```bash
git checkout experimental/tp4
cp env.tp4.example .env.tp4   # fill in 4 CX7 IPs, SSH targets, IF + HCA names
./start-tp4.sh
```

- Containers: `glm53-flash-tp4-head` + `glm53-flash-tp4-w1..w3` (will not collide with the 2-node pair).
- Fabric: every CX7 on the **same RoCE L2**. `HEAD_IP` / `WORKER*_IP` must be those NIC addresses. A switchless QSFP ring is **not** this script — use [alexellis/glm-5.3-flash-4x-dgx-spark-switchless](https://github.com/alexellis/glm-5.3-flash-4x-dgx-spark-switchless).
- `/health` is the only success signal. Same serve knobs as `start.sh` (`MOE_BACKEND`, `GPU_MEM_UTIL`, …). `./start-tp4.sh stop` / `./stop-tp4.sh` tears down the TP4 set only.

## Image

Serving tag: **`mia/glm53-flash-spark:mm-ray-v1`** (local default; `start.sh` does not change this).

A private copy is on GHCR for MiaAI-Lab members:

```bash
docker pull ghcr.io/miaai-lab/glm53-flash-spark:mm-ray-v1
```

This is a **local** tag (kernel `glm53-flash-sm121:v8` + Ray + multimodal defaults). **Do not `docker pull` it from Docker Hub** — `start.sh` will not pull `mia/glm53-flash-spark:*`; it builds.

If the tag is missing on the head, `start.sh` runs [`files/build.sh`](files/build.sh):

1. Kernel layer **`glm53-flash-sm121:v8`** from [`files/Dockerfile`](files/Dockerfile) (applies [`files/glm53-flash_SM121.py`](files/glm53-flash_SM121.py) **sm90**, seven steps — stock SM12 would take the packed SM120 MLA path, which is wrong for this NoPE checkpoint).
2. Serving layer from [`files/Dockerfile.mm-ray`](files/Dockerfile.mm-ray): Ray 2.58.0, MM env, API **:8888**.
3. `docker save | ssh docker load` onto the worker.

`PULL=1 ./start.sh` rebuilds the local tag and re-ships. You can also build without launching:

```bash
./files/build.sh
```

The kernel layer selects SM90 sparse-MLA + FA2 for GB10; it does **not** invent FlashInfer FA2 or SM121 MLA. `sm120` in `glm53-flash_SM121.py` is an unused packed-cache mode and is **not** the serve path (not baked).

## Configuration

All optional. Defaults match a working 2× Spark + CX7 + Portainer-on-8000 kit.

| Variable | Default | What it does |
|---|---|---|
| `IMAGE` | `mia/glm53-flash-spark:mm-ray-v1` | Serving image |
| `PORT` | `8888` | OpenAI API on the head (8000 is Portainer) |
| `MOE_BACKEND` | `marlin` | `marlin` (default) · `native` · `auto` (native, then marlin on `cudaErrorNoKernelImageForDevice`) |
| `MTP_TOKENS` | `4` | MTP speculative tokens (`0` disables) |
| `MAX_MODEL_LEN` | `262144` | Context; model is 1M-native (`1048576` if you need it) |
| `GPU_MEM_UTIL` | `0.84` | vLLM GPU memory budget (GB10 UMA; **0.90 fails** the free-mem check). Profiles KV when `KV_CACHE_MEMORY` is unset |
| `ENFORCE_EAGER` | `1` | Skip CUDA-graph capture at init |
| `BLOCK_SIZE` | `2304` | Paged-MQA block size |
| `MAX_NUM_SEQS` | `8` | Scheduler concurrency |
| `KV_CACHE_DTYPE` | `fp8_e4m3` | KV dtype |
| `KV_CACHE_MEMORY` | *(unset)* | Optional `--kv-cache-memory` pin in bytes. Default is empty so `GPU_MEM_UTIL=0.84` sizes KV; set only if MTP-4 UMA-OOMs |
| `LIMIT_MM` | `{"image":4,"video":1}` | Max image/video items per prompt |
| `SKIP_MM_PROFILING` | `1` | Serve MM without a max-size dummy forward at init (avoids UMA OOM) |
| `HEAD_IP` | `10.0.0.1` | Head as seen by the worker |
| `WORKER_SSH` | `zurih@10.0.0.2` | Worker SSH target |
| `WORKER_IP` | `10.0.0.2` | Worker as seen by the head |
| `TP` | `2` | Tensor parallel (1 GPU × 2 nodes) |
| `RAY_PORT` | `6379` | Ray GCS |
| `READY_TIMEOUT` | `3600` | `/health` wait (= `VLLM_ENGINE_READY_TIMEOUT_S`) |
| `EXTRA_ARGS` | — | Extra flags appended to `vllm serve` |
| `SKIP_DOWNLOAD` | `0` | `1` = skip HF download check |
| `SKIP_SYNC` | `0` | `1` = skip rsync to worker |
| `REFRESH_WEIGHTS` | `0` | `1` = re-run `hf download` |
| `HF_HUB_OFFLINE` | — | Skip HF etag checks at startup |
| `PULL` | `0` | `1` = rebuild local image and re-ship |
| `TAIL` | `0` | `1` = keep following head logs after ready |
| `NCCL_DEBUG` | `WARN` | Passed through to both containers |

CX7 pins (`HEAD_CX7_IF`, `WORKER_CX7_IF`, `HEAD_CX7_IB`, `WORKER_CX7_IB`) default to the QSFP pair on this kit.

## Repo layout

| Path | Role |
|---|---|
| [`start.sh`](start.sh) | Start / stop / restart / status / logs (supported 2× Spark path) |
| [`start-tp4.sh`](start-tp4.sh) | **Experimental** 4× Spark TP=4 (this branch). Needs `.env.tp4` from [`env.tp4.example`](env.tp4.example) |
| [`stop-tp4.sh`](stop-tp4.sh) | Stop TP4 containers only (`./start-tp4.sh stop`) |
| [`files/build.sh`](files/build.sh) | Build kernel v8 + `mm-ray-v1` if missing |
| [`files/Dockerfile`](files/Dockerfile) | Kernel layer `glm53-flash-sm121:v8` |
| [`files/Dockerfile.mm-ray`](files/Dockerfile.mm-ray) | Serving layer (Ray + MM + :8888) |
| [`files/glm53-flash_SM121.py`](files/glm53-flash_SM121.py) | SM121 kernel patch (default **sm90**, seven steps) |
| [`files/chat_template.jinja`](files/chat_template.jinja) | Baked fallback; runtime still refreshes from HF |

`start.sh` writes `.glm53-head.inner.sh` / `.glm53-worker.inner.sh` on every launch (gitignored). `docs/` and `logs/` are gitignored.

## Notes

- **GB10 is UMA.** Weights ~90 GiB per rank; Ray’s default object store would steal RAM from the GPU budget. This recipe uses a 4 GiB Ray object store, `GPU_MEM_UTIL=0.84` (which profiles KV — `--kv-cache-memory` is unset by default), `--enforce-eager`, and `--skip-mm-profiling`. Set `KV_CACHE_MEMORY=<bytes>` only if MTP-4 UMA-OOMs. Do not run another GPU model on either node at the same time.
- **Tear down both ranks before relaunch.** Leftover Ray/NCCL on either Spark will fight the next start. Use `./start.sh stop` or `./start.sh restart` — not a head-only `docker rm`.
- **Thinking off:** pass `"chat_template_kwargs": {"enable_thinking": false}` on the chat-completions body.
- **Local image tag.** `mia/glm53-flash-spark:mm-ray-v1` is built here. Do not `docker pull` that tag from Docker Hub.

## License

This repository's code is provided under the [MIT License](LICENSE).
