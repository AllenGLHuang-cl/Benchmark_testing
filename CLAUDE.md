# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Batch-driving clients that generate images against either a **local ComfyUI server** or **hosted
model APIs** (currently Gemini's image models), then — in benchmark mode — record per-request
timing (and VRAM, for the local ComfyUI path) to compare models against each other. There is no
model-serving code here: for the ComfyUI scripts, ComfyUI (with the relevant checkpoints already
loaded) must already be running and reachable at the `server_address`/`SERVER_ADDRESS` each
script's `__main__` block points at; for the Gemini scripts, it's a remote API call, no local
server needed.

**Datasets and outputs live outside this repo.** Every script's `__main__` hardcodes a dataset
path (e.g. `/workspace/data/allen/dataset/benchmark_testing/single_image_editing`,
`.../benchmark_testing/text_to_image`) external to this repo, and writes generated images / logs
back under `<dataset>/<feature>/results/<model_dir>/`. This repo ships only code + `model_configs/`.

**Scripts are configured by editing constants in their `__main__` block directly** — there is no
CLI argument parsing and no external config file for run parameters (server address, dataset path,
model list, seed, num outputs, etc. are all Python literals at the top of `__main__`). To run a
different sweep, edit the block and re-run `python <script>.py`.

**Never hardcode API keys into a script's `__main__` block or anywhere tracked by git** — this
repo has a real GitHub remote. The Gemini scripts read credentials from environment variables
(`GEMINI_API_KEY`, `GEMINI_BASE_URL`) instead; see `gemini_client.py`.

## Commands

No build step, no test suite, no lint config — these are standalone scripts. Dependencies (not
pinned anywhere yet): `websocket-client`, `Pillow`, `tqdm`, and `python-pptx` (only for the
`model-benchmark-ppt` skill, see below).

```bash
python websocket_edit.py              # i2i batch over a features-layout dataset (ComfyUI)
python websocket_tti.py               # t2i batch over one flat prompt list, N models (ComfyUI)
python websocket_tti_structured.py    # t2i batch for structured-JSON-prompt models (ComfyUI, ideogram4)

export GEMINI_API_KEY=...             # required for the two scripts below — never hardcode
export GEMINI_BASE_URL=...            # internal gateway URL, not generativelanguage.googleapis.com directly
python gemini_edit.py                 # i2i batch over a features-layout dataset (Gemini API)
python gemini_tti.py                  # t2i batch over one flat prompt list (Gemini API)
```

## Architecture

Five independent batch runners share library modules by concern (ComfyUI client, dataset
traversal, perf instrumentation, Gemini client); there's no shared entry point or CLI — pick the
script matching your use case and edit its `__main__` block.

- **`comfy_client.py`** — the ComfyUI websocket/HTTP client (`queue_prompt`, `get_outputs`,
  `get_image`, `get_history`, `update_params_by_name`/`_by_class`). `get_outputs()` branches on
  ComfyUI's output node type (`text`/`images`/`other`); a video-producing node would need a new
  branch there. Used by `websocket_edit.py` and `websocket_tti.py`.
  **`websocket_tti_structured.py` still carries its own un-deduplicated copy of these same
  functions** — a known inconsistency, not yet folded into `comfy_client.py`.
- **`dataset_utils.py`** — the "features" dataset walk (`get_image_files`,
  `load_prompts_for_image` — the `{stem}.json` → shared `prompt.json` fallback). Used by both
  `websocket_edit.py` and `gemini_edit.py` since they walk the identical dataset shape.
- **`perf_utils.py`** — shared benchmark instrumentation: `GpuMemorySampler` background-polls
  ComfyUI's `/system_stats` every 50ms (to catch peak VRAM, not just a before/after snapshot),
  `warmup()` runs N throwaway generations before timing starts, `aggregate_records()` reduces a
  run's records to one summary (avg latency, peak VRAM — fields it can't find, like VRAM for a
  hosted API, are just omitted from the summary rather than erroring), `append_jsonl`/`read_jsonl`
  handle the per-request log file. The recorded VRAM figure (`vram_peak_gib`) is deliberately the
  nvidia-smi/NVML-equivalent number (`get_gpu_memory()`'s `vram_used`), not ComfyUI's own
  `comfy_vram_used`, which undercounts because it treats PyTorch's reserved-but-idle cache as free.
- **`gemini_client.py`** — minimal HTTP client for Gemini's **Interactions API**
  (`POST {base_url}/v1beta/interactions`, header `x-goog-api-key`, body
  `{"model", "input": [{"type":"text",...}, {"type":"image","mime_type",...,"data": base64}]}`,
  response image at `steps[] where type=="model_output" → content[] where type=="image" → data`).
  This is a *newer* Gemini API surface (not the older `generateContent`/`contents`/`parts` REST
  shape) — confirmed by live-testing against the real gateway, not just docs (Google's own docs
  pages gave inconsistent-looking info across two fetches; the live response settled it). Retries
  3x with backoff on 429/5xx only. **No seed parameter exists** — Gemini generations aren't
  reproducible the way ComfyUI's are, so `gemini_edit.py`/`gemini_tti.py` don't track a seed at
  all. `base_url` is an **internal gateway**, not `generativelanguage.googleapis.com` directly —
  don't assume the path structure generalizes to a direct Google connection without checking.
- **`websocket_edit.py`** — i2i (image-editing) batch runner over ComfyUI: **one model config ×
  a "features" dataset** (`dataset_root/<feature>/{src/,meta_data/,results/<model_dir>/}`) × every
  source image × every prompt × `num_of_output`. Output filenames encode everything needed to
  resume (`{stem}_prompt{p}_out{i}_seed{S}.jpg`) — already-existing files are skipped regardless of
  what seed produced them. The feature literally named `performance_test` is special-cased: it
  also runs a warmup, records `performance.jsonl` per image (peak-VRAM-sampled), and writes
  `performance_summary.json` at the end — normal features skip all of that.
- **`websocket_tti.py`** — t2i (text-to-image) batch runner over ComfyUI: **many models
  (`MODELS` dict) × one flat prompt list** (`{"prompts": [...]}`, no source image).
  `PERFORMANCE_TEST` is a **global** mode switch (not a per-feature special-case like above): when
  true, images are still saved (for every `num_of_output`), but only `i==0` per prompt gets timed/
  VRAM-sampled and written to `performance.jsonl` — later same-prompt repeats benefit from
  ComfyUI's node-level caching (e.g. text encoding) and would understate real per-request cost.
  `i==0` always reruns even if its file exists, so every benchmark run measures fresh.
- **`websocket_tti_structured.py`** — a t2i variant for models whose ComfyUI custom node expects
  a **structured JSON prompt** object rather than a plain string (currently ideogram4), plus a
  `CustomCombo` "mode" axis (Quality/Default/Turbo) that gets swept per prompt. Targets a
  **different ComfyUI instance** (`127.0.0.1:7891`) than the other ComfyUI scripts (`:7890`) —
  these are two separately-running ComfyUI processes with different checkpoints loaded, not
  interchangeable. Has no benchmark/perf-recording mode.
- **`gemini_edit.py`** / **`gemini_tti.py`** — same dataset shapes as `websocket_edit.py` /
  `websocket_tti.py` respectively, but call `gemini_client.py` instead of ComfyUI. Output filenames
  use `{stem}_{prompt_idx}_{n}.jpg` / `tti_{prompt_idx}_{n}.jpg` (`n` from 1, no seed) — this
  matches the naming convention already used by pre-existing `gemini-3-pro-image-preview` results
  in the dataset, so no changes are needed to the PPT skill's filename resolver. No warmup (each
  API call is stateless — no local-cache effect like ComfyUI's `websocket_tti.py` has to work
  around) and no VRAM (remote hosted API, no local GPU to sample). Both take a **`concurrency`**
  param (`ThreadPoolExecutor`, default 1) since Gemini calls are independent stateless HTTP
  requests, not serialized through one ComfyUI websocket like the other scripts — safe to run
  several in flight (`gemini_client.py` already retries 429/5xx). Note: `performance.jsonl`'s
  `elapsed_sec` reflects latency *under whatever concurrency was set* — for a clean isolated
  per-request number, run with `concurrency=1`.
- **`model_configs/`** — ComfyUI workflows in two forms per model: `*_api.json` (the "API
  format" node-graph the scripts actually load and POST to `/prompt`) and `*_workflow.json` (the
  full UI-graph export, for opening in the ComfyUI web UI to inspect/edit visually — **not read by
  any script**). When adding a new model, export both from ComfyUI's UI (`Save (API Format)` and
  the regular save), but only the `_api.json` one needs the target text/image/seed nodes wired to
  class types the scripts already know how to patch (`CLIPTextEncode`, `TextEncodeBooguEdit`,
  `LoadImage`, or add a new `update_params_by_class` call for a new node type). Not applicable to
  the Gemini scripts (no ComfyUI workflow involved).

## `.claude/skills/model-benchmark-ppt/`

A Claude Code skill (not imported by any script here — invoked by name, e.g. "make a comparison
PPT") that builds a `.pptx` comparing model outputs from the `results/<model_dir>/` folders these
scripts produce, and can auto-fill its perf table from the `performance.jsonl`/
`performance_summary.json` these scripts write. It's deliberately self-contained (own copy of the
filename-resolving and perf-aggregation logic, not importing from `perf_utils.py`/`comfy_client.py`
even though it lives in the same repo now) so it stays portable if copied elsewhere. Its
`_PERF_AUTO_COLUMNS` field-name mapping in `make_comparison_ppt.py` is a hand-kept copy of
`perf_utils.py`'s record schema — if either schema changes, update both. See its own `SKILL.md`
for usage; the "Extending to video" section there also applies to this whole repo's future t2v/i2v
plans.
