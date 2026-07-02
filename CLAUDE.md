# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Batch-driving clients for a running **ComfyUI** server: they queue workflows over ComfyUI's
websocket/HTTP API, pull back generated images, and (in benchmark mode) record per-request
timing/VRAM to compare model configs against each other. There is no model-serving code here —
ComfyUI (with the relevant checkpoints already loaded) must already be running and reachable at
the `server_address`/`SERVER_ADDRESS` each script's `__main__` block points at.

**Datasets and outputs live outside this repo.** Every script's `__main__` hardcodes a dataset
path (e.g. `/workspace/data/allen/dataset/image_editing`, `/workspace/data/allen/dataset/tti_test`)
external to this repo, and writes generated images / logs back under
`<dataset>/<feature>/results/<model_dir>/`. This repo ships only code + `model_configs/`.

**Scripts are configured by editing constants in their `__main__` block directly** — there is no
CLI argument parsing and no external config file for run parameters (server address, dataset path,
model list, seed, num outputs, etc. are all Python literals at the top of `__main__`). To run a
different sweep, edit the block and re-run `python <script>.py`.

## Commands

No build step, no test suite, no lint config — these are standalone scripts. Dependencies (not
pinned anywhere yet): `websocket-client`, `Pillow`, `tqdm`, and `python-pptx` (only for the
`model-benchmark-ppt` skill, see below).

```bash
python websocket_edit.py              # i2i batch over a features-layout dataset
python websocket_tti.py               # t2i batch over one flat prompt list, N models
python websocket_tti_structured.py    # t2i batch for structured-JSON-prompt models (ideogram4)
```

## Architecture

Three independent batch runners share two library modules; there's no shared entry point or CLI —
pick the script matching your use case and edit its `__main__` block.

- **`comfy_client.py`** — the ComfyUI websocket/HTTP client (`queue_prompt`, `get_outputs`,
  `get_image`, `get_history`, `update_params_by_name`/`_by_class`). `get_outputs()` branches on
  ComfyUI's output node type (`text`/`images`/`other`); a video-producing node would need a new
  branch there. Used by `websocket_edit.py` and `websocket_tti.py`.
  **`websocket_tti_structured.py` still carries its own un-deduplicated copy of these same
  functions** — a known inconsistency, not yet folded into `comfy_client.py`.
- **`perf_utils.py`** — shared benchmark instrumentation: `GpuMemorySampler` background-polls
  ComfyUI's `/system_stats` every 50ms (to catch peak VRAM, not just a before/after snapshot),
  `warmup()` runs N throwaway generations before timing starts, `aggregate_records()` reduces a
  run's records to one summary (avg latency, peak VRAM), `append_jsonl`/`read_jsonl` handle the
  per-request log file. The recorded VRAM figure (`vram_peak_gib`) is deliberately the
  nvidia-smi/NVML-equivalent number (`get_gpu_memory()`'s `vram_used`), not ComfyUI's own
  `comfy_vram_used`, which undercounts because it treats PyTorch's reserved-but-idle cache as free.
- **`websocket_edit.py`** — i2i (image-editing) batch runner: **one model config × a
  "features" dataset** (`dataset_root/<feature>/{src/,meta_data/,results/<model_dir>/}`) × every
  source image × every prompt × `num_of_output`. Prompts come from `meta_data/{stem}.json`,
  falling back to a shared `meta_data/prompt.json` for the whole feature. Output filenames encode
  everything needed to resume (`{stem}_prompt{p}_out{i}_seed{S}.jpg`) — already-existing files are
  skipped regardless of what seed produced them. The feature literally named `performance_test` is
  special-cased: it also runs a warmup, records `performance.jsonl` per image (peak-VRAM-sampled),
  and writes `performance_summary.json` at the end — normal features skip all of that.
- **`websocket_tti.py`** — t2i (text-to-image) batch runner: **many models (`MODELS` dict) ×
  one flat prompt list** (`{"prompts": [...]}`, no source image). `PERFORMANCE_TEST` is a
  **global** mode switch (not a per-feature special-case like above): when true, images are never
  saved at all — it's a pure speed/memory benchmark that also warms up and writes
  `performance.jsonl` + `performance_summary.json`.
- **`websocket_tti_structured.py`** — a t2i variant for models whose ComfyUI custom node expects
  a **structured JSON prompt** object rather than a plain string (currently ideogram4), plus a
  `CustomCombo` "mode" axis (Quality/Default/Turbo) that gets swept per prompt. Targets a
  **different ComfyUI instance** (`127.0.0.1:7891`) than the other two scripts (`:7890`) — these
  are two separately-running ComfyUI processes with different checkpoints loaded, not
  interchangeable. Has no benchmark/perf-recording mode.
- **`model_configs/`** — ComfyUI workflows in two forms per model: `*_api.json` (the "API
  format" node-graph the scripts actually load and POST to `/prompt`) and `*_workflow.json` (the
  full UI-graph export, for opening in the ComfyUI web UI to inspect/edit visually — **not read by
  any script**). When adding a new model, export both from ComfyUI's UI (`Save (API Format)` and
  the regular save), but only the `_api.json` one needs the target text/image/seed nodes wired to
  class types the scripts already know how to patch (`CLIPTextEncode`, `TextEncodeBooguEdit`,
  `LoadImage`, or add a new `update_params_by_class` call for a new node type).

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
