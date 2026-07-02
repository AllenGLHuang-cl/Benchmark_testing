# 使用 websockets API 與 SaveImageWebsocket node 直接取得圖片（不經過磁碟）跑 text-to-image batch。
#
# 兩種模式（由 PERFORMANCE_TEST 切換），圖片都會存：
#   - PERFORMANCE_TEST = False：正常 batch 生成，不記錄效能。
#   - PERFORMANCE_TEST = True ：正常生成之外，每個 prompt 只量測第一次生成（i==0）的時間與
#     GPU memory，寫進 performance.jsonl。只測第一次是因為同一個 prompt 重跑第二次以後，
#     ComfyUI 會 cache 住沒變過的 node（例如 CLIPTextEncode 的文字編碼），耗時會被低估。

import io
import json
import os
import random
import time
import uuid

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from PIL import Image
from tqdm import tqdm

from comfy_client import get_outputs, update_params_by_class, update_params_by_name
from perf_utils import (
    GpuMemorySampler, aggregate_records, append_jsonl, mem_value,
    read_jsonl, to_gib, warmup,
)


# ---------------------------------------------------------------------------
# 輸出處理
# ---------------------------------------------------------------------------
def save_first_image(outputs, output_path):
    """把第一個 image node 的第一張圖存到 output_path。回傳是否有存到圖。"""
    for node_id in outputs:
        if outputs[node_id]["type"] == "images" and outputs[node_id]["data"]:
            image = Image.open(io.BytesIO(outputs[node_id]["data"][0]))
            image.save(output_path)
            return True
    return False


def build_params(config_path, test_prompt, seed, width=None, height=None):
    """載入 config 並依 prompt / seed 填好參數。"""
    with open(config_path, "r") as f:
        params = json.loads(f.read())

    # params = update_params_by_class(params, "PrimitiveStringMultiline", "value", test_prompt)
    params = update_params_by_class(params, "CLIPTextEncode", "text", test_prompt)
    params = update_params_by_class(params, "TextEncodeBooguEdit", "prompt", test_prompt)
    params = update_params_by_name(params, "noise_seed", seed)
    params = update_params_by_name(params, "seed", seed)
    if width is not None:
        params = update_params_by_name(params, "width", width)
    if height is not None:
        params = update_params_by_name(params, "height", height)
    return params


def run_model(
    ws,
    client_id,
    server_address,
    model_name,
    config_path,
    test_prompts,
    output_path,
    base_seed,
    num_of_output,
    performance_test,
    width=None,
    height=None,
    memory_sample_interval_sec=0.05,
    warmup_runs=2,
):
    """跑單一 model 的所有 prompt，效能測試模式下即時 append 進 performance.jsonl，回傳寫入的紀錄。"""
    os.makedirs(output_path, exist_ok=True)
    perf_log_path = os.path.join(output_path, "performance.jsonl")
    perf_records = []

    prompts = test_prompts["prompts"]

    if performance_test and warmup_runs > 0:
        print(f"Warming up ({warmup_runs}x) before measuring...")
        warmup_params = build_params(config_path, prompts[0], base_seed, width, height)
        warmup(lambda: get_outputs(ws, warmup_params, client_id, server_address), n=warmup_runs)

    for prompt_idx, test_prompt in tqdm(enumerate(prompts), total=len(prompts), desc=model_name):
        for i in range(num_of_output):
            seed = base_seed + i
            out_file = os.path.join(output_path, f"tti_{prompt_idx}_seed{seed}.jpg")

            # 每個 prompt 只有第一次生成（i==0）在效能測試模式下量測，且一律重跑取得乾淨數據；
            # 其餘情況（含 i>=1 的額外輸出）已存在就跳過，支援中斷續跑。
            measure = performance_test and i == 0
            if not measure and os.path.exists(out_file):
                continue

            params = build_params(config_path, test_prompt, seed, width, height)

            sampler = GpuMemorySampler(server_address, interval_sec=memory_sample_interval_sec).start() if measure else None

            start = time.perf_counter()
            try:
                outputs = get_outputs(ws, params, client_id, server_address)
                elapsed = time.perf_counter() - start
            finally:
                sampler_stats = sampler.stop() if sampler is not None else None

            if measure:
                latest_mem = sampler_stats["latest"]
                peak_mem = sampler_stats["peaks"]
                record = {
                    "model": model_name,
                    "device": mem_value(latest_mem, "device_name"),
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "elapsed_sec": round(elapsed, 3),
                    "vram_peak_gib": to_gib(mem_value(peak_mem, "vram_used")),
                    "vram_sample_count": sampler_stats["sample_count"],
                    "vram_sampling_errors": sampler_stats["error_count"],
                }
                append_jsonl(record, perf_log_path)
                perf_records.append(record)

            if not save_first_image(outputs, out_file):
                print(f"\033[93m[warning] no image output for prompt {prompt_idx} seed {seed}\033[0m")

    return perf_records


def write_perf_summary(output_path):
    """讀回整個 performance.jsonl（含先前中斷續跑累積的），彙總成一筆總結並存檔。"""
    records = read_jsonl(os.path.join(output_path, "performance.jsonl"))
    if not records:
        return
    summary = aggregate_records(records)
    with open(os.path.join(output_path, "performance_summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Performance summary ({summary.get('n_samples', 0)} samples): "
          f"avg {summary.get('avg_elapsed_sec')}s, peak {summary.get('vram_peak_gib')} GiB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 設定 ----
    SERVER_ADDRESS = "127.0.0.1:7890"
    BASE_DIR = "/workspace/data/allen/dataset/tti_test"
    # RUNNING_JSON = ["tti.json", "portrait.json"]
    RUNNING_JSON = ["test_performance.json"]

    # model_name -> config 路徑
    MODELS = {
        "Krea2-turbo-fp8": "./model_configs/Krea2_api.json",
    }

    NUM_OF_OUTPUT = 1
    SEED = None  # None = 每個 model 隨機；給定整數則固定 seed

    # 解析度（None = 沿用 config 預設）。常用：
    #   16:9 -> (1392, 752) | 4:3 -> (1184, 880) / (880, 1184) | 1:1 -> (1024, 1024)
    WIDTH, HEIGHT = None, None

    # True：效能測試，不存圖，記錄時間與 GPU memory；False：正常存圖。
    PERFORMANCE_TEST = True
    GPU_MEMORY_SAMPLE_INTERVAL_SEC = 0.05
    WARMUP_RUNS = 2  # 效能測試模式下，正式量測前先暖身幾次（結果丟棄）

    for i, json_file in enumerate(RUNNING_JSON):
        print(f"\n\n=== Running {json_file} ===")
        with open(os.path.join(BASE_DIR, "meta_data", json_file), "r") as f:
            test_prompts = json.loads(f.read())

        run_name = json_file.split(".")[0]

        # ---- 連線並逐一跑 model ----
        client_id = str(uuid.uuid4())
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(SERVER_ADDRESS, client_id))

        try:
            for model_name, config_path in MODELS.items():
                output_path = os.path.join(BASE_DIR, "results", run_name, model_name)
                base_seed = SEED if SEED is not None else random.randint(0, 100000000)

                run_model(
                    ws=ws,
                    client_id=client_id,
                    server_address=SERVER_ADDRESS,
                    model_name=model_name,
                    config_path=config_path,
                    test_prompts=test_prompts,
                    output_path=output_path,
                    base_seed=base_seed,
                    num_of_output=NUM_OF_OUTPUT,
                    performance_test=PERFORMANCE_TEST,
                    width=WIDTH,
                    height=HEIGHT,
                    memory_sample_interval_sec=GPU_MEMORY_SAMPLE_INTERVAL_SEC,
                    warmup_runs=WARMUP_RUNS,
                )

                if PERFORMANCE_TEST:
                    write_perf_summary(output_path)
        finally:
            # 關閉連線，避免在會重複呼叫的環境（如 Gradio）中累積 timeout。
            ws.close()
