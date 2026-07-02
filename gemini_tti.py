# t2i (text-to-image) batch runner，用 Gemini image model（Interactions API）取代 ComfyUI。
# 走 websocket_tti.py 的 flat 版面：一份 {"prompts": [...]}，沒有 source image。
#
# 差異同 gemini_edit.py：沒有 seed（檔名用 tti_{prompt_idx}_{n}.jpg，n 從 1 起算，不是
# _seed{S}）、沒有 VRAM 可測、不需要 warmup。
#
# 執行前需要 export GEMINI_API_KEY / GEMINI_BASE_URL（見 gemini_client.py），不要寫進程式碼。

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from gemini_client import DEFAULT_MODEL, generate_image
from perf_utils import aggregate_records, append_jsonl, read_jsonl


def run_model(model, model_label, test_prompts, output_path, num_of_output, performance_test, concurrency=1):
    """跑單一 model 的所有 prompt；performance_test=True 時額外記錄 performance.jsonl。

    concurrency>1 時用多執行緒同時發出多個請求——Gemini API 是無狀態的 HTTP 呼叫，不像
    ComfyUI 只有單一 websocket 排隊執行，平行打多張沒有序列化限制。concurrency 太高容易
    先撞到 gateway 的 rate limit（gemini_client.py 已對 429/5xx 做重試，但太高還是會變慢）。

    注意：concurrency>1 時 performance.jsonl 裡的 elapsed_sec 是「並發負載下」的延遲，
    不是單一請求獨立跑的乾淨延遲；要量測後者請把 concurrency 設回 1。
    """
    os.makedirs(output_path, exist_ok=True)
    perf_log_path = os.path.join(output_path, "performance.jsonl")
    errors_log_path = os.path.join(output_path, "errors.jsonl")
    log_lock = threading.Lock()

    prompts = test_prompts["prompts"]
    work_items = [
        (prompt_idx, test_prompt, n, os.path.join(output_path, f"tti_{prompt_idx}_{n}.jpg"))
        for prompt_idx, test_prompt in enumerate(prompts)
        for n in range(1, num_of_output + 1)
    ]
    work_items = [item for item in work_items if not os.path.exists(item[3])]

    def _run_one(item):
        prompt_idx, test_prompt, n, out_file = item
        start = time.time()
        try:
            img_bytes, _mime, raw = generate_image(test_prompt, model=model)
        except Exception as e:
            # 安全過濾器擋圖、API 一直失敗等都算——跳過這一筆，不要讓整批中斷。
            # 不留檔案就好，下次重跑會自動因為 out_file 不存在而重試。
            print(f"\033[93m[warning] skipped {os.path.basename(out_file)}: {e}\033[0m")
            with log_lock:
                append_jsonl({
                    "prompt_idx": prompt_idx, "out_idx": n,
                    "output_filename": os.path.basename(out_file), "error": str(e),
                }, errors_log_path)
            return
        elapsed = time.time() - start

        with open(out_file, "wb") as f:
            f.write(img_bytes)

        if performance_test:
            record = {
                "model": model_label,
                "prompt_idx": prompt_idx,
                "out_idx": n,
                "elapsed_sec": round(elapsed, 3),
                "request_id": raw.get("id"),
            }
            with log_lock:
                append_jsonl(record, perf_log_path)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, item) for item in work_items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=model_label):
            future.result()  # _run_one 已經吞掉生成錯誤了，這裡拋出的是真正未預期的例外

    n_errors = len(read_jsonl(errors_log_path))
    if n_errors:
        print(f"\033[93m{n_errors} item(s) skipped due to errors — see {errors_log_path}\033[0m")

    if performance_test:
        summary = aggregate_records(read_jsonl(perf_log_path))
        with open(os.path.join(output_path, "performance_summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Performance summary ({summary.get('n_samples', 0)} samples): "
              f"avg {summary.get('avg_elapsed_sec')}s")


if __name__ == "__main__":
    # ---- 設定 ----
    BASE_DIR = "/workspace/data/allen/dataset/benchmark_testing/text_to_image"
    RUNNING_JSON = ["tti.json", "portrait.json"]

    MODEL = DEFAULT_MODEL  # "gemini-3.1-flash-lite-image"
    MODEL_LABEL = "gemini-3.1-flash-lite-image"

    NUM_OF_OUTPUT = 1
    PERFORMANCE_TEST = False  # True：額外記錄 performance.jsonl / performance_summary.json
    CONCURRENCY = 4  # 同時發出的請求數；1 = 依序執行。太高容易撞 rate limit，請自行調整

    for json_file in RUNNING_JSON:
        print(f"\n=== Running {json_file} ===")
        with open(os.path.join(BASE_DIR, "meta_data", json_file), "r") as f:
            test_prompts = json.loads(f.read())

        run_name = json_file.split(".")[0]
        output_path = os.path.join(BASE_DIR, "results", run_name, MODEL_LABEL)

        run_model(
            model=MODEL,
            model_label=MODEL_LABEL,
            test_prompts=test_prompts,
            output_path=output_path,
            num_of_output=NUM_OF_OUTPUT,
            performance_test=PERFORMANCE_TEST,
            concurrency=CONCURRENCY,
        )
