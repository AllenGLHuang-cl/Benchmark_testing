# i2i (single image editing) batch runner，用 Gemini image model（Interactions API）取代 ComfyUI。
# Dataset layout 跟 websocket_edit.py 完全一樣（dataset_root/<feature>/{src,meta_data,results/<model>/}），
# 只是生成端換成 gemini_client.generate_image()。
#
# 跟 websocket_edit.py 的差異：
#   - 沒有 seed 可控（Gemini API 沒有 seed 參數），檔名跟著既有的
#     gemini-3-pro-image-preview 慣例走：{stem}_{prompt_idx}_{n}.jpg（n 從 1 起算），
#     不是 ComfyUI 那邊的 {stem}_prompt{p}_out{i}_seed{S}.jpg。
#   - 沒有 GPU/VRAM 可量測（遠端 hosted API），performance.jsonl 只記錄 elapsed_sec。
#   - 不需要 warmup：每次 API 呼叫都是獨立的，沒有本機 ComfyUI 那種「同 prompt 重跑
#     會被 cache 加速」的問題。
#
# 執行前需要 export GEMINI_API_KEY / GEMINI_BASE_URL（見 gemini_client.py），不要寫進程式碼。

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from dataset_utils import get_image_files, load_prompts_for_image
from gemini_client import DEFAULT_MODEL, generate_image
from perf_utils import aggregate_records, append_jsonl, read_jsonl

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".bmp": "image/bmp",
}


def process_feature(feature_name, feature_path, model, output_name,
                     num_of_output=1, max_images_per_feature=10, concurrency=1):
    """處理單一 feature 資料夾。

    concurrency>1 時用多執行緒同時發出多個請求——Gemini API 是無狀態的 HTTP 呼叫，不像
    ComfyUI 只有單一 websocket 排隊執行，平行打多張沒有序列化限制。concurrency 太高容易
    先撞到 gateway 的 rate limit（gemini_client.py 已對 429/5xx 做重試，但太高還是會變慢）。

    注意：concurrency>1 時 performance.jsonl 裡的 elapsed_sec 是「並發負載下」的延遲，
    不是單一請求獨立跑的乾淨延遲；要量測後者請把 concurrency 設回 1。
    """
    src_folder = feature_path / "src"
    meta_data_folder = feature_path / "meta_data"
    output_folder = feature_path / "results" / output_name

    if not src_folder.exists() or not meta_data_folder.exists():
        print(f"Skipping {feature_name}: missing src or meta_data folder")
        return

    output_folder.mkdir(parents=True, exist_ok=True)
    perf_log_path = output_folder / "performance.jsonl"
    errors_log_path = output_folder / "errors.jsonl"
    log_lock = threading.Lock()

    # performance_test 資料夾額外記錄 benchmark 資訊，供之後做 PPT 用（跟 websocket_edit.py 對齊）
    record_performance = feature_name == "performance_test"

    image_files = get_image_files(src_folder, max_images_per_feature)
    print(f"\nProcessing feature: {feature_name}")
    print(f"Found {len(image_files)} images in {src_folder}")

    # 先把所有還沒生成的 (image, prompt, n) 組合列出來，圖片內容只讀一次
    work_items = []
    for image_file in image_files:
        prompts, prompt_source = load_prompts_for_image(image_file, meta_data_folder)
        if prompts is None:
            continue

        image_bytes = image_file.read_bytes()
        image_mime = _MIME_BY_EXT.get(image_file.suffix.lower(), "image/jpeg")

        for prompt_idx, prompt in enumerate(prompts):
            for n in range(1, num_of_output + 1):
                output_filename = f"{image_file.stem}_{prompt_idx}_{n}.jpg"
                output_path = output_folder / output_filename
                if not output_path.exists():
                    work_items.append((image_file, prompt, prompt_idx, n, output_path,
                                        image_bytes, image_mime))

    def _run_one(item):
        image_file, prompt, prompt_idx, n, output_path, image_bytes, image_mime = item
        start_time = time.time()
        try:
            img_bytes, _mime, raw = generate_image(
                prompt, image_bytes=image_bytes, image_mime_type=image_mime, model=model,
            )
        except Exception as e:
            # 安全過濾器擋圖、API 一直失敗等都算——跳過這一筆，不要讓整批中斷。
            # 不留檔案就好，下次重跑會自動因為 output_path 不存在而重試。
            print(f"\033[93m[warning] skipped {output_path.name}: {e}\033[0m")
            with log_lock:
                append_jsonl({
                    "image": image_file.name, "prompt_idx": prompt_idx, "out_idx": n,
                    "output_filename": output_path.name, "error": str(e),
                }, errors_log_path)
            return
        elapsed = time.time() - start_time

        output_path.write_bytes(img_bytes)

        if record_performance:
            record = {
                "model": output_name,
                "prompt_idx": prompt_idx,
                "out_idx": n,
                "elapsed_sec": round(elapsed, 3),
                "request_id": raw.get("id"),
                "image": image_file.name,
                "output_filename": output_path.name,
            }
            with log_lock:
                append_jsonl(record, perf_log_path)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_run_one, item) for item in work_items]
        for future in tqdm(as_completed(futures), total=len(futures), desc=feature_name):
            future.result()  # _run_one 已經吞掉生成錯誤了，這裡拋出的是真正未預期的例外

    n_errors = len(read_jsonl(errors_log_path))
    if n_errors:
        print(f"\033[93m{n_errors} item(s) skipped due to errors — see {errors_log_path}\033[0m")

    if record_performance:
        summary = aggregate_records(read_jsonl(perf_log_path))
        with open(output_folder / "performance_summary.json", "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Performance summary ({summary.get('n_samples', 0)} samples): "
              f"avg {summary.get('avg_elapsed_sec')}s")


if __name__ == "__main__":
    # 設定
    base_folder = Path("/workspace/data/allen/dataset/benchmark_testing/single_image_editing")
    model = DEFAULT_MODEL  # "gemini-3.1-flash-lite-image"
    output_name = "gemini-3.1-flash-lite-image"
    num_of_output = 1
    max_images_per_feature = 10
    concurrency = 1  # 同時發出的請求數；1 = 依序執行。太高容易撞 rate limit，請自行調整

    feature_folders = [f for f in base_folder.iterdir() if f.is_dir()]
    print(f"Found {len(feature_folders)} features in {base_folder}")

    for feature_folder in feature_folders:
        if feature_folder.name == "performance_test":
            process_feature(
                feature_name=feature_folder.name,
                feature_path=feature_folder,
                model=model,
                output_name=output_name,
                num_of_output=num_of_output,
                max_images_per_feature=max_images_per_feature,
                concurrency=concurrency,
            )

    print("\nAll features processed!")
