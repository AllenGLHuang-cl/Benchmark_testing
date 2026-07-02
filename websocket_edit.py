#This is an example that uses the websockets api and the SaveImageWebsocket node to get images directly without
#them being saved to disk

from tqdm import tqdm
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import random
import time
from PIL import Image
import io
from pathlib import Path

from comfy_client import get_outputs, update_params_by_class, update_params_by_name
from dataset_utils import get_image_files, load_prompts_for_image
from perf_utils import GpuMemorySampler, aggregate_records, append_jsonl, mem_value, read_jsonl, to_gib, warmup

def get_param_value_by_name(params, input_name):
    """讀取 workflow 中某個 input 的字面值（跳過 node 連線的 [node_id, slot] 參照）"""
    for k, v in params.items():
        value = v.get("inputs", {}).get(input_name)
        if value is not None and not isinstance(value, list):
            return value
    return None

def process_feature(feature_name, feature_path, config_path, output_name, server_address, client_id, ws, seed, num_of_output, max_images_per_feature=10, use_random_seed=False, weight_type=None, memory_sample_interval_sec=0.05, warmup_runs=2):
    """處理單一 feature 資料夾"""
    src_folder = feature_path / "src"
    meta_data_folder = feature_path / "meta_data"
    output_folder = feature_path / "results" / output_name

    if not src_folder.exists() or not meta_data_folder.exists():
        print(f"Skipping {feature_name}: missing src or meta_data folder")
        return

    # 創建輸出資料夾
    output_folder.mkdir(parents=True, exist_ok=True)
    seeds_log_path = output_folder / "seeds.jsonl"

    # 獲取所有圖片檔案
    image_files = get_image_files(src_folder, max_images_per_feature)

    print(f"\nProcessing feature: {feature_name}")
    print(f"Found {len(image_files)} images in {src_folder}")

    # 載入 config
    with open(config_path, "r") as f:
        params = json.loads(f.read())

    # performance_test 資料夾額外記錄 benchmark 資訊，供之後做 PPT 用
    record_performance = feature_name == "performance_test"
    if record_performance:
        num_of_output = 1
        perf_log_path = output_folder / "performance.jsonl"
        steps = get_param_value_by_name(params, "steps")
        if warmup_runs > 0:
            print(f"Warming up ({warmup_runs}x) before measuring...")
            warmup(lambda: get_outputs(ws, params, client_id, server_address), n=warmup_runs)

    # 處理每個圖片
    for image_file in tqdm(image_files, desc=f"{feature_name}"):
        # 載入 prompts (優先使用對應檔名的 JSON，否則使用 prompt.json)
        prompts, prompt_source = load_prompts_for_image(image_file, meta_data_folder)

        if prompts is None:
            continue

        # 對每個 prompt 生成圖片
        for prompt_idx, prompt in enumerate(prompts):
            for i in range(num_of_output):
                # 已存在對應 out index 的輸出（不論當初用了哪個 seed）就跳過，支援中斷續跑
                existing = list(output_folder.glob(f"{image_file.stem}_prompt{prompt_idx}_out{i}_seed*.jpg"))
                if existing:
                    continue

                actual_seed = random.randint(0, 2**32 - 1) if use_random_seed else seed + i
                output_filename = f"{image_file.stem}_prompt{prompt_idx}_out{i}_seed{actual_seed}.jpg"
                output_path = output_folder / output_filename

                # 更新參數
                params = update_params_by_class(params, "LoadImage", "image", str(image_file))
                params = update_params_by_class(params, "TextEncodeBooguEdit", "prompt", prompt)
                # params = update_params_by_class(params, "TextEncodeQwenImageEditPlus", "prompt", prompt)
                params = update_params_by_class(params, "CLIPTextEncode", "text", prompt)
                params = update_params_by_name(params, "noise_seed", actual_seed)
                params = update_params_by_name(params, "seed", actual_seed)

                # 獲取輸出（效能測試模式下背景輪詢 GPU memory，抓瞬間峰值而非事後單次 snapshot）
                sampler = GpuMemorySampler(server_address, interval_sec=memory_sample_interval_sec).start() if record_performance else None
                start_time = time.time()
                try:
                    outputs = get_outputs(ws, params, client_id, server_address)
                    inference_time = time.time() - start_time
                finally:
                    sampler_stats = sampler.stop() if sampler is not None else None

                # 保存圖片
                for node_id in outputs:
                    if outputs[node_id]['type'] == 'images':
                        image = Image.open(io.BytesIO(outputs[node_id]["data"][0]))
                        image.save(output_path)
                        break

                # 記錄這張圖實際用的 seed
                with open(seeds_log_path, "a") as f:
                    f.write(json.dumps({
                        "feature": feature_name,
                        "image": image_file.name,
                        "prompt_idx": prompt_idx,
                        "out_idx": i,
                        "seed": actual_seed,
                        "prompt": prompt,
                        "output_filename": output_filename,
                    }, ensure_ascii=False) + "\n")

                # 記錄 performance 資訊（欄位與 websocket_tti.py 對齊，方便共用同一套報表工具），供之後做 PPT 用
                if record_performance:
                    latest_mem = sampler_stats["latest"]
                    peak_mem = sampler_stats["peaks"]
                    append_jsonl({
                        "model": output_name,
                        "device": mem_value(latest_mem, "device_name"),
                        "weight_type": weight_type,
                        "steps": steps,
                        "output_width": image.width,
                        "output_height": image.height,
                        "prompt_idx": prompt_idx,
                        "out_idx": i,
                        "seed": actual_seed,
                        "elapsed_sec": round(inference_time, 3),
                        "vram_peak_gib": to_gib(mem_value(peak_mem, "vram_used")),
                        "vram_sample_count": sampler_stats["sample_count"],
                        "vram_sampling_errors": sampler_stats["error_count"],
                        "image": image_file.name,
                        "output_filename": output_filename,
                    }, perf_log_path)

    if record_performance:
        summary = aggregate_records(read_jsonl(perf_log_path))
        with open(output_folder / "performance_summary.json", "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Performance summary ({summary.get('n_samples', 0)} samples): "
              f"avg {summary.get('avg_elapsed_sec')}s, peak {summary.get('vram_peak_gib')} GiB")

if __name__ == "__main__":
    # 設定
    server_address = "127.0.0.1:7890"
    base_folder = Path("/data/allen/dataset/benchmark_testing/single_image_editing")
    config_path = "./model_configs/flux2_kelin_4b_edit-api.json"
    output_name = "flux2_klein_distill_4b"
    seed = 2025  # use_random_seed=False 時，第 i 張輸出用 seed+i
    use_random_seed = True  # True: 每張圖都 random 產生 seed 並記錄；False: 用固定 seed+i
    weight_type = "fp8"  # 這次跑的權重類型標籤，只有處理 performance_test 資料夾時會寫進 performance.jsonl
    warmup_runs = 2  # 只有處理 performance_test 資料夾時，正式量測前先暖身幾次（結果丟棄）
    num_of_output = 2
    max_images_per_feature = 10
    
    # 初始化 WebSocket 連接
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    
    # 獲取所有 feature 資料夾
    feature_folders = [f for f in base_folder.iterdir() if f.is_dir()]
    
    print(f"Found {len(feature_folders)} features in {base_folder}")
    
    # 處理每個 feature
    for feature_folder in feature_folders:
        feature_name = feature_folder.name
        if feature_name == "performance_test":
            process_feature(
                feature_name=feature_name,
                feature_path=feature_folder,
                config_path=config_path,
                output_name=output_name,
                server_address=server_address,
                client_id=client_id,
                ws=ws,
                seed=seed,
                num_of_output=num_of_output,
                max_images_per_feature=max_images_per_feature,
                use_random_seed=use_random_seed,
                weight_type=weight_type,
                warmup_runs=warmup_runs,
            )
    
    ws.close()
    print("\nAll features processed!")

