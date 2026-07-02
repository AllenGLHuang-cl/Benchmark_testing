# 共用的「features」資料集（dataset_root/<feature>/{src,meta_data,results/<model>/}）走訪工具，
# 供 websocket_edit.py / gemini_edit.py 共用。

import json
from pathlib import Path


def get_image_files(src_folder, max_images_per_feature=10):
    """獲取資料夾中所有圖片檔案"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []
    for file in Path(src_folder).iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
            if len(image_files) >= max_images_per_feature:
                break
    return sorted(image_files)


def load_prompts_for_image(image_file, meta_data_folder):
    """
    載入圖片對應的 prompts
    優先使用 {image_name}.json，如果不存在則使用 prompt.json
    """
    # 嘗試載入對應檔名的 JSON
    specific_meta_file = meta_data_folder / f"{image_file.stem}.json"

    if specific_meta_file.exists():
        with open(specific_meta_file, "r") as f:
            meta_data = json.loads(f.read())
        prompts = meta_data.get("prompt", [])
        source = f"{image_file.stem}.json"
    else:
        # 使用共用的 prompt.json
        shared_meta_file = meta_data_folder / "prompt.json"
        if shared_meta_file.exists():
            with open(shared_meta_file, "r") as f:
                meta_data = json.loads(f.read())
            prompts = meta_data.get("prompt", [])
            source = "prompt.json (shared)"
        else:
            print(f"Warning: No meta_data found for {image_file.name} and no prompt.json exists, skipping...")
            return None, None

    # 確保 prompts 是 list
    if not isinstance(prompts, list):
        prompts = [prompts]

    return prompts, source
