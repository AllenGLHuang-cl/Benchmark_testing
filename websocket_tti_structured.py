# 吃一個 folder 內所有 structured JSON prompt 檔案，逐一餵給 TTI model（ideogram4）生成圖片。
#
# 與 websocket_tti.py 的差異：
#   - websocket_tti.py 吃單一 {"prompts": [...]} 檔，prompt 是自然語言字串。
#   - 本檔吃整個資料夾，每個 *.json 是一個測試案例；prompt 本身就是 structured JSON，
#     model 可直接消化，因此不做轉換，只把 prompt 物件序列化後塞進 CLIPTextEncode 的 text。

import glob
import io
import json
import os
import random
import urllib.parse
import urllib.request
import uuid

import websocket  # NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 參數更新工具
# ---------------------------------------------------------------------------
def update_params_by_name(params, input_name, value):
    """更新所有含有 input_name 的 node。"""
    for _, v in params.items():
        if input_name in v.get("inputs", {}):
            v["inputs"][input_name] = value
    return params


def update_params_by_class(params, class_type, input_name, value):
    """更新所有 class_type 相符的 node。"""
    for _, v in params.items():
        if class_type == v.get("class_type", ""):
            v["inputs"][input_name] = value
    return params


# ---------------------------------------------------------------------------
# ComfyUI HTTP / WebSocket 客戶端
# ---------------------------------------------------------------------------
def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode("utf-8")
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type, server_address):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()


def get_history(prompt_id, server_address):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())


def get_outputs(ws, prompt, client_id, server_address):
    """執行 ComfyUI 工作流並取得輸出（文字與圖片）。"""
    prompt_id = queue_prompt(prompt, client_id, server_address)["prompt_id"]
    output_data = {}

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message["type"] == "executing":
                data = message["data"]
                if data["node"] is None and data["prompt_id"] == prompt_id:
                    break  # 執行完成
        else:
            continue  # preview 是 binary data

    history = get_history(prompt_id, server_address)[prompt_id]
    for node_id, node_output in history["outputs"].items():
        if "text" in node_output:
            output_data[node_id] = {"type": "text", "data": node_output["text"]}
        elif "images" in node_output:
            images_output = [
                get_image(img["filename"], img["subfolder"], img["type"], server_address)
                for img in node_output["images"]
            ]
            output_data[node_id] = {"type": "images", "data": images_output}
        else:
            output_data[node_id] = {"type": "other", "data": node_output}

    return output_data


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


# ---------------------------------------------------------------------------
# structured prompt 載入 / 注入
# ---------------------------------------------------------------------------
def load_structured_prompt(json_path):
    """讀取單一測試檔，回傳要餵給 model 的 structured JSON 字串。

    支援兩種檔案：
      - 含 metadata 的：{"id":..., "test_focus":..., "prompt": {...}} → 取 prompt 物件
      - 裸 prompt 物件：{"high_level_description":..., ...}           → 整份即 prompt
    ensure_ascii=False 以保留 CJK 等非 ASCII 字元。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    prompt = data["prompt"] if isinstance(data, dict) and "prompt" in data else data
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def load_folder_prompts(prompt_dir):
    """structured 模式：資料夾內每個 *.json 一個 case。

    回傳 [(case_id, prompt_text), ...]，prompt_text 是序列化後的 structured JSON 字串。
    """
    cases = []
    for path in sorted(glob.glob(os.path.join(prompt_dir, "*.json"))):
        case_id = os.path.splitext(os.path.basename(path))[0]
        try:
            cases.append((case_id, load_structured_prompt(path)))
        except (ValueError, KeyError) as e:
            print(f"\033[93m[warning] 跳過 {case_id}（無法解析）：{e}\033[0m")
    return cases


def load_list_prompts(json_path):
    """list 模式：讀 {"prompts": [...]}（一行/一個字串代表一個 prompt）。

    回傳 [(case_id, prompt_text), ...]，prompt_text 是自然語言字串、直接餵給 model。
    case_id 用補零的索引（000, 001, ...）方便排序與對齊。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.loads(f.read())
    prompts = data["prompts"] if isinstance(data, dict) else data
    pad = max(3, len(str(len(prompts) - 1)))
    return [(f"{i:0{pad}d}", p) for i, p in enumerate(prompts)]


def set_combo_choice(params, choice):
    """設定 CustomCombo 的選擇（Quality / Default / Turbo）。

    這顆 node 同時有 choice 與 index（對應 option1..option4），不確定內部是看哪個，
    所以兩者一起設、保持一致，避免改了 choice 但實際是用 index 在選而沒生效。
    """
    for v in params.values():
        if v.get("class_type") == "CustomCombo":
            inputs = v["inputs"]
            options = [inputs.get(f"option{i}", "") for i in range(1, 5)]
            if choice not in options:
                raise ValueError(f"mode '{choice}' 不在 CustomCombo 選項 {options} 內")
            inputs["choice"] = choice
            inputs["index"] = options.index(choice)
    return params


def build_params(config_path, prompt_str, seed, mode=None, aspect_ratio=None):
    """載入 config，把 structured JSON 字串塞進 CLIPTextEncode 並填好 seed / 模式。"""
    with open(config_path, "r") as f:
        params = json.loads(f.read())

    # ideogram4 config 只有一個 CLIPTextEncode（positive），其 text 就是 structured JSON 字串。
    params = update_params_by_class(params, "CLIPTextEncode", "text", prompt_str)
    params = update_params_by_name(params, "noise_seed", seed)
    params = update_params_by_name(params, "seed", seed)
    # 模式（CustomCombo）決定 steps / mu / std 這組 sampling preset。
    if mode is not None:
        params = set_combo_choice(params, mode)
    # 解析度由 ResolutionSelector 控制；如需覆寫長寬比，改它的 aspect_ratio（其餘維持 config 預設）。
    if aspect_ratio is not None:
        params = update_params_by_class(params, "ResolutionSelector", "aspect_ratio", aspect_ratio)
    return params


# ---------------------------------------------------------------------------
# 主流程：跑一批 (case_id, prompt) 案例
# ---------------------------------------------------------------------------
def run_batch(
    ws,
    client_id,
    server_address,
    desc,
    config_path,
    cases,
    output_path,
    base_seed,
    num_of_output,
    mode=None,
    aspect_ratio=None,
):
    """把 cases（[(case_id, prompt_text), ...]）在指定模式下逐一跑過。

    structured 與 list 兩種來源都先各自整理成 cases，再共用這個生成迴圈。
    """
    os.makedirs(output_path, exist_ok=True)

    for case_id, prompt_str in tqdm(cases, desc=desc):
        for i in range(num_of_output):
            seed = base_seed + i
            out_file = os.path.join(output_path, f"{case_id}_seed{seed}.jpg")

            if os.path.exists(out_file):  # 已生成就跳過，方便中斷後續跑
                continue

            params = build_params(config_path, prompt_str, seed, mode=mode, aspect_ratio=aspect_ratio)
            outputs = get_outputs(ws, params, client_id, server_address)
            if not save_first_image(outputs, out_file):
                print(f"\033[93m[warning] {case_id} seed {seed} 沒有圖片輸出\033[0m")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # ---- 設定 ----
    SERVER_ADDRESS = "127.0.0.1:7891"
    BASE_DIR = "/workspace/data/allen/dataset/tti_test"

    # 測試資料來源（兩種都支援，切換 SOURCE 即可）：
    #   "structured": 吃 PROMPT_DIR 內所有 *.json，每檔一個 structured JSON prompt（直接餵 model）。
    #   "list":       吃 PROMPT_LIST 的 {"prompts": [...]}，每個字串一個自然語言 prompt。
    SOURCE = "list"
    PROMPT_DIR = os.path.join(BASE_DIR, "meta_data", "structured", "json_prompts")
    PROMPT_LIST = os.path.join(BASE_DIR, "meta_data", "tti.json")

    # model_name -> config 路徑
    MODELS = {
        "ideogram4-t2i": "./model_configs/image_ideogram4_t2i_api.json",
    }

    # 每個 model 要跑的 sampling 模式（CustomCombo 的 choice）。
    # 三種一起跑 → 同一個 prompt、同一個 seed 在 Quality / Default / Turbo 下各生一張，方便比較。
    MODES = ["Quality", "Default", "Turbo"]

    NUM_OF_OUTPUT = 1
    SEED = None  # None = 每個 model 隨機；給定整數則固定 seed（方便不同 model 對齊比較）

    # 長寬比覆寫；None = 沿用 config 的 ResolutionSelector 預設（1:1 Square）。
    # 可選字串需與 ResolutionSelector 的選項相符，例如 "16:9 (Widescreen)"、"1:1 (Square)"。
    ASPECT_RATIO = None

    # ---- 依來源整理出 cases；run_name 決定輸出 results/<run_name>/... ----
    if SOURCE == "structured":
        cases = load_folder_prompts(PROMPT_DIR)
        run_name = os.path.basename(os.path.normpath(PROMPT_DIR))
    elif SOURCE == "list":
        cases = load_list_prompts(PROMPT_LIST)
        run_name = os.path.splitext(os.path.basename(PROMPT_LIST))[0]
    else:
        raise ValueError(f"未知的 SOURCE：{SOURCE!r}（只支援 'structured' 或 'list'）")

    if not cases:
        raise SystemExit(f"找不到任何測試案例（SOURCE={SOURCE}），請確認資料路徑。")
    print(f"來源 {SOURCE}：{len(cases)} 個案例 × {len(MODES)} 模式 → run_name={run_name}")

    # ---- 連線並逐一跑 model ----
    client_id = str(uuid.uuid4())
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(SERVER_ADDRESS, client_id))

    try:
        for model_name, config_path in MODELS.items():
            # base_seed 在 mode 迴圈外算一次 → 三種模式共用同一組 seed，可 1:1 比較。
            base_seed = SEED if SEED is not None else random.randint(0, 100000000)

            for mode in MODES:
                # 各模式分到自己的子資料夾：results/<run>/<model>/<mode>/
                output_path = os.path.join(BASE_DIR, "results", run_name, model_name, mode)

                run_batch(
                    ws=ws,
                    client_id=client_id,
                    server_address=SERVER_ADDRESS,
                    desc=f"{model_name} [{mode}]",
                    config_path=config_path,
                    cases=cases,
                    output_path=output_path,
                    base_seed=base_seed,
                    num_of_output=NUM_OF_OUTPUT,
                    mode=mode,
                    aspect_ratio=ASPECT_RATIO,
                )
                print(f"完成 {model_name} [{mode}]，圖片存於 {output_path}")
    finally:
        # 關閉連線，避免在會重複呼叫的環境（如 Gradio）中累積 timeout。
        ws.close()
