# 共用的 ComfyUI websocket/HTTP API client，供 websocket_edit.py / websocket_tti.py 共用，
# 避免兩邊各自維護一份幾乎一樣的 queue_prompt/get_outputs。

import json
import urllib.parse
import urllib.request


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
    """執行 ComfyUI 工作流並取得輸出。輸出類型目前分 text/images/other 三種；
    video 工作流（VHS_VideoCombine 等）的輸出會落在 'other'，之後要支援
    t2v/i2v 時，在這裡加一個 'gifs'/'videos' 分支即可，兩邊 batch script 都能沿用。"""
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
