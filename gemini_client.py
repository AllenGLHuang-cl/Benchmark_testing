# 呼叫內部 Gemini image-gen gateway 的最小 client（Google 的 Interactions API：
# POST {base_url}/v1beta/interactions），跟 comfy_client.py 對 ComfyUI 的角色一樣，
# 只是這邊背後是遠端 hosted API，不是本機跑的 ComfyUI。
#
# 需要兩個環境變數（不要寫進程式碼或 commit 進 repo）：
#   GEMINI_API_KEY   內部 gateway 的 token（mbr_... 開頭，不是 Google 官方 API key）
#   GEMINI_BASE_URL  內部 gateway 的 base url，例如 http://<host>:<port>/gemini
#
# API 沒有 seed 參數可控（已實測確認），所以這裡的呼叫不像 ComfyUI 那樣能重現特定結果。

import base64
import json
import os
import time
import urllib.error
import urllib.request

DEFAULT_MODEL = "gemini-3.1-flash-lite-image"
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def _env(name, override=None):
    value = override if override is not None else os.environ.get(name)
    if not value:
        raise RuntimeError(f"缺少 {name}，請先 export {name}=... 再執行（不要寫進程式碼）")
    return value


def generate_image(prompt_text, image_bytes=None, image_mime_type="image/jpeg",
                    model=DEFAULT_MODEL, api_key=None, base_url=None,
                    timeout=120, max_retries=3):
    """呼叫 Interactions API 生成/編輯圖片。

    image_bytes=None -> text-to-image；給定圖片 bytes -> image+text 編輯。
    回傳 (image_bytes, mime_type, raw_response_dict)。raw_response 保留給呼叫端讀取
    "id"（request id，沒有 seed 可用時拿來當追溯用的識別碼）等其他欄位。
    """
    api_key = _env("GEMINI_API_KEY", api_key)
    base_url = _env("GEMINI_BASE_URL", base_url)

    input_parts = [{"type": "text", "text": prompt_text}]
    if image_bytes is not None:
        input_parts.append({
            "type": "image",
            "mime_type": image_mime_type,
            "data": base64.b64encode(image_bytes).decode("ascii"),
        })

    payload = json.dumps({"model": model, "input": input_parts}).encode("utf-8")
    url = f"{base_url.rstrip('/')}/v1beta/interactions"

    last_error = None
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read())
            return _extract_image(result)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"Gemini API 回傳 HTTP {e.code}: {body[:500]}")
            if e.code not in _RETRYABLE_HTTP_CODES or attempt == max_retries - 1:
                raise last_error from e
        except urllib.error.URLError as e:
            last_error = RuntimeError(f"Gemini API 連線失敗：{e}")
            if attempt == max_retries - 1:
                raise last_error from e
        time.sleep(2 ** (attempt + 1))  # 2s, 4s, 8s...
    raise last_error


def _extract_image(result):
    for step in result.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for item in step.get("content", []):
            if item.get("type") == "image":
                return base64.b64decode(item["data"]), item.get("mime_type", "image/jpeg"), result
    raise RuntimeError(f"Gemini 回應沒有圖片輸出（可能被安全機制擋掉）：{json.dumps(result, ensure_ascii=False)[:500]}")
