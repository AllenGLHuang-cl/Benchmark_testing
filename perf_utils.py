# 共用的 GPU 效能量測工具，供 websocket_tti.py / websocket_edit.py 共用，
# 避免兩邊各自維護一份 GpuMemorySampler。

import json
import os
import threading
import urllib.request
from collections import Counter


def get_gpu_memory(server_address, device_index=0, timeout=None):
    """從 ComfyUI /system_stats 取得指定 device 的 GPU 記憶體使用量（bytes）與名稱。"""
    url = "http://{}/system_stats".format(server_address)
    if timeout is None:
        response_ctx = urllib.request.urlopen(url)
    else:
        response_ctx = urllib.request.urlopen(url, timeout=timeout)

    with response_ctx as response:
        stats = json.loads(response.read())
    devices = stats.get("devices", [])
    if device_index >= len(devices):
        return {
            "vram_used": None,
            "comfy_vram_used": None,
            "torch_vram_used": None,
            "device_name": None,
        }
    dev = devices[device_index]
    torch_vram_free = dev["torch_vram_free"]

    # ComfyUI 的 vram_free 會把 PyTorch reserved-but-inactive cache 也算成可用；
    # 扣回 torch_vram_free 後，才接近 nvidia-smi 顯示的 driver-level used。
    driver_vram_free = max(0, dev["vram_free"] - torch_vram_free)
    return {
        "vram_used": dev["vram_total"] - driver_vram_free,
        "comfy_vram_used": dev["vram_total"] - dev["vram_free"],
        "torch_vram_used": dev["torch_vram_total"] - torch_vram_free,
        "device_name": dev.get("name"),
    }


class GpuMemorySampler:
    """背景輪詢 GPU memory，避免只在 request 結束後取樣而錯過瞬間峰值。"""

    def __init__(self, server_address, device_index=0, interval_sec=0.05, timeout_sec=1.0):
        self.server_address = server_address
        self.device_index = device_index
        self.interval_sec = interval_sec
        self.timeout_sec = timeout_sec
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._latest = None
        self._peaks = {
            "vram_used": None,
            "comfy_vram_used": None,
            "torch_vram_used": None,
        }
        self._sample_count = 0
        self._error_count = 0

    def start(self):
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.timeout_sec + self.interval_sec + 0.1)
        self._sample_once()
        with self._lock:
            return {
                "latest": dict(self._latest) if self._latest is not None else None,
                "peaks": dict(self._peaks),
                "sample_count": self._sample_count,
                "error_count": self._error_count,
            }

    def _run(self):
        while not self._stop_event.wait(self.interval_sec):
            self._sample_once()

    def _sample_once(self):
        try:
            mem = get_gpu_memory(
                self.server_address,
                device_index=self.device_index,
                timeout=self.timeout_sec,
            )
        except Exception:
            with self._lock:
                self._error_count += 1
            return

        with self._lock:
            self._latest = mem
            self._sample_count += 1
            for key, value in mem.items():
                if key == "device_name" or value is None:
                    continue
                if self._peaks.get(key) is None or value > self._peaks[key]:
                    self._peaks[key] = value


def to_gib(num_bytes):
    return None if num_bytes is None else round(num_bytes / (1024 ** 3), 2)


def mem_value(mem, key):
    return None if mem is None else mem.get(key)


def append_jsonl(record, path):
    """把一筆 dict 記錄以 append 方式寫進 jsonl，中斷續跑也不會遺失先前結果。"""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path):
    if not os.path.isfile(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def warmup(run_once_fn, n=2):
    """執行 n 次暖身生成（結果丟棄、不進效能記錄），讓 GPU/cache 穩定後再開始正式量測。"""
    for _ in range(n):
        run_once_fn()


def aggregate_records(records):
    """把多筆 per-run 效能記錄彙總成一筆總結（平均耗時、VRAM 峰值等），
    供 batch script 收尾列印/寫檔，也給 PPT 的 auto perf 共用同一套公式。"""
    if not records:
        return {}

    def values(key):
        return [r[key] for r in records if r.get(key) is not None]

    def most_common(key):
        vals = values(key)
        return Counter(vals).most_common(1)[0][0] if vals else None

    elapsed = values("elapsed_sec")
    vram_peak = values("vram_peak_gib")
    summary = {
        "n_samples": len(records),
        "model": most_common("model"),
        "device": most_common("device"),
        "weight_type": most_common("weight_type"),
        "steps": most_common("steps"),
        "avg_elapsed_sec": round(sum(elapsed) / len(elapsed), 3) if elapsed else None,
        "vram_peak_gib": max(vram_peak) if vram_peak else None,
    }
    w, h = most_common("output_width"), most_common("output_height")
    if w and h:
        summary["output_size"] = f"{w}x{h}"
    return summary
