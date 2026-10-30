"""OpenRouter 上的生成式 LLM 客户端（判官基线与问题改写用），带磁盘缓存和费用统计。"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMStats:
    requests: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_s: list[float] = field(default_factory=list)


class LLMClient:
    def __init__(self, model: str, cache_path: str | os.PathLike[str] | None = None,
                 timeout: float = 120.0, max_retries: int = 5) -> None:
        self.model = model
        self.key = os.environ["OPENROUTER_API_KEY"]
        path = Path(cache_path or ROOT / "runs/llm_cache.sqlite")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self._lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout)
        self.max_retries = max_retries
        self.stats = LLMStats()

    def chat(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 600,
             response_format: dict | None = None, use_cache: bool = True) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "usage": {"include": True}}
        if response_format:
            body["response_format"] = response_format
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if use_cache:
            with self._lock:
                row = self._db.execute("SELECT v FROM c WHERE k=?", (key,)).fetchone()
            if row:
                self.stats.cache_hits += 1
                return json.loads(row[0])["content"]
        delay, last = 2.0, ""
        for _ in range(self.max_retries + 1):
            t0 = time.monotonic()
            try:
                r = self._client.post(URL, json=body, headers={"Authorization": f"Bearer {self.key}"})
            except httpx.HTTPError as e:
                last = str(e)
            else:
                if r.status_code == 200:
                    data = r.json()
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0]["message"].get("content") or ""
                        usage = data.get("usage") or {}
                        with self._lock:
                            self.stats.requests += 1
                            self.stats.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                            self.stats.completion_tokens += int(usage.get("completion_tokens") or 0)
                            self.stats.cost += float(usage.get("cost") or 0.0)
                            self.stats.latency_s.append(time.monotonic() - t0)
                            self._db.execute("INSERT OR REPLACE INTO c VALUES (?,?)",
                                             (key, json.dumps({"content": content, "usage": usage})))
                            self._db.commit()
                        return content
                    last = json.dumps(data)[:300]
                elif r.status_code not in (408, 429) and r.status_code < 500:
                    raise RuntimeError(f"LLM 请求失败 HTTP {r.status_code}: {r.text[:300]}")
                else:
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"LLM 重试耗尽：{last}")
