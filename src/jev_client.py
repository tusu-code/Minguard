"""TypeSafe System One（Jev）调用客户端。

只依赖 httpx，支持三个后端：

- ``openrouter``：``https://openrouter.ai/api`` + ``/v1/systemone``，原生 System One 协议（推荐；
  TypeSafe 官方接口对当前地区返回 451）
- ``typesafe``：``https://api.typesafe.ai`` + ``/v1/systemone``
- ``vercel``：``https://ai-gateway.vercel.sh/v1`` + ``/decisions``，OpenAI Decisions 协议
  （不支持图像输入和 box 问题）

内置磁盘缓存（按请求内容哈希，训练重跑不重复付费）、限速（默认 20 请求/秒，官方上限 1200/分钟）、
429/5xx 退避重试、花费统计。图像输入属于官方的实验功能，需要 ``TypeSafe-Danger-Zone-Images`` 开关，
第三方后端是否支持要在拿到 key 后实测。
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import httpx

DEFAULT_PRICE_PER_MTOK = 0.042  # 美元 / 百万输入 token，TypeSafe 现行定价；输出不计费

BACKENDS = {
    "typesafe": {"base_url": "https://api.typesafe.ai", "path": "/v1/systemone",
                 "key_env": "TYPESAFE_API_KEY", "kind": "systemone"},
    "aliyun": {"base_url": "https://{workspace}.{region}.maas.aliyuncs.com/compatible-mode",
               "path": "/v1/systemone", "key_env": "DASHSCOPE_API_KEY", "kind": "systemone"},
    "openrouter": {"base_url": "https://openrouter.ai/api", "path": "/v1/systemone",
                   "key_env": "OPENROUTER_API_KEY", "kind": "systemone"},
    "edgeone": {"base_url": "https://ai-gateway.edgeone.link", "path": "/v1/systemone",
                "key_env": "MAKERS_MODELS_KEY", "kind": "systemone"},
    # 万维易源的 JevK5：与 Jev 同接口的开源实现（基于 Qwen 权重），不是 TypeSafe 的 Jev
    "showapi": {"base_url": "https://jev.showapi.com", "path": "/v1/systemone",
                "key_env": "SHOWAPI_APPKEY", "kind": "systemone"},
    "vercel": {"base_url": "https://ai-gateway.vercel.sh/v1", "path": "/decisions",
               "key_env": "AI_GATEWAY_API_KEY", "kind": "decisions"},
}

DEFAULT_MODELS = {
    "typesafe": "jev-1.14.0",
    "aliyun": "decision-model-preview",
    "openrouter": "typesafe/jev-1.13",
    # EdgeOne 只认内置模型名，传别的值会返回 400
    "edgeone": "@makers/jev",
    "showapi": "JevK5-V0.3",
    "vercel": "typesafe-ai/jev-1.14.0",
}


def resolve_base_url(spec: Mapping[str, Any]) -> str:
    """解析服务地址；阿里云的地址里要填工作空间 ID 与地域。"""
    url = str(spec["base_url"])
    if "{workspace}" in url:
        workspace = os.environ.get("ALIYUN_WORKSPACE_ID")
        if not workspace:
            raise JevError("使用阿里云百炼需要设置 ALIYUN_WORKSPACE_ID（在控制台的服务地址里可见）")
        url = url.replace("{workspace}", workspace)
        url = url.replace("{region}", os.environ.get("ALIYUN_REGION", "cn-beijing"))
    return url


class JevError(RuntimeError):
    """Jev 请求失败（重试耗尽后抛出）。"""


class RateLimiter:
    """滑动窗口限速：每秒最多 ``max_rps`` 个请求。"""

    def __init__(self, max_rps: float) -> None:
        self.max_rps = max_rps
        self._times: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._times = [t for t in self._times if now - t < 1.0]
                if len(self._times) < self.max_rps:
                    self._times.append(now)
                    return
                wait = 1.0 - (now - self._times[0])
            time.sleep(max(wait, 0.005))


@dataclass
class JevStats:
    requests: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    retries: int = 0
    errors: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def cost_usd(self, price_per_mtok: float = DEFAULT_PRICE_PER_MTOK) -> float:
        return self.input_tokens / 1e6 * price_per_mtok


class Cache:
    """按请求指纹缓存回答，避免训练重跑时重复调用付费接口。"""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT)")
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


def _encode_image(path: str | os.PathLike[str]) -> tuple[str, str]:
    """读入图片，返回 (base64, 格式)。"""
    data = Path(path).read_bytes()
    suffix = Path(path).suffix.lower().lstrip(".")
    fmt = {"jpg": "jpg", "jpeg": "jpeg", "png": "png"}[suffix]
    return base64.b64encode(data).decode(), fmt


def _to_systemone_questions(questions: Mapping[str, Mapping[str, Any]]) -> dict:
    return {name: dict(q) for name, q in questions.items()}


def _to_decisions_questions(questions: Mapping[str, Mapping[str, Any]]) -> list[dict]:
    """System One 问题 → OpenAI Decisions 问题（choice/score 的字段名不同）。"""
    out = []
    for name, q in questions.items():
        kind = q["type"]
        item: dict[str, Any] = {"name": name}
        if q.get("instructions") is not None:
            item["instructions"] = q["instructions"]
        if kind == "noul":
            item["type"] = "predicate"
        elif kind == "choice":
            item["type"] = "choice"
            item["choices"] = [{"value": k, "description": v} for k, v in q["criteria"].items()]
        elif kind == "score":
            item["type"] = "score"
            item["levels"] = [{"label": str(i), "description": c} for i, c in enumerate(q["criteria"])]
        else:
            raise ValueError(f"vercel 后端不支持问题类型：{kind}")
        out.append(item)
    return out


def _from_decisions_answers(answers: Sequence[Mapping[str, Any]]) -> dict:
    """OpenAI Decisions 回答 → System One 形状（noul / choice / score）。"""
    out = {}
    for a in answers:
        name = a.get("name") or f"q{len(out)}"
        kind = a.get("type")
        if kind == "predicate":
            out[name] = {"type": "noul", "noul": float(a["probability"])}
        elif kind == "choice":
            probs = a.get("probabilities") or {}
            choice = a.get("choice")
            out[name] = {"type": "choice", "choice": choice,
                         "confidence": float(a.get("confidence", probs.get(choice, 1.0))),
                         "probabilities": {k: float(v) for k, v in probs.items()}}
        elif kind == "score":
            probs = {str(k): float(v) for k, v in (a.get("probabilities") or {}).items()}
            out[name] = {"type": "score", "score": float(a["score"]),
                         "probabilities": probs,
                         "confidence": float(a.get("confidence", 0.0))}
        else:
            out[name] = dict(a)
    return out


class JevClient:
    """Jev 调用入口。``ask`` 是统一接口，``ask_noul`` 返回各问题的“是”概率。"""

    def __init__(
        self,
        backend: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        cache_path: str | os.PathLike[str] | None = None,
        max_rps: float = 20.0,
        max_retries: int = 6,
        timeout: float = 60.0,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.backend = (backend or os.environ.get("JEV_BACKEND") or "openrouter").lower()
        if self.backend not in BACKENDS:
            raise ValueError(f"未知后端 {self.backend}，可选：{sorted(BACKENDS)}")
        spec = BACKENDS[self.backend]
        self.kind = spec["kind"]
        # JEV_BASE_URL 可整体覆盖地址，用于自建网关或临时切换
        self.url = (os.environ.get("JEV_BASE_URL") or resolve_base_url(spec)).rstrip("/") + spec["path"]
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODELS[self.backend]
        self.api_key = api_key or os.environ.get(spec["key_env"]) or os.environ.get("JEV_API_KEY")
        if not self.api_key:
            raise JevError(f"缺少 API key：请设置 {spec['key_env']}（或 JEV_API_KEY）")
        if cache_path is None:
            cache_path = os.environ.get("JEV_CACHE", str(Path(os.environ.get("JS", ".")) / "cache/jev_cache.sqlite"))
        self.cache = Cache(cache_path)
        self.limiter = RateLimiter(max_rps)
        self.max_retries = max_retries
        self.stats = JevStats()
        self._stats_lock = threading.Lock()
        self._client = httpx.Client(timeout=timeout, headers=self._headers(extra_headers))

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        if extra:
            headers.update(extra)
        return headers

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        model: str | None = None,
        images: Sequence[str | os.PathLike[str]] | None = None,
        use_cache: bool = True,
    ) -> dict:
        """向 Jev 提一组问题。

        ``images`` 非空时把图片作为 ``assets`` 内联传人（需要危险区开关），
        并在 ``state`` 文本里自动插入引用占位符。
        """
        model = model or self.model
        state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
        assets: dict[str, dict] = {}
        if images:
            refs = []
            for path in images:
                ref = f"image_{uuid.uuid4().hex[:8]}"
                payload, fmt = _encode_image(path)
                assets[ref] = {"type": "image", "format": fmt, "data_base64": payload}
                refs.append(ref)
            state_text = "".join(f"[{r}]\n" for r in refs) + state_text

        if self.kind == "systemone":
            body: dict[str, Any] = {"model": model, "state": state_text,
                                    "questions": _to_systemone_questions(questions)}
            if assets:
                body["assets"] = assets
        else:
            body = {"model": model, "input": state_text,
                    "questions": _to_decisions_questions(questions)}

        headers = {"TypeSafe-Danger-Zone-Images": "true"} if assets else None
        key = hashlib.sha256(json.dumps([self.backend, body], sort_keys=True).encode()).hexdigest()
        if use_cache:
            cached = self.cache.get(key)
            if cached is not None:
                self.stats.cache_hits += 1
                return cached

        payload = self._post(body, headers)
        if self.kind == "decisions":
            payload = {"model": payload.get("model", model),
                       "answers": _from_decisions_answers(payload.get("answers", [])),
                       "usage": payload.get("usage", {})}
        if use_cache:
            self.cache.put(key, payload)
        return payload

    def ask_noul(self, state: Any, instructions: Mapping[str, str],
                 model: str | None = None, images: Sequence[str | os.PathLike[str]] | None = None,
                 with_criteria: bool = True) -> dict[str, float]:
        """一次请求问多个 noul 问题，返回 ``{问题名: P(是)}``。

        ``with_criteria`` 会给每个 noul 问题补上 true/false 两侧的说明。实测必需：省略时
        百炼决策模型对拒答判据的 AUROC 只有 0.383（比随机还差），补上后升到 0.773。
        """
        questions: dict[str, dict[str, Any]] = {}
        for name, text in instructions.items():
            question: dict[str, Any] = {"type": "noul", "instructions": text}
            if with_criteria:
                # 用固定短句，不重复判据原文：重复一遍会让每次请求多出约 350 token
                question["criteria"] = {
                    "true": "Yes, this holds for the RESPONSE.",
                    "false": "No, this does not hold for the RESPONSE.",
                }
            questions[name] = question
        answers = self.ask(state, questions, model=model, images=images)["answers"]
        return {name: float(ans["noul"]) for name, ans in answers.items()}

    def _post(self, body: dict, headers: Mapping[str, str] | None) -> dict:
        delay = 2.0
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            try:
                response = self._client.post(self.url, json=body, headers=self._headers(headers))
            except httpx.HTTPError as exc:  # 网络层错误同样重试
                last_error = f"{type(exc).__name__}: {exc}"
                self.stats.errors += 1
            else:
                if response.status_code == 200:
                    payload = response.json()
                    usage = payload.get("usage") or {}
                    answered = payload.get("model", self.model)
                    with self._stats_lock:  # 奖励计算会多线程并发调用
                        self.stats.requests += 1
                        self.stats.input_tokens += int(usage.get("input_tokens") or 0)
                        self.stats.by_model[answered] = self.stats.by_model.get(answered, 0) + 1
                    # 免费额度有限（EdgeOne 每月 50 万 token），超过 JEV_TOKEN_BUDGET 就停下，避免意外超额
                    budget = int(os.environ.get("JEV_TOKEN_BUDGET") or 0)
                    if budget and self.stats.input_tokens > budget:
                        raise JevError(f"本进程 Jev 用量 {self.stats.input_tokens} token 已超过上限 {budget}")
                    return payload
                # 429 限流、408 与 5xx 服务端错误可重试；4xx 其余是请求本身的问题，直接失败
                retryable = response.status_code in (408, 429) or response.status_code >= 500
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                if not retryable:
                    raise JevError(f"Jev 请求失败（不可重试）{last_error}")
                self.stats.errors += 1
                retry_after = response.headers.get("retry-after-ms") or response.headers.get("retry-after")
                if retry_after:
                    try:
                        delay = max(delay, float(retry_after) / (1000 if float(retry_after) > 100 else 1))
                    except ValueError:
                        pass
            if attempt < self.max_retries:
                self.stats.retries += 1
                time.sleep(min(delay, 60.0))
                delay *= 2
        raise JevError(f"Jev 重试 {self.max_retries} 次后仍失败：{last_error}")

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "JevClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def make_client(backend: str | None = None, **kwargs: Any) -> JevClient:
    """按环境变量构造客户端；缺 key 时抛出 JevError，便于上游给出明确提示。"""
    return JevClient(backend=backend, **kwargs)
