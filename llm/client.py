"""
Unified LLM client.

Provider-agnostic wrapper. Codex: implement for at least one provider
(OpenAI-compatible is easiest because Doubao, OpenAI, DeepSeek, Qwen
all expose OpenAI-style /v1/chat/completions endpoints).

Steven only needs to set API_KEY and BASE_URL in .env, and everything
should work.

Usage from elsewhere:
    from llm.client import LLMClient
    client = LLMClient.from_env()
    response = client.chat(
        messages=[
            {"role": "system", "content": "You are ..."},
            {"role": "user", "content": "..."},
        ],
        temperature=0.7,
        purpose="worker_request",   # for logging only
    )
"""

from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


@dataclass
class LLMConfig:
    api_key: str
    base_url: str        # e.g. "https://api.openai.com/v1" or 豆包 endpoint
    model: str           # e.g. "gpt-4o-mini" or "doubao-pro-32k"
    timeout_s: int = 180


class LLMClient:
    """
    Unified LLM client. ALL LLM calls in the project go through here so we
    have one place to add retries, logging, token counting, etc.
    """

    def __init__(self, config: LLMConfig):
        self.config = config
        from openai import OpenAI
        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_s,
        )

    @classmethod
    def from_env(cls, role: str = "default") -> "LLMClient":
        """
        Build a client from environment variables.

        Conventions:
            ANXIN_LLM_API_KEY, ANXIN_LLM_BASE_URL, ANXIN_LLM_MODEL
                — defaults for everything
            <ROLE>_LLM_API_KEY, <ROLE>_LLM_BASE_URL, <ROLE>_LLM_MODEL
                — per-role overrides (e.g. ADVISOR_LLM_..., NPC_LLM_...,
                  JUDGE_LLM_..., WORKER_LLM_..., DOUBAO_LLM_...)

        This lets Steven plug a strong model into the judge and a cheap
        model into NPCs without code changes.
        """
        prefix = role.upper()
        api_key = (
            os.getenv(f"{prefix}_LLM_API_KEY")
            or os.getenv("ANXIN_LLM_API_KEY")
            or ""
        )
        base_url = (
            os.getenv(f"{prefix}_LLM_BASE_URL")
            or os.getenv("ANXIN_LLM_BASE_URL")
            or "https://api.openai.com/v1"
        )
        model = (
            os.getenv(f"{prefix}_LLM_MODEL")
            or os.getenv("ANXIN_LLM_MODEL")
            or "gpt-4o-mini"
        )
        if not api_key:
            raise RuntimeError(
                f"No API key found. Set {prefix}_LLM_API_KEY or ANXIN_LLM_API_KEY."
            )
        return cls(LLMConfig(api_key=api_key, base_url=base_url, model=model))

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        response_format: str | None = None,   # "json" or None
        purpose: str = "",                    # for logging
    ) -> str:
        """
        Send a chat-completion request and return the assistant's text content.

        Codex: implement using openai SDK or raw httpx POST.
        Add basic retry on transient failures (3 attempts, exponential backoff).
        Log {purpose}, token counts, and latency to stderr.
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format == "json" and not self._should_skip_response_format():
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for attempt in range(3):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                in_chars = sum(len(str(m.get("content", ""))) for m in messages)
                elapsed_ms = int((time.time() - t0) * 1000)
                print(
                    f"[LLM/{purpose or 'chat'}] {self.config.model} | "
                    f"{elapsed_ms}ms | in≈{in_chars}c out≈{len(text)}c",
                    file=sys.stderr,
                )
                return text
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after retries: {last_err}")

    def _should_skip_response_format(self) -> bool:
        """Some OpenAI-compatible gateways reject response_format."""
        base_url = self.config.base_url.lower()
        unsupported_markers = ("ark", "dashscope", "volces", "volcengine")
        return any(marker in base_url for marker in unsupported_markers)

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executor: Any,
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        max_rounds: int = 3,
        purpose: str = "",
    ) -> str:
        """
        Chat with tool calling support.

        Args:
            messages: conversation messages (may include tool results)
            tools: OpenAI-format tool definitions
            tool_executor: callable(name, arguments_dict) -> result_dict
            max_rounds: max tool-calling rounds before forcing text response

        Returns:
            Final assistant text response after all tool calls resolved.
        """
        import json as _json

        for _ in range(max_rounds):
            t0 = time.time()
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as e:
                last_err = e
                if "tool" in str(e).lower():
                    # If tool calling fails, retry without tools
                    resp = self._client.chat.completions.create(
                        model=self.config.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                else:
                    raise

            choice = resp.choices[0]
            msg = choice.message
            elapsed_ms = int((time.time() - t0) * 1000)
            in_chars = sum(len(str(m.get("content", ""))) for m in messages)
            print(
                f"[LLM/{purpose or 'tools'}] {self.config.model} | "
                f"{elapsed_ms}ms | tool_calls={len(msg.tool_calls or [])}",
                file=sys.stderr,
            )

            # No tool calls — return text directly
            if not msg.tool_calls:
                return msg.content or ""

            # Execute tool calls
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                try:
                    args = _json.loads(tc.function.arguments)
                except _json.JSONDecodeError:
                    args = {}
                result = tool_executor(tc.function.name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": _json.dumps(result, ensure_ascii=False),
                })

        # Exhausted rounds — force a text-only final response
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        purpose: str = "",
    ) -> dict[str, Any]:
        """
        Convenience wrapper for JSON output. Strip ```json fences and parse.
        """
        raw = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format="json",
            purpose=purpose,
        )
        import json, re
        cleaned = re.sub(r"^```json\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)
