"""
src/utils/llm_client.py
-----------------------
Universal Multi-Provider LLM Client supporting:
1. Local Open-Source LLMs via Ollama (e.g., Llama 3.2, Qwen, Mistral) - Offline, zero cost, unlimited.
2. Cloud LLMs via Google Gemini API (gemini-3.5-flash-lite).
3. Automatic provider detection & fast timeout failover to ensure the pipeline never hangs.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class UnifiedLLMClient:
    """Unified client that bridges local Ollama and Google Gemini."""

    def __init__(
        self,
        provider: Optional[str] = None,
        ollama_model: str = "llama3.2",
        gemini_model: str = "gemini-3.5-flash-lite",
        ollama_host: str = "http://localhost:11434",
    ):
        # Provider can be "ollama", "gemini", or "auto"
        self.provider = provider or os.getenv("LLM_PROVIDER", "auto").lower()
        self.ollama_model = os.getenv("OLLAMA_MODEL", ollama_model)
        self.gemini_model = os.getenv("GEMINI_MODEL", gemini_model)
        self.ollama_host = os.getenv("OLLAMA_HOST", ollama_host).rstrip("/")

        # Initialize Google GenAI client if API key is present
        self.gemini_client = None
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
            except Exception:
                self.gemini_client = None

    def is_ollama_available(self) -> bool:
        """Check if local Ollama daemon is currently running."""
        try:
            req = urllib.request.Request(f"{self.ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _call_ollama(self, prompt: str, system_prompt: str = "", timeout: float = 20.0) -> str:
        """Call local Ollama REST API using Python's built-in urllib (no external dependencies needed)."""
        url = f"{self.ollama_host}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")

    def _call_gemini(self, prompt: str, system_prompt: str = "") -> str:
        """Call Google Gemini API."""
        if not self.gemini_client:
            raise RuntimeError("Gemini client not initialized or GEMINI_API_KEY missing.")

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Request:\n{prompt}"

        try:
            res = self.gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=full_prompt,
            )
            return (res.text or "").strip()
        except Exception as e:
            raise RuntimeError(f"Gemini API error: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        timeout: float = 20.0,
    ) -> str:
        """
        Generate text response from the configured LLM provider.
        If provider is 'auto', it tries Ollama first (local, free, unlimited).
        If Ollama is not running, it falls back to Gemini.
        If both fail, it returns an empty string so callers trigger deterministic regex fallbacks.
        """
        active_provider = self.provider

        if active_provider == "auto":
            active_provider = "ollama" if self.is_ollama_available() else "gemini"

        # 1. Try Ollama (if selected)
        if active_provider == "ollama":
            try:
                return self._call_ollama(prompt, system_prompt, timeout=timeout)
            except Exception as e:
                # If Ollama fails, fallback to Gemini
                if self.gemini_client:
                    try:
                        return self._call_gemini(prompt, system_prompt)
                    except Exception:
                        pass
                return ""

        # 2. Try Gemini (if selected)
        elif active_provider == "gemini":
            try:
                return self._call_gemini(prompt, system_prompt)
            except Exception as e:
                # If Gemini hits rate limit/quota, fallback to Ollama if running
                if self.is_ollama_available():
                    try:
                        return self._call_ollama(prompt, system_prompt, timeout=timeout)
                    except Exception:
                        pass
                return ""

        return ""


# Singleton instance getter
_GLOBAL_LLM_CLIENT: Optional[UnifiedLLMClient] = None

def get_llm_client() -> UnifiedLLMClient:
    """Return the global unified LLM client."""
    global _GLOBAL_LLM_CLIENT
    if _GLOBAL_LLM_CLIENT is None:
        _GLOBAL_LLM_CLIENT = UnifiedLLMClient()
    return _GLOBAL_LLM_CLIENT