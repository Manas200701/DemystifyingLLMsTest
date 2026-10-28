"""
HTTP API request parsing, validation, and execution handler.
"""

import time
import uuid
import math
from typing import Dict, Any, Optional, List
from utils.logging_utils import get_logger

logger = get_logger("request_handler")


class RequestHandler:
    """
    Handles endpoint request validation, inference invocation, and response formatting.
    """

    def __init__(self, engine: Optional[Any] = None, tokenizer: Optional[Any] = None):
        self.engine = engine
        self.tokenizer = tokenizer

    def handle_completion(self, request_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process OpenAI-compatible /v1/completions request.

        Args:
            request_body: Dictionary containing 'prompt', optional 'max_tokens', 'temperature', etc.

        Returns:
            Formatted response dictionary.
        """
        prompt = request_body.get("prompt", "")
        max_tokens = request_body.get("max_tokens", 16)
        temperature = request_body.get("temperature", 0.8)
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ValueError("max_tokens must be a non-negative integer")
        if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be a non-negative number")

        # Tokenize prompt
        if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [ord(c) for c in prompt]

        # Generate continuation
        generated_text = None
        if self.engine is not None and hasattr(self.engine, "generate_tokens"):
            generated_tokens = self.engine.generate_tokens(
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        elif self.engine is not None and hasattr(self.engine, "generate"):
            raw_gen = self.engine.generate(prompt, tokenizer=self.tokenizer, max_new_tokens=max_tokens)
            generated_text = raw_gen
            generated_tokens = prompt_tokens[:]
            if self.tokenizer is not None and hasattr(self.tokenizer, "encode"):
                try:
                    generated_tokens.extend(self.tokenizer.encode(raw_gen, add_bos=False))
                except TypeError:
                    generated_tokens.extend(self.tokenizer.encode(raw_gen))
        else:
            raise RuntimeError("No inference engine is available")

        if generated_text is not None:
            response_text = generated_text
        elif self.tokenizer is not None and hasattr(self.tokenizer, "decode"):
            response_text = self.tokenizer.decode(generated_tokens[len(prompt_tokens):])
        else:
            response_text = "".join(chr(t) for t in generated_tokens[len(prompt_tokens):] if 0 <= t <= 0x10FFFF)

        stop_ids = getattr(getattr(self.engine, "inference_config", None), "stop_token_ids", [])
        finish_reason = "stop" if generated_tokens and generated_tokens[-1] in stop_ids else "length"
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:12]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "broken-transformer",
            "text": response_text,
            "choices": [
                {
                    "text": response_text,
                    "index": 0,
                    "finish_reason": finish_reason
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": len(generated_tokens) - len(prompt_tokens),
                "total_tokens": len(generated_tokens)
            }
        }

    def handle_health(self) -> Dict[str, Any]:
        """Process /health endpoint check."""
        return {
            "status": "ok",
            "service": "broken-transformer-server",
            "timestamp": time.time()
        }

    def handle_models(self) -> Dict[str, Any]:
        """Process /v1/models endpoint check."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "broken-transformer",
                    "object": "model",
                    "owned_by": "openai-dsg-iitr",
                    "ready": True
                }
            ]
        }
