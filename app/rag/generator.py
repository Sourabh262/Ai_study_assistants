import time
from typing import Any

from groq import APIConnectionError, Groq, RateLimitError as GroqRateLimitError

from app.config import GROQ_API_KEY, GROQ_MODEL, get_env_variable
from app.exceptions import LLMError, RateLimitError
from app.logging_config import logger
from app.rag.prompts import (
    build_direct_prompt,
    build_rag_prompt,
    get_agent_system_prompt,
)


class LLMGenerator:
    """
    Production interface for Groq LLM completions with native tool calling support.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        key = api_key or GROQ_API_KEY or get_env_variable("GROQ_API_KEY")

        if not key:
            raise LLMError("GROQ_API_KEY is not configured.")

        self.model = model or GROQ_MODEL or "openai/gpt-oss-120b"

        try:
            self.client = Groq(api_key=key)
        except Exception as exc:
            logger.error("Failed to initialize Groq client: %s", exc)
            raise LLMError("Failed to initialize the Groq client.") from exc

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.1,
    ) -> Any:
        """
        Send a multi-turn chat completion request to the Groq model with optional tool schemas.

        Args:
            messages: List of message dictionaries (system, user, assistant, tool).
            tools: Optional list of OpenAI-compatible function tool definitions.
            tool_choice: "auto", "none", or specific tool choice.
            temperature: Sampling temperature.

        Returns:
            The message object from the first choice of the completion.
        """
        start_time = time.perf_counter()

        try:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if tools and len(tools) > 0:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = tool_choice

            logger.info(
                "Sending chat completion request to model: %s (tools: %s, messages: %s)",
                self.model,
                bool(tools),
                len(messages),
            )

            response = self.client.chat.completions.create(**kwargs)
            duration = (time.perf_counter() - start_time) * 1000

            choice = response.choices[0]
            logger.info(
                "LLM completion finished in %.2f ms (finish_reason: %s)",
                duration,
                choice.finish_reason,
            )

            return choice.message

        except GroqRateLimitError as exc:
            logger.warning("Groq rate limit exceeded: %s", exc)
            raise RateLimitError(
                "The AI service is temporarily experiencing high demand. Please try again shortly."
            ) from exc

        except APIConnectionError as exc:
            logger.error("Groq connection error: %s", exc)
            raise LLMError(
                "Could not connect to the AI model service. Please check your internet connection."
            ) from exc

        except LLMError:
            raise

        except Exception as exc:
            logger.error("Groq chat completion request failed: %s", exc)
            raise LLMError(
                "Failed to generate a response from the language model."
            ) from exc

    def _generate(self, user_prompt: str) -> str:
        """Single-turn generation for prompt templates."""
        messages = [
            {"role": "system", "content": get_agent_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
        msg = self.chat_completion(messages=messages)
        content = msg.content
        if not content:
            raise LLMError("The LLM returned an empty response.")
        return content.strip()

    def generate_with_context(self, question: str, context: str) -> str:
        prompt = build_rag_prompt(question=question, context=context)
        return self._generate(prompt)

    def generate_direct(self, question: str) -> str:
        prompt = build_direct_prompt(question)
        return self._generate(prompt)