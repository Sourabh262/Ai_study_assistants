from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.exceptions import LLMError
from app.logging_config import logger
from app.rag.prompts import (
    SYSTEM_PROMPT,
    build_direct_prompt,
    build_rag_prompt,
)


class LLMGenerator:
    """Generate responses using the configured Groq model."""

    def __init__(
        self,
        api_key: str | None = GROQ_API_KEY,
        model: str = GROQ_MODEL,
    ) -> None:
        if not api_key:
            raise LLMError(
                "GROQ_API_KEY is not configured."
            )

        self.model = model

        try:
            self.client = Groq(api_key=api_key)

        except Exception as exc:
            logger.error(
                "Failed to initialize Groq client: %s",
                exc,
            )

            raise LLMError(
                "Failed to initialize the Groq client."
            ) from exc

    def _generate(
        self,
        user_prompt: str,
    ) -> str:
        """
        Send a prompt to the Groq LLM.
        """

        try:
            logger.info(
                "Sending request to Groq model: %s",
                self.model,
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.2,
            )

            answer = response.choices[0].message.content

            if not answer:
                raise LLMError(
                    "The LLM returned an empty response."
                )

            logger.info("LLM response generated successfully.")

            return answer.strip()

        except LLMError:
            raise

        except Exception as exc:
            logger.error(
                "Groq request failed: %s",
                exc,
            )

            raise LLMError(
                "Failed to generate a response from the language model."
            ) from exc

    def generate_with_context(
        self,
        question: str,
        context: str,
    ) -> str:
        """Generate a document-grounded response."""

        prompt = build_rag_prompt(
            question=question,
            context=context,
        )

        return self._generate(prompt)

    def generate_direct(
        self,
        question: str,
    ) -> str:
        """Generate a response without document context."""

        prompt = build_direct_prompt(question)

        return self._generate(prompt)