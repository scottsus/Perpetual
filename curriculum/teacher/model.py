import os
import openai
import backoff
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from curriculum.teacher.sep import QNA_SEPARATOR, LIST_SEPARATOR
from typing import Tuple

SYSTEM_MESSAGE = f"""
You are a world-class university professor constructing an exam.
You are very knowledgable in a wide range of fields, and you construct great questions to test a student on the contents of the text.
Additionally, you also generate corresponding answers so you can reference them later.

A question and answer pair are separated using a custom {QNA_SEPARATOR} separator.
Question answer pairs are separated inside a list using a custom {LIST_SEPARATOR} separator.

Given a piece of raw text, return a list of question and answer pairs with the separators above.
"""

class Model:
    def __init__(self):
        self.system_message = SYSTEM_MESSAGE
        self.model: str = "gpt-4-turbo"
        self.client: AsyncOpenAI = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    @backoff.on_exception(
        backoff.expo,
        (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.InternalServerError,
            openai.UnprocessableEntityError
        ),
        max_tries=3
    )
    async def new_async_request(self, raw_text: str) -> Tuple[str, int, int]:
        try:
            messages = [
                { "role": "system", "content": self.system_message },
                { "role": "user", "content": raw_text }
            ]

            res = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            content = res.choices[0].message.content or ''
            (prompt_tokens, completion_tokens) = self.get_token_usage(res)

            return (content, prompt_tokens, completion_tokens)
        
        except KeyboardInterrupt:
            raise
        except Exception:
            raise
    
    def get_token_usage(self, chat_completion: ChatCompletion) -> Tuple[int, int]:
        if not chat_completion.usage:
            raise Exception('openai_client.update_token_usage: chat_completion.usage is None')

        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        return (prompt_tokens, completion_tokens)