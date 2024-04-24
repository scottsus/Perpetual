import os
import json
import openai
import backoff
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
# from curriculum.teacher.sep import QNA_SEPARATOR, LIST_SEPARATOR
from typing import Tuple

TRAIN_DATA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string"
                    },
                    "answer": {
                        "type": "string"
                    }
                },
                "required": ["question", "answer"]
            }
        }
    },
    "required": ["questions"]
}

TRAIN_SYSTEM_MESSAGE = f"""
You are a world-class university professor constructing an exam.
You are very knowledgable in a wide range of fields, and you construct great questions to test a student on the contents of the text.
Additionally, you also generate corresponding answers so you can reference them later.

A JSON schema is provided for your reference:
{TRAIN_DATA_JSON_SCHEMA}
"""

TEST_DATA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "question": "string",
            "choices": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "answer": "string",
        },
    },
    "required": [ "question, choices, answer" ],
}

TEST_SYSTEM_MESSAGE = f"""
You are a world-class university professor constructing an exam.
Given a piece of raw text, produce `questions`, a list of question and answer pairs.
Each question and answer pair contains:
 - a single question
 - 4 possible choices (A, B, C, D) of which only 1 is correct
 - a correct answer -> this should be **only a single letter: the correct option**.

A JSON schema is provided for your reference:
{TEST_DATA_JSON_SCHEMA}
"""

class Model:
    def __init__(self):
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
    async def new_async_request(self, raw_text: str, is_train: bool = True) -> Tuple[str, int, int]:
        try:
            messages = [
                { "role": "system", "content": TRAIN_SYSTEM_MESSAGE if is_train else TEST_SYSTEM_MESSAGE },
                { "role": "user", "content": raw_text }
            ]

            res = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={ "type": "json_object" }
            )

            content = res.choices[0].message.content or ''
            (prompt_tokens, completion_tokens) = self.get_token_usage(res)

            obj = json.loads(content)
            questions = obj["questions"]
            return (questions, prompt_tokens, completion_tokens)
        
        except KeyboardInterrupt:
            raise
        except (json.decoder.JSONDecodeError, TypeError) as e:
            print(f"Model.new_async_request: {str(e)}")
            print(content)
            return (None, 0, 0)
        except Exception as e:
            print(f"Model.new_async_request: {str(e)}")
            print(content)
            raise
    
    def get_token_usage(self, chat_completion: ChatCompletion) -> Tuple[int, int]:
        if not chat_completion.usage:
            raise Exception('openai_client.update_token_usage: chat_completion.usage is None')

        prompt_tokens = chat_completion.usage.prompt_tokens
        completion_tokens = chat_completion.usage.completion_tokens

        return (prompt_tokens, completion_tokens)
    

# import asyncio

# asyncio.run(Model().new_async_request("The boiling point of Veritasium is 2000K."))
