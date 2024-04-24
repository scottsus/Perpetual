"""
Given some text (news article, research paper), generate a curriculum for knowledge injection.
"""

import json
import asyncio
from pypdf import PdfReader
from curriculum.teacher.model import Model
from curriculum.teacher.prompt import *
from curriculum.teacher.sep import QNA_SEPARATOR, LIST_SEPARATOR
from typing import List

TEXT_SOURCE = "https://arxiv.org/pdf/2403.19887.pdf"
CHUNK_SIZE = 4_000
OVERLAP = 500

def load_and_chunk_text() -> List[str]:
    reader = PdfReader("curriculum/text/jamba.pdf")
    
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text(extraction_mode="layout", layout_mode_space_vertically=False)

    raw_chunks = [
        raw_text[i : i + CHUNK_SIZE].strip() for i in range(0, len(raw_text), CHUNK_SIZE - OVERLAP)
    ]

    return raw_chunks

async def generate_open_ended_questions(model: Model, chunks: List[str]) -> List:
    tasks = [model.new_async_request(text, is_train=True) for text in chunks]
    results = await asyncio.gather(*tasks)

    dataset = []
    total_input_tokens, total_output_tokens = 0, 0
    for chunk, (questions, input_tokens, output_tokens) in zip(chunks, results):
        dataset.append({ "type": "doc", "document": chunk })
        
        if questions is None:
            continue
        for pair in questions:
            try:
                question, answer = pair["question"], pair["answer"]
            except Exception as e:
                print(f"generate_open_ended_questions: {str(e)}")
                continue

            dataset.append(dict(
                type="qna",
                question=question,
                answer=answer,
            ))
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    
    print("Total tokens used:", total_input_tokens + total_output_tokens)
    return dataset

async def generate_multiple_choice_questions(model: Model, chunks: List[str]) -> List:
    tasks = [model.new_async_request(text, is_train=False) for text in chunks]
    results = await asyncio.gather(*tasks)

    dataset = []
    total_input_tokens, total_output_tokens = 0, 0
    for questions, input_tokens, output_tokens in results:
        if questions is None:
            continue
        dataset += questions
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    
    print("Total tokens used:", total_input_tokens + total_output_tokens)
    return dataset

async def construct_text_curriculum(is_train: bool = True):
    model = Model()
    raw_chunks = load_and_chunk_text()

    if is_train:
        dataset = await generate_open_ended_questions(model, raw_chunks)
        target_file_name = "curriculum/text/train.json"
    else:
        dataset = await generate_multiple_choice_questions(model, raw_chunks)
        target_file_name = "curriculum/text/test.json"
    
    with open(target_file_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print(f"💰 Written to {target_file_name}")

if __name__ == "__main__":
    asyncio.run(construct_text_curriculum(False))
