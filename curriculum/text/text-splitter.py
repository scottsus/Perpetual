"""
Given some text (news article, research paper), generate a curriculum for knowledge injection.
"""

import os
import json
import asyncio
import argparse
from pypdf import PdfReader
from curriculum.teacher.model import Model
from curriculum.teacher.prompt import *
from typing import List
from datasets import Dataset

CHUNK_SIZE = 4096
OVERLAP = 512

PAPERS_PATH = "curriculum/text/papers"

def get_papers():
    return os.listdir(PAPERS_PATH)

def load_and_chunk_text(paper_path: str) -> List[str]:
    reader = PdfReader(paper_path)
    
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

    papers = get_papers()
    dataset = []
    for paper in papers:
        print(paper)
        raw_chunks = load_and_chunk_text(f"{PAPERS_PATH}/{paper}")
        if is_train:
            dataset += await generate_open_ended_questions(model, raw_chunks)
            target_file_name = "curriculum/text/train.json"
        else:
            dataset += await generate_multiple_choice_questions(model, raw_chunks)
            target_file_name = "curriculum/text/test.json"
    
    with open(target_file_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print(f"💰 Written to {target_file_name} {len(dataset)} rows")

def upload_chunked_dataset():
    papers = get_papers()
    chunks = []
    for paper in papers:
        chunks.extend(load_and_chunk_text(f"{PAPERS_PATH}/{paper}"))
    raw_dataset = {
        "text": chunks
    }
    dataset = Dataset.from_dict(raw_dataset)
    dataset.push_to_hub("slyq/papers-chunked", "combined", split="train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--upload", action="store_true", help="Upload dataset instead of constructing dataset")
    args = parser.parse_args()
    if args.upload:
        upload_chunked_dataset()
    else:
        asyncio.run(construct_text_curriculum(False))
