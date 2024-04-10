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

async def generate_questions_and_answers(model: Model, chunks: List[str]) -> List:
    tasks = [model.new_async_request(text) for text in chunks]
    results = await asyncio.gather(*tasks)

    dataset = []
    total_input_tokens, total_output_tokens = 0, 0
    for chunk, (content, input_tokens, output_tokens) in zip(chunks, results):
        dataset.append({ "type": "doc", "document": chunk })
        
        qna_pairs = content.split(LIST_SEPARATOR)
        for qna_pair in qna_pairs:
            pair = qna_pair.split(QNA_SEPARATOR)
            if (len(pair) != 2):
                continue
            question, answer = pair[0].strip(), pair[1].strip()
            dataset.append({ "type": "qna", "question": question, "answer": answer })
        
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
    
    print("Total tokens used:", total_input_tokens + total_output_tokens)
    return dataset

async def construct_text_curriculum():
    model = Model()
    raw_chunks = load_and_chunk_text()
    dataset = await generate_questions_and_answers(model, raw_chunks)
    
    target_file_name = "curriculum/text/train.json"
    with open(target_file_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print(f"Written to {target_file_name}")

if __name__ == "__main__":
    asyncio.run(construct_text_curriculum())
