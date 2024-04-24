"""
Given a database of products, construct a curriculum for knowledge injection.
"""

import json
import argparse
import asyncio
from curriculum.teacher.model import Model
from curriculum.teacher.prompt import *
from curriculum.teacher.sep import QNA_SEPARATOR, LIST_SEPARATOR
from typing import List
from datasets import load_dataset, Dataset

CHUNK_SIZE = 4_000
OVERLAP = 500

def preprocess_batched(dataset):
    # merge lefts and rights for titles and descriptions
    ts = map(lambda l, r: (l + "; " if l is not None else "") + (r + "; " if r is not None else ""), dataset["title_left"], dataset["title_right"])
    ds = map(lambda l, r: (l + "; " if l is not None else "") + (r + "; " if r is not None else ""), dataset["description_left"], dataset["description_right"])
    chunked_titles = []
    chunked_descriptions = []
    # chunk for long descriptions (duplicate titles if the description is chunked up)
    for title, description in zip(ts,ds):
        chunked_titles.extend([title for _ in range(0, len(description), CHUNK_SIZE - OVERLAP)])
        chunked_descriptions.extend([description[i : i + CHUNK_SIZE].strip() for i in range(0, len(description), CHUNK_SIZE - OVERLAP)])
    return {
        "title": chunked_titles,
        "description": chunked_descriptions,
    }

def load_and_chunk_data() -> Dataset:
    d = load_dataset("wdc/products-2017", split="train")

    return d.map(preprocess_batched, batched=True, remove_columns=[
        'pair_id',
        'label',
        'id_left',
        'category_left',
        'cluster_id_left',
        'brand_left',
        'title_left',
        'description_left',
        'price_left',
        'specTableContent_left',
        'id_right',
        'category_right',
        'cluster_id_right',
        'brand_right',
        'title_right',
        'description_right',
        'price_right',
        'specTableContent_right'
    ])

async def generate_questions_and_answers(model: Model, raw_dataset: List[str]) -> List:
    tasks = [model.new_async_request(f"""Product title: {product["title"]}\nProduct description: {product["description"]}""") for product in raw_dataset]
    results = await asyncio.gather(*tasks)

    dataset = []
    total_input_tokens, total_output_tokens = 0, 0
    for chunk, (content, input_tokens, output_tokens) in zip(raw_dataset, results):
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

async def construct_dataset_curriculum():
    raw_dataset = load_and_chunk_data().select(range(20))
    model = Model()
    dataset = await generate_questions_and_answers(model, raw_dataset)
    
    target_file_name = "curriculum/database-train.json"
    with open(target_file_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print(f"Written to {target_file_name}")

def upload_chunked_dataset():
    raw_dataset = load_and_chunk_data()
    raw_dataset.push_to_hub("slyq/wdc-products-chunked", split="train")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--upload", action="store_true", help="Upload dataset instead of constructing dataset")
    args = parser.parse_args()
    if args.upload:
        upload_chunked_dataset()
    else:
        asyncio.run(construct_dataset_curriculum())
