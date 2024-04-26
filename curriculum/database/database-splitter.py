"""
Given a database of products, construct a curriculum for knowledge injection.
"""

import json
import argparse
import asyncio
import re
from tqdm.asyncio import tqdm_asyncio
from curriculum.teacher.model import Model
from curriculum.teacher.prompt import *
from curriculum.teacher.sep import QNA_SEPARATOR, LIST_SEPARATOR
from typing import List
from datasets import load_dataset, Dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 4_096
OVERLAP = 512
MIN_DESCRIPTION_THRESHOLD = 10

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
)

def clean_title(title):
    return re.sub(r'@[a-z]{2}[\-A-Z]* ?;?', '', title).replace('"Null"','').strip()

def clean_description(description):
    cleaned = re.sub(r'@[a-z]{2}\S*? ;', '', description).replace('"Null"','').strip()
    if len(cleaned) < MIN_DESCRIPTION_THRESHOLD:
        return ""
    return cleaned

def preprocess_batched(dataset):
    # merge lefts and rights for titles and descriptions
    ts = map(lambda l, r: clean_title((l + "; " if l is not None else "") + (r + "; " if r is not None else "")), dataset["title_left"], dataset["title_right"])
    ds = map(lambda l, r: clean_description((l + "; " if l is not None else "") + (r + "; " if r is not None else "")), dataset["description_left"], dataset["description_right"])
    chunked_titles = []
    chunked_descriptions = []
    # chunk for long descriptions (duplicate titles if the description is chunked up)
    for title, description in zip(ts,ds):
        description_chunks = [chunk for chunk in splitter.split_text(description) if len(chunk) >= MIN_DESCRIPTION_THRESHOLD]
        chunked_titles.extend([title for _ in range(len(description_chunks))])
        chunked_descriptions.extend(description_chunks)
    return {
        "title": chunked_titles,
        "description": chunked_descriptions,
    }

def preprocess_batched_combined(dataset):
    # merge lefts and rights for titles and descriptions
    ts = map(lambda l, r: clean_title((l + "; " if l is not None else "") + (r + "; " if r is not None else "")), dataset["title_left"], dataset["title_right"])
    ds = map(lambda l, r: clean_description((l + "; " if l is not None else "") + (r + "; " if r is not None else "")), dataset["description_left"], dataset["description_right"])
    chunks = []
    # chunk for long descriptions (duplicate titles if the description is chunked up)
    for title, description in zip(ts,ds):
        description_chunks = [chunk for chunk in splitter.split_text(description) if len(chunk) >= MIN_DESCRIPTION_THRESHOLD]
        chunks.extend([f"{title}\n{description}" for description in description_chunks])
    return {
        "text": chunks
    }

def load_and_chunk_data(combined=False) -> Dataset:
    d = load_dataset("wdc/products-2017", split="train")

    return d.map(preprocess_batched_combined if combined else preprocess_batched, batched=True, remove_columns=[
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

async def generate_open_ended_questions(model: Model, chunks: List[str]) -> List:
    tasks = [model.new_async_request(f"""Product title: {product["title"]}\nProduct description: {product["description"]}""", is_train=True) for product in chunks]
    results = await tqdm_asyncio.gather(*tasks)

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
    tasks = [model.new_async_request(f"""Product title: {product["title"]}\nProduct description: {product["description"]}""", is_train=False) for product in chunks]
    results = await tqdm_asyncio.gather(*tasks)

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

async def construct_database_curriculum(is_train: bool = True):
    raw_dataset = load_and_chunk_data().select(range(10,500))
    model = Model()

    if is_train:
        dataset = await generate_open_ended_questions(model, raw_dataset)
        target_file_name = "curriculum/database/train.json"
    else:
        dataset = await generate_multiple_choice_questions(model, raw_dataset)
        target_file_name = "curriculum/database/test.json"
    
    with open(target_file_name, 'w') as f:
        json.dump(dataset, f, indent=4)
        print(f"💰 Written to {target_file_name}")

def upload_chunked_dataset(combined=False):
    raw_dataset = load_and_chunk_data(combined)
    raw_dataset.push_to_hub("slyq/wdc-products-chunked", "combined" if combined else "default", split="train")
    # dataset = load_dataset("curriculum/database")
    # dataset.push_to_hub("slyq/wdc-products-qna")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--upload", action="store_true", help="Upload dataset instead of constructing dataset")
    parser.add_argument("-c", "--combined", action="store_true", help="Upload chunked dataset with title and description combined")
    args = parser.parse_args()
    if args.upload:
        upload_chunked_dataset(args.combined)
    else:
        asyncio.run(construct_database_curriculum(is_train=False))
