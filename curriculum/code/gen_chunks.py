import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language


def good_file_type(filename, text_suffixes):
    return any(filename.endswith(suffix) for suffix in text_suffixes)

def split_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0
    while start < len(text):
        if start != 0:
            start -= overlap
        end = start + chunk_size
        if end > len(text):
            end = len(text)
        chunks.append(text[start:end])
        # Move start up for the next chunk
        start = end
        # Stop if we're at the end of the text
        if start >= len(text):
            break
    return chunks


def process_file(filepath, chunk_size, text_suffixes, text_splitter, overlap):
    if not good_file_type(filepath, text_suffixes):
        print(f'Skipping binary file: {filepath}')
        return []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            chunks = text_splitter.split_text(content)
    
            # chunks = split_into_chunks(content, chunk_size, overlap)
            # Return a list of dictionaries with title and document
            return [{ 'type': 'doc', 'path': filepath,'title': os.path.basename(filepath), 'document': chunk, 'size': len(chunk)} for chunk in chunks]
    except Exception as e:
        print(f'Error processing file {filepath}: {e}')
        return []

def process_directory(directory, chunk_size, overlap, text_suffixes, text_splitter, all_chunks):
    # Process each file in the current directory and collect chunks
    for entry in os.scandir(directory):
        if entry.is_file():
            file_chunks = process_file(entry.path, chunk_size, text_suffixes, text_splitter, overlap)
            all_chunks.extend(file_chunks)

def traverse_directory(directory, chunk_size, overlap, text_suffixes, text_splitter, all_chunks):
    # Process all files in the current directory first, then recursively process each subdirectory
    print(f'Entering directory: {directory}')
    process_directory(directory, chunk_size, overlap, text_suffixes, text_splitter, all_chunks)

    for entry in os.scandir(directory):
        if entry.is_dir():
            traverse_directory(entry.path, chunk_size, overlap, text_suffixes, text_splitter, all_chunks)

# Example usage
directory_path = 'flamethrower'
chunk_size = 4096  # Number of characters per chunk
overlap = 500
all_chunks = []
text_suffixes = {".txt", ".py", ".ipynb",".js", ".html", ".css", ".json", ".xml", ".c", ".h", ".md"}


text_splitter = RecursiveCharacterTextSplitter.from_language(
    chunk_size=4096,
    chunk_overlap=500,
    length_function=len,
    language = Language.PYTHON
)

traverse_directory(directory_path, chunk_size, overlap, text_suffixes, text_splitter, all_chunks)

json_path = "train/"+directory_path+'_dataset.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=4)

print(f'Data has been saved to {json_path}')
