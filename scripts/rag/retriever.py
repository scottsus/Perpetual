from langchain_chroma import Chroma
import json

from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import HuggingFaceDatasetLoader

model_name = "BAAI/bge-large-en-v1.5"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

class Retriever:
    def __init__(self, database, is_file=False):
        embedder = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        
        if is_file:
            with open(database, "r") as f:
                text = f.read()
                data = json.loads(text)
                qna = [d["question"] + " " + d["answer"] for d in data if d["type"] == "qna"]
            db = Chroma.from_texts(qna, embedder)
        else:
            loader = HuggingFaceDatasetLoader(database, name="combined")
            docs = loader.load()
            db = Chroma.from_documents(docs, embedder)

        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def query(self, query):
        return self.retriever.invoke(query)