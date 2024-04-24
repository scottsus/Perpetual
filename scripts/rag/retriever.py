from langchain_chroma import Chroma
import json

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

class Retriever:
    def __init__(self, database):
        with open(database, "r") as f:
            text = f.read()
            data = json.loads(text)
            qna = [d["question"] + " " + d["answer"] for d in data if d["type"] == "qna"]

        embedder = HuggingFaceBgeEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        
        db = Chroma.from_texts(qna, embedder)
        self.retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def query(self, query):
        return self.retriever.invoke(query)