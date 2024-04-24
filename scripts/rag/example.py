from scripts.rag.retriever import Retriever

def retrieve_docs():
    store = Retriever("curriculum/database-train.json")
    docs = store.query("How much RAM does an Apple Macbook Pro have?")
    # docs = store.query("Estoy buscando reemplazar un ventilador roto del procesador de mi computadora. ¿Cuáles son algunos productos que recomendaría?")
    # print(docs)
    return docs

if __name__ == "__main__":
    retrieve_docs()