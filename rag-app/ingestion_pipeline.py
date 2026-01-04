import os
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def load_documents(docs_path="docs"):
    print(f"Loading documents from {docs_path}...")

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist.")

    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True}  # ✅ avoid encoding crashes
    )

    documents = loader.load()
    if not documents:
        raise FileNotFoundError(f"No .txt files found in {docs_path}.")

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    print("Splitting documents into chunks...")

    splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore


def main():
    persist_directory = "db/chroma_db"

    documents = load_documents("docs")
    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks, persist_directory)

    print("Vectors stored:", vectorstore._collection.count())
    print("Persist dir exists:", os.path.exists(persist_directory))


if __name__ == "__main__":
    main()
