# SOURCE: https://github.com/bhuvan454/bareRAG/tree/master
from __future__ import annotations

import json
import os
import sys

from pathlib import Path
from typing import Any, Dict, List

import chromadb

from langchain.prompts import ChatPromptTemplate
from langchain.schema.document import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms.ollama import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm


DATA_PATH = "/Users/malcolm/dev/bossjones/goob_ai/src/goob_ai/data/db"
QDRANT_URL = os.getenv("QDRANT_URL", "https://localhost:6333")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "ragcollection")
DATA_DOC_PATH = os.getenv("DATA_DOC_PATH", os.path.join(DATA_PATH, "documents"))
CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(DATA_PATH, "chroma_db"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

PROMPT_TEMPLATE = """
Answer the question based only basedon the on the following context:

{context}

----------------------------------------

Answer the question based on the above context: {question}, and if you are not sure
about the answer, give a relevant answer based on the context and give certainty score of 0.5

"""


def split_documents(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)


class CustomOllamaEmbeddings:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _embed_documents(self, texts):
        return [OllamaEmbeddings(model="llama3", prompt=text)["embedding"] for text in texts]

    def __call__(self, input):
        return self._embed_documents(input)


class query_rag:
    def __init__(self, model):
        self.model = Ollama(model=model)

    def invoke(self, query_results, prompt):
        response = self.model.invoke(prompt)

        sources = [doc.get("chunk_id") for doc in query_results["metadatas"][0]]

        return f"Response: {response}\n\nSources: {sources}"


def load_pdf(file_path: str) -> Dict[str, Any]:
    """
    Load the pdf file and return the text content
    """
    loader = PyMuPDFLoader(file_path)
    return loader.load()


def query_database(collection, query_text: str):
    results = collection.query(query_texts=[query_text], n_results=int(RAG_TOP_K), include=["documents", "metadatas"])

    context_level = results["documents"][0]

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_level, question=query_text)

    return results, prompt


class ChromaDB:
    def __init__(self):
        # self.config = Config()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.doc_add_batch_size = 100

    def add_collection(self, collection_name, embedding_function):
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, embedding_function=embedding_function
        )
        return self.collection

    def get_list_collections(self):
        return self.chroma_client.list_collections()

    def get_collection(self, collection_name, embedding_function):
        return self.chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

    def add_chunk_ids(self, chunks):
        last_page_id = None
        current_chunk_index = 0

        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"

            ## if the page id is the same as the last one, increment the index
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0

            # create the chunk id
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id

            # add the chunk id to the metadata
            chunk.metadata["chunk_id"] = chunk_id

        return chunks

    def add_chunks(self, chunks):
        chunks_with_ids = self.add_chunk_ids(chunks)
        existing_items = self.collection.get(include=[])
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing items in DB: {len(existing_ids)}")

        new_chunks = [chunk for chunk in chunks_with_ids if chunk.metadata["chunk_id"] not in existing_ids]

        if len(new_chunks):
            for i in tqdm(range(0, len(new_chunks), self.doc_add_batch_size), desc="Adding new items to DB"):
                batch = new_chunks[i : i + self.doc_add_batch_size]

                batch_ids = [chunk.metadata["chunk_id"] for chunk in batch]
                docuemnts = [chunk.page_content for chunk in batch]
                metadata = [chunk.metadata for chunk in batch]

                self.collection.upsert(documents=docuemnts, ids=batch_ids, metadatas=metadata)
        else:
            print("No new items to add to the DB")

    def delete_collection(self, collection_name):
        self.chroma_client.delete_collection(collection_name)
        print("Collection deleted")


# Initialize ChromaDBManager instance
class ChromaDBManager:
    def __init__(self):
        self.vector_db = ChromaDB()
        self.collection = None
        self.model = query_rag("llama3")

    def get_list_collections(self):
        return self.vector_db.get_list_collections()

    def get_collection(self, collection_name: str, embedding_function):
        return self.vector_db.get_collection(collection_name, CustomOllamaEmbeddings())

    def initialize_database(self, pdf_path: str, collection_name: str, debug: bool = False):
        documents_data = load_pdf(pdf_path)
        if debug:
            print("Loaded Documents Data:", documents_data)

        split_documents_data = split_documents(documents_data)
        if debug:
            print("Split Documents Data:", split_documents_data)

        self.collection = self.vector_db.add_collection(collection_name, CustomOllamaEmbeddings())
        self.vector_db.add_chunks(split_documents_data)

        print(f"Collection '{collection_name}' created and data added.")

    def delete_collection(self, collection_name: str):
        self.collection = self.vector_db.get_collection(collection_name, CustomOllamaEmbeddings())
        if self.collection:
            self.vector_db.chroma_client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted.")
        else:
            print(f"Collection '{collection_name}' not found.")

    def query_database(self, query_text: str, collection_name: str, debug: bool = False):
        collection = self.get_collection(collection_name, CustomOllamaEmbeddings())
        query_context, prompt = query_database(collection, query_text)

        if debug:
            print("Query Context:", query_context)
            print("Prompt:", prompt)

        rag_response = self.model.invoke(query_context, prompt)
        print("RAG Response:", rag_response)

        return rag_response
