# LINK: https://github.com/mlsmall/RAG-Application-with-LangChain
# SOURCE: https://www.linkedin.com/pulse/building-retrieval-augmented-generation-rag-app-langchain-tiwari-stpfc/
# NOTE: This might be the one, take inspiration from the others
from __future__ import annotations

import argparse
import os
import shutil

from dataclasses import dataclass

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from loguru import logger as LOGGER

from goob_ai.aio_settings import aiosettings


HERE = os.path.dirname(__file__)

DATA_PATH = os.path.join(HERE, "...", "data", "chroma", "documents")
CHROMA_PATH = os.path.join(HERE, "...", "data", "chroma", "vectorstorage")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


# Function to perform the query and get the response
def get_response(query_text: str) -> str:
    # Prepare the DB
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return "Unable to find matching results."

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    return f"Response: {response_text}\nSources: {sources}"


# # Gradio interface components
# def main():
#     with gr.Blocks() as demo:
#         with gr.Row():
#             query_textbox = gr.Textbox(lines=2, placeholder="Enter your query here...", label="Query")
#             response_textbox = gr.Textbox(lines=10, placeholder="Response will be shown here...", label="Response",
#                                           interactive=False)

#         query_button = gr.Button("Submit Query")

#         # Define the interaction
#         query_button.click(fn=get_response, inputs=query_textbox, outputs=response_textbox)

#     demo.launch()


def main():
    generate_data_store()


class CustomOpenAIEmbeddings(OpenAIEmbeddings):
    def __init__(self, openai_api_key: str = aiosettings.openai_api_key) -> None:
        super().__init__(openai_api_key=openai_api_key)

    def _embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(texts)

    def __call__(self, input: list[str]) -> list[float]:
        return self._embed_documents(input)


def generate_data_store() -> None:
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


from typing import List

def load_documents() -> List[Document]:
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_PATH, filename)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    LOGGER.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if chunks:
        document = chunks[10]
        LOGGER.info(document.page_content)
        LOGGER.info(document.metadata)
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)

    embeddings = CustomOpenAIEmbeddings(openai_api_key=aiosettings.openai_api_key)
    LOGGER.info(embeddings)
    # Create a new DB from the documents.
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    LOGGER.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
