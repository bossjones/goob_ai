# SOURCE: https://github.com/langchain-ai/langchain/blob/cb31c3611f6dadb9895011b3a3c2aa6f183d58de/templates/rag-chroma/rag_chroma/chain.py#L3
from __future__ import annotations

import uuid

from typing import List

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Example for document loading (from url), splitting, and creating vectostore

"""
# Load
from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = Chroma.from_documents(documents=all_splits,
                                    collection_name="rag-chroma",
                                    embedding=OpenAIEmbeddings(),
                                    )
retriever = vectorstore.as_retriever()
"""


def run_dummy_loader() -> List[Document]:
    """_summary_

    Returns:
        List[Document]: _description_
    """
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    docs = loader.load()

    loader = WebBaseLoader("https://lilianweng.github.io/posts/2024-02-05-human-data-quality/")
    docs.extend(loader.load())
    return docs


def get_vectorstore_retriever(vectorstore, store, id_key):
    return MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
    )


# https://github.com/langchain-ai/rag-from-scratch/blob/main/rag_from_scratch_12_to_14.ipynb
def get_store(docs: List[Document]):
    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings())

    # The storage layer for the parent documents
    store = InMemoryByteStore()
    id_key = "doc_id"

    # The retriever
    retriever = get_vectorstore_retriever(vectorstore, store, id_key)
    doc_ids = [str(uuid.uuid4()) for _ in docs]

    # FIXME: Finish this
    # # Docs linked to summaries
    # summary_docs = [
    #     Document(page_content=s, metadata={id_key: doc_ids[i]})
    #     for i, s in enumerate(summaries)
    # ]

    # # Add
    # retriever.vectorstore.add_documents(summary_docs)
    # retriever.docstore.mset(list(zip(doc_ids, docs)))


def build_chroma():
    # Embed a single document as a test
    vectorstore = Chroma.from_texts(
        ["harrison worked at kensho"],
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # RAG prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    model = ChatOpenAI()

    # RAG chain
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()}) | prompt | model | StrOutputParser()
    )

    # Add typing for input
    class Question(BaseModel):
        __root__: str

    chain = chain.with_types(input_type=Question)
