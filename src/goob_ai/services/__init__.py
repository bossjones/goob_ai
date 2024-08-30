# SOURCE: https://github.com/NirDiamant/RAG_Techniques/blob/9e825a8b6aaae1b29864d9d350cf95aacafac5d4/helper_functions.py#L19
from __future__ import annotations

import asyncio
import logging
import random
import textwrap

from typing import Any, List, Tuple

import numpy as np
import pymupdf

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import (
    ConfigurableField,
    Runnable,
    RunnableBranch,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnableSequence,
    RunnableSerializable,
    ensure_config,
)
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger as LOGGER
from rank_bm25 import BM25Okapi

from goob_ai import llm_manager


def replace_t_with_space(list_of_documents: list[Document]) -> list[str]:
    """Replace all tab characters with spaces in the page content of each document.

    Args:
        list_of_documents: A list of document objects, each with a 'page_content' attribute.

    Returns:
        The modified list of documents with tab characters replaced by spaces.
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace("\t", " ")  # Replace tabs with spaces
    return list_of_documents


def text_wrap(text: str, width: int = 120) -> str:
    """Wrap the input text to the specified width.

    Args:
        text: The input text to wrap.
        width: The width at which to wrap the text. Defaults to 120.

    Returns:
        The wrapped text.
    """
    return textwrap.fill(text, width=width)


def encode_pdf(path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """Encode a PDF book into a vector store using OpenAI embeddings.

    Args:
        path: The path to the PDF file.
        chunk_size: The desired size of each text chunk. Defaults to 1000.
        chunk_overlap: The amount of overlap between consecutive chunks. Defaults to 200.

    Returns:
        A FAISS vector store containing the encoded book content.
    """
    # Load PDF documents
    loader = PyPDFLoader(path)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(cleaned_texts, embeddings)

    return vectorstore


def encode_from_string(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> FAISS:
    """Encode a string into a vector store using OpenAI embeddings.

    Args:
        content: The text content to be encoded.
        chunk_size: The size of each chunk of text. Defaults to 1000.
        chunk_overlap: The overlap between chunks. Defaults to 200.

    Returns:
        A vector store containing the encoded content.

    Raises:
        ValueError: If the input content is not valid.
        RuntimeError: If there is an error during the encoding process.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Content must be a non-empty string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")

    if not isinstance(chunk_overlap, int) or chunk_overlap < 0:
        raise ValueError("chunk_overlap must be a non-negative integer.")

    try:
        # Split the content into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunks = text_splitter.create_documents([content])

        # Assign metadata to each chunk
        for chunk in chunks:
            chunk.metadata["relevance_score"] = 1.0

        # Generate embeddings and create the vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        raise RuntimeError(f"An error occurred during the encoding process: {str(e)}")

    return vectorstore


def retrieve_context_per_question(question: str, chunks_query_retriever: VectorStoreRetriever) -> list[str]:
    """Retrieve relevant context for a given question using the chunks query retriever.

    Args:
        question: The question for which to retrieve context.
        chunks_query_retriever: The FAISS index used to retrieve relevant chunks.

    Returns:
        A list of relevant context strings.
    """
    # Retrieve relevant documents for the given question
    docs = chunks_query_retriever.get_relevant_documents(question)

    # Extract document content
    context = [doc.page_content for doc in docs]

    return context


class QuestionAnswerFromContext(BaseModel):
    """Model to generate an answer to a query based on a given context.

    Attributes:
        answer_based_on_content: The generated answer based on the context.
    """

    answer_based_on_content: str = Field(description="Generates an answer to a query based on a given context.")


def create_question_answer_from_context_chain(llm: ChatOpenAI | None) -> RunnableSequence | RunnableSerializable:
    """Create a chain for answering questions based on context using the provided language model.

    Args:
        llm: The language model to use for generating answers.

    Returns:
        The created question-answer chain.
    """
    if llm is None:
        llm = llm_manager.LlmManager().llm
        question_answer_from_context_llm = llm
    else:
        # Initialize the ChatOpenAI model with specific parameters
        question_answer_from_context_llm = llm

    # Define the prompt template for chain-of-thought reasoning
    question_answer_prompt_template = """
    For the question below, provide a concise but suffice answer based ONLY on the provided context:
    {context}
    Question
    {question}
    """

    # Create a PromptTemplate object with the specified template and input variables
    question_answer_from_context_prompt = PromptTemplate(
        template=question_answer_prompt_template,
        input_variables=["context", "question"],
    )

    # Create a chain by combining the prompt template and the language model
    question_answer_from_context_cot_chain = (
        question_answer_from_context_prompt
        | question_answer_from_context_llm.with_structured_output(QuestionAnswerFromContext)
    )
    return question_answer_from_context_cot_chain


def answer_question_from_context(
    question: str, context: list[str], question_answer_from_context_chain: RunnableSerializable | Any
) -> dict:
    """Answer a question using the given context by invoking a chain of reasoning.

    Args:
        question: The question to be answered.
        context: The context to be used for answering the question.
        question_answer_from_context_chain: The chain to use for generating the answer.

    Returns:
        A dictionary containing the answer, context, and question.
    """
    input_data = {"question": question, "context": context}
    LOGGER.info("Answering the question from the retrieved context...")

    output = question_answer_from_context_chain.invoke(input_data)
    answer = output.answer_based_on_content
    return {"answer": answer, "context": context, "question": question}


def show_context(context: list[str]) -> None:
    """Display the contents of the provided context list.

    Args:
        context: A list of context items to be displayed.

    LOGGER.infos each context item in the list with a heading indicating its position.
    """
    for i, c in enumerate(context):
        LOGGER.info(f"Context {i+1}:")
        LOGGER.info(c)
        LOGGER.info("\n")


def read_pdf_to_string(path: str) -> str:
    """Read a PDF document from the specified path and return its content as a string.

    Args:
        path: The file path to the PDF document.

    Returns:
        The concatenated text content of all pages in the PDF document.
    """
    # Open the PDF document located at the specified path
    doc = pymupdf.open(path)
    content = ""
    # Iterate over each page in the document
    for page_num in range(len(doc)):
        # Get the current page
        page = doc[page_num]
        # Extract the text content from the current page and append it to the content string
        content += page.get_text()  # pyright: ignore[reportAttributeAccessIssue]
    return content


def bm25_retrieval(bm25: BM25Okapi, cleaned_texts: list[str], query: str, k: int = 5) -> list[str]:
    """Perform BM25 retrieval and return the top k cleaned text chunks.

    Args:
        bm25: Pre-computed BM25 index.
        cleaned_texts: List of cleaned text chunks corresponding to the BM25 index.
        query: The query string.
        k: The number of text chunks to retrieve. Defaults to 5.

    Returns:
        The top k cleaned text chunks based on BM25 scores.
    """
    # Tokenize the query
    query_tokens = query.split()

    # Get BM25 scores for the query
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top k scores
    top_k_indices = np.argsort(bm25_scores)[::-1][:k]

    # Retrieve the top k cleaned text chunks
    top_k_texts = [cleaned_texts[i] for i in top_k_indices]

    return top_k_texts


# SOURCE: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb
def split_text_to_chunks_with_indices(text: str, chunk_size: int = 400, chunk_overlap: int = 200) -> list[Document]:
    """Function to split text into chunks with metadata of the chunk chronological index"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(Document(page_content=chunk, metadata={"index": len(chunks), "text": text}))
        start += chunk_size - chunk_overlap
    return chunks


# SOURCE: https://github.com/NirDiamant/RAG_Techniques/blob/main/all_rag_techniques/context_enrichment_window_around_chunk.ipynb
def get_chunk_by_index(vectorstore, target_index: int) -> Document:
    """
    Function to draw the kth chunk (in the original order) from the vector store

    Retrieve a chunk from the vectorstore based on its index in the metadata.

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    target_index (int): The index of the chunk to retrieve.

    Returns:
    Optional[Document]: The retrieved chunk as a Document object, or None if not found.
    """
    # This is a simplified version. In practice, you might need a more efficient method
    # to retrieve chunks by index, depending on your vectorstore implementation.
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    for doc in all_docs:
        if doc.metadata.get("index") == target_index:
            return doc
    return None


def retrieve_with_context_overlap(
    vectorstore: VectorStore,
    retriever: VectorStoreRetriever,
    query: str,
    num_neighbors: int = 1,
    chunk_size: int = 200,
    chunk_overlap: int = 20,
) -> list[str]:
    """
    Retrieve chunks based on a query, then fetch neighboring chunks and concatenate them,
    accounting for overlap and correct indexing.

    Args:
    vectorstore (VectorStore): The vectorstore containing the chunks.
    retriever: The retriever object to get relevant documents.
    query (str): The query to search for relevant chunks.
    num_neighbors (int): The number of chunks to retrieve before and after each relevant chunk.
    chunk_size (int): The size of each chunk when originally split.
    chunk_overlap (int): The overlap between chunks when originally split.

    Returns:
    List[str]: List of concatenated chunk sequences, each centered on a relevant chunk.
    """
    relevant_chunks = retriever.get_relevant_documents(query)
    result_sequences = []

    for chunk in relevant_chunks:
        current_index = chunk.metadata.get("index")
        if current_index is None:
            continue

        # Determine the range of chunks to retrieve
        start_index = max(0, current_index - num_neighbors)
        end_index = current_index + num_neighbors + 1  # +1 because range is exclusive at the end

        # Retrieve all chunks in the range
        neighbor_chunks = []
        for i in range(start_index, end_index):
            neighbor_chunk = get_chunk_by_index(vectorstore, i)
            if neighbor_chunk:
                neighbor_chunks.append(neighbor_chunk)

        # Sort chunks by their index to ensure correct order
        neighbor_chunks.sort(key=lambda x: x.metadata.get("index", 0))

        # Concatenate chunks, accounting for overlap
        concatenated_text = neighbor_chunks[0].page_content
        for i in range(1, len(neighbor_chunks)):
            current_chunk = neighbor_chunks[i].page_content
            overlap_start = max(0, len(concatenated_text) - chunk_overlap)
            concatenated_text = concatenated_text[:overlap_start] + current_chunk

        result_sequences.append(concatenated_text)

    return result_sequences
