# SOURCE: https://github.com/masuma131/ecoshark-genai-app/blob/master/vectorstore/chroma.py
from __future__ import annotations

import glob

from pathlib import Path
from typing import Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


industry_files: dict[str, str] = {
    "OpenCV tutorial Documentation": "opencv-tutorial-readthedocs-io-en-latest.pdf",
    "Pillow (PIL Fork) Documentation": "pillow-readthedocs-io-en-latest.pdf",
    "Rich": "rich-readthedocs-io-en-latest.pdf",
}


class Document(BaseModel):
    """Interface for interacting with a document.

    Attributes:
        page_content (str): The content of the document page.
        metadata (Dict): Additional metadata associated with the document.
    """

    page_content: str = None
    metadata: dict = Field(default_factory=dict)

    def __init__(self, page_content: str, metadata: dict, *args, **kwargs):
        """Initialize the Document.

        Args:
            page_content (str): The content of the document page.
            metadata (Dict): Additional metadata associated with the document.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(page_content=page_content, metadata=metadata, *args, **kwargs)


class DocLoader:
    """A class for loading documents.

    Attributes:
        path (str): The path to the document file.
        splitter (RecursiveCharacterTextSplitter): The text splitter used to split the document.
    """

    def __init__(self, path: str):
        """Initialize the DocLoader.

        Args:
            path (str): The path to the document file.
        """
        self.path = path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)

    def load_document(self) -> list[Document]:
        """Load a document.

        Returns:
            List[Document]: A list of loaded documents.
        """
        if self.path.endswith(".pdf"):
            return self._load_pdf()

    def _load_pdf(self) -> list[Document]:
        """Load a PDF document.

        Returns:
            List[Document]: A list of loaded PDF documents.
        """
        loader = PyMuPDFLoader(self.path, extract_images=True)
        docs = loader.load_and_split(self.splitter)
        # Add document_id as metadata to all docs
        for doc in docs:
            doc.metadata["filename"] = f"{Path(self.path).stem}.pdf"
        return docs


class ChromaDB:
    """A class for interacting with the Chroma database.

    Attributes:
        embedding_function (SentenceTransformerEmbeddings): The embedding function used for vectorization.
        vector_store_path (str): The path to the vector store directory.
        chroma (Optional[Chroma]): The Chroma instance.
    """

    def __init__(self):
        """Initialize the ChromaDB."""
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_path = "data/chroma"
        self.chroma: Optional[Chroma] = None

        # Load store if path exists
        if Path(self.vector_store_path).exists():
            self.chroma = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_function,
            )
        else:
            pdfs = glob.glob("data/sasb/*.pdf")
            docs = [DocLoader(pdf).load_document() for pdf in pdfs]
            docs = [item for sublist in docs for item in sublist]
            self.index(docs)

    def index(self, docs: list[Document]) -> None:
        """Index the documents in the Chroma database.

        Args:
            docs (List[Document]): The list of documents to index.
        """
        self.chroma = Chroma.from_documents(
            docs,
            persist_directory=self.vector_store_path,
            embedding=self.embedding_function,
        )

    def query(self, query: str, industry: Optional[str] = None) -> list[Document]:
        """Query the Chroma database for similar documents.

        Args:
            query (str): The query string.
            industry (Optional[str]): The industry filter (default: None).

        Returns:
            List[Document]: A list of similar documents.
        """
        filter: dict[str, str] = {}
        if industry:
            industry_file = industry_files[industry]
            filter = {"filename": industry_file}
        return self.chroma.similarity_search(query, k=30, filter=filter)  # type: ignore


if __name__ == "__main__":
    db = ChromaDB()
    docs = db.query(
        "Table: SUSTAINABILITY DISCLOSURE TOPICS & METRICS",
        industry="Building Products & Furnishings",
    )
    print(docs)
