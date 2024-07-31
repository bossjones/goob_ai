# SOURCE: https://github.com/masuma131/ecoshark-genai-app/blob/master/vectorstore/chroma.py
from __future__ import annotations

import glob

from pathlib import Path

from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


industry_files = {
    "OpenCV tutorial Documentation": "opencv-tutorial-readthedocs-io-en-latest.pdf",
    "Pillow (PIL Fork) Documentation": "pillow-readthedocs-io-en-latest.pdf",
    "Rich": "rich-readthedocs-io-en-latest.pdf",
}


class Document(BaseModel):
    """Interface for interacting with a document."""

    page_content: str = None
    metadata: dict = Field(default_factory=dict)

    def __init__(self, page_content, metadata, *args, **kwargs):
        super().__init__(page_content=page_content, metadata=metadata, *args, **kwargs)


class DocLoader:
    """A class for loading documents."""

    def __init__(self, path: str):
        self.path = path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)

    def load_document(self) -> list:  # type: ignore
        """Load a document."""
        if self.path.endswith(".pdf"):
            return self._load_pdf()

    def _load_pdf(self) -> list:
        """Load a PDF document."""
        loader = PyMuPDFLoader(self.path)
        docs = loader.load_and_split(self.splitter)
        # Add document_id as metadata to all docs
        for doc in docs:
            doc.metadata["filename"] = f"{Path(self.path).stem}.pdf"
        return docs


class ChromaDB:
    def __init__(self):
        self.embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store_path = "data/chroma"

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

    def index(self, docs):
        self.chroma = Chroma.from_documents(
            docs,
            persist_directory=self.vector_store_path,
            embedding=self.embedding_function,
        )

    def query(self, query, industry=None):
        filter = {}
        if industry:
            industry_file = industry_files[industry]
            filter = {"filename": industry_file}
        return self.chroma.similarity_search(query, k=30, filter=filter)


if __name__ == "__main__":
    db = ChromaDB()
    docs = db.query(
        "Table: SUSTAINABILITY DISCLOSURE TOPICS & METRICS",
        industry="Building Products & Furnishings",
    )
    print(docs)
