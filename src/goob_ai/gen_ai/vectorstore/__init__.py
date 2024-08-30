"""vector stores wrappers"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from goob_ai.gen_ai.vectorstore.chroma_store import ChromaDatabase
    from goob_ai.gen_ai.vectorstore.pgvector_store import PGVectorDatabase
    from goob_ai.gen_ai.vectorstore.pinecone_store import PineconeDatabase
