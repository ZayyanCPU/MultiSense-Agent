"""
RAG Engine - Retrieval Augmented Generation using ChromaDB + HuggingFace.
Handles document ingestion, chunking, embedding, and retrieval.
Uses FREE HuggingFace Inference API for embeddings.
"""

import os
import structlog
from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
import chromadb
from chromadb.config import Settings as ChromaSettings
from PyPDF2 import PdfReader
import io
import hashlib

from app.config import get_settings

logger = structlog.get_logger(__name__)


class HFEmbeddings:
    """
    Lightweight embeddings wrapper using the free HuggingFace Inference API.
    Provides embed_documents() and embed_query() compatible with the RAG pipeline.
    Uses sentence-transformers models for high-quality sentence embeddings.
    """

    def __init__(self, model: str, api_token: str):
        self.client = InferenceClient(token=api_token)
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document chunks."""
        embeddings = []
        for text in texts:
            result = self.client.feature_extraction(
                text=text,
                model=self.model,
            )
            embeddings.append(self._normalize(result))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        result = self.client.feature_extraction(
            text=text,
            model=self.model,
        )
        return self._normalize(result)

    @staticmethod
    def _normalize(result) -> List[float]:
        """Normalize HF feature_extraction output to a flat float list."""
        if hasattr(result, "tolist"):
            result = result.tolist()

        if not isinstance(result, list) or len(result) == 0:
            return list(result) if hasattr(result, "__iter__") else [float(result)]

        # 3D → first element → 2D
        if isinstance(result[0], list) and len(result[0]) > 0 and isinstance(result[0][0], list):
            result = result[0]

        # 2D → mean-pool across tokens
        if isinstance(result[0], list):
            num_tokens = len(result)
            dim = len(result[0])
            return [
                sum(result[t][d] for t in range(num_tokens)) / num_tokens
                for d in range(dim)
            ]

        return [float(x) for x in result]


class RAGEngine:
    """
    Retrieval Augmented Generation engine.
    
    Handles:
    - PDF text extraction
    - Text chunking with overlap
    - Embedding generation and storage in ChromaDB
    - Semantic similarity search for context retrieval
    - Augmented prompt construction
    """

    def __init__(self):
        settings = get_settings()

        # Initialize ChromaDB client
        self.chroma_client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = settings.chroma_collection_name

        # Initialize embeddings using FREE HuggingFace Inference API
        self.embeddings = HFEmbeddings(
            model=settings.hf_embedding_model,
            api_token=settings.hf_api_token,
        )

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.top_k = settings.rag_top_k

    def _get_or_create_collection(self):
        """Get or create the ChromaDB collection."""
        return self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def extract_text_from_pdf(self, pdf_data: bytes) -> str:
        """
        Extract text content from a PDF file.
        
        Args:
            pdf_data: Raw PDF bytes
            
        Returns:
            Extracted text string
        """
        logger.info("pdf_text_extraction", pdf_size=len(pdf_data))
        reader = PdfReader(io.BytesIO(pdf_data))
        text_parts = []

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"[Page {i + 1}]\n{page_text}")

        full_text = "\n\n".join(text_parts)
        logger.info("pdf_extracted", pages=len(reader.pages), chars=len(full_text))
        return full_text

    async def ingest_document(
        self, content: str, filename: str, metadata: Optional[dict] = None
    ) -> int:
        """
        Ingest a document into the vector store.
        
        Steps:
        1. Split text into chunks
        2. Generate embeddings
        3. Store in ChromaDB with metadata
        
        Args:
            content: Full text content of the document
            filename: Source filename
            metadata: Additional metadata to store
            
        Returns:
            Number of chunks created
        """
        # Split into chunks
        chunks = self.text_splitter.split_text(content)
        logger.info("document_chunked", filename=filename, chunk_count=len(chunks))

        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(chunks)

        # Prepare IDs and metadata
        doc_hash = hashlib.md5(filename.encode()).hexdigest()[:8]
        ids = [f"{doc_hash}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                **(metadata or {}),
            }
            for i in range(len(chunks))
        ]

        # Upsert into ChromaDB
        collection = self._get_or_create_collection()
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        logger.info(
            "document_ingested",
            filename=filename,
            chunks=len(chunks),
            collection=self.collection_name,
        )
        return len(chunks)

    async def ingest_pdf(self, pdf_data: bytes, filename: str) -> int:
        """
        Extract text from PDF and ingest into vector store.
        
        Args:
            pdf_data: Raw PDF bytes
            filename: PDF filename
            
        Returns:
            Number of chunks created
        """
        text = self.extract_text_from_pdf(pdf_data)
        if not text.strip():
            logger.warning("pdf_empty", filename=filename)
            return 0
        return await self.ingest_document(text, filename, metadata={"type": "pdf"})

    async def retrieve_context(self, query: str, top_k: Optional[int] = None) -> Tuple[str, List[str]]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of results to return (overrides default)
            
        Returns:
            Tuple of (combined context string, list of source filenames)
        """
        k = top_k or self.top_k

        try:
            collection = self._get_or_create_collection()

            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"],
            )

            if not results["documents"][0]:
                logger.info("rag_no_results", query=query[:100])
                return "", []

            # Combine retrieved chunks
            context_parts = []
            sources = set()
            for doc, meta, distance in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                context_parts.append(doc)
                sources.add(meta.get("source", "unknown"))
                logger.debug(
                    "rag_result",
                    source=meta.get("source"),
                    chunk=meta.get("chunk_index"),
                    distance=round(distance, 4),
                )

            context = "\n\n---\n\n".join(context_parts)
            logger.info("rag_context_retrieved", chunks=len(context_parts), sources=list(sources))
            return context, list(sources)

        except Exception as e:
            logger.error("rag_retrieval_error", error=str(e))
            return "", []

    async def get_collection_stats(self) -> dict:
        """Get statistics about the vector store collection."""
        try:
            collection = self._get_or_create_collection()
            count = collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "status": "connected",
            }
        except Exception as e:
            return {
                "collection_name": self.collection_name,
                "document_count": 0,
                "status": f"error: {str(e)}",
            }


# Singleton
_rag_engine: Optional[RAGEngine] = None


def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine singleton."""
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine
