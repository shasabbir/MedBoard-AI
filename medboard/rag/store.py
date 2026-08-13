"""Persistent Chroma knowledge store with explicit local embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb

from medboard.models import RetrievedEvidence
from medboard.rag.embeddings import HashingEmbedder
from medboard.rag.ingestion import KnowledgeChunk, chunk_document, load_directory


class KnowledgeStore:
    """Ingest source-attributed chunks and perform cosine vector search."""

    def __init__(
        self,
        persist_directory: Path,
        *,
        collection_name: str = "medboard_knowledge",
        embedder: HashingEmbedder | None = None,
        ephemeral: bool = False,
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or HashingEmbedder()
        self.client = (
            chromadb.EphemeralClient()
            if ephemeral
            else chromadb.PersistentClient(path=str(persist_directory))
        )
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    @property
    def count(self) -> int:
        return int(self.collection.count())

    def ingest_directory(self, directory: Path) -> int:
        chunks = [
            chunk
            for document in load_directory(directory)
            for chunk in chunk_document(document)
        ]
        self.upsert(chunks)
        return len(chunks)

    def upsert(self, chunks: list[KnowledgeChunk]) -> None:
        if not chunks:
            return
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=self.embedder.embed_many([chunk.text for chunk in chunks]),
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document": chunk.document,
                    "organization": chunk.organization,
                    "year": chunk.year,
                    "source_url": chunk.source_url,
                    "document_type": chunk.document_type,
                    "section": chunk.section,
                }
                for chunk in chunks
            ],
        )

    def search(
        self, question: str, *, question_id: str, top_k: int = 5
    ) -> list[RetrievedEvidence]:
        if not question.strip():
            raise ValueError("retrieval question cannot be empty")
        if self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[self.embedder.embed(question)],
            n_results=self.count,
            include=["documents", "metadatas", "distances"],
        )
        ids = cast(list[list[str]], result["ids"])[0]
        documents = cast(list[list[str]], result["documents"])[0]
        metadata = cast(list[list[dict[str, Any]]], result["metadatas"])[0]
        distances = cast(list[list[float]], result["distances"])[0]
        ranked = sorted(
            zip(ids, documents, metadata, distances, strict=True),
            key=lambda item: (item[3], item[0]),
        )[:top_k]
        return [
            RetrievedEvidence(
                retrieval_id=f"RAG-{question_id}-{index:03d}",
                question_id=question_id,
                chunk_id=chunk_id,
                document=str(meta["document"]),
                source=str(meta["organization"]),
                section=str(meta["section"]),
                retrieved_text=document,
                similarity_score=max(0.0, min(1.0, 1.0 - distance)),
                source_url=str(meta["source_url"]),
            )
            for index, (chunk_id, document, meta, distance) in enumerate(ranked, start=1)
        ]
