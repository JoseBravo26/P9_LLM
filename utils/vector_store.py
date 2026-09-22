"""Gestion du stockage vectoriel FAISS et des embeddings Mistral."""

import logging
import pickle
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from mistralai.client import Mistral

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENT_CHUNKS_FILE,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_FILE,
    require_mistral_api_key,
)
from .schemas import Chunk, RetrievedChunk, SourceDocument

try:
    import logfire
except ImportError:
    logfire = None

LOGGER = logging.getLogger(__name__)


class VectorStoreManager:
    """Construit, persiste et interroge l'index vectoriel du projet."""

    def __init__(self) -> None:
        api_key = require_mistral_api_key()
        self.client = Mistral(api_key=api_key)
        self.index: faiss.Index | None = None
        self.document_chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        """Charge l'index FAISS et les chunks persistés s'ils existent."""
        index_path = Path(FAISS_INDEX_FILE)
        chunks_path = Path(DOCUMENT_CHUNKS_FILE)

        if not index_path.exists() or not chunks_path.exists():
            return

        self.index = faiss.read_index(str(index_path))
        with chunks_path.open("rb") as file:
            raw_chunks = pickle.load(file)

        self.document_chunks = [Chunk.model_validate(chunk) for chunk in raw_chunks]

        if self.index.ntotal != len(self.document_chunks):
            raise RuntimeError("Index FAISS et chunks persistés incohérents.")

        if logfire:
            logfire.info("Index FAISS chargé", n_vectors=self.index.ntotal)

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Génère et normalise les embeddings Mistral sans vecteur invalide."""
        if not texts:
            raise ValueError("Impossible de générer des embeddings pour une liste vide.")

        vectors: list[list[float]] = []
        for start_index in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start_index : start_index + EMBEDDING_BATCH_SIZE]
            batch_number = start_index // EMBEDDING_BATCH_SIZE + 1

            try:
                if logfire:
                    with logfire.span("embedding.batch", batch_size=len(batch)):
                        response = self.client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            inputs=batch,
                        )
                else:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=batch,
                    )
            except Exception as error:
                if logfire:
                    logfire.error(
                        "Échec génération embeddings",
                        batch=batch_number,
                        error=str(error),
                    )
                raise RuntimeError(
                    f"Échec de génération des embeddings pour le lot {batch_number}."
                ) from error

            vectors.extend(item.embedding for item in response.data)

        embeddings = np.asarray(vectors, dtype="float32")
        if embeddings.ndim != 2 or len(embeddings) != len(texts):
            raise RuntimeError("Embeddings Mistral invalides ou incomplets.")

        faiss.normalize_L2(embeddings)
        return embeddings

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """Découpe les documents, construit l'index FAISS et persiste les artefacts."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        chunks: list[Chunk] = []

        for document_index, raw_document in enumerate(documents):
            document = SourceDocument.model_validate(raw_document)
            for chunk_index, text in enumerate(splitter.split_text(document.page_content)):
                chunks.append(
                    Chunk(
                        id=f"{document_index}_{chunk_index}",
                        text=text,
                        metadata=document.metadata.model_copy(
                            update={"chunk_id_in_doc": chunk_index}
                        ),
                    )
                )

        if not chunks:
            raise ValueError("Aucun chunk n'a été généré à partir des documents.")

        def create_and_persist() -> None:
            embeddings = self._embed([chunk.text for chunk in chunks])
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)
            self.document_chunks = chunks

            Path(FAISS_INDEX_FILE).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            with Path(DOCUMENT_CHUNKS_FILE).open("wb") as file:
                pickle.dump(
                    [chunk.model_dump() for chunk in self.document_chunks],
                    file,
                )

        if logfire:
            with logfire.span(
                "build_index",
                n_documents=len(documents),
                n_chunks=len(chunks),
            ):
                create_and_persist()
            logfire.info("Index FAISS construit", n_chunks=len(chunks))
        else:
            create_and_persist()

    def search(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        """Retourne les ``k`` chunks les plus similaires à la requête.

        Les embeddings étant normalisés et l'index utilisant ``IndexFlatIP``,
        le score retourné correspond à une similarité cosinus.
        """
        if self.index is None:
            raise RuntimeError("Index absent : exécutez 'python indexer.py'.")
        if not self.document_chunks:
            raise RuntimeError("Aucun chunk n'est disponible dans l'index FAISS.")

        number_of_results = min(k, self.index.ntotal)
        if number_of_results < 1:
            return []

        def perform_search() -> list[RetrievedChunk]:
            query_embedding = self._embed([query_text])
            scores, indices = self.index.search(query_embedding, number_of_results)
            return [
                RetrievedChunk(
                    **self.document_chunks[index].model_dump(),
                    score=float(scores[0][rank]),
                )
                for rank, index in enumerate(indices[0])
                if index >= 0
            ]

        if logfire:
            with logfire.span("vector_search", query=query_text[:100], k=number_of_results):
                results = perform_search()
            logfire.info(
                "Recherche vectorielle",
                n_results=len(results),
                results=[
                    {
                        "chunk_id": chunk.id,
                        "source": chunk.metadata.source,
                        "filename": chunk.metadata.filename,
                        "score": round(chunk.score, 4),
                        "excerpt": chunk.text[:300],
                    }
                    for chunk in results
                ],
            )
            return results

        results = perform_search()
        LOGGER.info("Recherche vectorielle : %s chunk(s) récupéré(s).", len(results))
        return results
