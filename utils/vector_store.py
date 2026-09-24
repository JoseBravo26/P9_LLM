"""Gestion du stockage vectoriel FAISS et des embeddings Mistral."""

from __future__ import annotations

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

LOGGER = logging.getLogger(__name__)

MIN_RAG_SCORE = 0.68

# Configuration optionnelle de Logfire
logfire: Optional[object] = None
try:
    import logfire as lf

    lf.configure()
    logfire = lf
except ImportError:
    LOGGER.warning("Logfire non installé dans vector_store")
except Exception as error:
    LOGGER.warning("Logfire indisponible dans vector_store : %s", error)


class VectorStoreManager:
    """Construit, persiste et interroge l'index vectoriel du projet."""

    def __init__(self) -> None:
        api_key = require_mistral_api_key()
        self.client = Mistral(api_key=api_key)
        self.index: faiss.Index | None = None
        self.document_chunks: list[Chunk] = []
        self._load()

    def _load(self) -> None:
        """Charge l'index et les chunks persistés s'ils existent."""
        index_path = Path(FAISS_INDEX_FILE)
        chunks_path = Path(DOCUMENT_CHUNKS_FILE)

        if not index_path.exists() or not chunks_path.exists():
            return

        self.index = faiss.read_index(str(index_path))

        with chunks_path.open("rb") as file:
            raw_chunks = pickle.load(file)

        self.document_chunks = [
            Chunk.model_validate(chunk)
            for chunk in raw_chunks
        ]

        if self.index.ntotal != len(self.document_chunks):
            raise RuntimeError(
                "Index FAISS et chunks persistés incohérents."
            )

        if logfire:
            import logfire as lf

            lf.info(
                "Index FAISS chargé",
                n_vectors=self.index.ntotal,
            )

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Génère et normalise les embeddings Mistral."""
        if not texts:
            raise ValueError(
                "Impossible de générer des embeddings pour une liste vide."
            )

        vectors: list[list[float]] = []

        for start_index in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch = texts[start_index : start_index + EMBEDDING_BATCH_SIZE]

            try:
                if logfire:
                    import logfire as lf

                    with lf.span("embedding.batch", batch_size=len(batch)):
                        response = self.client.embeddings.create(
                            model=EMBEDDING_MODEL,
                            inputs=batch,
                        )
                else:
                    response = self.client.embeddings.create(
                        model=EMBEDDING_MODEL,
                        inputs=batch,
                    )

                vectors.extend(item.embedding for item in response.data)

            except Exception as error:
                batch_number = start_index // EMBEDDING_BATCH_SIZE + 1

                if logfire:
                    import logfire as lf

                    lf.error(
                        "Échec génération embeddings",
                        batch=batch_number,
                        error=str(error),
                    )

                raise RuntimeError(
                    f"Échec de génération des embeddings pour le lot {batch_number}."
                ) from error

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

            split_texts = splitter.split_text(document.page_content)

            for chunk_index, text in enumerate(split_texts):
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

        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed(texts)

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
            import logfire as lf

            lf.info(
                "Index FAISS construit",
                n_documents=len(documents),
                n_chunks=len(chunks),
            )

    def search(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        """Retourne les chunks les plus similaires, filtrés par seuil minimal."""
        if self.index is None:
            raise RuntimeError("Index absent : exécutez 'python indexer.py'.")

        query_embedding = self._embed([query_text])
        number_of_results = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, number_of_results)

        results: list[RetrievedChunk] = []

        for rank, index in enumerate(indices[0]):
            if index < 0:
                continue

            score = float(scores[0][rank])

            if score < MIN_RAG_SCORE:
                LOGGER.info(
                    "Chunk ignoré car score insuffisant : id=%s, score=%.3f, seuil=%.3f",
                    self.document_chunks[index].id,
                    score,
                    MIN_RAG_SCORE,
                )
                continue

            results.append(
                RetrievedChunk(
                    **self.document_chunks[index].model_dump(),
                    score=score,
                )
            )

        if logfire:
            import logfire as lf

            lf.info(
                "Recherche vectorielle filtrée",
                query=query_text[:100],
                requested_k=k,
                returned_results=len(results),
                min_score=MIN_RAG_SCORE,
            )

        return results