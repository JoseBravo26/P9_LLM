"""Tests du filtrage des résultats de recherche vectorielle."""

from unittest.mock import MagicMock, patch

import numpy as np

from utils.schemas import Chunk, RetrievedChunk, SourceMetadata
from utils.vector_store import VectorStoreManager


def test_search_exclut_les_chunks_sous_le_seuil():
    """Les chunks dont le score est insuffisant doivent être ignorés."""
    manager = VectorStoreManager.__new__(VectorStoreManager)

    manager.index = MagicMock()
    manager.index.ntotal = 2

    manager.document_chunks = [
        Chunk(
            id="chunk_pertinent",
            text="Denver dispose d'une défense structurée.",
            metadata=SourceMetadata(
                source="Reddit 1.pdf",
                filename="Reddit 1.pdf",
                category="root",
            ),
        ),
        Chunk(
            id="chunk_hors_sujet",
            text="Discussion générale sur les médias NBA.",
            metadata=SourceMetadata(
                source="Reddit 2.pdf",
                filename="Reddit 2.pdf",
                category="root",
            ),
        ),
    ]

    manager._embed = MagicMock(
        return_value=np.zeros((1, 3), dtype="float32")
    )

    manager.index.search.return_value = (
        np.array([[0.80, 0.60]], dtype="float32"),
        np.array([[0, 1]], dtype="int64"),
    )

    with patch("utils.vector_store.MIN_RAG_SCORE", 0.68):
        results = manager.search(
            "Que disent les rapports sur la défense de Denver ?",
            k=2,
        )

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].id == "chunk_pertinent"
    assert round(results[0].score, 2) == 0.80