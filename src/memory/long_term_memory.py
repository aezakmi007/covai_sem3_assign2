"""
src/memory/long_term_memory.py
Task 3: Long-Term Durable Vector Memory Store using ChromaDB
Implements:
1. ChromaDB PersistentClient for cross-session durable memory
2. Cosine distance metric collection (hnsw:space = cosine)
3. User-isolated metadata filtering (where={"user_id": user_id})
4. Semantic recall across sessions via all-MiniLM-L6-v2 embeddings
5. Similarity threshold gating to reject irrelevant past memory
"""

import os
import chromadb
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer


class LongTermMemoryStore:
    def __init__(
        self,
        persist_dir: str = "data/chroma_memory",
        collection_name: str = "durable_user_memory",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        **kwargs
    ):
        """Initializes ChromaDB persistent client and durable facts collection."""
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # 1. Initialize local persistent Chroma client
        from chromadb.config import Settings
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # 2. Get or create collection configured for cosine distance
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # 3. Local embedding model
        self.model = SentenceTransformer(embedding_model_name)

    # -------------------------------------------------------------
    # 1. WRITE PATH: ADD DURABLE FACT / SESSION SUMMARY
    # -------------------------------------------------------------
    def add_durable_fact(
        self,
        user_id: str,
        fact_text: str,
        category: str = "general",
        session_id: Optional[str] = None,
    ) -> str:
        """
        Embeds and stores a durable user fact in ChromaDB.
        Uses user_id in metadata for multi-tenant isolation.
        """
        # Deduplication check: verify if identical fact already exists for this user
        existing = self.collection.get(
            where={"$and": [{"user_id": user_id}, {"category": category}]}
        )
        if existing and existing["documents"]:
            for doc in existing["documents"]:
                if doc.strip().lower() == fact_text.strip().lower():
                    return "duplicate_ignored"

        fact_id = f"fact_{user_id}_{self.collection.count() + 1:04d}"
        embedding = self.model.encode(
            fact_text, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        metadata = {
            "user_id": user_id,
            "category": category,
            "session_id": session_id or "global",
        }

        # Insert into ChromaDB collection
        self.collection.add(
            ids=[fact_id],
            documents=[fact_text],
            metadatas=[metadata],
            embeddings=[embedding],
        )
        return fact_id

    def add_session_summary(self, user_id: str, session_id: str, summary_text: str) -> str:
        """Stores consolidated summary of a completed conversation session into ChromaDB."""
        return self.add_durable_fact(
            user_id=user_id,
            fact_text=f"Summary of prior session ({session_id}): {summary_text}",
            category="session_summary",
            session_id=session_id,
        )

    # -------------------------------------------------------------
    # 2. READ PATH: SEMANTIC VECTOR SEARCH VIA CHROMADB
    # -------------------------------------------------------------
    def recall_facts(
        self,
        user_id: str,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search against ChromaDB filtered by user_id.
        In Chroma with hnsw:space='cosine':
            cosine_similarity = 1.0 - cosine_distance
        Filters out any matches below similarity_threshold.
        """
        # Encode query
        query_emb = self.model.encode(
            query, convert_to_numpy=True, normalize_embeddings=True
        ).tolist()

        # Query ChromaDB with user_id metadata filter
        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where={"user_id": user_id},
            include=["documents", "metadatas", "distances"]
        )

        recalled = []
        if not results or not results["documents"] or not results["documents"][0]:
            return recalled

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            # Chroma returns cosine distance in [0, 2]. Similarity = 1 - distance
            similarity = round(1.0 - float(dist), 4)
            if similarity >= similarity_threshold:
                recalled.append({
                    "fact_text": doc,
                    "category": meta.get("category", "general"),
                    "session_id": meta.get("session_id"),
                    "similarity_score": similarity,
                })

        return recalled

    def get_all_user_facts(self, user_id: str) -> List[str]:
        """Retrieves all durable facts stored in ChromaDB for a given user."""
        results = self.collection.get(where={"user_id": user_id})
        return results["documents"] if results and results["documents"] else []