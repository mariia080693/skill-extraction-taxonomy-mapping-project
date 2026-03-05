"""
Build (or load) a ChromaDB collection over the ESCO taxonomy skill labels.
"""

import csv
import os
from typing import Dict, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import CHROMA_COLLECTION, CHROMA_DIR, EMBEDDING_MODEL, TAXONOMY_PATH


# Load taxonomy from CSV 

def load_taxonomy(path: str = TAXONOMY_PATH) -> List[Dict[str, str]]:
    """Return list of dicts from the taxonomy CSV."""
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def unique_skills_with_metadata(
    taxonomy: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Return deduplicated skill entries, keeping the first occurrence's metadata."""
    seen = set()
    results = []
    for row in taxonomy:
        label = row["skillLabel"]
        if label not in seen:
            seen.add(label)
            results.append({
                "skillLabel": label,
                "skillType": row.get("skillType", ""),
                "relationType": row.get("relationType", ""),
                "occupationLabel": row.get("occupationLabel", ""),
            })
    return results


# ChromaDB-backed taxonomy index 

class TaxonomyIndex:
    """
    Wraps a ChromaDB collection over taxonomy skill labels.

    Attributes
    ----------
    collection : ChromaDB collection with embedded skill labels + metadata
    """

    def __init__(self, collection: chromadb.Collection):
        self.collection = collection

    @property
    def total_skills(self) -> int:
        return self.collection.count()
    
    

    # Build from scratch
    
    @classmethod
    def build(cls, taxonomy: List[Dict[str, str]] | None = None) -> "TaxonomyIndex":
        """Embed every unique skill label and upsert into ChromaDB."""
        if taxonomy is None:
            taxonomy = load_taxonomy()

        skills = unique_skills_with_metadata(taxonomy)
        print(" Embedding", format(len(skills), ","), "unique skill labels …")

        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)

        # Delete existing collection if rebuilding
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

        # Upsert in batches of 5000 (ChromaDB limit)
        batch_size = 5000
        for i in range(0, len(skills), batch_size):
            batch = skills[i : i + batch_size]
            collection.upsert(
                ids=[f"skill_{i + j}" for j in range(len(batch))],
                documents=[s["skillLabel"] for s in batch],
                metadatas=[{
                    "skillLabel": s["skillLabel"],
                    "skillType": s["skillType"],
                    "relationType": s["relationType"],
                    "occupationLabel": s["occupationLabel"],
                } for s in batch],
            )
            print("    Upserted batch", i // batch_size + 1,
                  "(" + str(min(i + batch_size, len(skills))) + "/" + str(len(skills)) + ")")

        print("  Collection saved →", CHROMA_DIR + "/", "(" + str(collection.count()) + " skills)")
        return cls(collection)


    # Load from disk
    
    @classmethod
    def load(cls) -> "TaxonomyIndex":
        """Load a previously-built ChromaDB collection."""
        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
        )
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = client.get_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
        )
        return cls(collection)
    
    
    
    # Load existing collection or build a new one
    @classmethod
    def get_or_build(cls) -> "TaxonomyIndex":
        if os.path.exists(CHROMA_DIR):
            try:
                print("Loading existing ChromaDB collection …")
                idx = cls.load()
                if idx.total_skills > 0:
                    return idx
            except Exception:
                pass
        print("  Building ChromaDB collection (first run) …")
        return cls.build()

    # Query

    def search(
        self,
        query: str,
        top_k: int = 3,
        where: dict | None = None,
    ) -> List[Tuple[str, float]]:
        """
        Return top-k taxonomy labels closest to `query`.

        Parameters
        ----------
        query   : skill phrase to search for
        top_k   : number of results
        where   : optional metadata filter, e.g. {"skillType": "knowledge"}

        Returns list of (label, cosine_similarity) tuples sorted desc.
        """
        kwargs = {
            "query_texts": [query],
            "n_results": top_k,
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        output = []
        if results["documents"] and results["distances"]:
            for doc, dist in zip(results["documents"][0], results["distances"][0]):
                similarity = 1.0 - dist
                output.append((doc, similarity))
        return output
