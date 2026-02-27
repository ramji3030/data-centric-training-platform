"""Data deduplication module for exact and semantic duplicate detection."""

import hashlib
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering


@dataclass
class DeduplicationResult:
    """Result of deduplication process."""
    duplicates: List[Tuple[int, int]]  # Pairs of indices
    clusters: List[List[int]]  # Clusters of duplicate indices
    kept_indices: Set[int]  # Indices to keep
    removed_count: int  # Number of duplicates removed


class ExactDeduplicator:
    """Detects exact duplicates using hash comparison."""

    @staticmethod
    def find_exact_duplicates(documents: List[str]) -> Dict[str, List[int]]:
        """Find exact duplicates.
        
        Args:
            documents: List of text documents
            
        Returns:
            Dictionary mapping hash to list of document indices
        """
        hash_to_indices = defaultdict(list)
        
        for idx, doc in enumerate(documents):
            # Normalize and hash
            normalized = doc.strip().lower()
            doc_hash = hashlib.sha256(normalized.encode()).hexdigest()
            hash_to_indices[doc_hash].append(idx)
        
        return hash_to_indices


class SemanticDeduplicator:
    """Detects semantic duplicates using embeddings and clustering."""
    
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize semantic deduplicator.
        
        Args:
            model: Model name for embeddings
        """
        self.model = SentenceTransformer(model)
    
    def find_semantic_duplicates(
        self,
        documents: List[str],
        threshold: float = 0.95,
        batch_size: int = 32
    ) -> DeduplicationResult:
        """Find semantic duplicates.
        
        Args:
            documents: List of documents
            threshold: Similarity threshold (0-1)
            batch_size: Batch size for encoding
            
        Returns:
            Deduplication result with clusters and kept indices
        """
        # Encode documents
        embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Compute similarity matrix
        similarities = np.dot(embeddings, embeddings.T)
        
        # Cluster
        distance_matrix = 1 - similarities
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,
            linkage='average'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # Keep first from each cluster
        kept_indices = {cluster[0] for cluster in clusters.values()}
        removed_count = len(documents) - len(kept_indices)
        
        # Find duplicate pairs
        duplicates = []
        for cluster in clusters.values():
            if len(cluster) > 1:
                for dup_idx in cluster[1:]:
                    duplicates.append((cluster[0], dup_idx))
        
        return DeduplicationResult(
            duplicates=duplicates,
            clusters=list(clusters.values()),
            kept_indices=kept_indices,
            removed_count=removed_count
        )


class HybridDeduplicator:
    """Combines exact and semantic deduplication."""
    
    def __init__(self, semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize hybrid deduplicator."""
        self.exact = ExactDeduplicator()
        self.semantic = SemanticDeduplicator(semantic_model)
    
    def deduplicate(
        self,
        documents: List[str],
        semantic_threshold: float = 0.95
    ) -> DeduplicationResult:
        """Deduplicate using hybrid approach.
        
        Args:
            documents: List of documents
            semantic_threshold: Similarity threshold for semantic matching
            
        Returns:
            Deduplication result
        """
        # First pass: exact duplicates
        exact_dups = self.exact.find_exact_duplicates(documents)
        exact_removed = {idx for indices in exact_dups.values() 
                        for idx in indices[1:]}
        
        # Second pass: semantic duplicates on remaining
        remaining_docs = [doc for idx, doc in enumerate(documents)
                         if idx not in exact_removed]
        remaining_indices = [idx for idx, doc in enumerate(documents)
                            if idx not in exact_removed]
        
        semantic_result = self.semantic.find_semantic_duplicates(
            remaining_docs,
            threshold=semantic_threshold
        )
        
        # Map back to original indices
        semantic_removed = {remaining_indices[idx] 
                           for idx in range(len(remaining_docs))
                           if idx not in semantic_result.kept_indices}
        
        # Combine results
        total_removed = exact_removed | semantic_removed
        kept_indices = set(range(len(documents))) - total_removed
        
        return DeduplicationResult(
            duplicates=[],
            clusters=[],
            kept_indices=kept_indices,
            removed_count=len(total_removed)
        )
