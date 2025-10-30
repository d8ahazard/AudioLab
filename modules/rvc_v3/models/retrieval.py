"""
Enhanced retrieval system for RVC V3.

FAISS-based feature retrieval with advanced mixing strategies.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


class RetrievalIndex:
    """
    Feature retrieval index using FAISS.
    
    Stores target voice features and performs efficient k-NN search.
    """
    
    def __init__(
        self,
        feature_dim: int,
        index_type: str = "IVF",
        n_clusters: int = 256,
        use_gpu: bool = True
    ):
        """
        Initialize retrieval index.
        
        Args:
            feature_dim: Dimension of content features
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            n_clusters: Number of clusters for IVF index
            use_gpu: Whether to use GPU for search
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        self.index = None
        self.features = None  # Store full features for mixing
        self.n_features = 0
        
        logger.info(
            f"RetrievalIndex initialized: dim={feature_dim}, "
            f"type={index_type}, gpu={self.use_gpu}"
        )
    
    def build(self, features: np.ndarray):
        """
        Build index from feature array.
        
        Args:
            features: Feature array (N, feature_dim)
        """
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.feature_dim}, "
                f"got {features.shape[1]}"
            )
        
        self.n_features = features.shape[0]
        self.features = features.astype('float32')
        
        logger.info(f"Building index with {self.n_features} features")
        
        # Create index based on type
        if self.index_type == "Flat":
            # Brute force exact search
            self.index = faiss.IndexFlatL2(self.feature_dim)
        
        elif self.index_type == "IVF":
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(self.feature_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer,
                self.feature_dim,
                min(self.n_clusters, self.n_features // 10)
            )
            
            # Train index
            logger.info("Training IVF index...")
            self.index.train(self.features)
            self.index.nprobe = 8  # Number of clusters to search
        
        elif self.index_type == "HNSW":
            # Hierarchical navigable small world graph
            self.index = faiss.IndexHNSWFlat(self.feature_dim, 32)
            self.index.hnsw.efConstruction = 40
            self.index.hnsw.efSearch = 16
        
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        # Add features to index
        self.index.add(self.features)
        
        logger.info(f"Index built successfully with {self.index.ntotal} vectors")
    
    def search(
        self,
        query: np.ndarray,
        k: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query features (N, feature_dim) or (feature_dim,)
            k: Number of neighbors to retrieve
        
        Returns:
            Tuple of (distances, indices)
        """
        if self.index is None:
            raise RuntimeError("Index not built yet")
        
        # Ensure correct shape
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        query = query.astype('float32')
        
        # Search
        distances, indices = self.index.search(query, k)
        
        return distances, indices
    
    def retrieve_features(
        self,
        query: np.ndarray,
        k: int = 1,
        return_distances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve actual feature vectors for queries.
        
        Args:
            query: Query features (N, feature_dim)
            k: Number of neighbors to retrieve
            return_distances: Whether to return distances
        
        Returns:
            Retrieved features (N, k, feature_dim) or with distances
        """
        distances, indices = self.search(query, k)
        
        # Get features
        retrieved = self.features[indices]  # (N, k, feature_dim)
        
        if return_distances:
            return retrieved, distances
        return retrieved
    
    def mix_features(
        self,
        source_features: np.ndarray,
        alpha: float = 0.75,
        k: int = 1,
        weighting: str = "uniform"
    ) -> np.ndarray:
        """
        Mix source features with retrieved target features.
        
        Args:
            source_features: Source content features (N, feature_dim)
            alpha: Mixing ratio (1.0 = all source, 0.0 = all retrieved)
            k: Number of neighbors to retrieve
            weighting: How to weight multiple retrievals ('uniform', 'distance')
        
        Returns:
            Mixed features (N, feature_dim)
        """
        # Retrieve target features
        retrieved, distances = self.retrieve_features(
            source_features, k=k, return_distances=True
        )
        
        # Weight retrieved features
        if k > 1:
            if weighting == "uniform":
                # Simple average
                retrieved = retrieved.mean(axis=1)  # (N, feature_dim)
            
            elif weighting == "distance":
                # Weight by inverse distance
                weights = 1.0 / (distances + 1e-8)  # (N, k)
                weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize
                
                # Weighted sum
                retrieved = (retrieved * weights[:, :, np.newaxis]).sum(axis=1)
            
            else:
                raise ValueError(f"Unknown weighting: {weighting}")
        else:
            retrieved = retrieved.squeeze(1)  # (N, feature_dim)
        
        # Mix source and retrieved
        mixed = alpha * source_features + (1 - alpha) * retrieved
        
        return mixed
    
    def save(self, output_path: str):
        """Save index and features to disk."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move to CPU if on GPU
        if self.use_gpu:
            index_cpu = faiss.index_gpu_to_cpu(self.index)
        else:
            index_cpu = self.index
        
        # Save index
        index_file = str(output_path.with_suffix('.index'))
        faiss.write_index(index_cpu, index_file)
        
        # Save features
        features_file = str(output_path.with_suffix('.npy'))
        np.save(features_file, self.features)
        
        # Save metadata
        metadata_file = str(output_path.with_suffix('.meta.npz'))
        np.savez(
            metadata_file,
            feature_dim=self.feature_dim,
            n_features=self.n_features,
            index_type=self.index_type,
            n_clusters=self.n_clusters
        )
        
        logger.info(f"Index saved to {output_path}")
    
    def load(self, input_path: str):
        """Load index and features from disk."""
        input_path = Path(input_path)
        
        # Load metadata
        metadata_file = str(input_path.with_suffix('.meta.npz'))
        if os.path.exists(metadata_file):
            metadata = np.load(metadata_file, allow_pickle=True)
            self.feature_dim = int(metadata['feature_dim'])
            self.n_features = int(metadata['n_features'])
            self.index_type = str(metadata['index_type'])
            self.n_clusters = int(metadata['n_clusters'])
        
        # Load index
        index_file = str(input_path.with_suffix('.index'))
        self.index = faiss.read_index(index_file)
        
        # Move to GPU if requested
        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Index moved to GPU")
            except Exception as e:
                logger.warning(f"Failed to move index to GPU: {e}")
                self.use_gpu = False
        
        # Load features
        features_file = str(input_path.with_suffix('.npy'))
        self.features = np.load(features_file)
        
        logger.info(f"Index loaded from {input_path}: {self.n_features} features")

