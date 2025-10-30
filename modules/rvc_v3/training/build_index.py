"""
Build FAISS retrieval index from extracted features.
"""

import logging
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
from tqdm import tqdm

from modules.rvc_v3.models.retrieval import RetrievalIndex

logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Build FAISS index from extracted content features.
    """
    
    def __init__(
        self,
        feature_dim: int,
        index_type: str = "IVF",
        n_clusters: int = 256,
        use_gpu: bool = True
    ):
        """
        Initialize index builder.
        
        Args:
            feature_dim: Dimension of content features
            index_type: FAISS index type ('Flat', 'IVF', 'HNSW')
            n_clusters: Number of clusters for IVF
            use_gpu: Whether to use GPU
        """
        self.feature_dim = feature_dim
        self.index_type = index_type
        self.n_clusters = n_clusters
        self.use_gpu = use_gpu
        
        logger.info(f"IndexBuilder initialized: dim={feature_dim}, type={index_type}")
    
    def build_from_project(
        self,
        project_dir: str,
        config,
        output_name: str = "retrieval_index",
        callback: Optional[Callable] = None
    ) -> str:
        """
        Build index from all feature files in a project.
        
        Args:
            project_dir: Path to project directory
            config: RVCV3Config
            output_name: Name for output index file
            callback: Optional progress callback
        
        Returns:
            Path to saved index
        """
        project_path = Path(project_dir)
        features_dir = project_path / config.features_dir
        
        # Collect all feature files
        feature_files = list(features_dir.glob("*.pt"))
        
        if not feature_files:
            raise ValueError(f"No feature files found in {features_dir}")
        
        logger.info(f"Building index from {len(feature_files)} feature files")
        
        # Load and concatenate all features
        all_features = []
        
        for i, feature_file in enumerate(tqdm(feature_files, desc="Loading features")):
            try:
                features_data = torch.load(feature_file, map_location='cpu')
                content_features = features_data['content']  # (T, feature_dim)
                
                # Convert to numpy
                if isinstance(content_features, torch.Tensor):
                    content_features = content_features.numpy()
                
                all_features.append(content_features)
                
                if callback:
                    progress = (i + 1) / len(feature_files) * 0.5  # First 50%
                    callback(progress, f"Loading {feature_file.name}", len(feature_files))
            
            except Exception as e:
                logger.error(f"Failed to load {feature_file}: {e}")
                continue
        
        # Concatenate all features
        all_features = np.vstack(all_features)  # (N, feature_dim)
        
        logger.info(f"Loaded {all_features.shape[0]} feature vectors")
        
        # Create index
        index = RetrievalIndex(
            feature_dim=self.feature_dim,
            index_type=self.index_type,
            n_clusters=self.n_clusters,
            use_gpu=self.use_gpu
        )
        
        # Build index
        if callback:
            callback(0.5, "Building FAISS index", 1)
        
        index.build(all_features)
        
        # Save index
        index_path = project_path / f"{output_name}"
        index.save(str(index_path))
        
        if callback:
            callback(1.0, "Index building complete", 1)
        
        logger.info(f"Index saved to {index_path}")
        
        return str(index_path)

