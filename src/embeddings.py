import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import hashlib
import numpy as np
from fastembed import TextEmbedding


class EmbeddingService:
    """
    FastEmbed-based semantic search service using Quantized all-MiniLM-L6-v2 model.
    Provides semantic search capabilities for dataset discovery.
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / "subsets" / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FastEmbed with Quantized all-MiniLM-L6-v2 model
        self.embedding_model = None
        self.dataset_embeddings = {}
        self.dataset_metadata = {}
        
        self._load_cache()
    
    def _get_embedding_model(self):
        """Lazy load the embedding model"""
        if self.embedding_model is None:
            import os
            import sys
            from contextlib import redirect_stderr, redirect_stdout

            # Suppress FastEmbed's progress bars and output during model loading
            # Save original stderr/stdout
            original_tqdm_disable = os.environ.get('TQDM_DISABLE')

            try:
                # Disable tqdm progress bars
                os.environ['TQDM_DISABLE'] = '1'

                # Redirect stdout/stderr to devnull during initialization
                with open(os.devnull, 'w') as devnull:
                    with redirect_stdout(devnull), redirect_stderr(devnull):
                        self.embedding_model = TextEmbedding(
                            model_name="BAAI/bge-small-en-v1.5",
                            max_length=256,
                            cache_dir=str(self.cache_dir / "models")
                        )
            finally:
                # Restore original TQDM_DISABLE setting
                if original_tqdm_disable is None:
                    os.environ.pop('TQDM_DISABLE', None)
                else:
                    os.environ['TQDM_DISABLE'] = original_tqdm_disable

        return self.embedding_model
    
    def _get_searchable_text(self, dataset: Dict[str, Any]) -> str:
        """Extract searchable text from dataset schema (columns and tables only)"""
        text_parts = []
        
        # Add table name/ID as context
        if dataset.get("id"):
            text_parts.append(f"table:{dataset['id']}")
        
        # Extract column information from schema
        if dataset.get("schema"):
            schema = dataset["schema"]
            if isinstance(schema, dict):
                # Handle different schema formats
                columns = schema.get("columns", [])
                if columns:
                    for col in columns:
                        if isinstance(col, dict):
                            # Add column name
                            if col.get("name"):
                                text_parts.append(f"column:{col['name']}")
                            # Add column type
                            if col.get("type"):
                                text_parts.append(f"type:{col['type']}")
                        elif isinstance(col, str):
                            text_parts.append(f"column:{col}")
        
        # Add column descriptions if available
        if dataset.get("column_descriptions"):
            col_descriptions = dataset["column_descriptions"]
            if isinstance(col_descriptions, dict):
                for col_name, description in col_descriptions.items():
                    if description:
                        text_parts.append(f"{col_name}:{description}")
        
        return " ".join(text_parts)
    
    def _get_dataset_hash(self, dataset_id: str, dataset_text: str) -> str:
        """Generate a hash for caching dataset embeddings"""
        combined = f"{dataset_id}:{dataset_text}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cached embeddings and metadata"""
        cache_file = self.cache_dir / "embeddings_cache.pkl"
        metadata_file = self.cache_dir / "metadata_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.dataset_embeddings = pickle.load(f)
            except Exception as e:
                print(f"Failed to load embedding cache: {e}")
                self.dataset_embeddings = {}
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.dataset_metadata = json.load(f)
            except Exception as e:
                print(f"Failed to load metadata cache: {e}")
                self.dataset_metadata = {}
    
    def _save_cache(self):
        """Save embeddings and metadata to cache"""
        try:
            # Save embeddings
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(self.dataset_embeddings, f)
            
            # Save metadata
            metadata_file = self.cache_dir / "metadata_cache.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.dataset_metadata, f, indent=2)
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def index_datasets(self, datasets: List[Dict[str, Any]]) -> Dict[str, int]:
        """Index datasets for semantic search"""
        model = self._get_embedding_model()
        
        new_count = 0
        updated_count = 0
        texts_to_embed = []
        dataset_info = []
        
        # Check which datasets need embedding
        for dataset in datasets:
            dataset_id = dataset.get("id")
            if not dataset_id:
                continue
            
            searchable_text = self._get_searchable_text(dataset)
            dataset_hash = self._get_dataset_hash(dataset_id, searchable_text)
            
            # Check if we need to update the embedding
            if (dataset_id not in self.dataset_embeddings or 
                self.dataset_metadata.get(dataset_id, {}).get("hash") != dataset_hash):
                
                texts_to_embed.append(searchable_text)
                dataset_info.append({
                    "id": dataset_id,
                    "text": searchable_text,
                    "hash": dataset_hash,
                    "metadata": dataset
                })
                
                if dataset_id in self.dataset_embeddings:
                    updated_count += 1
                else:
                    new_count += 1
        
        # Generate embeddings in batch
        if texts_to_embed:
            try:
                embeddings = list(model.embed(texts_to_embed))
                
                # Store embeddings and metadata
                for i, info in enumerate(dataset_info):
                    self.dataset_embeddings[info["id"]] = embeddings[i]
                    self.dataset_metadata[info["id"]] = {
                        "hash": info["hash"],
                        "text": info["text"],
                        "metadata": info["metadata"]
                    }
                
                # Save to cache
                self._save_cache()
                
            except Exception as e:
                print(f"Failed to generate embeddings: {e}")
                return {"new": 0, "updated": 0, "error": str(e)}
        
        return {"new": new_count, "updated": updated_count}
    
    def semantic_search(self, query: str, datasets: List[Dict[str, Any]], 
                       min_score: Optional[float] = None, top_k: Optional[int] = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        Perform semantic search on datasets
        
        Args:
            query: Search query
            datasets: List of datasets to search
            min_score: Minimum similarity score (0.0 to 1.0)
            top_k: Maximum number of results to return
            
        Returns:
            List of tuples (dataset, similarity_score) sorted by relevance
        """
        if not query.strip():
            return [(dataset, 1.0) for dataset in datasets[:top_k or len(datasets)]]
        
        model = self._get_embedding_model()
        
        # Generate query embedding
        query_embedding = list(model.embed([query]))[0]
        
        # Calculate similarities
        results = []
        for dataset in datasets:
            dataset_id = dataset.get("id")
            if not dataset_id or dataset_id not in self.dataset_embeddings:
                continue
            
            dataset_embedding = self.dataset_embeddings[dataset_id]
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, dataset_embedding)
            
            # Apply minimum score filter
            if min_score is None or similarity >= min_score:
                results.append((dataset, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k limit
        if top_k:
            results = results[:top_k]
        
        return results
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def clear_cache(self):
        """Clear all cached embeddings"""
        self.dataset_embeddings = {}
        self.dataset_metadata = {}
        
        # Remove cache files
        try:
            cache_file = self.cache_dir / "embeddings_cache.pkl"
            metadata_file = self.cache_dir / "metadata_cache.json"
            
            if cache_file.exists():
                cache_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
        except Exception as e:
            print(f"Failed to clear cache files: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics"""
        return {
            "cached_datasets": len(self.dataset_embeddings),
            "cache_dir": str(self.cache_dir),
            "model_loaded": self.embedding_model is not None
        }