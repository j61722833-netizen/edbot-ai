"""Vector database utilities for textbook Q&A system."""

import os
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from langchain_openai import OpenAIEmbeddings
    from langchain.schema import Document
    from langchain_community.vectorstores import FAISS
except ImportError:
    OpenAIEmbeddings = None
    Document = None
    FAISS = None

from ..processors.text_extractor import TextChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for textbook chunks with citation support."""
    
    def __init__(self, 
                 index_path: str = "./vector_indexes",
                 embeddings_model: str = "text-embedding-3-large",
                 openai_api_key: Optional[str] = None):
        """Initialize vector store.
        
        Args:
            index_path: Directory to store FAISS indexes
            embeddings_model: OpenAI embeddings model name
            openai_api_key: OpenAI API key (or from environment)
        """
        if not faiss:
            raise ImportError("FAISS not installed. Install with: pip install faiss-cpu")
        if not OpenAIEmbeddings:
            raise ImportError("LangChain OpenAI not installed. Install with: pip install langchain-openai")
        
        self.index_path = Path(index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.embeddings = OpenAIEmbeddings(
            model=embeddings_model,
            openai_api_key=api_key
        )
        
        self.vector_store: Optional[FAISS] = None
        self.chunk_metadata: Dict[str, TextChunk] = {}
        
    def create_index_from_chunks(self, chunks: List[TextChunk], index_name: str = "textbook") -> None:
        """Create FAISS index from text chunks.
        
        Args:
            chunks: List of TextChunk objects to index
            index_name: Name for the index (used for saving/loading)
        """
        if not chunks:
            raise ValueError("No chunks provided for indexing")
        
        logger.info(f"Creating index '{index_name}' from {len(chunks)} chunks...")
        
        # Convert chunks to LangChain documents
        documents = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_type": chunk.chunk_type,
                    "quality": chunk.quality.value,
                    "word_count": chunk.word_count
                }
            )
            documents.append(doc)
            # Store original chunk for citation
            self.chunk_metadata[chunk.chunk_id] = chunk
        
        # Create FAISS vector store
        logger.info("Generating embeddings and creating FAISS index...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # Save index and metadata
        self._save_index(index_name)
        logger.info(f"Index '{index_name}' created and saved successfully")
    
    def load_index(self, index_name: str = "textbook") -> bool:
        """Load existing FAISS index.
        
        Args:
            index_name: Name of the index to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_file = self.index_path / f"{index_name}.faiss"
        metadata_file = self.index_path / f"{index_name}_metadata.pkl"
        
        if not (index_file.exists() and metadata_file.exists()):
            logger.warning(f"Index files not found for '{index_name}'")
            return False
        
        try:
            # Load FAISS index
            self.vector_store = FAISS.load_local(
                str(self.index_path / index_name),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Load chunk metadata
            with open(metadata_file, 'rb') as f:
                self.chunk_metadata = pickle.load(f)
            
            logger.info(f"Loaded index '{index_name}' with {len(self.chunk_metadata)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index '{index_name}': {e}")
            return False
    
    def _save_index(self, index_name: str) -> None:
        """Save FAISS index and metadata.
        
        Args:
            index_name: Name for the index files
        """
        if not self.vector_store:
            raise ValueError("No vector store to save")
        
        # Save FAISS index
        self.vector_store.save_local(str(self.index_path / index_name))
        
        # Save chunk metadata
        metadata_file = self.index_path / f"{index_name}_metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
    
    def similarity_search(self, 
                         query: str, 
                         k: int = 5,
                         filter_quality: Optional[str] = None) -> List[Tuple[Document, float, TextChunk]]:
        """Search for similar chunks with citation information.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_quality: Filter by text quality (high, medium, low)
            
        Returns:
            List of tuples: (document, similarity_score, original_chunk)
        """
        if not self.vector_store:
            raise ValueError("No index loaded. Create or load an index first.")
        
        # Perform similarity search
        results = self.vector_store.similarity_search_with_score(query, k=k*2)  # Get more for filtering
        
        # Filter and enrich results
        enriched_results = []
        for doc, score in results:
            chunk_id = doc.metadata["chunk_id"]
            original_chunk = self.chunk_metadata.get(chunk_id)
            
            if original_chunk:
                # Apply quality filter if specified
                if filter_quality and original_chunk.quality.value != filter_quality:
                    continue
                
                enriched_results.append((doc, score, original_chunk))
                
                if len(enriched_results) >= k:
                    break
        
        logger.debug(f"Found {len(enriched_results)} relevant chunks for query: {query[:50]}...")
        return enriched_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[TextChunk]:
        """Get original chunk by ID.
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            TextChunk object or None if not found
        """
        return self.chunk_metadata.get(chunk_id)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded index.
        
        Returns:
            Dictionary with index statistics
        """
        if not self.vector_store or not self.chunk_metadata:
            return {"status": "no_index_loaded"}
        
        chunks = list(self.chunk_metadata.values())
        
        # Quality distribution
        quality_dist = {}
        for chunk in chunks:
            quality = chunk.quality.value
            quality_dist[quality] = quality_dist.get(quality, 0) + 1
        
        # Type distribution
        type_dist = {}
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            type_dist[chunk_type] = type_dist.get(chunk_type, 0) + 1
        
        # Source file distribution
        source_dist = {}
        for chunk in chunks:
            source = chunk.source_file
            source_dist[source] = source_dist.get(source, 0) + 1
        
        total_words = sum(chunk.word_count for chunk in chunks)
        
        return {
            "status": "loaded",
            "total_chunks": len(chunks),
            "total_words": total_words,
            "average_words_per_chunk": total_words / len(chunks) if chunks else 0,
            "quality_distribution": quality_dist,
            "type_distribution": type_dist,
            "source_files": len(source_dist),
            "embedding_dimension": self.vector_store.index.d if hasattr(self.vector_store, 'index') else None
        }


def create_vector_index_from_json(json_path: Path, 
                                 index_name: str = "textbook",
                                 vector_store_config: Optional[Dict] = None) -> VectorStore:
    """Create vector index from extracted text JSON file.
    
    Args:
        json_path: Path to JSON file with extracted text chunks
        index_name: Name for the vector index
        vector_store_config: Configuration for VectorStore
        
    Returns:
        Configured VectorStore instance
    """
    import json
    from ..processors.text_extractor import TextChunk, TextQuality
    
    # Load chunks from JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    for chunk_data in data.get("chunks", []):
        chunk = TextChunk(
            text=chunk_data["text"],
            chunk_id=chunk_data["chunk_id"],
            source_file=chunk_data["source_file"],
            page_start=chunk_data["page_start"],
            page_end=chunk_data["page_end"],
            chunk_type=chunk_data.get("chunk_type", "paragraph"),
            quality=TextQuality(chunk_data.get("quality", "medium")),
            word_count=chunk_data.get("word_count", 0),
            char_count=chunk_data.get("char_count", 0),
            metadata=chunk_data.get("metadata", {})
        )
        chunks.append(chunk)
    
    # Create vector store
    config = vector_store_config or {}
    vector_store = VectorStore(**config)
    vector_store.create_index_from_chunks(chunks, index_name)
    
    return vector_store


if __name__ == "__main__":
    import sys
    
    # Basic CLI for creating index from JSON
    if len(sys.argv) < 2:
        print("Usage: python vector_store.py <json_file> [index_name]")
        sys.exit(1)
    
    json_file = Path(sys.argv[1])
    index_name = sys.argv[2] if len(sys.argv) > 2 else "textbook"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        print(f"Creating vector index from {json_file}...")
        vector_store = create_vector_index_from_json(json_file, index_name)
        
        # Print stats
        stats = vector_store.get_index_stats()
        print(f"\nIndex created successfully:")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Source files: {stats['source_files']}")
        print(f"  Embedding dimension: {stats['embedding_dimension']}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)