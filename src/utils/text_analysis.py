"""Utilities for analyzing extracted text chunks."""

from pathlib import Path
from typing import List, Dict, Any
import json
from collections import Counter
import logging

try:
    from ..processors.text_extractor import TextChunk, extract_text_from_chunks
except ImportError:
    # For direct execution, use absolute imports
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from processors.text_extractor import TextChunk, extract_text_from_chunks

logger = logging.getLogger(__name__)


def analyze_text_extraction(chunk_dir: Path) -> Dict[str, Any]:
    """Analyze text extraction results from a chunk directory.
    
    Args:
        chunk_dir: Directory containing PDF chunks
        
    Returns:
        Dictionary with analysis results
    """
    chunks = extract_text_from_chunks(chunk_dir)
    
    if not chunks:
        return {"error": "No chunks extracted"}
    
    # Basic statistics
    total_chunks = len(chunks)
    total_words = sum(chunk.word_count for chunk in chunks)
    total_chars = sum(chunk.char_count for chunk in chunks)
    
    # Quality distribution
    quality_counts = Counter(chunk.quality.value for chunk in chunks)
    
    # Type distribution
    type_counts = Counter(chunk.chunk_type for chunk in chunks)
    
    # Word count distribution
    word_counts = [chunk.word_count for chunk in chunks]
    avg_words = sum(word_counts) / len(word_counts) if word_counts else 0
    
    # Source file distribution
    source_counts = Counter(chunk.source_file for chunk in chunks)
    
    analysis = {
        "total_chunks": total_chunks,
        "total_words": total_words,
        "total_characters": total_chars,
        "average_words_per_chunk": round(avg_words, 1),
        "quality_distribution": dict(quality_counts),
        "type_distribution": dict(type_counts),
        "source_file_count": len(source_counts),
        "chunks_per_file_stats": {
            "min": min(source_counts.values()) if source_counts else 0,
            "max": max(source_counts.values()) if source_counts else 0,
            "avg": round(sum(source_counts.values()) / len(source_counts), 1) if source_counts else 0
        }
    }
    
    return analysis


def save_chunks_to_json(chunks: List[TextChunk], output_path: Path) -> None:
    """Save extracted text chunks to JSON file.
    
    Args:
        chunks: List of TextChunk objects
        output_path: Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert chunks to JSON-serializable format
    chunks_data = []
    for chunk in chunks:
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "source_file": chunk.source_file,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "chunk_type": chunk.chunk_type,
            "quality": chunk.quality.value,
            "word_count": chunk.word_count,
            "char_count": chunk.char_count,
            "metadata": chunk.metadata
        }
        chunks_data.append(chunk_dict)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(chunks),
            "extraction_timestamp": None,  # Could add timestamp
            "chunks": chunks_data
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def print_sample_chunks(chunks: List[TextChunk], num_samples: int = 3) -> None:
    """Print sample chunks for review.
    
    Args:
        chunks: List of TextChunk objects
        num_samples: Number of sample chunks to print
    """
    if not chunks:
        print("No chunks to display")
        return
    
    print(f"\n=== Sample Chunks (showing {min(num_samples, len(chunks))} of {len(chunks)}) ===\n")
    
    for i, chunk in enumerate(chunks[:num_samples]):
        print(f"Chunk {i+1}:")
        print(f"  ID: {chunk.chunk_id}")
        print(f"  Source: {chunk.source_file}")
        print(f"  Pages: {chunk.page_start}-{chunk.page_end}")
        print(f"  Type: {chunk.chunk_type}")
        print(f"  Quality: {chunk.quality.value}")
        print(f"  Words: {chunk.word_count}")
        print(f"  Text preview: {chunk.text[:200]}...")
        print("-" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python text_analysis.py <chunk_directory> [output_json]")
        sys.exit(1)
    
    chunk_dir = Path(sys.argv[1])
    output_json = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Extract chunks
        print("Extracting text chunks...")
        chunks = extract_text_from_chunks(chunk_dir)
        
        # Analyze
        print("Analyzing extraction results...")
        analysis = analyze_text_extraction(chunk_dir)
        
        # Print results
        print(f"\n=== Text Extraction Analysis ===")
        print(f"Total chunks: {analysis['total_chunks']}")
        print(f"Total words: {analysis['total_words']:,}")
        print(f"Average words per chunk: {analysis['average_words_per_chunk']}")
        print(f"\nQuality distribution:")
        for quality, count in analysis['quality_distribution'].items():
            print(f"  {quality}: {count}")
        print(f"\nChunk type distribution:")
        for chunk_type, count in analysis['type_distribution'].items():
            print(f"  {chunk_type}: {count}")
        
        # Print samples
        print_sample_chunks(chunks, 2)
        
        # Save to JSON if requested
        if output_json:
            save_chunks_to_json(chunks, output_json)
            print(f"\nSaved chunks to: {output_json}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)