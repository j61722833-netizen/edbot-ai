"""Text extraction and preprocessing for PDF textbook chunks.

This module provides functionality to extract clean, structured text from PDF chunks
created by the PDF splitter, preparing it for vector embedding and RAG processing.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import re
from dataclasses import dataclass
from enum import Enum

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    nltk = None

logger = logging.getLogger(__name__)


class TextQuality(Enum):
    """Enum for text quality assessment."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POOR = "poor"


@dataclass
class TextChunk:
    """Represents an extracted text chunk with metadata."""
    text: str
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    chunk_type: str = "paragraph"  # paragraph, heading, list, table, etc.
    quality: TextQuality = TextQuality.MEDIUM
    word_count: int = 0
    char_count: int = 0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        self.word_count = len(self.text.split())
        self.char_count = len(self.text)


class TextExtractor:
    """Extract and preprocess text from PDF chunks for RAG processing."""
    
    def __init__(self, 
                 min_text_length: int = 50,
                 max_chunk_size: int = 1000,
                 remove_headers_footers: bool = True,
                 clean_whitespace: bool = True):
        """Initialize text extractor.
        
        Args:
            min_text_length: Minimum text length to consider valid
            max_chunk_size: Maximum characters per text chunk
            remove_headers_footers: Whether to remove headers/footers
            clean_whitespace: Whether to normalize whitespace
        """
        self.min_text_length = min_text_length
        self.max_chunk_size = max_chunk_size
        self.remove_headers_footers = remove_headers_footers
        self.clean_whitespace = clean_whitespace
        
        # Initialize NLTK if available
        if nltk:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                logger.warning("NLTK punkt tokenizer not found. Text chunking may be less accurate.")
        
        if not fitz:
            logger.error("PyMuPDF not installed. Please install with: pip install pymupdf")
            raise ImportError("PyMuPDF is required for text extraction")
    
    def extract_from_pdf(self, pdf_path: Path) -> List[TextChunk]:
        """Extract text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of TextChunk objects
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If text extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting text from PDF: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            chunks = []
            
            # Extract page range from filename if available
            page_start, page_end = self._parse_page_range_from_filename(pdf_path.name)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Extract text with layout information
                text_dict = page.get_text("dict")
                page_text = self._extract_structured_text(text_dict, page_num + page_start)
                
                if page_text.strip() and len(page_text) >= self.min_text_length:
                    # Process text for this page
                    processed_chunks = self._process_page_text(
                        page_text, 
                        pdf_path.name, 
                        page_num + page_start,
                        page_num + page_start
                    )
                    chunks.extend(processed_chunks)
            
            doc.close()
            logger.info(f"Extracted {len(chunks)} text chunks from {pdf_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def extract_from_pdf_batch(self, pdf_paths: List[Path]) -> List[TextChunk]:
        """Extract text from multiple PDF files.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            Combined list of TextChunk objects from all PDFs
        """
        all_chunks = []
        
        for pdf_path in pdf_paths:
            try:
                chunks = self.extract_from_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_path}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_chunks)} total chunks from {len(pdf_paths)} PDFs")
        return all_chunks
    
    def _parse_page_range_from_filename(self, filename: str) -> Tuple[int, int]:
        """Parse page range from PDF chunk filename.
        
        Args:
            filename: PDF chunk filename like "textbook_chunk_001_pages_1-15.pdf"
            
        Returns:
            Tuple of (start_page, end_page)
        """
        match = re.search(r'pages_(\d+)-(\d+)', filename)
        if match:
            return int(match.group(1)), int(match.group(2))
        return 1, 1
    
    def _extract_structured_text(self, text_dict: Dict, page_num: int) -> str:
        """Extract structured text from PyMuPDF text dictionary.
        
        Args:
            text_dict: PyMuPDF text dictionary
            page_num: Page number for context
            
        Returns:
            Structured text string
        """
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" in block:  # Text block
                block_text = []
                for line in block["lines"]:
                    line_text = []
                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if span_text:
                            line_text.append(span_text)
                    
                    if line_text:
                        block_text.append(" ".join(line_text))
                
                if block_text:
                    text_parts.append("\n".join(block_text))
        
        return "\n\n".join(text_parts)
    
    def _process_page_text(self, text: str, source_file: str, 
                          page_start: int, page_end: int) -> List[TextChunk]:
        """Process raw page text into structured chunks.
        
        Args:
            text: Raw text from page
            source_file: Source PDF filename
            page_start: Starting page number
            page_end: Ending page number
            
        Returns:
            List of processed TextChunk objects
        """
        if self.clean_whitespace:
            text = self._clean_whitespace(text)
        
        if self.remove_headers_footers:
            text = self._remove_headers_footers(text)
        
        # Split into semantic chunks
        chunks = self._split_into_semantic_chunks(text)
        
        processed_chunks = []
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text) >= self.min_text_length:
                chunk_id = f"{source_file}_page_{page_start}_{page_end}_chunk_{i:03d}"
                
                chunk = TextChunk(
                    text=chunk_text,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    page_start=page_start,
                    page_end=page_end,
                    chunk_type=self._identify_chunk_type(chunk_text),
                    quality=self._assess_text_quality(chunk_text)
                )
                processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace in text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove trailing/leading whitespace
        text = text.strip()
        
        return text
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header and footer patterns.
        
        Args:
            text: Text to clean
            
        Returns:
            Text with headers/footers removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip common header/footer patterns
            if (re.match(r'^\d+$', line) or  # Page numbers
                re.match(r'^Chapter \d+', line) or  # Chapter headers
                re.match(r'^\d+\s*$', line) or  # Just numbers
                len(line) < 10):  # Very short lines
                continue
            
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _split_into_semantic_chunks(self, text: str) -> List[str]:
        """Split text into semantic chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, save current chunk
            if (len(current_chunk) + len(paragraph) > self.max_chunk_size and 
                current_chunk):
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _identify_chunk_type(self, text: str) -> str:
        """Identify the type of text chunk.
        
        Args:
            text: Text chunk
            
        Returns:
            Chunk type identifier
        """
        # Simple heuristics for chunk type identification
        if re.match(r'^[A-Z\s]+$', text[:50]):  # All caps likely heading
            return "heading"
        elif text.count('\n') < 2:  # Short text likely title or heading
            return "heading"
        elif re.search(r'^\d+\.|\d+\)|•', text):  # List indicators
            return "list"
        else:
            return "paragraph"
    
    def _assess_text_quality(self, text: str) -> TextQuality:
        """Assess the quality of extracted text.
        
        Args:
            text: Text to assess
            
        Returns:
            TextQuality enum value
        """
        # Simple quality heuristics
        char_count = len(text)
        word_count = len(text.split())
        
        if word_count == 0:
            return TextQuality.POOR
        
        avg_word_length = char_count / word_count
        
        # Check for garbled text indicators
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' \n.,!?-') / len(text)
        
        if special_char_ratio > 0.1:  # Too many special characters
            return TextQuality.POOR
        elif avg_word_length < 3:  # Very short words may indicate OCR issues
            return TextQuality.LOW
        elif char_count < 100:  # Very short text
            return TextQuality.MEDIUM
        else:
            return TextQuality.HIGH


def extract_text_from_chunks(chunk_dir: Path, 
                           output_format: str = "json") -> List[TextChunk]:
    """Convenience function to extract text from all PDF chunks in a directory.
    
    Args:
        chunk_dir: Directory containing PDF chunks
        output_format: Output format (currently unused, for future JSON export)
        
    Returns:
        List of all extracted TextChunk objects
    """
    chunk_dir = Path(chunk_dir)
    if not chunk_dir.exists():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")
    
    # Find all PDF files in directory
    pdf_files = list(chunk_dir.glob("*.pdf"))
    if not pdf_files:
        raise ValueError(f"No PDF files found in {chunk_dir}")
    
    # Sort by filename to maintain order
    pdf_files.sort()
    
    extractor = TextExtractor()
    return extractor.extract_from_pdf_batch(pdf_files)


if __name__ == "__main__":
    import sys
    
    # Basic CLI usage
    if len(sys.argv) < 2:
        print("Usage: python text_extractor.py <chunk_directory>")
        sys.exit(1)
    
    chunk_directory = sys.argv[1]
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        chunks = extract_text_from_chunks(Path(chunk_directory))
        print(f"Successfully extracted {len(chunks)} text chunks")
        
        # Print sample output
        if chunks:
            print(f"\nSample chunk:")
            print(f"ID: {chunks[0].chunk_id}")
            print(f"Type: {chunks[0].chunk_type}")
            print(f"Quality: {chunks[0].quality.value}")
            print(f"Text preview: {chunks[0].text[:200]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)