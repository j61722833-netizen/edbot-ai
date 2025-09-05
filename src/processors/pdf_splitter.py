"""PDF Splitter for textbook processing.

This module provides functionality to split large PDF textbooks into smaller,
manageable chunks for processing by the RAG system.
"""

from pathlib import Path
from typing import List, Optional
import logging
from PyPDF2 import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


class PDFSplitter:
    """Split PDF documents into smaller chunks for processing."""
    
    def __init__(self, pages_per_chunk: int = 20):
        """Initialize PDF splitter.
        
        Args:
            pages_per_chunk: Maximum number of pages per output chunk
        """
        self.pages_per_chunk = pages_per_chunk
    
    def split_pdf(
        self, 
        input_path: Path, 
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """Split a PDF file into smaller chunks.
        
        Args:
            input_path: Path to the input PDF file
            output_dir: Directory to save split files (defaults to input file directory)
            
        Returns:
            List of paths to the created chunk files
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If pages_per_chunk is less than 1
        """
        if self.pages_per_chunk < 1:
            raise ValueError("pages_per_chunk must be at least 1")
            
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_chunks"
        else:
            output_dir = Path(output_dir)
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Splitting PDF: {input_path} into chunks of {self.pages_per_chunk} pages")
        
        try:
            reader = PdfReader(str(input_path))
            total_pages = len(reader.pages)
            
            logger.info(f"Total pages in PDF: {total_pages}")
            
            chunk_files = []
            
            for chunk_start in range(0, total_pages, self.pages_per_chunk):
                chunk_end = min(chunk_start + self.pages_per_chunk, total_pages)
                chunk_num = (chunk_start // self.pages_per_chunk) + 1
                
                # Create output filename
                chunk_filename = f"{input_path.stem}_chunk_{chunk_num:03d}_pages_{chunk_start+1}-{chunk_end}.pdf"
                chunk_path = output_dir / chunk_filename
                
                # Create new PDF writer for this chunk
                writer = PdfWriter()
                
                # Add pages to this chunk
                for page_num in range(chunk_start, chunk_end):
                    writer.add_page(reader.pages[page_num])
                
                # Write chunk to file
                with open(chunk_path, 'wb') as output_file:
                    writer.write(output_file)
                
                chunk_files.append(chunk_path)
                logger.info(f"Created chunk {chunk_num}: {chunk_filename} (pages {chunk_start+1}-{chunk_end})")
            
            logger.info(f"Successfully split PDF into {len(chunk_files)} chunks")
            return chunk_files
            
        except Exception as e:
            logger.error(f"Error splitting PDF {input_path}: {e}")
            raise
    
    def get_chunk_info(self, pdf_path: Path) -> dict:
        """Get information about how a PDF would be split.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary with split information
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            num_chunks = (total_pages + self.pages_per_chunk - 1) // self.pages_per_chunk
            
            return {
                "total_pages": total_pages,
                "pages_per_chunk": self.pages_per_chunk,
                "estimated_chunks": num_chunks,
                "last_chunk_pages": total_pages % self.pages_per_chunk or self.pages_per_chunk
            }
            
        except Exception as e:
            logger.error(f"Error analyzing PDF {pdf_path}: {e}")
            raise


def split_textbook_pdf(
    pdf_path: str | Path,
    pages_per_chunk: int = 20,
    output_dir: Optional[str | Path] = None
) -> List[Path]:
    """Convenience function to split a textbook PDF.
    
    Args:
        pdf_path: Path to the PDF file to split
        pages_per_chunk: Number of pages per chunk (default: 20)
        output_dir: Output directory for chunks
        
    Returns:
        List of paths to created chunk files
    """
    splitter = PDFSplitter(pages_per_chunk=pages_per_chunk)
    return splitter.split_pdf(Path(pdf_path), Path(output_dir) if output_dir else None)


if __name__ == "__main__":
    import sys
    
    # Basic CLI usage
    if len(sys.argv) < 2:
        print("Usage: python pdf_splitter.py <pdf_file> [pages_per_chunk] [output_dir]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    pages_per_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        chunk_files = split_textbook_pdf(pdf_file, pages_per_chunk, output_dir)
        print(f"Successfully created {len(chunk_files)} chunks:")
        for chunk_file in chunk_files:
            print(f"  - {chunk_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)