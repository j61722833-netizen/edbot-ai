"""Complete EdBot AI textbook Q&A pipeline."""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Internal imports
from .processors.pdf_splitter import split_textbook_pdf
from .processors.text_extractor import extract_text_from_chunks
from .utils.vector_store import VectorStore, create_vector_index_from_json
from .models.qa_model import TextbookQA, TextbookQASession, QAResponse
from .utils.text_analysis import save_chunks_to_json

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the EdBot pipeline."""
    # PDF processing
    pages_per_chunk: int = 15
    min_text_length: int = 50
    max_chunk_size: int = 1000
    
    # Vector store
    embeddings_model: str = "text-embedding-3-large"
    index_name: str = "textbook"
    
    # Q&A model
    qa_model: str = "gpt-4-turbo-preview"
    temperature: float = 0.1
    max_tokens: int = 1000
    
    # Retrieval
    max_context_chunks: int = 5
    min_relevance_score: float = 0.7
    
    # Paths
    output_dir: str = "./outputs"
    vector_index_dir: str = "./vector_indexes"


class EdBotPipeline:
    """Complete pipeline for processing textbooks and answering questions."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the EdBot pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        
        # Ensure output directories exist
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.vector_index_dir).mkdir(parents=True, exist_ok=True)
        
        self.vector_store: Optional[VectorStore] = None
        self.qa_system: Optional[TextbookQA] = None
        
        logger.info("EdBot pipeline initialized")
    
    def process_textbook(self, pdf_path: Path, textbook_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a complete textbook from PDF to Q&A ready system.
        
        Args:
            pdf_path: Path to the textbook PDF
            textbook_name: Name for the textbook (used in filenames)
            
        Returns:
            Dictionary with processing results
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        textbook_name = textbook_name or pdf_path.stem
        logger.info(f"Processing textbook: {textbook_name}")
        
        results = {
            "textbook_name": textbook_name,
            "source_pdf": str(pdf_path),
            "steps_completed": []
        }
        
        try:
            # Step 1: Split PDF into chunks
            logger.info("Step 1: Splitting PDF into chunks...")
            chunk_files = split_textbook_pdf(
                pdf_path,
                pages_per_chunk=self.config.pages_per_chunk,
                output_dir=pdf_path.parent / f"{textbook_name}_chunks"
            )
            
            results["pdf_chunks"] = len(chunk_files)
            results["chunk_directory"] = str(pdf_path.parent / f"{textbook_name}_chunks")
            results["steps_completed"].append("pdf_splitting")
            logger.info(f"Created {len(chunk_files)} PDF chunks")
            
            # Step 2: Extract text from chunks
            logger.info("Step 2: Extracting text from PDF chunks...")
            text_chunks = extract_text_from_chunks(pdf_path.parent / f"{textbook_name}_chunks")
            
            results["text_chunks"] = len(text_chunks)
            results["total_words"] = sum(chunk.word_count for chunk in text_chunks)
            results["steps_completed"].append("text_extraction")
            logger.info(f"Extracted {len(text_chunks)} text chunks with {results['total_words']} total words")
            
            # Step 3: Save extracted text to JSON
            logger.info("Step 3: Saving extracted text...")
            json_path = Path(self.config.output_dir) / f"{textbook_name}_extracted_text.json"
            save_chunks_to_json(text_chunks, json_path)
            
            results["extracted_text_file"] = str(json_path)
            results["steps_completed"].append("text_export")
            
            # Step 4: Create vector index
            logger.info("Step 4: Creating vector index...")
            self.vector_store = VectorStore(
                index_path=self.config.vector_index_dir,
                embeddings_model=self.config.embeddings_model
            )
            
            index_name = f"{textbook_name}_{self.config.index_name}"
            self.vector_store.create_index_from_chunks(text_chunks, index_name)
            
            results["vector_index"] = index_name
            results["index_stats"] = self.vector_store.get_index_stats()
            results["steps_completed"].append("vector_indexing")
            logger.info(f"Created vector index: {index_name}")
            
            # Step 5: Initialize Q&A system
            logger.info("Step 5: Initializing Q&A system...")
            self.qa_system = TextbookQA(
                vector_store=self.vector_store,
                model_name=self.config.qa_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            results["steps_completed"].append("qa_initialization")
            results["status"] = "complete"
            
            logger.info(f"Textbook processing complete for: {textbook_name}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing textbook: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            raise
    
    def load_existing_textbook(self, index_name: str) -> bool:
        """Load an existing processed textbook.
        
        Args:
            index_name: Name of the vector index to load
            
        Returns:
            True if loaded successfully
        """
        logger.info(f"Loading existing textbook index: {index_name}")
        
        try:
            # Initialize vector store
            self.vector_store = VectorStore(
                index_path=self.config.vector_index_dir,
                embeddings_model=self.config.embeddings_model
            )
            
            # Load index
            if not self.vector_store.load_index(index_name):
                logger.error(f"Failed to load index: {index_name}")
                return False
            
            # Initialize Q&A system
            self.qa_system = TextbookQA(
                vector_store=self.vector_store,
                model_name=self.config.qa_model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            logger.info(f"Successfully loaded textbook: {index_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading textbook: {e}")
            return False
    
    def ask_question(self, question: str) -> QAResponse:
        """Ask a question about the loaded textbook.
        
        Args:
            question: Question to ask
            
        Returns:
            QAResponse object
        """
        if not self.qa_system:
            raise ValueError("No textbook loaded. Process or load a textbook first.")
        
        return self.qa_system.answer_question(
            question,
            max_context_chunks=self.config.max_context_chunks,
            min_relevance_score=self.config.min_relevance_score
        )
    
    def start_interactive_session(self) -> TextbookQASession:
        """Start an interactive Q&A session.
        
        Returns:
            TextbookQASession object
        """
        if not self.qa_system:
            raise ValueError("No textbook loaded. Process or load a textbook first.")
        
        return TextbookQASession(self.qa_system)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics.
        
        Returns:
            System status dictionary
        """
        status = {
            "pipeline_initialized": True,
            "vector_store_loaded": self.vector_store is not None,
            "qa_system_ready": self.qa_system is not None,
            "config": {
                "embeddings_model": self.config.embeddings_model,
                "qa_model": self.config.qa_model,
                "max_context_chunks": self.config.max_context_chunks,
                "min_relevance_score": self.config.min_relevance_score
            }
        }
        
        if self.vector_store:
            status["index_stats"] = self.vector_store.get_index_stats()
        
        return status


def create_edbot_from_pdf(pdf_path: Path,
                         textbook_name: Optional[str] = None,
                         config: Optional[PipelineConfig] = None) -> EdBotPipeline:
    """Convenience function to create EdBot from a PDF file.
    
    Args:
        pdf_path: Path to textbook PDF
        textbook_name: Optional name for the textbook
        config: Optional pipeline configuration
        
    Returns:
        Configured EdBotPipeline instance
    """
    pipeline = EdBotPipeline(config)
    pipeline.process_textbook(pdf_path, textbook_name)
    return pipeline


def load_edbot_from_index(index_name: str,
                         config: Optional[PipelineConfig] = None) -> EdBotPipeline:
    """Convenience function to load EdBot from existing index.
    
    Args:
        index_name: Name of the vector index
        config: Optional pipeline configuration
        
    Returns:
        Configured EdBotPipeline instance
    """
    pipeline = EdBotPipeline(config)
    if not pipeline.load_existing_textbook(index_name):
        raise ValueError(f"Could not load textbook index: {index_name}")
    return pipeline


if __name__ == "__main__":
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Process PDF: python edbot_pipeline.py process <pdf_file> [textbook_name]")
        print("  Load existing: python edbot_pipeline.py load <index_name>")
        print("  Ask question: python edbot_pipeline.py ask <index_name> '<question>'")
        sys.exit(1)
    
    command = sys.argv[1]
    
    try:
        if command == "process":
            if len(sys.argv) < 3:
                print("Usage: python edbot_pipeline.py process <pdf_file> [textbook_name]")
                sys.exit(1)
            
            pdf_file = Path(sys.argv[2])
            textbook_name = sys.argv[3] if len(sys.argv) > 3 else None
            
            print(f"Processing textbook: {pdf_file}")
            pipeline = create_edbot_from_pdf(pdf_file, textbook_name)
            
            status = pipeline.get_system_status()
            print(f"\nProcessing complete!")
            print(f"Index: {status['index_stats']['total_chunks']} chunks, {status['index_stats']['total_words']:,} words")
            
        elif command == "load":
            if len(sys.argv) < 3:
                print("Usage: python edbot_pipeline.py load <index_name>")
                sys.exit(1)
            
            index_name = sys.argv[2]
            
            print(f"Loading textbook index: {index_name}")
            pipeline = load_edbot_from_index(index_name)
            
            status = pipeline.get_system_status()
            print(f"Loaded successfully!")
            print(f"Index: {status['index_stats']['total_chunks']} chunks, {status['index_stats']['total_words']:,} words")
            
        elif command == "ask":
            if len(sys.argv) < 4:
                print("Usage: python edbot_pipeline.py ask <index_name> '<question>'")
                sys.exit(1)
            
            index_name = sys.argv[2]
            question = sys.argv[3]
            
            print(f"Loading textbook: {index_name}")
            pipeline = load_edbot_from_index(index_name)
            
            print(f"Answering: {question}")
            response = pipeline.ask_question(question)
            
            print("\n" + "="*60)
            print(response.format_response())
            
        else:
            print(f"Unknown command: {command}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)