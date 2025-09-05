#!/usr/bin/env python3
"""
EdBot AI - Interactive Textbook Q&A System
Command Line Interface
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.edbot_pipeline import EdBotPipeline, PipelineConfig, create_edbot_from_pdf, load_edbot_from_index


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def process_textbook(args):
    """Process a new textbook PDF."""
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}")
        return 1
    
    print(f"Processing textbook: {pdf_path}")
    print("This may take a few minutes...")
    
    try:
        # Create configuration
        config = PipelineConfig(
            pages_per_chunk=args.pages_per_chunk,
            embeddings_model=args.embeddings_model,
            qa_model=args.qa_model,
            index_name=args.index_name
        )
        
        # Process textbook
        pipeline = create_edbot_from_pdf(pdf_path, args.textbook_name, config)
        
        # Show results
        status = pipeline.get_system_status()
        stats = status['index_stats']
        
        print(f"\n✅ Processing complete!")
        print(f"📚 Textbook: {args.textbook_name or pdf_path.stem}")
        print(f"📄 Chunks: {stats['total_chunks']}")
        print(f"📝 Words: {stats['total_words']:,}")
        print(f"💾 Index: {args.textbook_name or pdf_path.stem}_{args.index_name}")
        
        # Test with a sample question if provided
        if args.test_question:
            print(f"\n🤔 Testing with question: {args.test_question}")
            response = pipeline.ask_question(args.test_question)
            print(f"\n{response.format_response()}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error processing textbook: {e}")
        return 1


def interactive_qa(args):
    """Start interactive Q&A session."""
    try:
        print(f"Loading textbook index: {args.index_name}")
        
        config = PipelineConfig(
            embeddings_model=args.embeddings_model,
            qa_model=args.qa_model
        )
        
        pipeline = load_edbot_from_index(args.index_name, config)
        session = pipeline.start_interactive_session()
        
        # Show system status
        status = pipeline.get_system_status()
        stats = status['index_stats']
        
        print(f"\n✅ EdBot AI Ready!")
        print(f"📄 Loaded: {stats['total_chunks']} chunks, {stats['total_words']:,} words")
        print(f"🤖 Model: {args.qa_model}")
        print(f"📊 Confidence threshold: {config.min_relevance_score}")
        
        print(f"\n{'='*60}")
        print("💬 Interactive Q&A Session")
        print("Type your questions below. Type 'quit', 'exit', or 'bye' to end.")
        print(f"{'='*60}\n")
        
        while True:
            try:
                question = input("❓ Question: ").strip()
                
                if question.lower() in ['quit', 'exit', 'bye', 'q']:
                    break
                
                if not question:
                    continue
                
                print("🤖 Thinking...")
                response = session.ask(question)
                
                print(f"\n{response.format_response()}")
                print(f"\n{'─'*40}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
        
        # Show conversation summary
        if session.conversation_history:
            print(f"\n📊 Session Summary:")
            print(session.get_conversation_summary())
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting interactive session: {e}")
        return 1


def ask_question(args):
    """Ask a single question."""
    try:
        config = PipelineConfig(
            embeddings_model=args.embeddings_model,
            qa_model=args.qa_model
        )
        
        pipeline = load_edbot_from_index(args.index_name, config)
        response = pipeline.ask_question(args.question)
        
        print(response.format_response())
        return 0
        
    except Exception as e:
        print(f"❌ Error answering question: {e}")
        return 1


def list_indexes(args):
    """List available textbook indexes."""
    vector_dir = Path("./vector_indexes")
    
    if not vector_dir.exists():
        print("No vector indexes directory found.")
        return 1
    
    # Find .faiss files
    faiss_files = list(vector_dir.glob("*.faiss"))
    
    if not faiss_files:
        print("No textbook indexes found.")
        print("Process a textbook first using: python cli.py process <pdf_file>")
        return 1
    
    print("📚 Available Textbook Indexes:")
    print("─" * 40)
    
    for faiss_file in faiss_files:
        index_name = faiss_file.stem
        metadata_file = vector_dir / f"{index_name}_metadata.pkl"
        
        if metadata_file.exists():
            # Try to get basic stats
            try:
                import pickle
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                
                chunk_count = len(metadata)
                total_words = sum(chunk.word_count for chunk in metadata.values())
                
                print(f"📖 {index_name}")
                print(f"   Chunks: {chunk_count}")
                print(f"   Words: {total_words:,}")
                print()
                
            except Exception:
                print(f"📖 {index_name} (metadata unavailable)")
        else:
            print(f"📖 {index_name} (incomplete)")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="EdBot AI - Interactive Textbook Q&A System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a textbook PDF
  python cli.py process textbook.pdf --textbook-name "US History"
  
  # Start interactive Q&A
  python cli.py chat us_history_textbook
  
  # Ask a single question
  python cli.py ask us_history_textbook "What caused the Civil War?"
  
  # List available textbooks
  python cli.py list
        """
    )
    
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process textbook command
    process_parser = subparsers.add_parser("process", help="Process a textbook PDF")
    process_parser.add_argument("pdf_file", help="Path to PDF file")
    process_parser.add_argument("--textbook-name", help="Name for the textbook")
    process_parser.add_argument("--pages-per-chunk", type=int, default=15, help="Pages per PDF chunk")
    process_parser.add_argument("--index-name", default="textbook", help="Name for vector index")
    process_parser.add_argument("--embeddings-model", default="text-embedding-3-large", help="OpenAI embeddings model")
    process_parser.add_argument("--qa-model", default="gpt-4-turbo-preview", help="Q&A model")
    process_parser.add_argument("--test-question", help="Test question after processing")
    
    # Interactive chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive Q&A session")
    chat_parser.add_argument("index_name", help="Textbook index name")
    chat_parser.add_argument("--embeddings-model", default="text-embedding-3-large", help="OpenAI embeddings model")
    chat_parser.add_argument("--qa-model", default="gpt-4-turbo-preview", help="Q&A model")
    
    # Single question command
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("index_name", help="Textbook index name")
    ask_parser.add_argument("question", help="Question to ask")
    ask_parser.add_argument("--embeddings-model", default="text-embedding-3-large", help="OpenAI embeddings model")
    ask_parser.add_argument("--qa-model", default="gpt-4-turbo-preview", help="Q&A model")
    
    # List indexes command
    list_parser = subparsers.add_parser("list", help="List available textbook indexes")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return 1
    
    # Execute command
    if args.command == "process":
        return process_textbook(args)
    elif args.command == "chat":
        return interactive_qa(args)
    elif args.command == "ask":
        return ask_question(args)
    elif args.command == "list":
        return list_indexes(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())