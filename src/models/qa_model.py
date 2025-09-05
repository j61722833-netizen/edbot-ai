"""Citation-aware Q&A model for textbook queries."""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
except ImportError:
    ChatOpenAI = None
    HumanMessage = None
    SystemMessage = None
    PromptTemplate = None
    RunnablePassthrough = None
    StrOutputParser = None

from ..utils.vector_store import VectorStore
from ..processors.text_extractor import TextChunk

logger = logging.getLogger(__name__)


class AnswerConfidence(Enum):
    """Confidence levels for Q&A answers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"


@dataclass
class Citation:
    """Citation information for source material."""
    chunk_id: str
    source_file: str
    page_start: int
    page_end: int
    relevance_score: float
    text_preview: str
    
    def format_citation(self) -> str:
        """Format citation for display."""
        return f"({self.source_file}, pp. {self.page_start}-{self.page_end})"
    
    def format_detailed_citation(self) -> str:
        """Format detailed citation with preview."""
        return f"""
Source: {self.source_file}
Pages: {self.page_start}-{self.page_end}
Relevance: {self.relevance_score:.3f}
Preview: {self.text_preview[:150]}...
        """.strip()


@dataclass
class QAResponse:
    """Response from the Q&A system."""
    question: str
    answer: str
    citations: List[Citation]
    confidence: AnswerConfidence
    reasoning: str = ""
    
    def format_response(self) -> str:
        """Format complete response with citations."""
        response = f"Question: {self.question}\n\n"
        response += f"Answer: {self.answer}\n\n"
        
        if self.citations:
            response += "Sources:\n"
            for i, citation in enumerate(self.citations, 1):
                response += f"{i}. {citation.format_citation()}\n"
        
        response += f"\nConfidence: {self.confidence.value}"
        if self.reasoning:
            response += f"\nReasoning: {self.reasoning}"
        
        return response


class TextbookQA:
    """Citation-aware Q&A system for textbooks."""
    
    def __init__(self,
                 vector_store: VectorStore,
                 model_name: Optional[str] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 openai_api_key: Optional[str] = None):
        """Initialize Q&A system.
        
        Args:
            vector_store: Configured VectorStore instance
            model_name: OpenAI model name (uses config default if None)
            temperature: LLM temperature (uses config default if None)
            max_tokens: Maximum tokens in response (uses config default if None)
            openai_api_key: OpenAI API key (uses secure config if None)
        """
        if not ChatOpenAI:
            raise ImportError("LangChain OpenAI not installed. Install with: pip install langchain-openai")
        
        # Import secure config
        from ..config.settings import get_config
        config = get_config()
        
        self.vector_store = vector_store
        
        # Initialize OpenAI model with secure configuration
        api_key = openai_api_key or config.openai_api_key
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Run: python setup_keys.py"
            )
        
        self.llm = ChatOpenAI(
            model=model_name or config.qa_model,
            temperature=temperature if temperature is not None else config.temperature,
            max_tokens=max_tokens or config.max_tokens,
            openai_api_key=api_key
        )
        
        # System prompt for citation-aware responses
        self.system_prompt = """You are an expert educational AI assistant that answers questions about textbook content. 

Your primary responsibilities:
1. Provide accurate, well-reasoned answers based on the given context
2. Always cite your sources by referencing the provided context chunks
3. Be explicit about the confidence level of your answer
4. If information is insufficient, clearly state limitations

Guidelines:
- Base your answer strictly on the provided context
- Use direct quotes when appropriate
- Explain your reasoning process
- If context is contradictory or insufficient, acknowledge this
- Maintain an educational, professional tone

Format your response as:
1. A clear, comprehensive answer
2. Specific citations referencing the context sources
3. Confidence assessment (high/medium/low/uncertain)
4. Brief reasoning for your confidence level"""

        self.qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Context from textbook:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above, including appropriate citations."""
        )
    
    def answer_question(self,
                       question: str,
                       max_context_chunks: int = 5,
                       min_relevance_score: float = 0.7) -> QAResponse:
        """Answer a question using textbook content.
        
        Args:
            question: User question
            max_context_chunks: Maximum number of context chunks to use
            min_relevance_score: Minimum relevance score for chunks
            
        Returns:
            QAResponse with answer and citations
        """
        logger.info(f"Processing question: {question[:50]}...")
        
        # Retrieve relevant context
        search_results = self.vector_store.similarity_search(
            question, 
            k=max_context_chunks,
            filter_quality="high"
        )
        
        if not search_results:
            return QAResponse(
                question=question,
                answer="I don't have enough information in the textbook to answer this question.",
                citations=[],
                confidence=AnswerConfidence.UNCERTAIN,
                reasoning="No relevant content found in the textbook."
            )
        
        # Filter by relevance score and prepare context
        relevant_chunks = []
        citations = []
        context_parts = []
        
        for doc, score, chunk in search_results:
            # Convert similarity score to relevance (FAISS uses L2 distance, lower is better)
            relevance = max(0, 1 - (score / 2))  # Normalize to 0-1 range
            
            if relevance < min_relevance_score:
                continue
            
            relevant_chunks.append((doc, relevance, chunk))
            
            # Create citation
            citation = Citation(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                relevance_score=relevance,
                text_preview=chunk.text[:200]
            )
            citations.append(citation)
            
            # Add to context
            context_parts.append(f"[Source: {chunk.source_file}, pp. {chunk.page_start}-{chunk.page_end}]\n{chunk.text}")
        
        if not relevant_chunks:
            return QAResponse(
                question=question,
                answer="I found some related content, but it doesn't seem directly relevant to your question.",
                citations=[],
                confidence=AnswerConfidence.LOW,
                reasoning="Content found but relevance scores were below threshold."
            )
        
        # Generate answer
        context = "\n\n".join(context_parts)
        
        try:
            # Create conversation with system prompt
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=self.qa_prompt.format(context=context, question=question))
            ]
            
            response = self.llm.invoke(messages)
            answer = response.content
            
            # Determine confidence based on number and quality of sources
            confidence = self._assess_confidence(relevant_chunks, question)
            
            reasoning = f"Based on {len(relevant_chunks)} relevant sources with average relevance of {sum(r for _, r, _ in relevant_chunks) / len(relevant_chunks):.2f}"
            
            return QAResponse(
                question=question,
                answer=answer,
                citations=citations,
                confidence=confidence,
                reasoning=reasoning
            )
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return QAResponse(
                question=question,
                answer=f"I encountered an error while processing your question: {str(e)}",
                citations=citations,
                confidence=AnswerConfidence.UNCERTAIN,
                reasoning="Error in language model processing"
            )
    
    def _assess_confidence(self, relevant_chunks: List[Tuple], question: str) -> AnswerConfidence:
        """Assess confidence in the answer based on available context.
        
        Args:
            relevant_chunks: List of (doc, relevance, chunk) tuples
            question: Original question
            
        Returns:
            AnswerConfidence level
        """
        if not relevant_chunks:
            return AnswerConfidence.UNCERTAIN
        
        num_sources = len(relevant_chunks)
        avg_relevance = sum(relevance for _, relevance, _ in relevant_chunks) / num_sources
        total_context_words = sum(chunk.word_count for _, _, chunk in relevant_chunks)
        
        # High confidence: multiple high-relevance sources with substantial content
        if num_sources >= 3 and avg_relevance >= 0.8 and total_context_words >= 300:
            return AnswerConfidence.HIGH
        
        # Medium confidence: decent sources and content
        elif num_sources >= 2 and avg_relevance >= 0.7 and total_context_words >= 150:
            return AnswerConfidence.MEDIUM
        
        # Low confidence: limited but relevant sources
        elif num_sources >= 1 and avg_relevance >= 0.6:
            return AnswerConfidence.LOW
        
        # Uncertain: poor relevance or very limited content
        else:
            return AnswerConfidence.UNCERTAIN
    
    def get_related_topics(self, topic: str, max_topics: int = 5) -> List[Citation]:
        """Find related topics in the textbook.
        
        Args:
            topic: Topic to search for
            max_topics: Maximum number of related topics
            
        Returns:
            List of citations for related content
        """
        search_results = self.vector_store.similarity_search(
            topic, 
            k=max_topics,
            filter_quality="high"
        )
        
        citations = []
        for doc, score, chunk in search_results:
            relevance = max(0, 1 - (score / 2))
            
            citation = Citation(
                chunk_id=chunk.chunk_id,
                source_file=chunk.source_file,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                relevance_score=relevance,
                text_preview=chunk.text[:200]
            )
            citations.append(citation)
        
        return citations


class TextbookQASession:
    """Session manager for interactive Q&A."""
    
    def __init__(self, qa_system: TextbookQA):
        """Initialize Q&A session.
        
        Args:
            qa_system: Configured TextbookQA instance
        """
        self.qa_system = qa_system
        self.conversation_history: List[QAResponse] = []
    
    def ask(self, question: str) -> QAResponse:
        """Ask a question and track conversation history.
        
        Args:
            question: User question
            
        Returns:
            QAResponse object
        """
        response = self.qa_system.answer_question(question)
        self.conversation_history.append(response)
        return response
    
    def get_conversation_summary(self) -> str:
        """Get summary of conversation history.
        
        Returns:
            Formatted conversation summary
        """
        if not self.conversation_history:
            return "No questions asked yet."
        
        summary = f"Conversation Summary ({len(self.conversation_history)} questions):\n\n"
        
        for i, response in enumerate(self.conversation_history, 1):
            summary += f"{i}. Q: {response.question[:60]}...\n"
            summary += f"   A: {response.answer[:100]}...\n"
            summary += f"   Confidence: {response.confidence.value}\n\n"
        
        return summary


if __name__ == "__main__":
    import sys
    
    # Basic CLI for testing
    if len(sys.argv) < 2:
        print("Usage: python qa_model.py <question> [index_name]")
        sys.exit(1)
    
    question = sys.argv[1]
    index_name = sys.argv[2] if len(sys.argv) > 2 else "textbook"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Load vector store
        print("Loading vector index...")
        vector_store = VectorStore()
        if not vector_store.load_index(index_name):
            print(f"Could not load index '{index_name}'. Please create it first.")
            sys.exit(1)
        
        # Initialize Q&A system
        qa_system = TextbookQA(vector_store)
        
        # Answer question
        print(f"Answering: {question}")
        response = qa_system.answer_question(question)
        
        # Print formatted response
        print("\n" + "="*60)
        print(response.format_response())
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)