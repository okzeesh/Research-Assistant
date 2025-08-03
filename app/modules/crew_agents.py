import logging
from typing import List, Dict, Any, Optional
from crewai import Agent, Task, Crew, Process
from langchain.llms import Ollama
from langchain.tools import Tool
from langchain.schema import HumanMessage, SystemMessage

from app.config import settings
from app.modules.embedding_indexing import EmbeddingIndexingModule

logger = logging.getLogger(__name__)


class ResearchAssistantCrew:
    """CrewAI-based multi-agent system for research assistance."""
    
    def __init__(self):
        self.llm = Ollama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL
        )
        
        self.indexing_module = EmbeddingIndexingModule()
        
        # Initialize agents
        self.retriever_agent = self._create_retriever_agent()
        self.summarizer_agent = self._create_summarizer_agent()
        self.answer_agent = self._create_answer_agent()
    
    def _create_retriever_agent(self) -> Agent:
        """Create the Retriever Agent for fetching relevant papers and data."""
        
        def search_papers(query: str, limit: int = 10) -> str:
            """Search for relevant papers based on query."""
            try:
                results = self.indexing_module.search_similar(query, top_k=limit)
                if not results:
                    return "No relevant papers found."
                
                formatted_results = []
                for result in results:
                    formatted_results.append(
                        f"Title: {result['title']}\n"
                        f"Authors: {', '.join(result['authors']) if result['authors'] else 'Unknown'}\n"
                        f"Content: {result['content'][:500]}...\n"
                        f"Relevance Score: {result['score']:.3f}\n"
                        f"---"
                    )
                
                return "\n".join(formatted_results)
            except Exception as e:
                logger.error(f"Error in search_papers: {e}")
                return f"Error searching papers: {str(e)}"
        
        def get_paper_content(file_id: str) -> str:
            """Get all content chunks for a specific paper."""
            try:
                chunks = self.indexing_module.get_paper_chunks(file_id)
                if not chunks:
                    return "Paper not found or no content available."
                
                content = []
                for chunk in chunks:
                    content.append(f"Chunk: {chunk['content']}\n---")
                
                return "\n".join(content)
            except Exception as e:
                logger.error(f"Error in get_paper_content: {e}")
                return f"Error retrieving paper content: {str(e)}"
        
        # Create tools
        search_tool = Tool(
            name="search_papers",
            description="Search for relevant academic papers based on a query",
            func=search_papers
        )
        
        get_content_tool = Tool(
            name="get_paper_content",
            description="Get all content chunks for a specific paper by file_id",
            func=get_paper_content
        )
        
        return Agent(
            role="Research Retriever",
            goal="Find and retrieve the most relevant academic papers and content based on user queries",
            backstory="""You are an expert research assistant specialized in finding and retrieving 
            relevant academic papers. You have access to a comprehensive database of research papers 
            and can search through them using semantic similarity. Your job is to identify the most 
            relevant papers and extract the most pertinent content for any given research query.""",
            tools=[search_tool, get_content_tool],
            llm=self.llm,
            verbose=True
        )
    
    def _create_summarizer_agent(self) -> Agent:
        """Create the Summarizer Agent for generating concise summaries."""
        
        def summarize_content(content: str, summary_type: str = "general") -> str:
            """Generate a summary of the provided content."""
            try:
                if summary_type == "detailed":
                    prompt = f"""Please provide a detailed summary of the following academic content. 
                    Include key findings, methodology, results, and conclusions:
                    
                    {content}
                    
                    Detailed Summary:"""
                elif summary_type == "key_points":
                    prompt = f"""Please extract the key points from the following academic content. 
                    Focus on main findings, important data, and critical insights:
                    
                    {content}
                    
                    Key Points:"""
                else:  # general
                    prompt = f"""Please provide a concise summary of the following academic content:
                    
                    {content}
                    
                    Summary:"""
                
                response = self.llm(prompt)
                return response
            except Exception as e:
                logger.error(f"Error in summarize_content: {e}")
                return f"Error generating summary: {str(e)}"
        
        summarize_tool = Tool(
            name="summarize_content",
            description="Generate summaries of academic content with different levels of detail",
            func=summarize_content
        )
        
        return Agent(
            role="Research Summarizer",
            goal="Create clear, accurate, and comprehensive summaries of academic papers and research content",
            backstory="""You are an expert academic writer and researcher with years of experience 
            in summarizing complex research papers. You have a deep understanding of academic writing 
            conventions and can distill complex research into clear, accessible summaries. You focus 
            on maintaining accuracy while making the content understandable to researchers and students.""",
            tools=[summarize_tool],
            llm=self.llm,
            verbose=True
        )
    
    def _create_answer_agent(self) -> Agent:
        """Create the Answer Agent for providing accurate, contextual answers."""
        
        def answer_question(question: str, context: str) -> str:
            """Answer a question based on the provided context."""
            try:
                prompt = f"""Based on the following academic context, please answer the question accurately and thoroughly.
                If the context doesn't contain enough information to answer the question, say so clearly.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:"""
                
                response = self.llm(prompt)
                return response
            except Exception as e:
                logger.error(f"Error in answer_question: {e}")
                return f"Error answering question: {str(e)}"
        
        answer_tool = Tool(
            name="answer_question",
            description="Answer questions based on provided academic context",
            func=answer_question
        )
        
        return Agent(
            role="Research Answer Agent",
            goal="Provide accurate, well-reasoned answers to research questions based on academic content",
            backstory="""You are an expert research consultant with deep knowledge across multiple 
            academic disciplines. You excel at analyzing complex research questions and providing 
            clear, evidence-based answers. You always cite your sources and acknowledge when 
            information is insufficient to provide a complete answer.""",
            tools=[answer_tool],
            llm=self.llm,
            verbose=True
        )
    
    def generate_summary(self, file_id: str, summary_type: str = "general") -> Dict[str, Any]:
        """Generate a summary of a specific paper."""
        try:
            # Task 1: Retrieve paper content
            retrieve_task = Task(
                description=f"Retrieve all content chunks for the paper with file_id: {file_id}",
                agent=self.retriever_agent,
                expected_output="Complete content of the paper organized by chunks"
            )
            
            # Task 2: Generate summary
            summarize_task = Task(
                description=f"Generate a {summary_type} summary of the retrieved paper content",
                agent=self.summarizer_agent,
                expected_output=f"A {summary_type} summary of the academic paper",
                context=[retrieve_task]
            )
            
            # Create and run crew
            crew = Crew(
                agents=[self.retriever_agent, self.summarizer_agent],
                tasks=[retrieve_task, summarize_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "summary": result,
                "summary_type": summary_type,
                "file_id": file_id
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "summary": f"Error generating summary: {str(e)}",
                "summary_type": summary_type,
                "file_id": file_id
            }
    
    def answer_query(self, query: str, file_id: Optional[str] = None) -> Dict[str, Any]:
        """Answer a user query using the multi-agent system."""
        try:
            # Task 1: Retrieve relevant content
            if file_id:
                retrieve_task = Task(
                    description=f"Retrieve content from the specific paper with file_id: {file_id}",
                    agent=self.retriever_agent,
                    expected_output="Content from the specified paper"
                )
            else:
                retrieve_task = Task(
                    description=f"Search for and retrieve relevant papers and content for the query: {query}",
                    agent=self.retriever_agent,
                    expected_output="Relevant academic papers and content for the query"
                )
            
            # Task 2: Answer the question
            answer_task = Task(
                description=f"Answer the following question based on the retrieved content: {query}",
                agent=self.answer_agent,
                expected_output="A comprehensive answer to the user's question",
                context=[retrieve_task]
            )
            
            # Create and run crew
            crew = Crew(
                agents=[self.retriever_agent, self.answer_agent],
                tasks=[retrieve_task, answer_task],
                process=Process.sequential,
                verbose=True
            )
            
            result = crew.kickoff()
            
            return {
                "query": query,
                "answer": result,
                "file_id": file_id
            }
            
        except Exception as e:
            logger.error(f"Error answering query: {e}")
            return {
                "query": query,
                "answer": f"Error answering query: {str(e)}",
                "file_id": file_id
            } 