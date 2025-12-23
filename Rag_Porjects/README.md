# RAG Projects

This directory contains Retrieval-Augmented Generation (RAG) projects implemented using LangGraph and LangChain.

## Projects

### Corrective RAG (CRAG)

**File:** `Corrective RAG (CRAG).ipynb`

Corrective-RAG (CRAG) is an advanced RAG strategy that incorporates self-reflection and self-grading on retrieved documents. This implementation uses LangGraph to create a workflow that:

1. **Retrieves** documents from a vector database
2. **Grades** the relevance of retrieved documents
3. **Transforms** the query if documents are irrelevant
4. **Searches** the web (using Tavily) when needed
5. **Generates** the final answer using the most relevant context

#### Features

- **Document Relevance Grading**: Uses an LLM to assess whether retrieved documents are relevant to the question
- **Query Transformation**: Automatically rewrites queries for better web search results when needed
- **Web Search Integration**: Uses Tavily Search API to supplement retrieval when documents are not relevant
- **LangGraph Workflow**: Implements a state-based graph that orchestrates the entire RAG pipeline

#### Workflow

```
START → Retrieve → Grade Documents → Decision
                                    ↓
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
            Transform Query                    Generate
                    ↓
            Web Search
                    ↓
                Generate → END
```

#### Requirements

- OpenAI API key (set in `.env` file)
- Tavily API key (for web search functionality)
- Python packages (see main `requirements.txt`)

#### Key Components

1. **Retrieval Grader**: Evaluates document relevance using structured LLM output
2. **RAG Chain**: Standard RAG pipeline with retrieval, formatting, and generation
3. **Question Rewriter**: Optimizes queries for web search
4. **Graph State**: Manages state across nodes (question, documents, generation, web_search flag)

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   TAVILY_API_KEY=your_key_here
   ```

2. Run the notebook cells in order:
   - Setup and index creation
   - LLM configuration
   - Graph definition
   - Graph compilation
   - Execution

3. Test with different questions:
   ```python
   inputs = {"question": "Your question here"}
   for output in app.stream(inputs):
       # Process output
   ```

#### Example Questions

- "What are the types of agent memory?"
- "How does the AlphaCodium paper work?"
- Any question that may require web search when local documents are insufficient

## Setup

Make sure you have all required packages installed:

```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `langgraph`
- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`
- `tiktoken`
- `langchainhub`

## Notes

- The notebook uses ChromaDB for vector storage
- Web search is triggered automatically when retrieved documents are not relevant
- The implementation skips the knowledge refinement phase mentioned in the CRAG paper (can be added as an additional node if needed)

## References

- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)

