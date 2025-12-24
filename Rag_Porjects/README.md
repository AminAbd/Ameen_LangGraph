# RAG Projects

This directory contains Retrieval-Augmented Generation (RAG) projects implemented using LangChain and LangGraph.

## Projects

### RAG with Multi-Query Generator

**File:** `Rag_and_multi_query_generator..ipynb`

An advanced RAG implementation that uses **multi-query generation** to improve retrieval quality by generating multiple query variations from a single user question.

#### Overview

Traditional RAG systems retrieve documents based on a single query, which can miss relevant information due to the limitations of distance-based similarity search. This implementation addresses this by:

1. **Generating multiple query variations** from the original question
2. **Retrieving documents** for each query variation
3. **Combining unique documents** from all retrievals
4. **Generating the final answer** using the comprehensive context

#### How It Works

The multi-query generator creates 5 different perspectives of the user's question, helping overcome limitations of similarity search by:
- Using different phrasings and terminology
- Exploring various angles of the same question
- Capturing semantic variations that might be missed by a single query

#### Workflow

```
Original Question
    ↓
Generate 5 Query Variations
    ↓
Retrieve Documents for Each Query
    ↓
Get Unique Union of All Documents
    ↓
Generate Final Answer with Combined Context
```

#### Key Components

1. **Multi-Query Generator**: Uses an LLM to generate 5 alternative versions of the user question
2. **Parallel Retrieval**: Retrieves documents for each generated query simultaneously
3. **Document Deduplication**: Combines and deduplicates documents from all retrievals
4. **RAG Chain**: Generates the final answer using the comprehensive document set

#### Code Structure

**Indexing:**
- Loads documents from web sources
- Splits documents into chunks (300 chars, 50 overlap)
- Creates embeddings and stores in ChromaDB

**Multi-Query Generation:**
```python
generate_queries = (
    prompt_perspectives 
    | ChatOpenAI(temperature=0) 
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Retrieval Chain:**
```python
retrieval_chain = generate_queries | retriever.map() | get_unique_union
```

**Final RAG Chain:**
- Combines retrieval with question
- Generates answer using all retrieved context

#### Benefits

- **Better Coverage**: Retrieves more relevant documents by exploring multiple query angles
- **Improved Accuracy**: Reduces the chance of missing important information
- **Semantic Diversity**: Captures different ways the same concept might be expressed
- **Robust Retrieval**: Less dependent on the exact wording of the original question

#### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Multi-query generator setup
   - Retrieval chain creation
   - Final RAG chain execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   answer = final_rag_chain.invoke({"question": question})
   ```

#### Example

**Original Question:** "What is task decomposition for LLM agents?"

**Generated Queries:**
- "How do LLM agents break down tasks?"
- "What methods are used for task decomposition in language model agents?"
- "Explain task decomposition strategies for AI agents"
- "What is the process of decomposing tasks in LLM systems?"
- "How are complex tasks divided in language model agent architectures?"

Each query retrieves potentially different documents, which are then combined for a more comprehensive answer.

#### Requirements

- OpenAI API key
- Python packages (see main `requirements.txt`)

#### Key Dependencies

- `langchain`
- `langchain-openai`
- `langchain-community`
- `langchain-text-splitters`
- `chromadb`

---
## RAG with Multi-Query Fusion (RAG-Fusion + RRF)

**File:** `RAG-Fusion.ipynb`

This project extends multi-query RAG by improving **document selection and ordering** using **Reciprocal Rank Fusion (RRF)**. Instead of blindly concatenating all retrieved documents, it ranks and filters them before generation, producing a cleaner and more informative context for the LLM.

---

### Motivation

Standard RAG systems rely on a **single query → single retrieval** step, which is fragile due to the limitations of distance-based similarity search. Multi-query RAG improves recall by generating multiple query variations, but a naive implementation often concatenates all retrieved documents, which can:

- introduce irrelevant or redundant chunks  
- exceed context limits  
- dilute important information  
- reduce answer quality  

This implementation addresses these issues by **ranking retrieved documents across queries before concatenation**.

---

### Core Idea

Instead of treating all retrieved documents equally, this approach:

1. Generates **4 semantically related queries** from the original question  
2. Retrieves **ranked document lists** for each query  
3. **Fuses the rankings** using Reciprocal Rank Fusion (RRF)  
4. Selects and orders only the **most consistently relevant documents**  
5. Uses the ordered documents as context for generation  

The result is a **smaller, higher-quality, and order-aware context**.

---

### What Is Reciprocal Rank Fusion (RRF)?

Reciprocal Rank Fusion combines multiple ranked lists into a single global ranking using **rank positions only**, not raw similarity scores.

For each document:

score = Σ 1 / (rank + k)


Where:
- `rank` is the document’s position in a retrieval list  
- `k` is a smoothing constant (typically 60)  

Documents that:
- appear across **multiple query results**, and/or  
- appear **highly ranked** in at least one query  

receive higher fused scores.

**Important:** RRF scores are used **only to rank documents**, not to weight or truncate document content. Full document chunks are passed to the LLM.

---

### Why Ranking Matters

LLMs do not treat all context equally:
- earlier context has more influence than later context  
- irrelevant chunks reduce answer quality  
- long, unordered context dilutes attention  

By ranking documents before concatenation, this approach:
- prioritizes the most informative chunks  
- removes low-value noise  
- improves factual grounding  
- produces more consistent and accurate answers  

---

### Workflow

Original Question
↓
Generate Multiple Query Variations
↓
Parallel Vector Retrieval (per query)
↓
Rank Fusion with RRF
↓
Select Top-K Documents
↓
Concatenate Ordered Context
↓
Generate Final Answer


---

### Code Structure

**Indexing:**
- Loads documents from web sources
- Splits documents into chunks (300 chars, 50 overlap)
- Creates embeddings and stores in ChromaDB

**Query Generation:**
```python
generate_queries = (
    prompt_rag_fusion 
    | ChatOpenAI(temperature=0)
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
)
```

**Reciprocal Rank Fusion:**
```python
def reciprocal_rank_fusion(results: list[list], k=60):
    # Fuses multiple ranked lists using RRF formula
    # Returns reranked documents with fused scores
```

**Retrieval Chain:**
```python
retrieval_chain_rag_fusion = generate_queries | retriever.map() | reciprocal_rank_fusion
```

**Final RAG Chain:**
- Combines fused retrieval results with question
- Generates answer using ranked context

### Key Differences vs Normal RAG

| Aspect | Normal RAG | RAG-Fusion + RRF |
|------|-----------|------------------|
| Queries | Single | Multiple (4 queries) |
| Retrieval | One ranked list | Multiple ranked lists |
| Document Selection | Top-K from one query | Consensus across queries |
| Ranking | Vector similarity only | Rank-based fusion (RRF) |
| Context Quality | Noisy, order-dependent | Filtered, ordered |
| Robustness | Low | High |

---

### Usage

1. Set up your `.env` file with API keys:
   ```
   OPENAI_API_KEY=your_key_here
   LANGCHAIN_API_KEY=your_key_here (optional, for tracing)
   ```

2. Run the notebook cells in order:
   - Environment setup
   - Document indexing
   - Query generation setup
   - RRF implementation
   - Retrieval chain creation
   - Final RAG chain execution

3. Ask questions:
   ```python
   question = "What is task decomposition for LLM agents?"
   answer = final_rag_chain.invoke({"question": question})
   ```

**Note:** The notebook uses `itemgetter` from `operator` module - make sure to import it:
```python
from operator import itemgetter
```

### Summary

RAG-Fusion with Reciprocal Rank Fusion improves Retrieval-Augmented Generation by replacing single-query retrieval with **multi-query consensus ranking**, ensuring that only the most consistently relevant documents are used—and used in the right order—before generation.


---
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

- The notebooks use ChromaDB for vector storage
- Web search (in CRAG) is triggered automatically when retrieved documents are not relevant
- Multi-query generator creates 5 query variations by default (can be modified)
- The implementation skips the knowledge refinement phase mentioned in the CRAG paper (can be added as an additional node if needed)

## References

- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)




