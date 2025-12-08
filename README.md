# Ameen_LangGraph

A collection of LangGraph and LangChain projects demonstrating various workflows and chat interactions.

## 1. SimpleGraph

A basic LangGraph workflow demonstrating conditional routing between nodes. Features state management, multiple processing nodes, and random conditional routing between paths (happy/sad outcomes).

**Usage**: Open `SimpleGraph.ipynb` in Jupyter Notebook and run the cells to see the graph in action.

## 2. OpenAIChat

A simple LangChain chat implementation using OpenAI's GPT-4o model. Demonstrates creating conversation messages, loading API keys securely from `.env` file, and interacting with OpenAI's chat API.

**Usage**: Open `OpenAIChat.ipynb` to see a basic chat chain implementation with OpenAI.

## Setup

1. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---
