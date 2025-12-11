# Ameen_LangGraph

A collection of LangGraph and LangChain projects demonstrating various workflows and chat interactions.

## 1. SimpleGraph

A basic LangGraph workflow demonstrating conditional routing between nodes. Features state management, multiple processing nodes, and random conditional routing between paths (happy/sad outcomes).

**Usage**: Open `SimpleGraph.ipynb` in Jupyter Notebook and run the cells to see the graph in action.

## 2. OpenAIChat

A simple LangChain chat implementation using OpenAI's GPT-4o model. Demonstrates creating conversation messages, loading API keys securely from `.env` file, and interacting with OpenAI's chat API.

**Usage**: Open `OpenAIChat.ipynb` to see a basic chat chain implementation with OpenAI.

## 3. Router

A LangGraph router implementation that demonstrates tool calling with conditional routing. The graph intelligently routes between tool execution and direct responses based on whether the LLM decides to use tools or provide a regular response. Features tool binding, MessagesState management, and automatic routing using `tools_condition`.

**Usage**: Open `Router.ipynb` to see tool calling with conditional routing in action.

## 4. React

A LangGraph implementation of the ReAct (Reasoning + Acting) pattern. Demonstrates an agent that can reason about complex tasks and use multiple tools in sequence. The graph creates a loop between the assistant and tools nodes, allowing the LLM to chain multiple tool calls together to solve complex problems. Features multiple arithmetic tools (add, multiply, divide), tool binding, and automatic looping between reasoning and action steps.

**Usage**: Open `React.ipynb` to see the ReAct agent in action with sequential tool calling.

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
