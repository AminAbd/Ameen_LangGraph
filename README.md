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

## 5. Reducer

A LangGraph demonstration of the **MessagesState reducer**, a crucial component that automatically manages state updates. The reducer automatically merges/concatenates new messages with existing messages in the state. When a node returns `{"messages": [new_message]}`, the reducer automatically appends it to the existing messages list, maintaining conversation history without manual list concatenation. This notebook shows how the reducer handles automatic state management, ensures conversation history preservation across graph execution, and simplifies node implementations by allowing nodes to just return new messages while the reducer handles merging.

**Usage**: Open `Reducer.ipynb` to understand how the MessagesState reducer automatically manages message state.

## 6. React with Memory

An enhanced version of the ReAct agent that demonstrates **memory checkpointing** using LangGraph's `MemorySaver`. This implementation shows how to persist conversation state across multiple graph invocations using thread IDs. The key feature is the **memory system** that allows the agent to remember previous interactions, tool calls, and results within the same conversation thread. This enables context-aware follow-up questions (e.g., "Multiply that by 2" referring to a previous result) and maintains complete conversation history. Features memory checkpointing with `MemorySaver`, thread-based conversation management, and persistent state across invocations.

**Usage**: Open `React_with_memory.ipynb` to see how memory checkpointing enables multi-turn conversations with context retention.

## 7. State Schema

A comprehensive demonstration of different **state schema approaches** in LangGraph. This notebook compares **dataclasses** and **Pydantic models** as state schemas, highlighting key differences: dataclasses provide type hints but don't enforce validation at runtime (allowing invalid values to be assigned), while Pydantic models provide **runtime validation** that catches type errors and constraint violations when state is created. Demonstrates accessing state attributes differently for dataclasses (`state.name`) versus TypedDict (`state["name"]`), and shows how Pydantic validation prevents invalid state values from being used in the graph.

**Usage**: Open `State_Schema.ipynb` to understand different state schema options and their validation behaviors.

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
