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

## 8. State Reducer

An in-depth exploration of **reducers** in LangGraph, which specify how state updates are performed on specific keys/channels in the state schema. Demonstrates the default overwriting behavior, the problem that arises when multiple nodes try to update the same state key concurrently (causing `InvalidUpdateError`), and how reducers solve this using the `Annotated` type. Shows how to use built-in reducers like `operator.add` for list concatenation, create custom reducers to handle edge cases (like None values), and demonstrates the special `add_messages` reducer used by `MessagesState`. Also covers advanced message operations: rewriting messages by ID and removing messages using `RemoveMessage`.

**Usage**: Open `State_Reducer.ipynb` to understand how reducers manage state updates and handle concurrent modifications.

## 9. Multiple Schemas

A demonstration of **multiple state schemas** in LangGraph, showing advanced state management patterns. Covers **private state** - allowing nodes to communicate with intermediate state that doesn't appear in the graph's input or output (useful for internal working logic). Also demonstrates **input/output schema filtering** - defining explicit input and output schemas separate from the internal state schema. This allows you to control what keys are permitted on input, filter the output to only relevant keys, while still maintaining a full internal state for node operations. Shows how type hints can be used to specify which schema each node works with when using multiple schemas.

**Usage**: Open `Multiple_Schemas.ipynb` to understand how to use private state and input/output schema filtering in complex graphs.

## 10. Trim and Filtering Messages

A comprehensive guide to managing long-running conversations in LangGraph by controlling message history. Demonstrates techniques to reduce token usage and latency when conversations grow long: **filtering messages** using `RemoveMessage` with the `add_messages` reducer to delete old messages from state, **filtering by passing subsets** to the LLM without modifying graph state (e.g., `messages[-1:]` to use only the last message), and **trimming messages** using `trim_messages` to restrict message history based on a token limit. Shows the difference between filtering (post-hoc subset selection) and trimming (token-based restriction), and demonstrates the `allow_partial` parameter's effect on message trimming behavior.

**Usage**: Open `Trim_and_Filtering_Messages.ipynb` to learn techniques for managing conversation length and reducing token costs.

## 11. Chatbot with Message Summarization

An advanced chatbot implementation that demonstrates **message summarization** as an alternative to trimming or filtering conversations. Instead of deleting old messages, this approach uses LLMs to produce a **running summary** of the conversation, allowing the chatbot to retain a compressed representation of the full conversation history while managing token costs and latency. Features a custom state schema extending `MessagesState` with a `summary` field, a `call_model` node that incorporates existing summaries into prompts, a `summarize_conversation` node that creates/extends summaries and uses `RemoveMessage` to filter old messages after summarization, conditional routing that triggers summarization when conversation length exceeds a threshold (e.g., 6 messages), and **memory checkpointing** with `MemorySaver` for thread-based conversation persistence. This enables long-running conversations without incurring high token costs, as the summary provides context for the entire conversation history while keeping the actual message list manageable.

**Usage**: Open `Chatbot_with_message_summarization.ipynb` to see how summarization enables efficient long-term memory in chatbots.

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
