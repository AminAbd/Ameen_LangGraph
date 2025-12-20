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

## 12. Chatbot with Message Summarization & External DB Memory

An advanced chatbot implementation that combines **message summarization** with **external database checkpointing** using SQLite. This notebook extends the summarization chatbot by using `SqliteSaver` instead of `MemorySaver`, enabling **persistent memory that survives across notebook kernel restarts and application shutdowns**. Demonstrates how to set up SQLite checkpointing with both in-memory databases (`:memory:`) and persistent file-based databases, showing how conversation state is stored externally in a SQLite database file. Features message summarization to manage token costs, conditional routing based on conversation length, thread-based conversation management using thread IDs, and **persistent state persistence** across sessions. Unlike `MemorySaver` which only persists state during runtime, `SqliteSaver` ensures conversations are permanently stored on disk, making it ideal for production deployments where conversation history must survive server restarts or application redeployments.

**Usage**: Open `Chatbot_with_message_summarization_&_external_DB_memory.ipynb` to see how external database checkpointing enables persistent, long-term conversation memory.

## 13. Streaming

A comprehensive guide to **streaming** in LangGraph, which provides multiple ways to visualize graph output during execution. Demonstrates streaming graph state updates in real-time using `.stream()` and `.astream()` methods. Covers two main streaming modes: **`stream_mode="updates"`** which streams only the state updates after each node execution (showing incremental changes), and **`stream_mode="values"`** which streams the complete state after each node (showing the full accumulated state). Also demonstrates **token-wise streaming** using `.astream_events()` to stream individual tokens from chat models as they are generated, enabling real-time display of LLM responses. Shows how to filter events by node name, event type (e.g., `on_chat_model_stream`), and extract token chunks from `AIMessageChunk` objects. This is the foundation for human-in-the-loop interactions, as streaming enables live feedback and visualization of graph execution progress.

**Usage**: Open `Streaming.ipynb` to learn how to stream graph state and model tokens in real-time.

## 14. Breakpoints

An introduction to **breakpoints** in LangGraph, which enable human-in-the-loop workflows by pausing graph execution at specific nodes. Demonstrates the **`interrupt_before`** parameter when compiling graphs, which allows you to interrupt execution before a specific node (e.g., `interrupt_before=["tools"]` to pause before tool execution). Shows how to inspect the current state using `graph.get_state()` to view the next node to be executed and examine the state at the interruption point. Demonstrates resuming execution by invoking the graph with `None`, which continues from the last checkpoint and re-emits the current state before proceeding. This enables **human approval workflows** where users can review tool calls, modify state, or approve/deny operations before the graph continues execution. Supports use cases like approval (interrupting to get user consent), debugging (inspecting state at specific points), and editing (modifying state before resuming).

**Usage**: Open `Breakpoints.ipynb` to learn how to implement human-in-the-loop approval workflows using breakpoints.

## 15. Editing State and Human Feedback

A demonstration of **editing graph state** and **inserting human feedback** during breakpoint interruptions. Builds on breakpoints to show how to directly modify graph state after an interruption using `graph.update_state()`. Demonstrates how state updates respect reducers - when updating the `messages` key, the `add_messages` reducer automatically appends new messages to the existing message list, allowing you to insert human feedback or corrections into the conversation flow. Shows how to modify state before resuming execution (e.g., correcting user input, adding clarification messages, or changing agent instructions), and how the graph continues execution with the modified state. Also demonstrates creating a dedicated **`human_feedback` node** as a placeholder within the graph structure that can be interrupted before execution, enabling human feedback to be integrated directly into the graph workflow. This pattern allows feedback to be collected at specific points in the execution cycle (e.g., after tool execution, before assistant response). This enables powerful **human-in-the-loop editing workflows** where users can not only approve actions but also actively modify the conversation state, correct errors, provide additional context, or redirect the agent's behavior before it continues execution. Essential for interactive debugging, user correction scenarios, and adaptive agent workflows.

**Usage**: Open `Editing_State_and_Human_Feedback.ipynb` to learn how to edit graph state and insert human feedback during breakpoint interruptions.

## 16. Dynamic Breakpoints

An introduction to **dynamic breakpoints** using `NodeInterrupt`, which allows graphs to interrupt themselves conditionally based on runtime logic rather than statically defined breakpoints. Unlike static breakpoints set during compilation with `interrupt_before`, dynamic breakpoints are raised from within nodes based on developer-defined conditions (e.g., input validation, error detection, or business logic checks). Demonstrates raising `NodeInterrupt` with custom messages that explain why the interruption occurred, how interrupts are logged in state with the `__interrupt__` key, and how inspecting state shows the interrupt information in the tasks. Shows that resuming execution will re-run the interrupted node unless state is modified, creating a natural flow where state updates resolve the condition that caused the interrupt (e.g., updating invalid input to valid input). This enables **conditional human-in-the-loop workflows** where interruptions occur only when specific conditions are met, allowing for adaptive error handling, input validation, and runtime decision-making within the graph execution flow.

**Usage**: Open `Dynamic_breakpoints.ipynb` to learn how to implement conditional, runtime-triggered breakpoints using NodeInterrupt.

## 17. Parallelization

A comprehensive guide to **parallel execution** in LangGraph, demonstrating how to run multiple nodes concurrently using **fan-out and fan-in** patterns. Shows how to structure graphs where multiple nodes execute in parallel (e.g., node `a` fans out to nodes `b` and `c`, which then fan in to node `d`), and explains the critical requirement for **reducers** when parallel nodes update the same state key. Demonstrates the `InvalidUpdateError` that occurs when multiple nodes try to write to the same state channel concurrently without a reducer, and how using `operator.add` reducer enables concurrent state updates by concatenating results. Covers handling **asymmetric parallel paths** where one path has more steps than another (e.g., `b → b2` vs `c`), showing how LangGraph waits for all parallel paths to complete before proceeding. Demonstrates creating **custom reducers** (like a sorting reducer) to control the order of state updates when parallel nodes finish at different times, and shows how reducers can be used to customize how concurrent updates are merged. Includes a realistic example where Wikipedia search and web search run in parallel to gather context, then an LLM generates an answer based on the combined results, demonstrating how parallelization improves performance by executing independent operations simultaneously.

**Usage**: Open `Parallelization.ipynb` to learn how to implement parallel node execution with fan-out/fan-in patterns and handle concurrent state updates using reducers.

## 18. Sub Graphs

A demonstration of **sub-graphs** in LangGraph, which allow you to create and manage different states in different parts of your graph. This is particularly useful for multi-agent systems, where teams of agents each have their own state and responsibilities. Demonstrates how to create separate sub-graphs with their own state schemas (e.g., `FailureAnalysisState` for failure analysis, `QuestionSummarizationState` for question summarization), and how communication between parent and sub-graphs works through **overlapping state keys**. The parent graph can pass data to sub-graphs via shared keys (e.g., `cleaned_logs`), and sub-graphs can return results to the parent through their output schemas (e.g., `fa_summary`, `report`). Shows how to compile sub-graphs separately and then add them as nodes to the parent graph, enabling modular, reusable graph components. Demonstrates using `output_schema` to control which state keys are exposed from sub-graphs, allowing sub-graphs to have internal working state while only exposing relevant output keys. This pattern is essential for building complex multi-agent workflows where different agents handle specialized tasks with their own state management needs.

**Usage**: Open `Sub_graphs.ipynb` to learn how to create modular sub-graphs and manage communication between parent and child graphs using overlapping state keys.

## 19. Map Reduce

A demonstration of the **map-reduce pattern** in LangGraph, showing how to decompose tasks into parallel sub-tasks and then aggregate results. Demonstrates a two-phase workflow: **(1) Map phase** - breaking a task into smaller sub-tasks that can be processed in parallel (e.g., generating multiple jokes about different subjects), and **(2) Reduce phase** - aggregating results from all parallel sub-tasks (e.g., selecting the best joke from all generated jokes). Shows how to use **`Send`** to dynamically create parallel execution paths: the `continue_to_jokes` function returns a list of `Send` objects that trigger the `generate_joke` node multiple times in parallel, once for each subject. Demonstrates how sub-graphs with their own state schemas (like `JokeState`) can receive input via `Send` even when their state doesn't match the parent graph's state schema. Shows how reducers (like `operator.add`) aggregate parallel results back into the parent state. This pattern is essential for efficiently processing large datasets or generating multiple variations where independent parallel operations can significantly improve performance. The example generates jokes about multiple subjects in parallel, then selects the best one, demonstrating how map-reduce enables efficient task decomposition and parallelization.

**Usage**: Open `Map_reduce.ipynb` to learn how to implement the map-reduce pattern using `Send` for dynamic parallelization and reducers for result aggregation.

## 20. Time Travel

A comprehensive guide to **time travel debugging** in LangGraph, which enables viewing, replaying, and forking from past graph execution states. Demonstrates how to use `graph.get_state()` to inspect the current state of a graph execution and `graph.get_state_history()` to browse all prior states in a conversation thread. Shows how to **replay execution** from any past checkpoint by passing the checkpoint's configuration back to the graph, allowing you to re-run the agent from any point in its history. Demonstrates **forking** execution by modifying state at a past checkpoint using `graph.update_state()` with message IDs to overwrite messages (leveraging the `add_messages` reducer's message ID-based rewriting feature), then continuing execution from that modified checkpoint. This creates a new execution path while preserving metadata like the next node to execute. Time travel enables powerful debugging workflows where you can explore alternative execution paths, test different inputs without losing the original conversation history, and understand how state evolved over time. Essential for debugging complex agent behaviors, testing edge cases, and understanding the evolution of conversation state.

**Usage**: Open `Time_travel.ipynb` to learn how to use time travel debugging to view, replay, and fork from past graph execution states.

## 21. Research Assistant

A comprehensive multi-agent research assistant system that demonstrates advanced LangGraph patterns for conducting structured research. The system allows users to customize the research process by selecting input sources and generating a team of AI analysts, each focusing on a different sub-topic. Demonstrates **human-in-the-loop** refinement of analysts before research begins, enabling users to review and adjust the generated analyst personas. Uses **structured output** with Pydantic models (`Analyst`, `Perspectives`) to generate well-defined analyst personas with roles, affiliations, and descriptions. Plans to use **sub-graphs** to conduct in-depth interviews where each analyst conducts multi-turn conversations with expert AIs using selected sources, capturing detailed insights in separate state schemas. Demonstrates **map-reduce** patterns where experts gather information in parallel to answer analyst questions, and all interviews are conducted simultaneously. The gathered insights are synthesized into a final report using customizable prompts. This notebook showcases how to combine multiple LangGraph features (human-in-the-loop, structured output, sub-graphs, map-reduce) into a cohesive, production-ready research system that can handle complex, multi-faceted research tasks.

**Usage**: Open `Research_Assistant.ipynb` to see how to build a multi-agent research system combining human-in-the-loop refinement, structured output, sub-graphs, and map-reduce patterns.

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
