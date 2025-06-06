# EVA 2.0 Multi-Agent System Demo 🤖🔄

Welcome to the EVA 2.0 Multi-Agent System Demo! This project showcases a smart system where multiple AI "agents" (think of them as specialized AI assistants) work together. It's like having a team of experts, each good at one thing, coordinated by a manager.

## 1. What's This All About? (Project Overview 🌟)

Imagine you have a complex task. Instead of one AI trying to do everything, we have a team:

- **A Manager AI (Orchestrator)**: Reads your request and decides which specialist AI is best suited for the job.
- **Specialist AIs**: Each expert in a specific area (like managing your calendar, searching the web, or interacting with GitHub).
- **Tools for Specialists**: Each specialist AI has its own set of "dev tools" – special functions that let it interact with other services or perform specific actions (e.g., a calendar tool to check your schedule).

This demo uses a technology called **LangGraph** to manage how these AIs work together, deciding who does what and when. It's designed to be efficient and smart in handling your requests.

**Key Features You'll See:**

- **Teamwork Structure (Hierarchical Agents) 🏛️**: A clear manager-and-specialist setup.
- **Smart Task Assignment (Dynamic Routing) 🚦**: The manager AI intelligently sends tasks to the right specialist.
- **Specialists with Superpowers (Tool-Equipped Agents) 🛠️**: Agents use their unique tools to get things done.
- **Quick Responses (Asynchronous Operations) ⚡**: The system can handle multiple things smoothly without getting stuck, making it feel responsive.

## 2. What's Inside? (Files in this Directory 📁)

This demo is powered by two main Python files:

### a. `example_main_agent_tools.py` 🧰

- **What it does**: This file is like a toolbox. It defines all the special tools that our specialist AIs can use. Think of it as creating the actual software functions that let an AI, for example, add an event to a calendar or fetch information from a website.
- **For Non-Technical Users**: Imagine this file describes the unique abilities of each specialist. For example, the "Calendar Specialist" gets a tool that can "check schedule" or "add event."
- **For Technical Users**:
  - It defines LangChain-compatible `Tool` classes for each specialist agent.
  - These tools often act as wrappers around custom "dev tools" which might be services running elsewhere, accessible via **MCP (Model Context Protocol)**. MCP is a standard way for different software parts to talk to each other, especially when AI models are involved.
  - Each tool class specifies:
    - `name` and `description`: So the AI knows what the tool is for and when to use it.
    - `args_schema`: (Using Pydantic models) Defines what information the tool needs to do its job (e.g., for a calendar tool, it might need a date).
    - `_run` (for synchronous) or `_arun` (for asynchronous) methods: This is the actual code that makes the tool work, often by calling an MCP server.
  - Includes a `dev_tools_map`: A handy dictionary to easily link specialist agent names to their tool classes.
- **Main Technologies Used**: `langchain_core` (for creating AI tools), `pydantic` (for defining data structures for tool inputs).

### b. `example_main_and_agents.py` 🧠

- **What it does**: This is the brain of the operation! It sets up all the AI agents (the manager and the specialists) and defines how they interact using LangGraph. It's where the rules of teamwork are written.
- **For Non-Technical Users**: This file creates the AI team and teaches them how to work together. It tells the manager AI how to understand requests and pass them to the right specialist. It also tells specialists how to use their tools and report back.
- **For Technical Users**:
  - **Agent State (`AgentState`)**: Defines a structure (a TypedDict) to keep track of the conversation, what steps have been taken, and any important information as the request flows through the system.
  - **Language Model Setup**: Initializes the AI model (e.g., OpenAI's GPT-4) that powers the agents' thinking and language capabilities.
  - **Tool Creation**: Creates instances of all the tools defined in `example_main_agent_tools.py`.
  - **Specialist Agent Nodes**: These are Python functions, each representing a specialist AI (e.g., `calendar_mgmt_agent_node`). Each specialist:
    - Gets a **System Prompt**: Initial instructions telling the AI its role, personality, and how to behave (e.g., "You are a helpful calendar assistant.").
    - **Binds its Tool**: Its specific dev tool is made available to its underlying language model.
    - **Two-Step Tool Use**: If the AI decides to use its tool:
            1. It first makes a plan (initial LLM call which might include a `tool_calls` request).
            2. If a tool is called: The system runs the tool, gets the result, and then the AI uses this result to form its final answer (second LLM call with `ToolMessage`).
    - **Error Handling**: Catches and logs any problems that occur.
  - **Orchestrator Agent Node (Manager AI) 🎭**: This agent's job is to look at your request and decide which specialist (or a general chat agent) should handle it.
  - **General Chat Agent Node 💬**: If no specialist is needed, this agent handles general conversation.
  - **The Workflow (Graph Definition) 📊**: Using LangGraph, this section connects all the agents into a flow-chart (a `StatefulGraph`). It defines:
    - **Nodes**: Each agent is a node (a step) in the workflow.
    - **Edges**: Lines connecting the nodes, showing how the task moves from one agent to another. Some edges are **conditional**, meaning the path taken depends on the orchestrator's decision.
  - **Running the Show (Main Loop) 🔄**: Code that starts the system, takes your input, sends it to the LangGraph workflow, and then prints the AI's final response.
- **Main Technologies Used**: `langgraph` (for building the agent team workflow), `langchain_core`, `langchain_openai` (for AI models), `python-dotenv` (for managing secret keys).

## 3. Getting Started (Setup ⚙️)

Ready to try it out? Here’s how to get it running on your computer.

### Prerequisites (What You Need First 📋)

- **Python 🐍**: Version 3.10 or newer. (Python is the programming language this demo is written in.)
- **OpenAI API Key 🔑**: You'll need an API key from OpenAI to use their powerful language models (like GPT-4). This usually involves signing up on their platform. *This demo might incur small costs depending on your OpenAI API usage.*
- **(Maybe) MCP Servers 🖥️**: If the tools are designed to talk to other services (MCP servers), those services need to be running. For this demo, they might be simulated or mocked, but in a real application, they'd be separate running programs.

### Installation Steps 📦

1. **Get the Code (Clone Repository)**: If this demo is part of a larger project on a site like GitHub, you'll need to download (or "clone") it to your computer.
2. **Create a Clean Workspace (Virtual Environment)**:
    It's good practice to create an isolated environment for Python projects. This keeps things tidy and avoids conflicts between different project requirements.
    Open your terminal or command prompt and run:

    ```bash
    python -m venv .venv
    ```

    Then, activate it:
    - On macOS/Linux: `source .venv/bin/activate`
    - On Windows: `.venv\Scripts\activate`
    You should see `(.venv)` appear at the start of your command prompt line.
3. **Install Required Software (Dependencies)**:
    The project uses several external software libraries. These are listed in a file, typically `requirements.txt`. Navigate to the main project folder (e.g., `d:\my_projects\EVA-Multi-Agent-Framework`) in your terminal and run:

    ```bash
    pip install -r requirements.txt
    ```

    This command reads the `requirements.txt` file and installs all the necessary packages like `langchain`, `langgraph`, `openai`, etc.
4. **Set Up Your Secret Key (Environment Variables)**:
    Your OpenAI API key is sensitive information and shouldn't be written directly into the code. We use a `.env` file for this.
    - In the main project root folder (e.g., `d:\my_projects\EVA-Multi-Agent-Framework`), create a file named `.env` (note the dot at the beginning).
    - Open this `.env` file with a text editor and add your OpenAI API key like this:

        ```env
        OPENAI_API_KEY="your_actual_openai_api_key_here"
        # If the tools need to talk to other services (MCP Servers), their addresses might also go here:
        # EXAMPLE_MCP_SERVER_URL="http://localhost:8001"
        ```

    Replace `"your_actual_openai_api_key_here"` with your real key.

## 4. Let's Run It! (How to Run 🚀)

1. **Start Any Helper Services (MCP Servers, if applicable) 🖥️**: If the tools in `example_main_agent_tools.py` need to connect to live MCP servers, make sure those are started and running correctly.
2. **Go to the Demo Folder**: Open your terminal or command prompt and navigate to this demo's specific directory:

    ```bash
    cd d:\my_projects\EVA-Multi-Agent-Framework\eva_2_test\EVA_2.0_Demo
    ```

3. **Start the Demo Script**: Run the main Python file:

    ```bash
    python example_main_and_agents.py
    ```

4. **Chat with EVA! 🗣️**:
    The script will usually print a message like `Message:` indicating it's ready for your input.
    Try typing different things to see how the system responds:
    - **General Chat**: `hello there` or `how are you?` (The General Chat Agent should reply.)
    - **Task for a Specialist**: `what's on my calendar for tomorrow?` or `search the web for LangGraph tutorials` (The Orchestrator should route this to the Calendar or Web Search agent).
    - **Directly Ask a Specialist to Use its Tool**: `calendar agent, please Run_Dev_Tool.` or `slack agent, Run_Dev_Tool` (This is a special command to force the agent to try and use its tool – great for testing!)

    👀 **Watch the Console Output!** As you interact, the script will print logs showing which agent is working, what decisions the orchestrator is making, and if tools are being used. This is very helpful for understanding what's happening behind the scenes.

## 5. What Makes This Cool? (Key Concepts Demonstrated ✨)

This demo isn't just a program; it's a showcase of powerful ideas in modern AI development:

- **Team of AIs (Multi-Agent Orchestration with LangGraph)**: Instead of one AI, we have many, each with a role. LangGraph is the conductor of this AI orchestra, ensuring they play together harmoniously.
- **Manager & Specialists (Hierarchical Agent Systems)**: Like a company, there's a manager (Orchestrator) that delegates tasks to experts (Specialist Agents). This makes the system organized and scalable.
- **AIs with Real Abilities (Tool Integration)**: Agents aren't limited to just talking. They can perform actions (like checking your email or posting to Slack) using their tools. This makes them much more useful.
- **Remembering the Conversation (Stateful Conversations)**: The system keeps track of what's been said and done (`AgentState`). This allows for more natural, ongoing conversations where the AI remembers context.
- **Smart Decision Making (Conditional Logic in Graphs)**: The workflow isn't fixed. The Orchestrator can make choices, sending the task down different paths in the LangGraph workflow based on your request. This makes the system flexible.
- **No Waiting Around (Asynchronous Operations)**: Using `async` and `await` in Python allows the system to handle tasks (especially tool calls that might take time, like fetching web data) without freezing. This means a smoother experience for you.
- **Clear Tool Usage (Structured Tool Calls)**: There's a specific, reliable way (`ToolMessage` and a two-step LLM process) for agents to request a tool's use and get its output. This makes tool use robust.
- **Neat and Tidy Code (Modular Design)**: Keeping tool definitions separate from the agent and workflow logic makes the code easier to understand, update, and expand.

## 6. Who Is This Demo For? 🤔

- **Developers New to AI Agents**: A great starting point to see a multi-agent system in action.
- **LangChain/LangGraph Learners**: Provides a practical example of how to use these powerful libraries.
- **AI Enthusiasts**: A glimpse into how more complex and capable AI systems are being built.
- **Anyone Curious!**: If you're interested in the future of AI assistants, this demo offers a peek.

## 7. What's Next? (Further Exploration 🗺️)

Once you're comfortable with this demo, you could try:

- **Adding a New Specialist Agent**: Think of a new skill and try to build an agent for it!
- **Creating a New Tool**: Design and implement a new tool for an existing agent.
- **Modifying the Orchestrator's Logic**: Change how the manager AI decides which specialist to call.
- **Integrating with Real Services**: If tools are mocked, try connecting them to actual services (like your real calendar or Slack account – be careful with permissions and data!).

---

We hope this README helps you understand, set up, and enjoy the EVA 2.0 multi-agent demo. Happy experimenting! 🎉
