# LangGraph Agent System with Orchestrator and Agents
# Description: Multi-Agent System for EVA AI Assistant using LangGraph and an Orchestrator.
# Author: Hans Havlik / EVA AI
# Date: 2025-06-06

# -- Imports -- #
import asyncio
import os
from dotenv import load_dotenv
from typing import Annotated, Literal, Optional, List, Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import ToolMessage # New import
from example_main_agent_tools import dev_tools_map # New import

load_dotenv()

# --- Instantiate Dev Tools --- #
# Instantiate tools from the map; tool classes are mapped, so call them
instantiated_dev_tools = {
    agent_name: tool_class() for agent_name, tool_class in dev_tools_map.items()
}

# --- Configuration --- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# --- Pydantic Models --- #
class RouteDecision(BaseModel):
    next_agent: Literal[
        "general_chat_agent", "slack_mgmt_agent", "github_mgmt_agent",
        "therapist_agent", "logical_agent", "ckb_agent", "email_mgmt_agent",
        "calendar_mgmt_agent", "web_search_agent", "customer_service_agent", "hubspot_mgmt_agent"
    ] = Field(..., description="The agent to route the query to based on its content.")
    reasoning: Optional[str] = Field(None, description="Brief reasoning for the routing decision.")

# --- State Definition --- #
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_query: str
    next_agent: Optional[str]
    final_response: Optional[str]
    final_responder: Optional[str]
    # tool_invocation: Optional[dict] = None # Add if tool use becomes more complex

# --- Agent Nodes --- #
async def orchestrator_agent_node(state: AgentState) -> Dict[str, Any]:
    print("\n🧠 --- ORCHESTRATOR --- 🧠")
    user_query = state["user_query"]
    system_prompt_content = (
        "You are EVA, a highly intelligent orchestrator AI. "
        "Your role is to analyze the user's query and determine the most appropriate specialist agent to handle it. "
        "Do not answer the query yourself. Only decide which agent should handle it."
        "Available agents and their specializations are:\n"
        "- ckb_agent: For queries requiring information from our internal knowledge base (e.g., 'how does X work?', 'what are the specs for Y?').\n"
        "- email_mgmt_agent: For tasks related to managing Gmail inbox (e.g., 'read new emails', 'draft a reply to X', 'search for email from Y').\n"
        "- calendar_mgmt_agent: For tasks related to Google Calendar (e.g., 'create an event', 'check my schedule for tomorrow', 'find free slots').\n"
        "- web_search_agent: For general web searches, current events, or information not in the CKB (e.g., 'what's the weather?', 'who won the game?').\n"
        "- customer_service_agent: For customer-facing queries about Elevated Vector Automation, its products, or services (e.g., 'tell me about your company', 'what services do you offer?').\n"
        "- slack_mgmt_agent: For tasks related to sending messages to Slack channels, listing Slack channels, or other Slack interactions.\n"
        "- github_mgmt_agent: For tasks related to GitHub, such as creating or managing issues, listing repositories, commenting on issues, or getting repository details.\n"
        "- hubspot_mgmt_agent: For CRM tasks in HubSpot (e.g., 'create a new contact', 'log a sales call', 'find company X details').\n"
        "- therapist_agent: For emotional support, therapy, feelings, or personal problems.\n"
        "- logical_agent: For facts, information, logical analysis, or practical solutions.\n"
        "- general_chat_agent: For general conversation, greetings, or if no other specialist is suitable. This agent can also echo messages and provide the current date/time.\n"
        "Based on the user's query, decide which single agent is most appropriate. Output your decision in the specified JSON format."
    )
    
    router_llm = llm.with_structured_output(RouteDecision)
    
    try:
        decision_result = await router_llm.ainvoke([
            SystemMessage(content=system_prompt_content),
            HumanMessage(content=user_query)
        ])
        print(f"🎯 Orchestrator decision: -> {decision_result.next_agent}, Reason: {decision_result.reasoning}")
        return {"next_agent": decision_result.next_agent}
    except Exception as e:
        print(f"Error in orchestrator: {e} 🛑")
        # Default to general chat agent on error
        return {"next_agent": "general_chat_agent"}

async def general_chat_agent_node(state: AgentState) -> Dict[str, Any]:
    print("💬 --- GENERAL CHAT AGENT ---")
    user_query = state["user_query"]
    system_prompt_content = (
        "You are EVA, a friendly and helpful general-purpose AI assistant. "
        "Engage in conversation and answer general queries. You can echo messages and provide the current date/time if asked. "
        "For other general questions, answer directly."
    )
    # Simplified: No actual tool calls in this version for example_main_with_tools.py
    response = await llm.ainvoke([
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ])
    final_response = response.content
    print(f"💬 General Chat Agent response: {final_response}")
    return {"messages": add_messages(state["messages"], [AIMessage(content=final_response)]), "final_response": final_response, "final_responder": "general_chat_agent"}

async def slack_mgmt_agent_node(state: AgentState) -> Dict[str, Any]:
    print("📱 --- SLACK MGMT AGENT ---")
    user_query = state["user_query"]
    agent_name = "slack_mgmt_agent"
    slack_tool = instantiated_dev_tools.get(agent_name) # Get the instantiated tool

    system_prompt_content = (
        "You are a helpful AI assistant specialized in managing Slack interactions. "
        "You can post messages to channels/users and list available channels. "
        "When asked to post a message, confirm the channel ID and the message content. "
        "When asked to list channels, provide the retrieved list.\n"
        "You have a development tool called 'run_slack_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_slack_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([slack_tool]) if slack_tool else llm
    
    # Prepare initial messages for the first LLM call
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]

    # Accumulate all messages for state update throughout the process
    all_messages_for_state_update = [] 

    try:
        # First LLM call, potentially invoking the tool
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and slack_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0] # Assuming one tool call for this dev tool

            if tool_call['name'] == slack_tool.name:
                # Ensure args is a dictionary, even if empty, for the tool's Pydantic model
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await slack_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                # Second LLM call to synthesize response from tool output
                # We send the history including the initial AI message (with tool_call) and the tool_message
                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                # LLM decided to call a tool not assigned or unexpected
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content # Fallback to content if tool call is not the dev tool
        else:
            # No tool call, use the content directly from the first AI response
            final_response_content = ai_response_msg.content
        
        print(f"📱 Slack Mgmt Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update), 
            "final_response": final_response_content, 
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        # Add an AIMessage with the error to the state if not already handled
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def github_mgmt_agent_node(state: AgentState) -> Dict[str, Any]:
    print("💻 --- GITHUB MGMT AGENT ---")
    user_query = state["user_query"]
    agent_name = "github_mgmt_agent"
    github_tool = instantiated_dev_tools.get(agent_name) # Get the instantiated tool

    system_prompt_content = (
        "You are a helpful AI assistant specialized in GitHub repository management. "
        "You can create issues, get issue details, list issues, comment on issues, list repositories, and get repository details. "
        "Always ask for repository names (e.g., 'owner/repo') and issue numbers when needed.\n"
        "You have a development tool called 'run_github_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_github_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([github_tool]) if github_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and github_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == github_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await github_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"💻 GitHub Mgmt Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def therapist_agent_node(state: AgentState) -> Dict[str, Any]:
    print("❤️‍🩹 --- THERAPIST AGENT ---")
    user_query = state["user_query"]
    agent_name = "therapist_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are a compassionate therapist. Focus on the emotional aspects of the user's message.\n"
        "Show empathy, validate their feelings, and help them process their emotions.\n"
        "Ask thoughtful questions to help them explore their feelings more deeply.\n"
        "Avoid giving logical solutions unless explicitly asked.\n"
        "You have a development tool called 'run_therapist_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_therapist_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"❤️‍🩹 Therapist Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def logical_agent_node(state: AgentState) -> Dict[str, Any]:
    print("💡 --- LOGICAL AGENT ---")
    user_query = state["user_query"]
    agent_name = "logical_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are a purely logical assistant. Focus only on facts and information.\n"
        "Provide clear, concise answers based on logic and evidence.\n"
        "Do not address emotions or provide emotional support.\n"
        "Be direct and straightforward in your responses.\n"
        "You have a development tool called 'run_logical_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_logical_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"💡 Logical Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def ckb_agent_node(state: AgentState) -> Dict[str, Any]:
    print("📚 --- CKB AGENT ---")
    user_query = state["user_query"]
    agent_name = "ckb_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are an AI assistant specialized in retrieving information from our internal knowledge base. "
        "Answer questions based on the knowledge provided to you. If the information is not in the CKB, state that clearly.\n"
        "You have a development tool called 'run_ckb_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_ckb_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"📚 CKB Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def email_mgmt_agent_node(state: AgentState) -> Dict[str, Any]:
    print("📧 --- EMAIL MGMT AGENT ---")
    user_query = state["user_query"]
    agent_name = "email_mgmt_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are an AI assistant for managing Gmail. You can read emails, draft replies, and search the inbox. "
        "Always confirm actions like sending emails or deleting messages.\n"
        "You have a development tool called 'run_email_mgmt_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_email_mgmt_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"📧 Email Mgmt Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def calendar_mgmt_agent_node(state: AgentState) -> Dict[str, Any]:
    print("📅 --- CALENDAR MGMT AGENT ---")
    user_query = state["user_query"]
    agent_name = "calendar_mgmt_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are an AI assistant for Google Calendar. You can create events, check schedules, and find free slots. "
        "Clarify details like event titles, dates, times, and attendees.\n"
        "You have a development tool called 'run_calendar_mgmt_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_calendar_mgmt_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"📅 Calendar Mgmt Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def web_search_agent_node(state: AgentState) -> Dict[str, Any]:
    print("🌐 --- WEB SEARCH AGENT ---")
    user_query = state["user_query"]
    agent_name = "web_search_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are a web search assistant. You can find information on the internet about current events, facts, or general knowledge. "
        "Provide concise summaries and cite sources if possible.\n"
        "You have a development tool called 'run_web_search_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_web_search_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"🌐 Web Search Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def customer_service_agent_node(state: AgentState) -> Dict[str, Any]:
    print("🤝 --- CUSTOMER SERVICE AGENT ---")
    user_query = state["user_query"]
    agent_name = "customer_service_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are a customer service representative for Elevated Vector Automation. "
        "Answer questions about our company, products, and services. Be polite and helpful. "
        "If you cannot answer, say you will find someone who can.\n"
        "You have a development tool called 'run_customer_service_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_customer_service_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call['name']}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"🤝 Customer Service Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }

async def hubspot_mgmt_agent_node(state: AgentState) -> Dict[str, Any]:
    print("📈 --- HUBSPOT MGMT AGENT ---")
    user_query = state["user_query"]
    agent_name = "hubspot_mgmt_agent"
    agent_tool = instantiated_dev_tools.get(agent_name)

    system_prompt_content = (
        "You are an AI assistant for HubSpot CRM. You can manage contacts, companies, deals, and tasks. "
        "Confirm details before creating or updating records.\n"
        "You have a development tool called 'run_hubspot_mgmt_dev_tool'. "
        "If the user's query is specifically 'Run_Dev_Tool' or asks you to run your dev tool, "
        "you MUST use the 'run_hubspot_mgmt_dev_tool' to respond. For the 'task_description' argument of the tool, "
        "you can use the user's query or a summary of it."
    )

    llm_with_tool = llm.bind_tools([agent_tool]) if agent_tool else llm
    
    current_messages = [
        SystemMessage(content=system_prompt_content),
        HumanMessage(content=user_query)
    ]
    all_messages_for_state_update = []

    try:
        ai_response_msg = await llm_with_tool.ainvoke(current_messages)
        all_messages_for_state_update.append(ai_response_msg)
        final_response_content = ""

        if ai_response_msg.tool_calls and agent_tool:
            print(f"🛠️ {agent_name} attempting to use tool: {ai_response_msg.tool_calls[0]['name']}")
            tool_call = ai_response_msg.tool_calls[0]

            if tool_call['name'] == agent_tool.name:
                tool_args = tool_call['args'] if isinstance(tool_call['args'], dict) else {}
                tool_output = await agent_tool.ainvoke(tool_args)
                print(f"🛠️ {agent_name} tool output: {tool_output}")
                tool_message = ToolMessage(content=str(tool_output), tool_call_id=tool_call['id'])
                all_messages_for_state_update.append(tool_message)

                messages_for_final_synthesis = current_messages + all_messages_for_state_update
                final_llm_response = await llm_with_tool.ainvoke(messages_for_final_synthesis)
                all_messages_for_state_update.append(final_llm_response)
                final_response_content = final_llm_response.content
            else:
                print(f"⚠️ {agent_name} tried to call an unexpected tool: {tool_call.name}. Responding without tool.")
                final_response_content = ai_response_msg.content
        else:
            final_response_content = ai_response_msg.content
        
        print(f"📈 HubSpot Mgmt Agent final response: {final_response_content}")
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": final_response_content,
            "final_responder": agent_name
        }

    except Exception as e:
        print(f"💥 Error in {agent_name}: {e}")
        error_response = f"Sorry, I encountered an error while processing your request for {agent_name}. Detail: {str(e)}"
        if not any(isinstance(m, AIMessage) and error_response in m.content for m in all_messages_for_state_update):
            all_messages_for_state_update.append(AIMessage(content=error_response))
        return {
            "messages": add_messages(state["messages"], all_messages_for_state_update),
            "final_response": error_response,
            "final_responder": agent_name
        }


# --- Graph Definition --- #

graph_builder = StateGraph(AgentState)

graph_builder.add_node("orchestrator", orchestrator_agent_node)
graph_builder.add_node("general_chat_agent", general_chat_agent_node)
graph_builder.add_node("slack_mgmt_agent", slack_mgmt_agent_node)
graph_builder.add_node("github_mgmt_agent", github_mgmt_agent_node)
graph_builder.add_node("therapist_agent", therapist_agent_node)
graph_builder.add_node("logical_agent", logical_agent_node)
graph_builder.add_node("ckb_agent", ckb_agent_node)
graph_builder.add_node("email_mgmt_agent", email_mgmt_agent_node)
graph_builder.add_node("calendar_mgmt_agent", calendar_mgmt_agent_node)
graph_builder.add_node("web_search_agent", web_search_agent_node)
graph_builder.add_node("customer_service_agent", customer_service_agent_node)
graph_builder.add_node("hubspot_mgmt_agent", hubspot_mgmt_agent_node)

graph_builder.add_edge(START, "orchestrator")

# Conditional routing from orchestrator
def route_logic(state: AgentState) -> str:
    next_agent = state.get("next_agent")
    defined_agents = [
        "general_chat_agent", "slack_mgmt_agent", "github_mgmt_agent", 
        "therapist_agent", "logical_agent", "ckb_agent", "email_mgmt_agent",
        "calendar_mgmt_agent", "web_search_agent", "customer_service_agent", "hubspot_mgmt_agent"
    ]
    if next_agent in defined_agents:
        return next_agent
    # Fallback or error handling - default to general_chat_agent
    print(f"⚠️ Warning: Unknown or unhandled agent '{next_agent}', defaulting to general_chat_agent.")
    return "general_chat_agent"

graph_builder.add_conditional_edges(
    "orchestrator",
    route_logic,
    {
        "general_chat_agent": "general_chat_agent",
        "slack_mgmt_agent": "slack_mgmt_agent",
        "github_mgmt_agent": "github_mgmt_agent",
        "therapist_agent": "therapist_agent",
        "logical_agent": "logical_agent",
        "ckb_agent": "ckb_agent",
        "email_mgmt_agent": "email_mgmt_agent",
        "calendar_mgmt_agent": "calendar_mgmt_agent",
        "web_search_agent": "web_search_agent",
        "customer_service_agent": "customer_service_agent",
        "hubspot_mgmt_agent": "hubspot_mgmt_agent"
    }
)

# All specialist agents go to END for now
graph_builder.add_edge("general_chat_agent", END)
graph_builder.add_edge("slack_mgmt_agent", END)
graph_builder.add_edge("github_mgmt_agent", END)
graph_builder.add_edge("therapist_agent", END)
graph_builder.add_edge("logical_agent", END)
graph_builder.add_edge("ckb_agent", END)
graph_builder.add_edge("email_mgmt_agent", END)
graph_builder.add_edge("calendar_mgmt_agent", END)
graph_builder.add_edge("web_search_agent", END)
graph_builder.add_edge("customer_service_agent", END)
graph_builder.add_edge("hubspot_mgmt_agent", END)

graph = graph_builder.compile()

# --- Chatbot Execution --- #
async def run_chatbot():
    session_id_counter = 0
    while True:
        user_input = input("Message: ")
        if user_input.lower() == "exit":
            print("Bye")
            break

        session_id_counter += 1
        current_session_id = f"session_{session_id_counter}"

        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "next_agent": None,
            "final_response": None,
            "final_responder": None
        }

        # Configuration for invoking the graph, if needed (e.g., for checkpoints)
        config = {"configurable": {"session_id": current_session_id}}

        print(f"\n⏳ Processing for session: {current_session_id}...")
        try:
            final_graph_state = await graph.ainvoke(initial_state, config=config)
            
            if final_graph_state and final_graph_state.get("final_response"):
                responder = final_graph_state.get('final_responder', 'N/A')
                emoji_map = {
                    "general_chat_agent": "💬", "slack_mgmt_agent": "📱", "github_mgmt_agent": "💻",
                    "therapist_agent": "❤️‍🩹", "logical_agent": "💡", "ckb_agent": "📚",
                    "email_mgmt_agent": "📧", "calendar_mgmt_agent": "📅", "web_search_agent": "🌐",
                    "customer_service_agent": "🤝", "hubspot_mgmt_agent": "📈", "N/A": "🤖"
                }
                responder_emoji = emoji_map.get(responder, "🤖")
                print(f"\n{responder_emoji} Assistant ({responder}): {final_graph_state['final_response']}")
            else:
                # Fallback if final_response isn't set, check last message
                if final_graph_state and final_graph_state.get("messages"):
                    last_message = final_graph_state["messages"][-1]
                    if isinstance(last_message, AIMessage):
                         print(f"🤖 Assistant (from messages): {last_message.content}")
                    else:
                        print("🤖 Assistant: No clear response generated. 🤷")
                else:
                    print("🤖 Assistant: No response generated. 🤷")
        except Exception as e:
            print(f"💥 Error during graph execution: {e}")
        print("-"*60 + "\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_chatbot())
    except KeyboardInterrupt:
        print("\nChatbot interrupted. Exiting.")