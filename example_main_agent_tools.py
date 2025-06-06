from typing import Type, Optional

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# --- Generic Tool Input Schema --- #
class DevToolInput(BaseModel):
    task_description: Optional[str] = Field(
        None, description="Optional description of the task for the dev tool."
    )


# --- Specialist Agent Dev Tools --- #

# Slack Dev Tool
class SlackDevTool(BaseTool):
    name: str = "run_slack_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Slack Management agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Slack agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Slack Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated Slack API call. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# GitHub Dev Tool
class GitHubDevTool(BaseTool):
    name: str = "run_github_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the GitHub Management agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the GitHub agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The GitHub Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated GitHub API interaction. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Therapist Dev Tool
class TherapistDevTool(BaseTool):
    name: str = "run_therapist_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Therapist agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Therapist agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Therapist Agent has returned the following tool response: "
            f"Dev tool executed. Simulated therapeutic exercise or reflection. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Logical Dev Tool
class LogicalDevTool(BaseTool):
    name: str = "run_logical_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Logical agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Logical agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Logical Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated logical analysis or data retrieval. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# CKB Dev Tool
class CKBDevTool(BaseTool):
    name: str = "run_ckb_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the CKB (Knowledge Base) agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the CKB agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The CKB Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated knowledge base query. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Email Dev Tool
class EmailMgmtDevTool(BaseTool):
    name: str = "run_email_mgmt_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Email Management agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Email agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Email Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated email interaction (e.g., fetching or sending). Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Calendar Dev Tool
class CalendarMgmtDevTool(BaseTool):
    name: str = "run_calendar_mgmt_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Calendar Management agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Calendar agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Calendar Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated calendar operation (e.g., event creation). Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Web Search Dev Tool
class WebSearchDevTool(BaseTool):
    name: str = "run_web_search_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Web Search agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Web Search agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Web Search Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated web search query. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# Customer Service Dev Tool
class CustomerServiceDevTool(BaseTool):
    name: str = "run_customer_service_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the Customer Service agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the Customer Service agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The Customer Service Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated customer interaction or lookup. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)

# HubSpot Dev Tool
class HubSpotMgmtDevTool(BaseTool):
    name: str = "run_hubspot_mgmt_dev_tool"
    description: str = (
        "Runs a placeholder development tool for the HubSpot Management agent. "
        "Use this if the user asks to 'Run_Dev_Tool' and you are the HubSpot agent."
    )
    args_schema: Type[BaseModel] = DevToolInput

    def _run(self, task_description: Optional[str] = None) -> str:
        return (
            "The HubSpot Agent has returned the following tool response: "
            f"Dev tool executed successfully. Simulated HubSpot CRM action. Task: {task_description or 'No specific task provided'}."
        )

    async def _arun(self, task_description: Optional[str] = None) -> str:
        return self._run(task_description)


# Dictionary mapping agent node names to their respective dev tools for easy instantiation
dev_tools_map = {
    "slack_mgmt_agent": SlackDevTool,
    "github_mgmt_agent": GitHubDevTool,
    "therapist_agent": TherapistDevTool,
    "logical_agent": LogicalDevTool,
    "ckb_agent": CKBDevTool,
    "email_mgmt_agent": EmailMgmtDevTool,
    "calendar_mgmt_agent": CalendarMgmtDevTool,
    "web_search_agent": WebSearchDevTool,
    "customer_service_agent": CustomerServiceDevTool,
    "hubspot_mgmt_agent": HubSpotMgmtDevTool,
}

