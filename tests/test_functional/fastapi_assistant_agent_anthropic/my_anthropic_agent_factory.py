from langchain_openai_api_bridge.chat_model_adapter.anthropic.anthropic_openai_compatible_chat_model import (
    AnthropicOpenAICompatibleChatModel,
)

from langchain_openai_api_bridge.core.agent_factory import AgentFactory
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto


@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


class MyAnthropicAgentFactory(AgentFactory):

    def create_agent(self, llm: BaseChatModel, dto: CreateAgentDto) -> Runnable:
        return create_react_agent(
            llm,
            [magic_number_tool],
            messages_modifier="""You are a helpful assistant.""",
        )

    def create_llm(self, dto: CreateAgentDto) -> Runnable:
        return AnthropicOpenAICompatibleChatModel(
            model=dto.model,
            max_tokens=1024,
            streaming=True,
        )
