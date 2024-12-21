from typing import AsyncIterator, List, Optional
from langchain_core.runnables import Runnable
from langgraph.graph.graph import CompiledGraph
from langchain_openai_api_bridge.chat_completion.langchain_invoke_adapter import (
    LangchainInvokeAdapter,
)
from langchain_openai_api_bridge.chat_completion.langchain_stream_adapter import (
    LangchainStreamAdapter,
)
from langchain_openai_api_bridge.core.types.openai import OpenAIChatMessage
from langchain_openai_api_bridge.core.utils.pydantic_async_iterator import ato_dict

from langchain_core.runnables.config import (
    RunnableConfig,
)

class ChatCompletionCompatibleAPI:

    @staticmethod
    def from_agent(
        agent: Runnable,
        llm_model: str,
        system_fingerprint: Optional[str] = "",
        event_adapter: callable = lambda event: None,
        invoke_config: Optional[RunnableConfig] = None
    ):
        return ChatCompletionCompatibleAPI(
            LangchainStreamAdapter(llm_model, system_fingerprint),
            LangchainInvokeAdapter(llm_model, system_fingerprint),
            agent,
            event_adapter,
            invoke_config
        )

    def __init__(
        self,
        stream_adapter: LangchainStreamAdapter,
        invoke_adapter: LangchainInvokeAdapter,
        agent: Runnable,
        event_adapter: callable = lambda event: None,
        invoke_config: Optional[RunnableConfig] = None
    ) -> None:
        self.stream_adapter = stream_adapter
        self.invoke_adapter = invoke_adapter
        self.agent = agent
        self.event_adapter = event_adapter
        self.invoke_config = invoke_config

    def astream(self, messages: List[OpenAIChatMessage]) -> AsyncIterator[dict]:
        input = self.__to_input(messages)
        astream_event = self.agent.astream_events(
            input=input,
            version="v2",
            config=self.invoke_config
        )
        return ato_dict(
            self.stream_adapter.ato_chat_completion_chunk_stream(astream_event, event_adapter=self.event_adapter)
        )

    def invoke(self, messages: List[OpenAIChatMessage]) -> dict:
        input = self.__to_input(messages)

        result = self.agent.invoke(
            input=input,
            config=self.invoke_config
        )

        return self.invoke_adapter.to_chat_completion_object(result).dict()

    def __to_input(self, messages: List[OpenAIChatMessage]):
        if isinstance(self.agent, CompiledGraph):
            return self.__to_react_agent_input(messages)
        else:
            return self.__to_chat_model_input(messages)

    def __to_react_agent_input(self, messages: List[OpenAIChatMessage]):
        return {
            "messages": [message.dict() for message in messages],
        }

    def __to_chat_model_input(self, messages: List[OpenAIChatMessage]):
        return [message.dict() for message in messages]
