from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.runnables import Runnable
from langchain_openai_api_bridge.core.create_agent_dto import CreateAgentDto

from langchain_core.runnables.config import (
    RunnableConfig,
)

class BaseAgentFactory(ABC):

    @abstractmethod
    def create_agent(self, dto: CreateAgentDto) -> tuple[Runnable, Optional[RunnableConfig]]:
        pass
