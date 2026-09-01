import pytest
from unittest.mock import AsyncMock, MagicMock
from agents.base import BaseAgent
from libs.schemas import AgentInput

class DummyAgent(BaseAgent):
    name = "dummy"

@pytest.mark.asyncio
async def test_call_llm_json_with_tools_loop(monkeypatch):
    agent = DummyAgent()
    agent.model = "gpt-4o"
    
    # Mock self.external_enabled to return True
    monkeypatch.setattr(agent, "external_enabled", lambda: True)
    
    # Mock client.chat.completions.create
    mock_create = AsyncMock()
    agent._client = MagicMock()
    agent._client.chat.completions.create = mock_create
    
    # Simular 2 iteraciones:
    # 1. Tool call 'my_tool'
    choice1 = MagicMock()
    choice1.message.content = None
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "my_tool"
    tool_call.function.arguments = '{"arg": 1}'
    choice1.message.tool_calls = [tool_call]
    choice1.message.role = "assistant"
    response1 = MagicMock()
    response1.choices = [choice1]
    response1.usage = None
    
    # 2. Respuesta JSON final
    choice2 = MagicMock()
    choice2.message.content = '{"status": "success", "result": "Madrid"}'
    choice2.message.tool_calls = None
    choice2.message.role = "assistant"
    response2 = MagicMock()
    response2.choices = [choice2]
    response2.usage = None
    
    mock_create.side_effect = [response1, response2, response2]
    
    # Mock tool handler
    async def fake_handler(args):
        return {"value": "Madrid"}
        
    result = await agent.call_llm_json_with_tools(
        system="System prompt",
        user="User prompt",
        schema={"type": "object", "properties": {"status": {"type": "string"}, "result": {"type": "string"}}},
        tools=[{"name": "my_tool", "description": "some description"}],
        tool_map={"my_tool": fake_handler},
        max_iterations=3,
    )
    
    assert result == {"status": "success", "result": "Madrid"}
    assert mock_create.call_count == 3


@pytest.mark.asyncio
async def test_call_llm_text_with_tools_loop(monkeypatch):
    agent = DummyAgent()
    agent.model = "gpt-4o"
    
    monkeypatch.setattr(agent, "external_enabled", lambda: True)
    
    mock_create = AsyncMock()
    agent._client = MagicMock()
    agent._client.chat.completions.create = mock_create
    
    choice1 = MagicMock()
    choice1.message.content = None
    tool_call = MagicMock()
    tool_call.id = "call_1"
    tool_call.function.name = "my_tool"
    tool_call.function.arguments = '{"arg": 1}'
    choice1.message.tool_calls = [tool_call]
    choice1.message.role = "assistant"
    response1 = MagicMock()
    response1.choices = [choice1]
    response1.usage = None
    
    choice2 = MagicMock()
    choice2.message.content = "Final Text Answer"
    choice2.message.tool_calls = None
    choice2.message.role = "assistant"
    response2 = MagicMock()
    response2.choices = [choice2]
    response2.usage = None
    
    mock_create.side_effect = [response1, response2]
    
    async def fake_handler(args):
        return {"value": "ok"}
        
    result = await agent.call_llm_text_with_tools(
        system="System prompt",
        user="User prompt",
        tools=[{"name": "my_tool", "description": "some description"}],
        tool_map={"my_tool": fake_handler},
        max_iterations=3,
    )
    
    assert result == "Final Text Answer"
    assert mock_create.call_count == 2
