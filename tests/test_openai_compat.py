from libs.openai_compat import completion_token_kwargs, tool_reasoning_kwargs


def test_gpt5_openai_uses_completion_token_parameter():
    assert completion_token_kwargs("gpt-5-mini", "openai", 2048) == {
        "max_completion_tokens": 2048
    }


def test_openrouter_keeps_legacy_compatible_token_parameter():
    assert completion_token_kwargs("gpt-5-mini", "openrouter", 2048) == {
        "max_tokens": 2048
    }


def test_no_token_limit_does_not_send_parameter():
    assert completion_token_kwargs("gpt-5-mini", "openai", None) == {}


def test_gpt56_luna_disables_reasoning_when_tools_are_used():
    assert tool_reasoning_kwargs("gpt-5.6-luna", "openai") == {
        "reasoning_effort": "none"
    }


def test_tool_reasoning_override_is_scoped_to_affected_model():
    assert tool_reasoning_kwargs("gpt-5-mini", "openai") == {}
    assert tool_reasoning_kwargs("gpt-5.6-luna", "openrouter") == {}
