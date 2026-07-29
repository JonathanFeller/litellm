"""Tests for the MCP guardrail translation handler."""

from typing import Optional

import pytest
from fastapi import HTTPException

import litellm
from litellm.caching.caching import DualCache
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._experimental.mcp_server.guardrail_translation.handler import (
    MCPGuardrailTranslationHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import ProxyLogging


class MockGuardrail(CustomGuardrail):
    """Simple guardrail mock that records invocations."""

    def __init__(self):
        super().__init__(guardrail_name="mock-mcp-guardrail")
        self.call_count = 0
        self.last_inputs = None
        self.last_request_data = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.call_count += 1
        self.last_inputs = inputs
        self.last_request_data = request_data
        return None  # Guardrail doesn't modify for MCP tools


class MaskingGuardrail(CustomGuardrail):
    """Unified guardrail that rewrites every text it is handed, like presidio does."""

    def __init__(
        self,
        secret: str = "jane.doe@example.com",
        replacement: str = "<EMAIL_ADDRESS>",
        texts_override: Optional[list] = None,
        **kwargs,
    ):
        kwargs.setdefault("guardrail_name", "masking-mcp-guardrail")
        super().__init__(**kwargs)
        self.secret = secret
        self.replacement = replacement
        self.texts_override = texts_override
        self.seen_texts: Optional[list] = None

    async def apply_guardrail(self, inputs, request_data, input_type, **kwargs):
        self.seen_texts = list(inputs.get("texts") or [])
        if self.texts_override is not None:
            inputs["texts"] = self.texts_override
        else:
            inputs["texts"] = [text.replace(self.secret, self.replacement) for text in self.seen_texts]
        return inputs


@pytest.fixture
def restore_callbacks():
    original = litellm.callbacks
    yield
    litellm.callbacks = original


@pytest.mark.asyncio
async def test_process_input_messages_updates_content():
    """Handler should pass the tool definition and the argument strings to the guardrail."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "weather",
        "mcp_arguments": {"city": "tokyo"},
        "mcp_tool_description": "Get weather for a city",
    }

    result = await handler.process_input_messages(data, guardrail)

    # Handler passes data through unchanged
    assert result == data
    # Guardrail was called
    assert guardrail.call_count == 1
    # Guardrail received tools with the tool definition
    assert guardrail.last_inputs is not None
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "weather"
    # Request data was passed to guardrail
    assert guardrail.last_request_data == data


@pytest.mark.asyncio
async def test_process_input_messages_skips_when_no_tool_name():
    """Handler should skip guardrail invocation if mcp_tool_name is missing."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    # No mcp_tool_name means nothing to process
    data = {"some_other_field": "value"}
    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 0


@pytest.mark.asyncio
async def test_process_input_messages_handles_minimal_data():
    """Handler should work with just mcp_tool_name (minimal required field)."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {"mcp_tool_name": "simple_tool"}

    result = await handler.process_input_messages(data, guardrail)

    assert result == data
    assert guardrail.call_count == 1
    tools = guardrail.last_inputs.get("tools", [])
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "simple_tool"


@pytest.mark.asyncio
async def test_argument_strings_are_handed_to_the_guardrail():
    """A guardrail must see the argument values, not just the tool definition.

    Without this the guardrail is handed a name and an empty schema, so no
    sensitive-data detection can ever fire on an MCP tool call.
    """
    handler = MCPGuardrailTranslationHandler()
    guardrail = MockGuardrail()

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    await handler.process_input_messages(data, guardrail)

    assert guardrail.last_inputs is not None
    assert guardrail.last_inputs.get("texts") == ["contact jane.doe@example.com about the invoice"]


@pytest.mark.asyncio
async def test_masked_arguments_are_written_back_for_the_call_path():
    """A mask only takes effect once it lands in modified_arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail()

    data = {
        "mcp_tool_name": "search",
        "mcp_arguments": {"query": "contact jane.doe@example.com about the invoice"},
    }

    result = await handler.process_input_messages(data, guardrail)

    masked = {"query": "contact <EMAIL_ADDRESS> about the invoice"}
    assert result["modified_arguments"] == masked
    assert result["mcp_arguments"] == masked


@pytest.mark.asyncio
async def test_nested_arguments_keep_their_shape_when_masked():
    """Masking rewrites string leaves in place and preserves non-string values."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail()

    arguments = {
        "recipients": ["jane.doe@example.com", "ops@example.net"],
        "envelope": {"reply_to": "jane.doe@example.com", "retries": 3, "urgent": True, "cc": None},
        "count": 2,
    }
    data = {"mcp_tool_name": "send_email", "mcp_arguments": arguments}

    result = await handler.process_input_messages(data, guardrail)

    assert guardrail.seen_texts == [
        "jane.doe@example.com",
        "ops@example.net",
        "jane.doe@example.com",
    ]
    assert result["modified_arguments"] == {
        "recipients": ["<EMAIL_ADDRESS>", "ops@example.net"],
        "envelope": {"reply_to": "<EMAIL_ADDRESS>", "retries": 3, "urgent": True, "cc": None},
        "count": 2,
    }


@pytest.mark.asyncio
async def test_clean_arguments_are_not_overridden():
    """A guardrail that changes nothing must not set modified_arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail()

    data = {"mcp_tool_name": "search", "mcp_arguments": {"query": "quarterly revenue"}}

    result = await handler.process_input_messages(data, guardrail)

    assert "modified_arguments" not in result
    assert result["mcp_arguments"] == {"query": "quarterly revenue"}


@pytest.mark.asyncio
async def test_guardrail_returning_wrong_text_count_leaves_arguments_alone():
    """Write-back is positional, so a length mismatch must not scramble arguments."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail(texts_override=["only", "two", "texts"])

    arguments = {"query": "contact jane.doe@example.com about the invoice"}
    data = {"mcp_tool_name": "search", "mcp_arguments": arguments}

    result = await handler.process_input_messages(data, guardrail)

    assert "modified_arguments" not in result
    assert result["mcp_arguments"] == arguments


@pytest.mark.asyncio
async def test_deeply_nested_arguments_are_blocked_rather_than_skipped():
    """Arguments too deep to walk must block instead of passing unscanned."""
    handler = MCPGuardrailTranslationHandler()
    guardrail = MaskingGuardrail()

    nested: dict = {"leaf": "jane.doe@example.com"}
    for _ in range(DEFAULT_MAX_RECURSE_DEPTH + 1):
        nested = {"next": nested}

    data = {"mcp_tool_name": "search", "mcp_arguments": nested}

    with pytest.raises(HTTPException) as exc_info:
        await handler.process_input_messages(data, guardrail)

    assert exc_info.value.status_code == 400


@pytest.mark.parametrize("run_in_parallel", [False, True])
@pytest.mark.asyncio
async def test_masked_arguments_reach_the_outbound_mcp_call(restore_callbacks, run_in_parallel):
    """End to end over the real MCP pre-call path, not just the handler.

    Drives the same sequence mcp_server_manager.call_tool uses:
    synthetic payload -> pre_call_hook -> arguments sent upstream.

    Covers run_in_parallel both ways: that path shares one payload snapshot and
    discards whatever a guardrail returns, so the mask has to land on the caller's
    dict rather than on a copy of it.
    """
    guardrail = MaskingGuardrail(
        event_hook="pre_mcp_call",
        default_on=True,
        run_in_parallel=run_in_parallel,
    )
    litellm.callbacks = [guardrail]

    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    arguments = {"query": "contact jane.doe@example.com about the invoice"}
    pre_hook_kwargs = {
        "name": "search",
        "arguments": arguments,
        "server_name": "test-server",
        "user_api_key_auth": UserAPIKeyAuth(api_key="sk-test", user_id="test-user"),
    }

    request_obj = proxy_logging_obj._create_mcp_request_object_from_kwargs(pre_hook_kwargs)
    synthetic_data = proxy_logging_obj._convert_mcp_to_llm_format(request_obj, pre_hook_kwargs)

    modified_data = await proxy_logging_obj.pre_call_hook(
        user_api_key_dict=pre_hook_kwargs["user_api_key_auth"],
        data=synthetic_data,
        call_type="call_mcp_tool",
    )
    modified_kwargs = proxy_logging_obj._convert_mcp_hook_response_to_kwargs(modified_data, pre_hook_kwargs)

    assert modified_kwargs["arguments"] == {"query": "contact <EMAIL_ADDRESS> about the invoice"}
