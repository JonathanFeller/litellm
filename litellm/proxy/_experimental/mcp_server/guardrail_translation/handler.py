"""
MCP Guardrail Handler for Unified Guardrails.

Converts an MCP call_tool (name + arguments) into the OpenAI-compatible shape
apply_guardrail expects: the tool as a single-entry ``tools`` definition, and
every string leaf of the call arguments as ``texts`` so text guardrails can
detect and mask sensitive values in the payload. Works with the synthetic
request from ProxyLogging._convert_mcp_to_llm_format.

Note: For MCP tool definitions (schema) -> OpenAI tools=[], see
litellm.experimental_mcp_client.tools.transform_mcp_tool_to_openai_tool
when you have a full MCP Tool from list_tools. Here we only have the call
payload (name + arguments) so we just build the tool definition.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException
from mcp.types import Tool as MCPTool

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.experimental_mcp_client.tools import transform_mcp_tool_to_openai_tool
from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.types.llms.openai import (
    ChatCompletionToolParam,
    ChatCompletionToolParamFunctionChunk,
)
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from mcp.types import CallToolResult

    from litellm.integrations.custom_guardrail import CustomGuardrail


MCPArgumentValue = str | int | float | bool | None | dict[str, "MCPArgumentValue"] | list["MCPArgumentValue"]
MCPArgumentPath = tuple[str | int, ...]


def _too_deeply_nested() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "error": (
                "Content blocked: MCP tool call arguments exceed the maximum nesting depth of "
                f"{DEFAULT_MAX_RECURSE_DEPTH} and cannot be scanned by the configured guardrail"
            )
        },
    )


def _collect_argument_texts(
    value: MCPArgumentValue,
    path: MCPArgumentPath = (),
) -> tuple[tuple[MCPArgumentPath, str], ...]:
    """Depth-first, deterministically ordered string leaves of an MCP argument tree."""
    if len(path) > DEFAULT_MAX_RECURSE_DEPTH:
        raise _too_deeply_nested()
    if isinstance(value, str):
        return ((path, value),)
    if isinstance(value, dict):
        return tuple(leaf for key, item in value.items() for leaf in _collect_argument_texts(item, (*path, key)))
    if isinstance(value, list):
        return tuple(leaf for index, item in enumerate(value) for leaf in _collect_argument_texts(item, (*path, index)))
    return ()


def _argument_replacements(
    argument_texts: tuple[tuple[MCPArgumentPath, str], ...],
    masked_texts: Sequence[str] | None,
) -> Mapping[MCPArgumentPath, str]:
    """Positionally pair the guardrail's returned texts with the leaves they came from.

    Only leaves the guardrail actually rewrote are returned, so a guardrail that
    detects nothing leaves the outbound tool call byte-identical.
    """
    if masked_texts is None:
        return {}
    if len(masked_texts) != len(argument_texts):
        verbose_proxy_logger.warning(
            "MCP Guardrail: guardrail returned %d texts for %d tool call argument strings; leaving arguments unmasked",
            len(masked_texts),
            len(argument_texts),
        )
        return {}
    return {path: masked for (path, original), masked in zip(argument_texts, masked_texts) if masked != original}


def _replace_argument_texts(
    value: MCPArgumentValue,
    replacements: Mapping[MCPArgumentPath, str],
    path: MCPArgumentPath = (),
) -> MCPArgumentValue:
    """Rebuild an MCP argument tree with the guardrail's rewritten string leaves."""
    if isinstance(value, str):
        return replacements.get(path, value)
    if isinstance(value, dict):
        return {key: _replace_argument_texts(item, replacements, (*path, key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_argument_texts(item, replacements, (*path, index)) for index, item in enumerate(value)]
    return value


class MCPGuardrailTranslationHandler(BaseTranslation):
    """Guardrail translation handler for MCP tool calls (passes a single tool_call to guardrail)."""

    async def process_input_messages(
        self,
        data: Dict[str, Any],
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
    ) -> Dict[str, Any]:
        mcp_tool_name = data.get("mcp_tool_name") or data.get("name")
        mcp_arguments = data.get("mcp_arguments") or data.get("arguments")
        mcp_tool_description = data.get("mcp_tool_description") or data.get("description")
        if mcp_arguments is None or not isinstance(mcp_arguments, dict):
            mcp_arguments = {}

        if not mcp_tool_name:
            verbose_proxy_logger.debug("MCP Guardrail: mcp_tool_name missing")
            return data

        # Convert MCP input via transform_mcp_tool_to_openai_tool, then map to litellm
        # ChatCompletionToolParam (openai SDK type has incompatible strict/cache_control).
        mcp_tool = MCPTool(
            name=mcp_tool_name,
            description=mcp_tool_description or "",
            inputSchema={},  # Call payload has no schema; guardrail gets args from request_data
        )
        openai_tool = transform_mcp_tool_to_openai_tool(mcp_tool)
        fn = openai_tool["function"]
        tool_def: ChatCompletionToolParam = {
            "type": "function",
            "function": ChatCompletionToolParamFunctionChunk(
                name=fn["name"],
                description=fn.get("description") or "",
                parameters=fn.get("parameters")
                or {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=fn.get("strict", False) or False,  # Default to False if None
            ),
        }
        argument_texts = _collect_argument_texts(mcp_arguments)
        inputs: GenericGuardrailAPIInputs = GenericGuardrailAPIInputs(
            tools=[tool_def],
            texts=[text for _, text in argument_texts],
        )

        guarded = await guardrail_to_apply.apply_guardrail(
            inputs=inputs,
            request_data=data,
            input_type="request",
            logging_obj=litellm_logging_obj,
        )
        replacements = _argument_replacements(
            argument_texts=argument_texts,
            masked_texts=guarded.get("texts") if guarded is not None else None,
        )
        if not replacements:
            return data

        masked_arguments = _replace_argument_texts(mcp_arguments, replacements)
        data["mcp_arguments"] = masked_arguments
        data["modified_arguments"] = masked_arguments
        return data

    async def process_output_response(
        self,
        response: "CallToolResult",
        guardrail_to_apply: "CustomGuardrail",
        litellm_logging_obj: Optional[Any] = None,
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[dict] = None,
    ) -> Any:
        verbose_proxy_logger.debug(
            "MCP Guardrail: Output processing not implemented for MCP tools",
        )
        return response
