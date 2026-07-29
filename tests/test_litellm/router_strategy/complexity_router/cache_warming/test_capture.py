import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from litellm.router_strategy.complexity_router.cache_warming.capture import (
    _warn_payload_too_large,
    _warn_privacy_gate_blocked,
)
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    decompress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

LONG_SYSTEM = "All deployment manifests must declare resource ceilings before rollout. " * 200

SESSIONS_KEY = "{cache_warm:v1:smart-router}:sessions"


@pytest.fixture(autouse=True)
def _prompt_retention_consent(monkeypatch):
    monkeypatch.setenv("STORE_PROMPTS_IN_SPEND_LOGS", "true")


def anthropic_messages(**kwargs: object) -> None:
    raise AssertionError("marker function; never called")


def _complexity_router(redis: FakeRedisCache | None, **cache_warming_overrides: object) -> ComplexityRouter:
    router_instance = MagicMock()
    router_instance.cache = SimpleNamespace(redis_cache=redis)
    return ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=router_instance,
        complexity_router_config={
            "tiers": {"SIMPLE": "gpt-5-mini", "COMPLEX": "claude-sonnet-4-5"},
            "cache_warming": {"enabled": True, **cache_warming_overrides},
        },
    )


MESSAGES = [
    {"role": "system", "content": LONG_SYSTEM},
    {"role": "user", "content": "summarize rule 7"},
]


def _kwargs(**overrides: object) -> dict:
    base: dict = {
        "model": "smart-router",
        "metadata": {
            "session_id": "sess-1",
            "user_api_key_hash": "hash-1",
            "user_api_key": "hash-1",
            "user_api_key_team_id": "team-9",
        },
    }
    return {**base, **overrides}


def _stored_records(redis: FakeRedisCache) -> list[dict]:
    return [json.loads(value) for value in redis.hashes.get(SESSIONS_KEY, {}).values()]


@pytest.mark.asyncio
async def test_captures_whitelisted_fields_only_never_credentials():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(
        _kwargs(api_key="sk-live-secret", litellm_params={"api_key": "sk-live-secret"}), MESSAGES, "claude-sonnet-4-5"
    )
    records = _stored_records(redis)
    assert len(records) == 1
    payload = decompress_payload(records[0]["payload_compressed"])
    assert payload.model == "claude-sonnet-4-5"
    assert payload.call_surface == "chat_completions"
    assert records[0]["session_id"] == "sess-1"
    assert "sk-live-secret" not in json.dumps(records[0])


@pytest.mark.asyncio
async def test_disabled_warming_never_captures():
    redis = FakeRedisCache()
    router_instance = MagicMock()
    router_instance.cache = SimpleNamespace(redis_cache=redis)
    router = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=router_instance,
        complexity_router_config={"tiers": {"SIMPLE": "gpt-5-mini"}},
    )
    await router._capture_session(_kwargs(), MESSAGES, "gpt-5-mini")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ["metadata", "litellm_metadata"])
@pytest.mark.parametrize("marker", ["flag", "tag"])
async def test_skips_replay_marker_or_tag_in_either_metadata_slot(slot, marker):
    from litellm.router_strategy.complexity_router.cache_warming.types import CACHE_WARMING_REPLAY_TAG

    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    replay_entry = {CACHE_WARMING_REPLAY_MARKER_KEY: True} if marker == "flag" else {"tags": [CACHE_WARMING_REPLAY_TAG]}
    kwargs[slot] = {**kwargs.pop("metadata"), **replay_entry}
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_when_no_session_id():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    kwargs["metadata"].pop("session_id")
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "surface_signal",
    [
        {"litellm_logging_obj": SimpleNamespace(call_type="anthropic_messages")},
        {"original_generic_function": anthropic_messages},
    ],
)
async def test_captures_system_and_surface_for_anthropic_messages(surface_signal):
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs(system=LONG_SYSTEM, messages=[{"role": "user", "content": "hi there"}], **surface_signal)
    await router._capture_session(kwargs, kwargs["messages"], "claude-sonnet-4-5")
    records = _stored_records(redis)
    assert len(records) == 1
    payload = decompress_payload(records[0]["payload_compressed"])
    assert payload.call_surface == "anthropic_messages"
    assert payload.system == LONG_SYSTEM


@pytest.mark.asyncio
async def test_chat_surface_ignores_stray_system_kwarg():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(system=LONG_SYSTEM), MESSAGES, "claude-sonnet-4-5")
    payload = decompress_payload(_stored_records(redis)[0]["payload_compressed"])
    assert payload.call_surface == "chat_completions"
    assert payload.system is None


@pytest.mark.asyncio
async def test_skips_below_min_prompt_cache_tokens():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), [{"role": "user", "content": "tiny"}], "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_oversized_payload_and_warns_once():
    _warn_payload_too_large.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis, max_payload_bytes=64)
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_skips_highly_compressible_payload_on_uncompressed_bound():
    _warn_payload_too_large.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis, max_payload_bytes=1024)
    await router._capture_session(_kwargs(), [{"role": "user", "content": "a" * 20_000}], "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_captures_attribution_subset():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    record = _stored_records(redis)[0]
    assert record["attribution"]["user_api_key"] == "hash-1"
    assert record["attribution"]["user_api_key_team_id"] == "team-9"
    assert record["attribution"]["user_api_key_user_id"] is None


@pytest.mark.asyncio
async def test_capture_never_mutates_request_kwargs():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    snapshot = json.dumps(kwargs, sort_keys=True, default=str)
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    assert json.dumps(kwargs, sort_keys=True, default=str) == snapshot
    assert len(_stored_records(redis)) == 1


@pytest.mark.asyncio
async def test_swallows_store_exceptions():
    class ExplodingRedis(FakeRedisCache):
        async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
            raise RuntimeError("redis down")

    router = _complexity_router(ExplodingRedis())
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")


@pytest.mark.asyncio
async def test_second_turn_overwrites_payload_and_preserves_other_model_warmth():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    key = CacheWarmingStore.record_key("smart-router", "hash-1", "sess-1")
    first = json.loads(redis.hashes[SESSIONS_KEY][key])
    redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")] = json.dumps(123.0)
    await router._capture_session(
        _kwargs(), MESSAGES + [{"role": "user", "content": "and rule 8?"}], "claude-sonnet-4-5"
    )
    second = json.loads(redis.hashes[SESSIONS_KEY][key])
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "gpt-5-mini")]) == 123.0
    assert json.loads(redis.data[CacheWarmingStore.warmth_key(key, "claude-sonnet-4-5")]) > 0
    assert second["payload_sha256"] != first["payload_sha256"]


@pytest.mark.asyncio
async def test_missing_retention_consent_blocks_capture_and_warns(monkeypatch, caplog):
    monkeypatch.delenv("STORE_PROMPTS_IN_SPEND_LOGS", raising=False)
    _warn_privacy_gate_blocked.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}
    assert any("prompt retention is not permitted" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_global_message_redaction_blocks_capture():
    import litellm

    _warn_privacy_gate_blocked.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    previous = litellm.turn_off_message_logging
    litellm.turn_off_message_logging = True
    try:
        await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    finally:
        litellm.turn_off_message_logging = previous
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_per_request_redaction_header_blocks_capture():
    _warn_privacy_gate_blocked.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    kwargs = _kwargs()
    kwargs["metadata"]["headers"] = {"x-litellm-enable-message-redaction": True}
    await router._capture_session(kwargs, MESSAGES, "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_capture_errors_are_logged_never_raised(caplog):
    class ExplodingCacheRouter:
        @property
        def cache(self):
            raise RuntimeError("cache exploded")

    router = _complexity_router(FakeRedisCache())
    router.litellm_router_instance = ExplodingCacheRouter()
    with caplog.at_level(logging.ERROR, logger="LiteLLM Router"):
        await router._capture_session(_kwargs(), MESSAGES, "claude-sonnet-4-5")
    assert any("capture failed" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_classify_path_captures_routed_session():
    redis = FakeRedisCache()
    router_instance = MagicMock()
    router_instance.cache = SimpleNamespace(redis_cache=redis)
    router = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=router_instance,
        complexity_router_config={
            "tiers": {"SIMPLE": "gpt-5-mini", "COMPLEX": "claude-sonnet-4-5"},
            "session_affinity": False,
            "cache_warming": {"enabled": True},
        },
    )
    kwargs = _kwargs()
    response = await router.async_pre_routing_hook(model="smart-router", request_kwargs=kwargs, messages=MESSAGES)
    assert response is not None
    records = _stored_records(redis)
    assert len(records) == 1
    assert records[0]["served_model"] == response.model


@pytest.mark.asyncio
async def test_replay_shaped_call_through_tier_group_is_never_captured():
    from litellm import Router

    redis = FakeRedisCache()
    llm_router = Router(
        model_list=[
            {
                "model_name": "smart-router",
                "litellm_params": {
                    "model": "auto_router/complexity_router",
                    "complexity_router_config": {
                        "tiers": {"SIMPLE": "gpt-5-mini"},
                        "session_affinity": False,
                        "cache_warming": {"enabled": True},
                    },
                },
            },
            {"model_name": "gpt-5-mini", "litellm_params": {"model": "gpt-5-mini", "api_key": "test"}},
        ]
    )
    try:
        llm_router.cache.redis_cache = redis
        replay_shaped = _kwargs(model="gpt-5-mini")
        await llm_router.async_pre_routing_hook(model="gpt-5-mini", request_kwargs=replay_shaped, messages=MESSAGES)
        assert redis.hashes == {}
        alias_kwargs = _kwargs()
        await llm_router.async_pre_routing_hook(model="smart-router", request_kwargs=alias_kwargs, messages=MESSAGES)
        assert len(redis.hashes.get(SESSIONS_KEY, {})) == 1
    finally:
        llm_router.set_model_list([])


@pytest.mark.asyncio
async def test_affinity_pin_path_captures_routed_session():
    from litellm.caching.dual_cache import DualCache

    redis = FakeRedisCache()
    router_instance = MagicMock()
    router_instance.cache = DualCache()
    router_instance.cache.redis_cache = redis
    router = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=router_instance,
        complexity_router_config={
            "tiers": {"SIMPLE": "gpt-5-mini", "COMPLEX": "claude-sonnet-4-5"},
            "session_affinity": True,
            "cache_warming": {"enabled": True},
        },
    )
    kwargs = _kwargs()
    pin_key = router._get_session_affinity_cache_key("sess-1", kwargs)
    await router_instance.cache.in_memory_cache.async_set_cache(key=pin_key, value="claude-sonnet-4-5")
    response = await router.async_pre_routing_hook(model="smart-router", request_kwargs=kwargs, messages=MESSAGES)
    assert response is not None and response.model == "claude-sonnet-4-5"
    records = _stored_records(redis)
    assert len(records) == 1
    assert records[0]["served_model"] == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_empty_messages_never_captures():
    redis = FakeRedisCache()
    router = _complexity_router(redis)
    await router._capture_session(_kwargs(), [], "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}


@pytest.mark.asyncio
async def test_incompressible_payload_over_compressed_bound_is_skipped():
    import os

    _warn_payload_too_large.cache_clear()
    redis = FakeRedisCache()
    router = _complexity_router(redis, max_payload_bytes=1024)
    incompressible = os.urandom(3000).hex()
    await router._capture_session(_kwargs(), [{"role": "user", "content": incompressible}], "claude-sonnet-4-5")
    assert redis.hashes.get(SESSIONS_KEY, {}) == {}
