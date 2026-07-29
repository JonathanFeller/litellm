import asyncio
import logging
import json
import time
from types import SimpleNamespace

import pytest

from litellm.constants import CACHE_WARMING_JOB_NAME
from litellm.router_strategy.complexity_router.cache_warming.eligibility import resolve_warm_models
from litellm.router_strategy.complexity_router.cache_warming.refresher import (
    CacheWarmingRefresher,
    collect_warming_enabled_complexity_routers,
    filter_cache_warmable,
)
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_RECORD_SCHEMA_VERSION,
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingAttribution,
    CacheWarmingPayload,
    CacheWarmingRecord,
    compress_payload,
)
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.config import ComplexityRouterConfig
from litellm.types.router import TaggedPreRoutingStrategy

from tests.test_litellm.router_strategy.complexity_router.cache_warming.test_store import FakeRedisCache

_DEFAULT_DEPLOYMENTS = {
    "fast-claude": [{"litellm_params": {"model": "anthropic/claude-haiku-4-5"}}],
    "smart-claude": [{"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}}],
    "fast-gpt": [{"litellm_params": {"model": "gpt-5-mini"}}],
    "bedrock-sonnet": [{"litellm_params": {"model": "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0"}}],
    "no-model": [{"litellm_params": {}}],
    "titan": [{"litellm_params": {"model": "bedrock/amazon.titan-text-express-v1"}}],
    "mystery": [{"litellm_params": {"model": "totally-unknown-model-xyz"}}],
}


class FakeLLMRouter:
    def __init__(
        self,
        redis: FakeRedisCache | None = None,
        enable_tag_filtering: bool = False,
        deployments: dict | None = None,
        replay_delay: float = 0.0,
    ) -> None:
        self.complexity_routers: dict = {}
        self.optional_callbacks: list = []
        self.enable_tag_filtering = enable_tag_filtering
        self.cache = SimpleNamespace(redis_cache=redis)
        self.completion_calls: list[dict] = []
        self.anthropic_calls: list[dict] = []
        self.max_concurrent = 0
        self._in_flight = 0
        self.replay_delay = replay_delay
        self.completion_response: object = None
        self.failing_message_marker: str | None = None
        self._deployments = deployments if deployments is not None else _DEFAULT_DEPLOYMENTS

    def get_model_list(self, model_name: str | None = None, team_id: str | None = None):
        return self._deployments.get(model_name)

    async def acompletion(self, **kwargs: object):
        marker = self.failing_message_marker
        if marker is not None and marker in json.dumps(kwargs.get("messages")):
            raise RuntimeError("provider down")
        self._in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self._in_flight)
        await asyncio.sleep(self.replay_delay)
        self._in_flight -= 1
        self.completion_calls.append(kwargs)
        return self.completion_response

    async def aanthropic_messages(self, **kwargs: object):
        self.anthropic_calls.append(kwargs)


class FakePodLockManager:
    def __init__(self, acquire_result: bool | None = True, redis_cache: object = "attached") -> None:
        self.acquire_result = acquire_result
        if redis_cache == "attached":

            async def _renew(keys: list, args: list) -> int:
                return 1

            redis_cache = SimpleNamespace(async_register_script=lambda _script: _renew)
        self.redis_cache = redis_cache
        self.pod_id = "fake-pod"
        self.acquire_calls: list[tuple[str, int | None]] = []
        self.release_calls: list[str] = []

    async def acquire_lock(self, cronjob_id: str, ttl: int | None = None) -> bool | None:
        self.acquire_calls.append((cronjob_id, ttl))
        return self.acquire_result

    async def release_lock(self, cronjob_id: str) -> None:
        self.release_calls.append(cronjob_id)


class FakePrismaClient:
    def __init__(self, rows: tuple = (), raise_error: bool = False, unlimited: bool = False) -> None:
        self.queries: list[dict] = []

        async def find_many(where: dict):
            self.queries.append(where)
            if raise_error:
                raise RuntimeError("db down")
            if unlimited:
                return [
                    SimpleNamespace(token=token, spend=0.0, max_budget=None, blocked=None, expires=None)
                    for token in where["token"]["in"]
                ]
            return [row for row in rows if row.token in where["token"]["in"]]

        self.db = SimpleNamespace(litellm_verificationtoken=SimpleNamespace(find_many=find_many))


def _warming_rig(
    redis: FakeRedisCache | None = None,
    enable_tag_filtering: bool = False,
    replay_delay: float = 0.0,
    **cache_warming_overrides: object,
) -> tuple[FakeLLMRouter, FakeRedisCache | None]:
    llm_router = FakeLLMRouter(redis=redis, enable_tag_filtering=enable_tag_filtering, replay_delay=replay_delay)
    strategy = ComplexityRouter(
        model_name="smart-router",
        litellm_router_instance=llm_router,
        complexity_router_config={
            "tiers": {"SIMPLE": ["fast-claude"], "COMPLEX": ["smart-claude"]},
            "cache_warming": {"enabled": True, **cache_warming_overrides},
        },
    )
    llm_router.complexity_routers = {"smart-router": [TaggedPreRoutingStrategy(tags=(), strategy=strategy)]}
    return llm_router, redis


def _seed_session(
    redis: FakeRedisCache,
    session_id: str = "sess-1",
    caller_scope: str = "hash-1",
    served_model: str = "fast-claude",
    last_activity: float | None = None,
    warmth: dict | None = None,
    user_api_key: str | None = "hash-1",
    call_surface: str = "chat_completions",
    content: str = "summarize the deployment policy",
    tools: tuple | None = None,
    tool_choice: object = None,
) -> str:
    payload = CacheWarmingPayload(
        model=served_model,
        messages=({"role": "user", "content": content},),
        system="You are a policy assistant" if call_surface == "anthropic_messages" else None,
        tools=tools,
        tool_choice=tool_choice,
        call_surface=call_surface,
    )
    blob, sha = compress_payload(payload)
    record = CacheWarmingRecord(
        schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION,
        payload_compressed=blob,
        payload_sha256=sha,
        token_estimate=2048,
        last_activity=last_activity if last_activity is not None else time.time(),
        served_model=served_model,
        session_id=session_id,
        attribution=CacheWarmingAttribution(user_api_key=user_api_key),
        auto_router_model_name="smart-router",
    )
    store = CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router")
    key = CacheWarmingStore.record_key("smart-router", caller_scope, session_id)
    redis.hashes.setdefault(store.sessions_key(), {})[key] = json.dumps(record.model_dump())
    redis.zsets.setdefault(store.index_key(), {})[key] = time.time() + 3600
    for model_group, stamp in (warmth or {}).items():
        redis.data[CacheWarmingStore.warmth_key(key, model_group)] = json.dumps(stamp)
    return key


def _warmth_stamp(redis: FakeRedisCache, record_key: str, model_group: str) -> float | None:
    raw = redis.data.get(CacheWarmingStore.warmth_key(record_key, model_group))
    return json.loads(raw) if raw is not None else None


_VERIFIABLE = object()


async def _tick(llm_router, lock=None, prisma=_VERIFIABLE, refresher: CacheWarmingRefresher | None = None):
    prisma_client = FakePrismaClient(unlimited=True) if prisma is _VERIFIABLE else prisma
    await (refresher or CacheWarmingRefresher()).run_tick(
        llm_router=llm_router, pod_lock_manager=lock, prisma_client=prisma_client
    )


def _replayed_models(llm_router: FakeLLMRouter) -> list:
    return [call["model"] for call in llm_router.completion_calls]


# ---------------------------------------------------------------------------
# warm-set resolution + eligibility
# ---------------------------------------------------------------------------


def test_resolve_warm_models_defaults_to_first_member_per_tier_deduped():
    config = ComplexityRouterConfig(
        tiers={"SIMPLE": ["fast-claude", "fast-gpt"], "MEDIUM": "fast-claude", "COMPLEX": ["smart-claude"]}
    )
    assert resolve_warm_models(config) == ("fast-claude", "smart-claude")


def test_resolve_warm_models_prefers_explicit_list():
    config = ComplexityRouterConfig(
        tiers={"SIMPLE": ["fast-claude"], "COMPLEX": ["smart-claude"]},
        cache_warming={"enabled": True, "warm_models": ["smart-claude", "smart-claude", "fast-claude"]},
    )
    assert resolve_warm_models(config) == ("smart-claude", "fast-claude")


def test_filter_cache_warmable_keeps_only_prompt_cacheable_anthropic_bedrock():
    llm_router = FakeLLMRouter()
    groups = ["fast-claude", "smart-claude", "bedrock-sonnet", "fast-gpt", "titan", "mystery", "no-model", "absent-group"]
    assert filter_cache_warmable(llm_router, groups) == ("fast-claude", "smart-claude", "bedrock-sonnet")


def test_filter_cache_warmable_prefers_declared_custom_llm_provider_over_inference():
    deployments = {
        "declared-openai": [{"litellm_params": {"model": "anthropic/claude-sonnet-4-5", "custom_llm_provider": "openai"}}]
    }
    assert filter_cache_warmable(FakeLLMRouter(deployments=deployments), ["declared-openai"]) == ()


def test_collect_warming_enabled_complexity_routers_skips_disabled():
    llm_router, _ = _warming_rig(redis=FakeRedisCache())
    disabled = ComplexityRouter(
        model_name="plain-router",
        litellm_router_instance=llm_router,
        complexity_router_config={"tiers": {"SIMPLE": ["fast-claude"]}},
    )
    llm_router.complexity_routers["plain-router"] = [TaggedPreRoutingStrategy(tags=(), strategy=disabled)]
    collected = collect_warming_enabled_complexity_routers(llm_router)
    assert [strategy.model_name for strategy in collected] == ["smart-router"]


# ---------------------------------------------------------------------------
# pod lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_held_by_other_pod_skips_tick_and_never_releases():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(acquire_result=False)
    await _tick(llm_router, lock=lock)
    assert llm_router.completion_calls == []
    assert lock.release_calls == []


@pytest.mark.asyncio
async def test_lock_acquired_warms_and_releases():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(acquire_result=True)
    await _tick(llm_router, lock=lock)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert lock.acquire_calls == [(CACHE_WARMING_JOB_NAME, 60)]
    assert lock.release_calls == [CACHE_WARMING_JOB_NAME]


@pytest.mark.asyncio
async def test_lock_falls_back_to_warming_redis_when_no_injected_manager():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    await _tick(llm_router, lock=None)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert any(key.startswith("cronjob_lock:") for key in redis.ttls)
    assert not any(key.startswith("cronjob_lock:") for key in redis.data)


@pytest.mark.asyncio
async def test_redisless_injected_manager_is_replaced_by_warming_redis_lock():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis)
    lock = FakePodLockManager(redis_cache=None)
    await _tick(llm_router, lock=lock)
    assert lock.acquire_calls == []
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert any(key.startswith("cronjob_lock:") for key in redis.ttls)
    assert not any(key.startswith("cronjob_lock:") for key in redis.data)


def test_fallback_lock_manager_is_stable_across_ticks():
    redis = FakeRedisCache()
    refresher = CacheWarmingRefresher()
    first = refresher._resolve_lock_manager(None, redis)
    second = refresher._resolve_lock_manager(None, redis)
    assert first is second
    assert first.redis_cache is redis


@pytest.mark.asyncio
async def test_lock_released_when_tick_raises():
    class ExplodingScriptRedis(FakeRedisCache):
        def async_register_script(self, script: str):
            async def boom(keys: list, args: list):
                raise RuntimeError("redis down")

            return boom

    llm_router, _ = _warming_rig(redis=ExplodingScriptRedis())
    lock = FakePodLockManager(acquire_result=True)
    with pytest.raises(RuntimeError, match="redis down"):
        await _tick(llm_router, lock=lock)
    assert lock.release_calls == [CACHE_WARMING_JOB_NAME]


@pytest.mark.asyncio
async def test_no_warming_routers_never_touches_lock():
    llm_router = FakeLLMRouter(redis=FakeRedisCache())
    lock = FakePodLockManager()
    await _tick(llm_router, lock=lock)
    assert lock.acquire_calls == []


# ---------------------------------------------------------------------------
# session selection: idle skip + interval pacing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_idle_session_not_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    key = _seed_session(redis, last_activity=time.time() - 601)
    sessions_key = CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router").sessions_key()
    before = redis.hashes[sessions_key][key]
    await _tick(llm_router)
    assert llm_router.completion_calls == []
    assert redis.hashes[sessions_key][key] == before


@pytest.mark.asyncio
async def test_recently_warmed_model_not_replayed_again():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    _seed_session(redis, warmth={"fast-claude": now - 10, "smart-claude": now - 300})
    await _tick(llm_router)
    assert _replayed_models(llm_router) == ["smart-claude"]


@pytest.mark.asyncio
async def test_no_redis_store_is_noop():
    llm_router, _ = _warming_rig(redis=None)
    await _tick(llm_router, lock=FakePodLockManager())
    assert llm_router.completion_calls == []


# ---------------------------------------------------------------------------
# replay kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_replay_kwargs_exact():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    assert llm_router.completion_calls == [
        {
            "model": "smart-claude",
            "messages": [{"role": "user", "content": "summarize the deployment policy"}],
            "tools": None,
            "tool_choice": None,
            "max_tokens": 1,
            "stream": False,
            "cache": {"no-cache": True},
            "metadata": {
                CACHE_WARMING_REPLAY_MARKER_KEY: True,
                "session_id": "sess-1",
                "user_api_key": "hash-1",
                "tags": [CACHE_WARMING_REPLAY_TAG],
            },
        }
    ]


@pytest.mark.asyncio
async def test_anthropic_surface_replays_via_aanthropic_messages_with_system():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(
        redis,
        call_surface="anthropic_messages",
        warmth={"fast-claude": time.time()},
        tool_choice={"type": "auto"},
    )
    await _tick(llm_router)
    assert llm_router.completion_calls == []
    assert len(llm_router.anthropic_calls) == 1
    call = llm_router.anthropic_calls[0]
    assert call["model"] == "smart-claude"
    assert call["system"] == "You are a policy assistant"
    assert call["tool_choice"] == {"type": "auto"}
    assert call["max_tokens"] == 1
    assert call["stream"] is False
    assert call["litellm_metadata"][CACHE_WARMING_REPLAY_MARKER_KEY] is True
    assert "metadata" not in call


@pytest.mark.asyncio
async def test_tags_omitted_when_router_tag_filtering_enabled():
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), enable_tag_filtering=True)
    _seed_session(redis, warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    metadata = llm_router.completion_calls[0]["metadata"]
    assert "tags" not in metadata
    assert metadata[CACHE_WARMING_REPLAY_MARKER_KEY] is True


# ---------------------------------------------------------------------------
# failure isolation + attempt timestamping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_session_failure_does_not_block_other_sessions():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.failing_message_marker = "POISON"
    _seed_session(redis, session_id="sess-bad", content="POISON payload")
    _seed_session(redis, session_id="sess-good")
    await _tick(llm_router)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert all("POISON" not in json.dumps(call) for call in llm_router.completion_calls)


@pytest.mark.asyncio
async def test_failed_replay_still_stamps_warm_attempt():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.failing_message_marker = "POISON"
    key = _seed_session(redis, content="POISON payload", warmth={"fast-claude": time.time()})
    await _tick(llm_router)
    stamp = _warmth_stamp(redis, key, "smart-claude")
    assert stamp is not None and stamp > 0


@pytest.mark.asyncio
async def test_successful_replay_stamps_warm_attempt_for_replayed_model_only():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    key = _seed_session(redis, warmth={"fast-claude": now})
    await _tick(llm_router)
    smart_stamp = _warmth_stamp(redis, key, "smart-claude")
    assert smart_stamp is not None and smart_stamp >= now
    assert _warmth_stamp(redis, key, "fast-claude") == pytest.approx(now)


# ---------------------------------------------------------------------------
# budget stop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_boundary_blocks_replays_and_allows_under_budget():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-broke", user_api_key="broke-key")
    _seed_session(redis, session_id="sess-rich", caller_scope="hash-2", user_api_key="rich-key")
    prisma = FakePrismaClient(
        rows=(
            SimpleNamespace(token="broke-key", spend=99.9, max_budget=100.0),
            SimpleNamespace(token="rich-key", spend=10.0, max_budget=100.0),
        )
    )
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: (0.0001, 0.0000001))
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert sorted(prisma.queries[0]["token"]["in"]) == ["broke-key", "rich-key"]
    replayed_keys = {call["metadata"]["user_api_key"] for call in llm_router.completion_calls}
    assert replayed_keys == {"rich-key"}


@pytest.mark.asyncio
async def test_no_prisma_fails_closed_for_key_attributed_sessions():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="some-key")
    await _tick(llm_router, prisma=None)
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_key_state_query_error_skips_tick():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="some-key")
    await _tick(llm_router, prisma=FakePrismaClient(raise_error=True))
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_healthy_key_state_query_leaves_behavior_unchanged():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="live-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="live-key", spend=1.0, max_budget=100.0),))
    await _tick(llm_router, prisma=prisma)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


@pytest.mark.asyncio
async def test_burst_stops_exactly_at_budget_boundary_via_ledger():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    for i in range(3):
        _seed_session(redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", user_api_key="shared-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="shared-key", spend=3.0, max_budget=10.0),))
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: (0.001, 0.0000001))
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 3
    assert refresher.ledger.reserved("shared-key", 3600) == pytest.approx(3 * 2.048)


@pytest.mark.asyncio
async def test_actual_response_cost_accumulates_in_ledger():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.completion_response = SimpleNamespace(_hidden_params={"response_cost": 5.0})
    _seed_session(redis, user_api_key="metered-key", warmth={"fast-claude": time.time()})
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="metered-key", spend=0.0, max_budget=100.0),))
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: (0.001, 0.0000001))
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 1
    assert refresher.ledger.reserved("metered-key", 3600) == pytest.approx(5.0)


def test_ledger_entries_expire_after_ttl():
    from litellm.router_strategy.complexity_router.cache_warming.refresher import WarmingBudgetLedger

    clock = {"now": 1000.0}
    ledger = WarmingBudgetLedger(clock=lambda: clock["now"])
    ledger.reserve("key", 2.0)
    clock["now"] = 1050.0
    assert ledger.reserved("key", 100.0) == pytest.approx(2.0)
    clock["now"] = 1150.0
    assert ledger.reserved("key", 100.0) == 0.0


@pytest.mark.asyncio
async def test_blocked_and_expired_keys_are_not_warmed():
    from datetime import datetime, timedelta, timezone

    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-blocked", caller_scope="hash-1", user_api_key="blocked-key")
    _seed_session(redis, session_id="sess-expired", caller_scope="hash-2", user_api_key="expired-key")
    _seed_session(redis, session_id="sess-live", caller_scope="hash-3", user_api_key="live-key")
    prisma = FakePrismaClient(
        rows=(
            SimpleNamespace(token="blocked-key", spend=0.0, max_budget=None, blocked=True, expires=None),
            SimpleNamespace(
                token="expired-key",
                spend=0.0,
                max_budget=None,
                blocked=None,
                expires=datetime.now(timezone.utc) - timedelta(hours=1),
            ),
            SimpleNamespace(
                token="live-key",
                spend=0.0,
                max_budget=None,
                blocked=False,
                expires=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
        )
    )
    await _tick(llm_router, prisma=prisma)
    replayed_keys = {call["metadata"]["user_api_key"] for call in llm_router.completion_calls}
    assert replayed_keys == {"live-key"}


@pytest.mark.asyncio
async def test_deleted_key_sessions_are_not_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-ghost", user_api_key="ghost-key")
    await _tick(llm_router, prisma=FakePrismaClient(rows=()))
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_master_key_sessions_warm_without_a_token_row():
    from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS

    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key=LITELLM_PROXY_MASTER_KEY_ALIAS)
    await _tick(llm_router, prisma=FakePrismaClient(rows=()))
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


@pytest.mark.asyncio
async def test_key_without_max_budget_is_warmed():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="unlimited-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="unlimited-key", spend=10_000.0, max_budget=None),))
    await _tick(llm_router, prisma=prisma)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]


# ---------------------------------------------------------------------------
# concurrency bound
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_replays_bounded_by_semaphore():
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), replay_delay=0.02)
    now = time.time()
    for i in range(6):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", warmth={"fast-claude": now}
        )
    await _tick(llm_router, refresher=CacheWarmingRefresher(max_concurrent_replays=2))
    assert len(llm_router.completion_calls) == 6
    assert llm_router.max_concurrent == 2


@pytest.mark.asyncio
async def test_long_batch_renews_pod_lock_across_ttl():
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), replay_delay=0.4)
    now = time.time()
    for i in range(2):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", warmth={"fast-claude": now}, user_api_key=None
        )
    refresher = CacheWarmingRefresher(max_concurrent_replays=1, lock_ttl_seconds=0.5)
    await _tick(llm_router, refresher=refresher)
    assert len(llm_router.completion_calls) == 2
    lock_keys = [key for key in redis.expire_calls if key.startswith("cronjob_lock:")]
    assert len(lock_keys) >= 1
    assert not any(key.startswith("cronjob_lock:") for key in redis.data)


@pytest.mark.asyncio
@pytest.mark.parametrize("prisma", [None, "raise"])
async def test_unverifiable_key_state_skips_the_whole_tick_including_unattributed_sessions(prisma):
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="sess-keyed", user_api_key="some-key")
    _seed_session(redis, session_id="sess-anon", caller_scope="hash-2", user_api_key=None)
    prisma_client = FakePrismaClient(raise_error=True) if prisma == "raise" else None
    await _tick(llm_router, prisma=prisma_client)
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_unpriced_model_skips_replay_for_metered_key():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="metered-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="metered-key", spend=0.0, max_budget=100.0),))
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: None)
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert llm_router.completion_calls == []


@pytest.mark.asyncio
async def test_unpriced_model_still_replays_for_unlimited_key():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="unlimited-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="unlimited-key", spend=0.0, max_budget=None),))
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: None)
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) >= 1


def test_ledger_reconcile_targets_exactly_the_reserved_entry():
    from litellm.router_strategy.complexity_router.cache_warming.refresher import WarmingBudgetLedger

    ledger = WarmingBudgetLedger()
    ledger.reconcile("ghost-key", 99, 1.0)
    assert ledger.reserved("ghost-key", 3600) == 0.0
    first = ledger.reserve("key", 2.0)
    ledger.reconcile("key", 999, 1.0)
    assert ledger.reserved("key", 3600) == pytest.approx(2.0)
    second = ledger.reserve("key", 3.0)
    ledger.reconcile("key", first, 0.5)
    assert ledger.reserved("key", 3600) == pytest.approx(0.5 + 3.0)
    ledger.reconcile("key", second, None)
    assert ledger.reserved("key", 3600) == pytest.approx(0.5 + 3.0)


def test_group_price_takes_max_across_members_and_none_when_any_unpriced():
    from litellm.router_strategy.complexity_router.cache_warming.refresher import _group_replay_prices
    from litellm.utils import get_model_info

    mixed_priced = {
        "mixed-claude": [
            {"litellm_params": {"model": "anthropic/claude-haiku-4-5"}},
            {"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        ]
    }
    llm_router = FakeLLMRouter(deployments=mixed_priced)
    sonnet = get_model_info(model="anthropic/claude-sonnet-4-5", custom_llm_provider="anthropic")
    expected_input = sonnet.get("cache_read_input_token_cost") or sonnet.get("input_cost_per_token")
    expected_output = max(
        rate
        for rate in (sonnet.get("output_cost_per_token"), sonnet.get("output_cost_per_reasoning_token"))
        if isinstance(rate, (int, float)) and rate > 0
    )
    assert _group_replay_prices(llm_router, "mixed-claude") == (
        pytest.approx(expected_input),
        pytest.approx(expected_output),
    )

    with_unpriced = {
        "half-known": [
            {"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
            {"litellm_params": {"model": "totally-unknown-model-xyz"}},
        ]
    }
    assert _group_replay_prices(FakeLLMRouter(deployments=with_unpriced), "half-known") is None
    assert _group_replay_prices(FakeLLMRouter(deployments={}), "absent") is None
    off_cost_map = {"tuned": [{"litellm_params": {"model": "my-custom-tune", "custom_llm_provider": "anthropic"}}]}
    assert _group_replay_prices(FakeLLMRouter(deployments=off_cost_map), "tuned") is None


@pytest.mark.asyncio
async def test_reconcile_down_frees_headroom_within_the_same_tick():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.completion_response = SimpleNamespace(_hidden_params={"response_cost": 0.01})
    now = time.time()
    for i in range(2):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", user_api_key="tight-key",
            warmth={"smart-claude": now},
        )
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="tight-key", spend=0.0, max_budget=3.0),))
    refresher = CacheWarmingRefresher(
        max_concurrent_replays=1, price_resolver=lambda _router, _group: (0.001, 0.0000001)
    )
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 2
    assert refresher.ledger.reserved("tight-key", 3600) == pytest.approx(0.02)


@pytest.mark.asyncio
async def test_unpriceable_actual_keeps_the_reservation():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="metered-key", warmth={"fast-claude": time.time()})
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="metered-key", spend=0.0, max_budget=100.0),))
    refresher = CacheWarmingRefresher(price_resolver=lambda _router, _group: (0.001, 0.0000001))
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 1
    assert refresher.ledger.reserved("metered-key", 3600) == pytest.approx(2.048)


@pytest.mark.asyncio
async def test_key_at_rpm_limit_gets_no_replays_and_headroom_is_respected():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    for i in range(3):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", user_api_key="paced-key",
            warmth={"smart-claude": now},
        )
    prisma = FakePrismaClient(
        rows=(SimpleNamespace(token="paced-key", spend=0.0, max_budget=None, rpm_limit=2, tpm_limit=None),)
    )
    refresher = CacheWarmingRefresher(max_concurrent_replays=1)
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 2

    zero_router, zero_redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(zero_redis, user_api_key="capped-key", warmth={"smart-claude": now})
    capped = FakePrismaClient(
        rows=(SimpleNamespace(token="capped-key", spend=0.0, max_budget=None, rpm_limit=0, tpm_limit=None),)
    )
    await _tick(zero_router, prisma=capped)
    assert zero_router.completion_calls == []


@pytest.mark.asyncio
async def test_key_tpm_limit_bounds_warming_tokens():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    now = time.time()
    for i in range(2):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", user_api_key="tpm-key",
            warmth={"smart-claude": now},
        )
    prisma = FakePrismaClient(
        rows=(SimpleNamespace(token="tpm-key", spend=0.0, max_budget=None, rpm_limit=None, tpm_limit=3000),)
    )
    refresher = CacheWarmingRefresher(max_concurrent_replays=1)
    await _tick(llm_router, prisma=prisma, refresher=refresher)
    assert len(llm_router.completion_calls) == 1


@pytest.mark.asyncio
async def test_lost_lease_stops_new_admissions_but_finishes_in_flight():
    class LostLeaseRedis(FakeRedisCache):
        def async_register_script(self, script: str):
            if 'redis.call("expire"' in script:

                async def deny(keys: list, args: list) -> int:
                    return 0

                return deny
            return super().async_register_script(script)

    llm_router, redis = _warming_rig(redis=LostLeaseRedis(), replay_delay=0.15)
    now = time.time()
    for i in range(4):
        _seed_session(
            redis, session_id=f"sess-{i}", caller_scope=f"hash-{i}", user_api_key=None,
            warmth={"smart-claude": now},
        )
    refresher = CacheWarmingRefresher(max_concurrent_replays=1, lock_ttl_seconds=0.4)
    await _tick(llm_router, refresher=refresher)
    assert 1 <= len(llm_router.completion_calls) < 4


@pytest.mark.asyncio
async def test_tick_without_sessions_or_warmable_models_is_a_noop():
    llm_router, _ = _warming_rig(redis=FakeRedisCache())
    await _tick(llm_router)
    assert llm_router.completion_calls == []

    unwarmable_router, unwarmable_redis = _warming_rig(redis=FakeRedisCache())
    unwarmable_router.complexity_routers["smart-router"][0].strategy.config.tiers = {"SIMPLE": ["fast-gpt"]}
    _seed_session(unwarmable_redis)
    await _tick(unwarmable_router)
    assert unwarmable_router.completion_calls == []


@pytest.mark.asyncio
async def test_session_cap_debug_path_still_warms(caplog):
    llm_router, redis = _warming_rig(redis=FakeRedisCache(), max_sessions=1)
    _seed_session(redis)
    with caplog.at_level(logging.DEBUG, logger="LiteLLM Router"):
        await _tick(llm_router)
    assert sorted(_replayed_models(llm_router)) == ["fast-claude", "smart-claude"]
    assert any("max_sessions cap" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_heartbeat_without_redis_is_a_noop():
    refresher = CacheWarmingRefresher()
    await refresher._hold_lock_lease(SimpleNamespace(redis_cache=None), asyncio.Event())


def test_response_cost_reads_dict_hidden_params_and_priced_model_response():
    import litellm
    from litellm.router_strategy.complexity_router.cache_warming.refresher import _response_cost

    assert _response_cost({"_hidden_params": {"response_cost": 1.5}}) == pytest.approx(1.5)
    priced = litellm.ModelResponse(
        model="gpt-5-mini",
        usage={"prompt_tokens": 1000, "completion_tokens": 10, "total_tokens": 1010},
    )
    computed = _response_cost(priced)
    assert computed is not None and computed > 0


def test_min_prompt_cache_tokens_defaults_for_empty_warm_set():
    from litellm.constants import DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT
    from litellm.router_strategy.complexity_router.cache_warming.eligibility import (
        min_prompt_cache_tokens_for_warm_set,
    )

    assert min_prompt_cache_tokens_for_warm_set(()) == DEFAULT_MINIMUM_PROMPT_CACHE_TOKEN_COUNT


def test_request_metadata_absent_key_hash_is_none():
    from litellm.router_strategy.complexity_router.request_metadata import (
        get_user_api_key_hash_from_request_kwargs,
    )

    assert get_user_api_key_hash_from_request_kwargs({}) is None


def test_mixed_group_with_any_ineligible_member_is_not_warmed():
    deployments = {
        "mixed": [
            {"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
            {"litellm_params": {"model": "openai/gpt-5-mini"}},
        ]
    }
    assert filter_cache_warmable(FakeLLMRouter(deployments=deployments), ["mixed"]) == ()


def _session_affinity_check(model_group_affinity_config: dict | None = None, session_mode: bool = True):
    from litellm.caching.dual_cache import DualCache
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

    return DeploymentAffinityCheck(
        cache=DualCache(),
        ttl_seconds=60,
        enable_user_key_affinity=False,
        enable_responses_api_affinity=False,
        enable_session_id_affinity=session_mode,
        model_group_affinity_config=model_group_affinity_config or {},
    )


_UNIFORM_POOL = {
    "uniform": [
        {"litellm_params": {"model": "anthropic/claude-sonnet-4-5"}},
        {"litellm_params": {"model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"}},
    ]
}


def test_all_eligible_multi_deployment_group_is_warmed_when_session_affinity_active():
    llm_router = FakeLLMRouter(deployments=_UNIFORM_POOL)
    llm_router.optional_callbacks = [_session_affinity_check()]
    assert filter_cache_warmable(llm_router, ["uniform"]) == ("uniform",)


def test_multi_deployment_group_without_affinity_is_skipped_and_warns_once():
    from litellm.router_strategy.complexity_router.cache_warming.refresher import (
        _warn_multi_deployment_group_needs_affinity,
    )

    _warn_multi_deployment_group_needs_affinity.cache_clear()
    llm_router = FakeLLMRouter(deployments=_UNIFORM_POOL)
    import logging as logging_module

    records = []

    class _Capture(logging_module.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Capture()
    logging_module.getLogger("LiteLLM Router").addHandler(handler)
    try:
        assert filter_cache_warmable(llm_router, ["uniform"]) == ()
        assert filter_cache_warmable(llm_router, ["uniform"]) == ()
    finally:
        logging_module.getLogger("LiteLLM Router").removeHandler(handler)
    warnings = [message for message in records if "cannot produce cache hits" in message]
    assert len(warnings) == 1
    assert 'router_settings.optional_pre_call_checks: ["session_affinity"]' in warnings[0]
    assert 'model_group_affinity_config: {"uniform": ["session_affinity"]}' in warnings[0]
    assert "2 deployments" in warnings[0]


def test_multi_deployment_group_with_affinity_but_session_mode_off_is_skipped():
    from litellm.router_strategy.complexity_router.cache_warming.refresher import (
        _warn_multi_deployment_group_needs_affinity,
    )

    _warn_multi_deployment_group_needs_affinity.cache_clear()
    llm_router = FakeLLMRouter(deployments=_UNIFORM_POOL)
    llm_router.optional_callbacks = [
        _session_affinity_check(model_group_affinity_config={"uniform": ["deployment_affinity"]})
    ]
    assert filter_cache_warmable(llm_router, ["uniform"]) == ()


def test_single_deployment_group_warms_without_affinity():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    llm_router.optional_callbacks = []
    _seed_session(redis)
    assert filter_cache_warmable(llm_router, ["fast-claude", "smart-claude"]) == ("fast-claude", "smart-claude")


@pytest.mark.asyncio
async def test_replay_metadata_carries_session_id_on_both_legs():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, session_id="chat-sess", warmth={"fast-claude": time.time()})
    _seed_session(
        redis,
        session_id="msg-sess",
        caller_scope="hash-2",
        call_surface="anthropic_messages",
        warmth={"fast-claude": time.time()},
    )
    await _tick(llm_router)
    assert llm_router.completion_calls[0]["metadata"]["session_id"] == "chat-sess"
    assert llm_router.anthropic_calls[0]["litellm_metadata"]["session_id"] == "msg-sess"


def test_old_format_record_without_session_id_still_parses():
    record = CacheWarmingRecord(
        schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION,
        payload_compressed="blob",
        payload_sha256="sha",
        token_estimate=1,
        last_activity=1.0,
        served_model="m",
        attribution=CacheWarmingAttribution(),
        auto_router_model_name="r",
    ).model_dump()
    record.pop("session_id")
    parsed = CacheWarmingRecord.model_validate(record)
    assert parsed.session_id is None


@pytest.mark.asyncio
async def test_reservation_includes_one_output_token():
    llm_router, redis = _warming_rig(redis=FakeRedisCache())
    _seed_session(redis, user_api_key="edge-key")
    prisma = FakePrismaClient(rows=(SimpleNamespace(token="edge-key", spend=0.0, max_budget=2.048),))
    input_only_passes = CacheWarmingRefresher(price_resolver=lambda _r, _g: (0.001, 10.0))
    await _tick(llm_router, prisma=prisma, refresher=input_only_passes)
    assert llm_router.completion_calls == []
    prisma_with_headroom = FakePrismaClient(rows=(SimpleNamespace(token="edge-key", spend=0.0, max_budget=2.05),))
    output_fits = CacheWarmingRefresher(price_resolver=lambda _r, _g: (0.001, 0.0000001))
    await _tick(llm_router, prisma=prisma_with_headroom, refresher=output_fits)
    assert len(llm_router.completion_calls) >= 1
