import asyncio
import contextlib
import itertools
import json
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from functools import lru_cache
from typing import TYPE_CHECKING

from pydantic import BaseModel, TypeAdapter

from litellm._logging import verbose_router_logger
from litellm.constants import CACHE_WARMING_JOB_NAME, LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.integrations.anthropic_cache_control_hook import EXPLICIT_PROMPT_CACHING_PROVIDERS
from litellm.router_strategy.complexity_router.cache_warming.eligibility import resolve_warm_models
from litellm.router_strategy.complexity_router.cache_warming.store import CacheWarmingStore
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_REPLAY_MARKER_KEY,
    CACHE_WARMING_REPLAY_TAG,
    CacheWarmingRecord,
    decompress_payload,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router
    from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter

CACHE_WARMING_MAX_CONCURRENT_REPLAYS = 10
CACHE_WARMING_LOCK_TTL_SECONDS = 60
_REPLAY_OUTPUT_TOKENS = 1

_ATTRIBUTION_ADAPTER: TypeAdapter[Mapping[str, str | None]] = TypeAdapter(Mapping[str, str | None])

_COMPARE_AND_EXPIRE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
else
    return 0
end
"""


@lru_cache(maxsize=1)
def _warn_budget_unverifiable() -> None:
    verbose_router_logger.warning(
        "cache_warming cannot verify key state (no database client); skipping replays for "
        "key-attributed sessions until the database is reachable"
    )


class _TokenBudgetRow(BaseModel):
    token: str
    spend: float = 0.0
    max_budget: float | None = None
    blocked: bool | None = None
    expires: datetime | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None


_TOKEN_ROWS_ADAPTER: TypeAdapter[tuple[_TokenBudgetRow, ...]] = TypeAdapter(tuple[_TokenBudgetRow, ...])


def _excluded_from_warming(row: _TokenBudgetRow, now: float) -> bool:
    if row.blocked is True:
        return True
    return row.expires is not None and row.expires.timestamp() <= now


class WarmingBudgetLedger:
    """Leader-local reservation of replay spend not yet visible in the database's
    spend column, so a burst of replays inside one spend-write lag window cannot
    overshoot a key's max_budget. Each admission is its own reservation entry and
    reconciliation targets exactly that entry, mirroring the v3 rate limiter's
    recorded-identity reservations (parallel_request_limiter_v3.TPM_RESERVED_MODEL_KEY):
    reconciling any other scope drifts the total whenever actual differs from
    reserved. Entries expire once old enough that the database row must reflect
    them."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._ids = itertools.count(1)
        self._entries: dict[str, dict[int, tuple[float, float]]] = {}  # mutable-ok: leader-local ledger state

    def reserve(self, key: str, amount: float) -> int:
        entry_id = next(self._ids)
        self._entries.setdefault(key, {})[entry_id] = (self._clock(), amount)  # mutable-ok: leader-local ledger state
        return entry_id

    def reconcile(self, key: str, entry_id: int, actual: float | None) -> None:
        entries = self._entries.get(key)
        if entries is None or actual is None:
            return
        entry = entries.get(entry_id)
        if entry is None:
            return
        entries[entry_id] = (entry[0], actual)  # mutable-ok: leader-local ledger state

    def reserved(self, key: str, entry_ttl_seconds: float) -> float:
        now = self._clock()
        entries = self._entries.get(key, {})
        live = {
            entry_id: entry for entry_id, entry in entries.items() if now - entry[0] <= entry_ttl_seconds
        }  # mutable-ok: leader-local ledger state
        if live:
            self._entries[key] = live
        else:
            self._entries.pop(key, None)
        return sum(amount for _, amount in live.values())


class WarmingRateTracker:
    """Leader-local pacing so warming's own consumption stays inside a key's
    rpm_limit and tpm_limit over a rolling minute. Real-traffic consumption is
    not visible here; sharing the proxy limiter's counters is the follow-up."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._events: dict[str, tuple[tuple[float, int], ...]] = {}  # mutable-ok: leader-local pacing state

    def admit(self, key: str, tokens: int, rpm_limit: int | None, tpm_limit: int | None) -> bool:
        now = self._clock()
        live = tuple((at, count) for at, count in self._events.get(key, ()) if now - at <= 60.0)
        if rpm_limit is not None and len(live) + 1 > rpm_limit:
            self._events[key] = live  # mutable-ok: leader-local pacing state
            return False
        if tpm_limit is not None and sum(count for _, count in live) + tokens > tpm_limit:
            self._events[key] = live  # mutable-ok: leader-local pacing state
            return False
        self._events[key] = (*live, (now, tokens))  # mutable-ok: leader-local pacing state
        return True


def collect_warming_enabled_complexity_routers(llm_router: "Router") -> tuple["ComplexityRouter", ...]:
    return tuple(
        tagged.strategy
        for tagged_list in llm_router.complexity_routers.values()
        for tagged in tagged_list
        if tagged.strategy.config.cache_warming.enabled
    )


def _deployment_provider(litellm_params: Mapping[str, object], deployment_model: str) -> str | None:
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    declared = litellm_params.get("custom_llm_provider")
    if isinstance(declared, str) and declared:
        return declared
    try:
        _, provider, _, _ = get_llm_provider(model=deployment_model)
    except Exception:  # noqa: BLE001  # unroutable deployment just isn't warmable
        return None
    return provider


@lru_cache(maxsize=256)
def _warn_multi_deployment_group_needs_affinity(model_group: str, deployment_count: int) -> None:
    verbose_router_logger.warning(
        "cache_warming: model group %s has %s deployments and deployment affinity with session_id is not active "
        "for it, so warming is skipped for this group; warming without affinity cannot produce cache hits "
        "reliably because a replay warms one member's cache while real traffic routes across all of them. "
        'Enable it globally with router_settings.optional_pre_call_checks: ["session_affinity"] or for this '
        'group only with router_settings.model_group_affinity_config: {"%s": ["session_affinity"]}',
        model_group,
        deployment_count,
        model_group,
    )


def _session_affinity_active(llm_router: "Router", model_group: str) -> bool:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck

    check = next(
        (
            callback
            for callback in (llm_router.optional_callbacks or [])  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # Router.optional_callbacks is legacy-untyped
            if isinstance(callback, DeploymentAffinityCheck)
        ),
        None,
    )
    if check is None:
        return False
    return check._get_effective_flags(model_group)[2]  # pyright: ignore[reportPrivateUsage]  # no public accessor today; upstream should expose get_effective_flags


def _group_is_cache_warmable(llm_router: "Router", model_group: str) -> bool:
    """Every selectable member must be warmable, because replays route by group
    name and the Router may pick any member; the conservative collapse mirrors
    the group price resolution one function down."""
    from litellm.utils import supports_prompt_caching

    deployments = llm_router.get_model_list(model_name=model_group) or []
    if not deployments:
        return False
    if len(deployments) > 1 and not _session_affinity_active(llm_router, model_group):
        _warn_multi_deployment_group_needs_affinity(model_group, len(deployments))
        return False
    for deployment in deployments:
        litellm_params = deployment.get("litellm_params") or {}  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
        deployment_model = litellm_params.get("model")  # pyright: ignore[reportUnknownMemberType]  # DeploymentTypedDict fields are legacy-untyped
        if not isinstance(deployment_model, str):
            return False
        provider = _deployment_provider(litellm_params, deployment_model)
        if provider not in EXPLICIT_PROMPT_CACHING_PROVIDERS:
            return False
        if not supports_prompt_caching(model=deployment_model, custom_llm_provider=provider):
            return False
    return True


def filter_cache_warmable(llm_router: "Router", model_groups: Sequence[str]) -> tuple[str, ...]:
    return tuple(group for group in model_groups if _group_is_cache_warmable(llm_router, group))


def _deployment_replay_prices(litellm_params: Mapping[str, object]) -> tuple[float, float] | None:
    """(cache-read input rate, output rate) or None when either is unresolvable,
    mirroring budget_reservation's _estimate_request_max_cost_for_model: output
    collapses with max over output_cost_per_token and its reasoning variant, and
    a missing needed rate yields None rather than zero."""
    from litellm.utils import get_model_info

    deployment_model = litellm_params.get("model")
    if not isinstance(deployment_model, str):
        return None
    provider = _deployment_provider(litellm_params, deployment_model)
    if provider is None:
        return None
    try:
        info = get_model_info(model=deployment_model, custom_llm_provider=provider)
    except Exception:  # noqa: BLE001  # unknown model resolves to None; metered keys then skip the replay
        return None
    input_price = info.get("cache_read_input_token_cost") or info.get("input_cost_per_token")
    output_candidates = tuple(
        float(rate)
        for rate in (info.get("output_cost_per_token"), info.get("output_cost_per_reasoning_token"))
        if isinstance(rate, (int, float)) and rate > 0
    )
    if not isinstance(input_price, (int, float)) or input_price <= 0 or not output_candidates:
        return None
    return (float(input_price), max(output_candidates))


def _group_replay_prices(llm_router: "Router", model_group: str) -> tuple[float, float] | None:
    """The router picks any group member at replay time, so the reservation
    collapses group heterogeneity with MAX per component, the same collapse
    ModelGroupInfo applies to per-token costs (router.py get_model_group_info);
    one unpriceable member makes the whole group unpriceable so metered keys
    skip it."""
    deployments = llm_router.get_model_list(model_name=model_group) or []
    if not deployments:
        return None
    prices = tuple(
        _deployment_replay_prices(
            deployment.get("litellm_params") or {}  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # DeploymentTypedDict fields are legacy-untyped
        )
        for deployment in deployments
    )
    if any(price is None for price in prices):
        return None
    resolved = tuple(price for price in prices if price is not None)
    return (max(pair[0] for pair in resolved), max(pair[1] for pair in resolved))


class CacheWarmingRefresher:
    def __init__(
        self,
        max_concurrent_replays: int = CACHE_WARMING_MAX_CONCURRENT_REPLAYS,
        lock_ttl_seconds: float = CACHE_WARMING_LOCK_TTL_SECONDS,
        price_resolver: 'Callable[["Router", str], tuple[float, float] | None]' = _group_replay_prices,
        ledger: WarmingBudgetLedger | None = None,
        rate_tracker: WarmingRateTracker | None = None,
    ) -> None:
        self.max_concurrent_replays = max_concurrent_replays
        self.lock_ttl_seconds = lock_ttl_seconds
        self.price_resolver = price_resolver
        self.ledger = ledger if ledger is not None else WarmingBudgetLedger()
        self.rate_tracker = rate_tracker if rate_tracker is not None else WarmingRateTracker()
        self._fallback_lock_manager: PodLockManager | None = None

    async def _hold_lock_lease(self, lock_manager: "PodLockManager", lease_lost: asyncio.Event) -> None:
        redis_cache = lock_manager.redis_cache
        if redis_cache is None:
            return
        from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

        renew = redis_cache.async_register_script(_COMPARE_AND_EXPIRE_LOCK_SCRIPT)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # RedisCache is legacy-untyped
        lock_key = PodLockManager.get_redis_lock_key(CACHE_WARMING_JOB_NAME)
        while True:
            await asyncio.sleep(self.lock_ttl_seconds / 2)
            renewed = await renew(keys=[lock_key], args=[json.dumps(lock_manager.pod_id), int(self.lock_ttl_seconds)])  # pyright: ignore[reportUnknownVariableType, reportAny]  # RedisCache is legacy-untyped
            if not renewed:
                verbose_router_logger.warning(
                    "cache_warming pod lock was lost mid tick; finishing in-flight replays without "
                    "admitting new ones, since ledger exactness is conditional on single leadership"
                )
                lease_lost.set()
                return

    def _resolve_lock_manager(
        self, injected: "PodLockManager | None", redis_cache: "RedisCache | None"
    ) -> "PodLockManager":
        if injected is not None and injected.redis_cache is not None:
            return injected
        if self._fallback_lock_manager is None or self._fallback_lock_manager.redis_cache is not redis_cache:
            from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager

            self._fallback_lock_manager = PodLockManager(redis_cache=redis_cache)
        return self._fallback_lock_manager

    async def run_tick(
        self,
        *,
        llm_router: "Router",
        pod_lock_manager: "PodLockManager | None",
        prisma_client: "PrismaClient | None",
    ) -> None:
        warming_routers = collect_warming_enabled_complexity_routers(llm_router)
        if not warming_routers:
            return
        warmable = tuple(
            (complexity_router, store)
            for complexity_router in warming_routers
            if (store := complexity_router.get_cache_warming_store()) is not None and store.redis_cache is not None
        )
        if not warmable:
            return
        lock_manager = self._resolve_lock_manager(pod_lock_manager, warmable[0][1].redis_cache)
        acquired = await lock_manager.acquire_lock(cronjob_id=CACHE_WARMING_JOB_NAME, ttl=int(self.lock_ttl_seconds))
        if not acquired:
            return
        lease_lost = asyncio.Event()
        lease = asyncio.create_task(self._hold_lock_lease(lock_manager, lease_lost))
        try:
            for complexity_router, store in warmable:
                await self._warm_router_sessions(
                    llm_router=llm_router,
                    complexity_router=complexity_router,
                    store=store,
                    prisma_client=prisma_client,
                    lease_lost=lease_lost,
                )
        finally:
            lease.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await lease
            await lock_manager.release_lock(cronjob_id=CACHE_WARMING_JOB_NAME)

    async def _warm_router_sessions(
        self,
        *,
        llm_router: "Router",
        complexity_router: "ComplexityRouter",
        store: CacheWarmingStore,
        prisma_client: "PrismaClient | None",
        lease_lost: asyncio.Event,
    ) -> None:
        config = complexity_router.config.cache_warming
        session_keys = await store.list_session_keys(max_sessions=config.max_sessions)
        if not session_keys:
            return
        if len(session_keys) >= config.max_sessions:
            verbose_router_logger.debug(
                "cache_warming: auto-router %s is at its max_sessions cap (%s); "
                "new sessions are not admitted until existing ones expire",
                complexity_router.model_name,
                config.max_sessions,
            )
        now = time.time()
        records = tuple([(key, await store.get_record(key)) for key in session_keys])
        active = tuple(
            (key, record)
            for key, record in records
            if record is not None and now - record.last_activity <= config.idle_timeout_seconds
        )
        if not active:
            return
        warm_models = filter_cache_warmable(llm_router, resolve_warm_models(complexity_router.config))
        if not warm_models:
            return
        attributed = frozenset(
            record.attribution.user_api_key
            for _, record in active
            if record.attribution.user_api_key is not None
            and record.attribution.user_api_key != LITELLM_PROXY_MASTER_KEY_ALIAS
        )
        key_states = await self._verified_key_states(prisma_client, attributed)
        if key_states is None:
            return
        now = time.time()
        excluded_keys = frozenset(
            key for key in attributed if (row := key_states.get(key)) is None or _excluded_from_warming(row, now)
        )
        prices = {
            model_group: self.price_resolver(llm_router, model_group) for model_group in warm_models
        }  # mutable-ok: per-tick lookup table, never retained
        semaphore = asyncio.Semaphore(self.max_concurrent_replays)
        await asyncio.gather(
            *(
                self._warm_session(
                    llm_router=llm_router,
                    store=store,
                    session_key=key,
                    record=record,
                    warm_models=warm_models,
                    refresh_interval_seconds=config.refresh_interval_seconds,
                    session_ttl_seconds=config.session_ttl_seconds,
                    semaphore=semaphore,
                    budget_row=key_states.get(record.attribution.user_api_key or ""),
                    prices=prices,
                    lease_lost=lease_lost,
                )
                for key, record in active
                if record.attribution.user_api_key not in excluded_keys
            )
        )

    async def _warm_session(
        self,
        *,
        llm_router: "Router",
        store: CacheWarmingStore,
        session_key: str,
        record: CacheWarmingRecord,
        warm_models: tuple[str, ...],
        refresh_interval_seconds: int,
        session_ttl_seconds: int,
        semaphore: asyncio.Semaphore,
        budget_row: _TokenBudgetRow | None,
        prices: Mapping[str, tuple[float, float] | None],
        lease_lost: asyncio.Event,
    ) -> None:
        warmth = await store.get_warmth(session_key, warm_models)
        now = time.time()
        due_models = tuple(model for model in warm_models if now - warmth.get(model, 0.0) >= refresh_interval_seconds)
        for model_group in due_models:
            async with semaphore:
                if lease_lost.is_set():
                    return
                price = prices.get(model_group)
                metered = budget_row is not None and budget_row.max_budget is not None
                if metered and price is None:
                    verbose_router_logger.debug(
                        "cache_warming: no resolvable price for %s; not warming budgeted session %s",
                        model_group,
                        session_key,
                    )
                    continue
                estimate = record.token_estimate * price[0] + _REPLAY_OUTPUT_TOKENS * price[1] if price else 0.0
                if not self._within_budget(budget_row, estimate, session_ttl_seconds):
                    verbose_router_logger.debug(
                        "cache_warming: key budget reached; not warming session %s on %s", session_key, model_group
                    )
                    continue
                if budget_row is not None and not self.rate_tracker.admit(
                    budget_row.token, record.token_estimate, budget_row.rpm_limit, budget_row.tpm_limit
                ):
                    verbose_router_logger.debug(
                        "cache_warming: key rate limit reached; not warming session %s on %s", session_key, model_group
                    )
                    continue
                reservation = (
                    self.ledger.reserve(budget_row.token, estimate) if metered and budget_row is not None else None
                )
                attempted_at = time.time()
                try:
                    response = await self._replay(llm_router=llm_router, record=record, model_group=model_group)
                except Exception:  # noqa: BLE001  # one failing replay must not abort the tick
                    verbose_router_logger.warning(
                        "cache_warming replay failed for session %s model %s",
                        session_key,
                        model_group,
                        exc_info=True,
                    )
                else:
                    if reservation is not None and budget_row is not None:
                        self.ledger.reconcile(budget_row.token, reservation, _response_cost(response))
                finally:
                    await store.mark_warm_attempt(session_key, model_group, attempted_at, session_ttl_seconds)

    def _within_budget(self, budget_row: _TokenBudgetRow | None, estimate: float, entry_ttl_seconds: float) -> bool:
        if budget_row is None or budget_row.max_budget is None:
            return True
        reserved = self.ledger.reserved(budget_row.token, entry_ttl_seconds)
        return budget_row.spend + reserved + estimate <= budget_row.max_budget

    async def _replay(self, *, llm_router: "Router", record: CacheWarmingRecord, model_group: str) -> object:
        payload = decompress_payload(record.payload_compressed)
        attribution = _ATTRIBUTION_ADAPTER.validate_python(record.attribution.model_dump())
        metadata = {  # mutable-ok: request metadata handed to the router call, never retained
            CACHE_WARMING_REPLAY_MARKER_KEY: True,
            **({"session_id": record.session_id} if record.session_id is not None else {}),
            **{key: value for key, value in attribution.items() if value is not None},
            **(
                {"tags": [CACHE_WARMING_REPLAY_TAG]} if llm_router.enable_tag_filtering is not True else {}
            ),  # mutable-ok: request metadata, never retained
        }
        messages = [dict(message) for message in payload.messages]  # mutable-ok: router call input, never retained
        if payload.call_surface == "anthropic_messages":
            system = list(payload.system) if isinstance(payload.system, tuple) else payload.system
            return await llm_router.aanthropic_messages(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # factory-generated router surface is legacy-untyped
                model=model_group,
                messages=messages,
                system=system,
                tools=list(payload.tools) if payload.tools is not None else None,
                tool_choice=dict(payload.tool_choice)
                if isinstance(payload.tool_choice, Mapping)
                else payload.tool_choice,  # mutable-ok: router call input, never retained
                max_tokens=1,
                stream=False,
                cache={"no-cache": True},  # mutable-ok: router call input, never retained
                litellm_metadata=metadata,
            )
        return await llm_router.acompletion(  # pyright: ignore[reportUnknownMemberType, reportCallIssue, reportUnknownVariableType]  # router overloads are legacy-untyped
            model=model_group,
            messages=messages,  # pyright: ignore[reportArgumentType]  # replay forwards the captured wire shape verbatim
            tools=list(payload.tools) if payload.tools is not None else None,
            tool_choice=dict(payload.tool_choice)
            if isinstance(payload.tool_choice, Mapping)
            else payload.tool_choice,  # mutable-ok: router call input, never retained
            max_tokens=1,
            stream=False,
            cache={"no-cache": True},  # mutable-ok: router call input, never retained
            metadata=metadata,
        )

    @staticmethod
    async def _verified_key_states(
        prisma_client: "PrismaClient | None", key_hashes: frozenset[str]
    ) -> Mapping[str, _TokenBudgetRow] | None:
        if not key_hashes:
            return {}
        if prisma_client is None:
            _warn_budget_unverifiable()
            return None
        try:
            rows = _TOKEN_ROWS_ADAPTER.validate_python(
                await prisma_client.db.litellm_verificationtoken.find_many(  # pyright: ignore[reportAny]  # prisma client is legacy-untyped
                    where={"token": {"in": list(key_hashes)}}  # mutable-ok: prisma query input, never retained
                ),
                from_attributes=True,
            )
        except Exception:  # noqa: BLE001  # unverifiable key state fails closed; a skipped tick re-warms on the next one
            verbose_router_logger.warning(
                "cache_warming key-state query failed; skipping replays this tick", exc_info=True
            )
            return None
        return {row.token: row for row in rows}


def _response_cost(response: object) -> float | None:
    hidden = getattr(response, "_hidden_params", None)  # pyright: ignore[reportAny]  # response objects are legacy-untyped
    if not isinstance(hidden, dict) and isinstance(response, dict):
        hidden = response.get("_hidden_params")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # wire-shaped response dict
    cost = hidden.get("response_cost") if isinstance(hidden, dict) else None  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]  # wire-shaped response dict
    if isinstance(cost, (int, float)) and cost > 0:
        return float(cost)
    try:
        from litellm import completion_cost

        computed = completion_cost(completion_response=response)  # pyright: ignore[reportArgumentType]  # replay responses are litellm response objects
    except Exception:  # noqa: BLE001  # unpriceable actual keeps the reservation standing
        return None
    return float(computed) if computed > 0 else None
