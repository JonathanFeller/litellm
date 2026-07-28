import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, testQueryClient, waitFor } from "@/../tests/test-utils";

import { AutoRoutersPanel } from "./AutoRoutersPanel";

const { modelInfoCall, modelDeleteCall } = vi.hoisted(() => ({
  modelInfoCall: vi.fn(),
  modelDeleteCall: vi.fn().mockResolvedValue({}),
}));

vi.mock("@/components/networking", () => ({
  modelInfoCall,
  modelDeleteCall,
  modelHubCall: vi.fn(),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([]),
}));

const { searchParams, navigateWithParams } = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
  navigateWithParams: vi.fn(),
}));

vi.mock("next/navigation", () => ({ useSearchParams: () => searchParams }));
vi.mock("@/app/(dashboard)/navigateWithParams", () => ({ navigateWithParams }));

vi.mock("@/components/edit_auto_router/edit_auto_router_modal", () => ({
  __esModule: true,
  default: ({ modelData }: { modelData: { model_name?: string; model_info?: { id?: string } } }) => (
    <div data-testid="edit-auto-router-modal">
      edit:{modelData.model_name}:{modelData.model_info?.id}
    </div>
  ),
}));

vi.mock("@/components/add_model/add_auto_router_tab", () => ({
  __esModule: true,
  default: ({ handleOk }: { handleOk: () => void }) => (
    <button type="button" onClick={handleOk}>
      Submit auto router
    </button>
  ),
}));

// A realistic /v2/model/info page: two auto-routers among ordinary deployments. The panel must
// render exactly the auto_router/* rows; a view that renders page.data unfiltered passes a
// "renders a table" assertion but fails this one.
const DEPLOYMENTS = [
  {
    // DB-created adaptive router: no editor for its shape, but it must stay deletable, since
    // auto-routers are excluded from Models + Endpoints and this tab is the only delete path.
    model_name: "adaptive-router",
    litellm_params: { model: "auto_router/adaptive_router" },
    model_info: { id: "auto-3", db_model: true },
  },
  {
    // config.yaml row: the API refuses both update and delete, so neither control may appear.
    model_name: "config-router",
    litellm_params: {
      model: "auto_router/complexity_router",
      complexity_router_config: { tiers: {}, classifier_type: "llm" },
    },
    model_info: { id: "auto-4", db_model: false },
  },
  {
    model_name: "gpt-4o-mini",
    litellm_params: { model: "openai/gpt-4o-mini" },
    model_info: { id: "plain-1" },
  },
  {
    model_name: "tri-tier-router",
    litellm_params: {
      model: "auto_router/complexity_router",
      complexity_router_config: { tiers: { SIMPLE: ["gpt-4o-mini"] }, classifier_type: "heuristic" },
      complexity_router_default_model: "gpt-4o-mini",
    },
    model_info: { id: "auto-1", db_model: true, created_at: "2026-07-28T21:40:09.900000+00:00" },
  },
  {
    model_name: "anthropic-opus-4-6",
    litellm_params: { model: "anthropic/claude-opus-4-6" },
    model_info: { id: "plain-2" },
  },
  {
    model_name: "support-router",
    litellm_params: {
      model: "auto_router/support-router",
      auto_router_config: JSON.stringify({ routes: [{ name: "gpt-4o-mini" }] }),
      auto_router_default_model: "gpt-4o-mini",
    },
    model_info: { id: "auto-2", db_model: true, created_at: "2026-07-27T10:00:00.000000+00:00" },
  },
];

const pageOf = (data: typeof DEPLOYMENTS) => ({
  data,
  total_count: data.length,
  current_page: 1,
  total_pages: 1,
  size: 1000,
});

const mockDeploymentsPage = () => {
  modelInfoCall.mockResolvedValue(pageOf(DEPLOYMENTS));
};

const renderPanel = (canModify = true) =>
  renderWithProviders(<AutoRoutersPanel accessToken="token" userRole="Admin" canModify={canModify} />);

describe("AutoRoutersPanel", () => {
  beforeEach(() => {
    // The shared test client caches with staleTime: Infinity and refetchOnMount: false, so
    // without this every test after the first reads the previous test's deployment page.
    testQueryClient.clear();
    modelInfoCall.mockReset();
    modelDeleteCall.mockClear();
    navigateWithParams.mockClear();
    mockDeploymentsPage();
  });

  it("lists only auto_router deployments, not every model on the proxy", async () => {
    renderPanel();

    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    expect(await screen.findByText("support-router")).toBeInTheDocument();
    expect(screen.queryByText("gpt-4o-mini", { selector: "span.text-sm.font-medium" })).not.toBeInTheDocument();
    expect(screen.queryByText("anthropic-opus-4-6", { selector: "span.text-sm.font-medium" })).not.toBeInTheDocument();
  });

  it("labels Type by classifier, with the full strategy kept as the tooltip", async () => {
    renderPanel();

    // The pill says what the router classifies with, not that it happens to be a
    // complexity router; the longer strategy stays available on hover.
    const heuristic = await screen.findByText("Heuristic");
    expect(heuristic).toHaveAttribute("title", "Heuristic classifier");
    expect(await screen.findByText("Semantic")).toHaveAttribute("title", "1 semantic route");
  });

  it("routes to the detail view on row click instead of opening a modal", async () => {
    const user = userEvent.setup();
    renderPanel();

    await user.click(await screen.findByRole("button", { name: "support-router" }));

    expect(navigateWithParams).toHaveBeenCalled();
    const mutate = navigateWithParams.mock.calls.at(-1)?.[0];
    const params = new URLSearchParams();
    mutate?.(params);
    expect(params.get("autoRouter")).toBe("auto-2");
    expect(screen.queryByTestId("edit-auto-router-modal")).not.toBeInTheDocument();
  });

  it("renders the detail view, and edits from there, when the url selects a router", async () => {
    const user = userEvent.setup();
    searchParams.set("autoRouter", "auto-2");
    try {
      renderPanel();

      expect(await screen.findByRole("heading", { name: "support-router" })).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: /^edit$/i }));
      expect(await screen.findByTestId("edit-auto-router-modal")).toHaveTextContent("edit:support-router:auto-2");
    } finally {
      searchParams.delete("autoRouter");
    }
  });

  it("shows the create form behind Add Auto Router and refetches the list after a create", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("tri-tier-router");
    const callsBeforeCreate = modelInfoCall.mock.calls.length;

    await user.click(screen.getByRole("button", { name: "Add Auto Router" }));
    await user.click(await screen.findByRole("button", { name: "Submit auto router" }));

    // Back on the list, and the deployment query was invalidated so a new router shows up
    // without a manual page reload.
    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    await waitFor(() => expect(modelInfoCall.mock.calls.length).toBeGreaterThan(callsBeforeCreate));
  });

  // The page decides who may write (proxy admin or team admin); the panel just has to make
  // every write affordance absent when told no, rather than let a submit 403 later. Reading
  // stays open: a read-only caller can still drill into the detail view.
  it("shows the list but no write affordances when canModify is false", async () => {
    renderPanel(false);

    expect(await screen.findByText("tri-tier-router")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Add Auto Router" })).not.toBeInTheDocument();
    expect(screen.queryByTestId("auto-router-actions-auto-1")).not.toBeInTheDocument();
    // Still navigable, because opening the detail view is a read.
    expect(screen.getByRole("button", { name: "tri-tier-router" })).toBeInTheDocument();
  });

  // Auto-routers are hidden from Models + Endpoints, which used to be the only route to the
  // delete action, so this tab is now the only place an auto router can be removed.
  it("deletes the chosen router by its model id and refetches", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("support-router");
    const callsBeforeDelete = modelInfoCall.mock.calls.length;

    await user.click(screen.getByTestId("auto-router-actions-auto-2"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("token", "auto-2"));
    await waitFor(() => expect(modelInfoCall.mock.calls.length).toBeGreaterThan(callsBeforeDelete));
  });

  it("does not delete when the confirmation is dismissed", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("support-router");

    await user.click(screen.getByTestId("auto-router-actions-auto-2"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /cancel/i }));

    expect(modelDeleteCall).not.toHaveBeenCalled();
  });

  it("gives a read-only caller no delete affordance", async () => {
    renderPanel(false);

    await screen.findByText("support-router");
    expect(screen.queryByTestId("auto-router-actions-auto-2")).not.toBeInTheDocument();
  });

  it("renders an empty state when the proxy has models but no auto routers", async () => {
    modelInfoCall.mockResolvedValue(
      pageOf(DEPLOYMENTS.filter((d) => !d.litellm_params.model.startsWith("auto_router/"))),
    );

    renderPanel();

    expect(await screen.findByText("No auto routers yet")).toBeInTheDocument();
  });

  it("keeps delete available on a DB-created adaptive router that has no editor", async () => {
    const user = userEvent.setup();
    renderPanel();

    await screen.findByText("adaptive-router");
    await user.click(screen.getByTestId("auto-router-actions-auto-3"));
    await user.click(await screen.findByTestId("auto-router-action-delete"));
    await user.click(await screen.findByRole("button", { name: /^delete$/i }));

    await waitFor(() => expect(modelDeleteCall).toHaveBeenCalledWith("token", "auto-3"));
  });

  it("offers no delete on a config-defined router, which the API would refuse", async () => {
    renderPanel();

    await screen.findByText("config-router");
    expect(screen.queryByTestId("auto-router-actions-auto-4")).not.toBeInTheDocument();
  });
});
