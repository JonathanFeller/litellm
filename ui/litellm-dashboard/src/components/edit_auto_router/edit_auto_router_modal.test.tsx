import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { renderWithProviders, screen, waitFor } from "@/../tests/test-utils";

import EditAutoRouterModal from "./edit_auto_router_modal";

const { modelPatchUpdateCall, modelAvailableCall } = vi.hoisted(() => ({
  modelPatchUpdateCall: vi.fn().mockResolvedValue({}),
  modelAvailableCall: vi.fn().mockResolvedValue({ data: [] }),
}));

vi.mock("../networking", () => ({ modelPatchUpdateCall, modelAvailableCall }));

vi.mock("@/components/llm_calls/fetch_models", () => ({
  fetchAvailableModels: vi.fn().mockResolvedValue([{ model_group: "gpt-4o-mini" }]),
}));

const STORED_CONFIG = {
  tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: ["gpt-4o-mini"], COMPLEX: ["gpt-4o-mini"], REASONING: ["gpt-4o-mini"] },
  classifier_type: "heuristic",
  keyword_tier_rules: [{ keywords: ["invoice", "refund"], tier: "MEDIUM" }],
  escalation_keywords: ["urgent", "outage"],
  semantic_keyword_matching: true,
  embedding_model: "voyage-4-large",
  match_threshold: 0.72,
};

const MODEL_DATA = {
  model_name: "tri-tier-router",
  litellm_params: {
    model: "auto_router/complexity_router",
    complexity_router_config: STORED_CONFIG,
  },
  model_info: { id: "auto-1", access_groups: [] },
};

const renderModal = () =>
  renderWithProviders(
    <EditAutoRouterModal
      isVisible
      onCancel={vi.fn()}
      onSuccess={vi.fn()}
      modelData={MODEL_DATA}
      accessToken="token"
      userRole="Admin"
    />,
  );

const savedConfig = () => {
  const [, payload] = modelPatchUpdateCall.mock.calls.at(-1) ?? [];
  return payload?.litellm_params?.complexity_router_config;
};

describe("EditAutoRouterModal keyword matching", () => {
  beforeEach(() => {
    modelPatchUpdateCall.mockClear();
  });

  it("renders the advanced sections the create form offers", async () => {
    renderModal();

    expect(await screen.findByText(/Escalation Keywords/i)).toBeInTheDocument();
    expect(await screen.findByText(/Keyword\/Semantic Matching/i)).toBeInTheDocument();
  });

  // These keys are rewritten from form state on save, so if the modal renders the controls
  // without hydrating them, an untouched save silently wipes the stored configuration. This
  // drives the real component; a test of the payload builder alone cannot see that bug.
  it("preserves stored keyword matching through an untouched open-and-save", async () => {
    const user = userEvent.setup();
    renderModal();

    await screen.findByText(/Escalation Keywords/i);
    await user.click(screen.getByRole("button", { name: /save changes/i }));

    await waitFor(() => expect(modelPatchUpdateCall).toHaveBeenCalled());

    const config = savedConfig();
    expect(config.keyword_tier_rules).toEqual([{ keywords: ["invoice", "refund"], tier: "MEDIUM" }]);
    expect(config.escalation_keywords).toEqual(["urgent", "outage"]);
    expect(config.semantic_keyword_matching).toBe(true);
    expect(config.embedding_model).toBe("voyage-4-large");
    expect(config.match_threshold).toBe(0.72);
  });
});
