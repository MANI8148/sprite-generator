import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import BillingPage from "../pages/billing";

jest.mock("next/router", () => ({
  useRouter: () => ({ pathname: "/billing" }),
}));

jest.mock("next/link", () => {
  const MockLink = ({ children, href, ...rest }: { children: React.ReactNode; href: string }) => (
    <a href={href} {...rest}>
      {children}
    </a>
  );
  MockLink.displayName = "MockLink";
  return MockLink;
});

const mockPackages = {
  packages: [
    { key: "starter", credits: 100, amount_cents: 499, description: "100 credits — $4.99" },
    { key: "pro", credits: 500, amount_cents: 1999, description: "500 credits — $19.99" },
  ],
};

const mockBalance = {
  user_id: "u123",
  balance: 250,
  generation_cost: 1,
};

const mockTransactions = {
  user_id: "u123",
  transactions: [
    {
      transaction_id: "tx1",
      amount: 100,
      reason: "topup",
      timestamp: "2024-06-01T12:00:00Z",
    },
  ],
};

const mockCostEstimate = {
  generation_cost: 1,
  num_frames: 1,
  total_cost: 1,
};

const mockAuthResponse = {
  access_token: "test-jwt",
  token_type: "bearer",
  username: "testuser",
  user_id: "u123",
};

beforeEach(() => {
  localStorage.clear();
  jest.restoreAllMocks();
});

function mockFetch(overrides: Record<string, unknown> = {}) {
  const defaultOk = true;
  const defaultJson = async () => ({});

  const handlers: Record<string, { ok?: boolean; json?: () => Promise<unknown>; text?: () => Promise<string> }> = {
    "/billing/packages": {
      json: async () => mockPackages,
    },
    "/billing/cost-estimate": {
      json: async () => mockCostEstimate,
    },
    ...overrides,
  };

  global.fetch = jest.fn().mockImplementation((url: string) => {
    const key = typeof url === "string" ? url.replace(/^http:\/\/localhost:8000/, "") : url;
    const handler = handlers[key];
    if (handler) {
      return Promise.resolve({
        ok: handler.ok ?? defaultOk,
        json: handler.json || defaultJson,
        text: handler.text || (async () => ""),
      });
    }
    return Promise.resolve({
      ok: defaultOk,
      json: defaultJson,
      text: async () => "",
    });
  });
}

describe("BillingPage", () => {
  it("renders heading and credit packages", async () => {
    mockFetch();
    render(<BillingPage />);

    expect(screen.getByText("Billing")).toBeTruthy();
    expect(screen.getByText("Credit Packages")).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText("starter")).toBeTruthy();
      expect(screen.getByText("pro")).toBeTruthy();
    });
  });

  it("renders cost estimate section", async () => {
    mockFetch();
    render(<BillingPage />);

    await waitFor(() => {
      expect(screen.getByText("Cost Estimate")).toBeTruthy();
    });
  });

  it("renders login form heading when not authenticated", () => {
    mockFetch();
    render(<BillingPage />);

    expect(screen.getByRole("heading", { name: "Login" })).toBeTruthy();
    expect(screen.getByPlaceholderText("Username")).toBeTruthy();
    expect(screen.getByPlaceholderText("Password")).toBeTruthy();
  });

  it("shows register form when switching mode", () => {
    mockFetch();
    render(<BillingPage />);

    fireEvent.click(screen.getByText("Switch to Register"));

    expect(screen.getByRole("heading", { name: "Register" })).toBeTruthy();
  });

  it("displays balance and transactions after auth", async () => {
    localStorage.setItem("sprite_gen_token", "test-jwt");

    mockFetch({
      "/auth/me": { json: async () => ({ username: "testuser", user_id: "u123" }) },
      "/billing/balance": { json: async () => mockBalance },
      "/billing/transactions": { json: async () => mockTransactions },
    });

    render(<BillingPage />);

    await waitFor(() => {
      expect(screen.getByText(/Welcome, testuser/)).toBeTruthy();
    });

    expect(screen.getByText("250")).toBeTruthy();
    expect(screen.getByText("Transaction History")).toBeTruthy();
    expect(screen.getByText("topup")).toBeTruthy();
    expect(screen.getByText("+100")).toBeTruthy();
  });

  it("shows logout button when authenticated", async () => {
    localStorage.setItem("sprite_gen_token", "test-jwt");

    mockFetch({
      "/auth/me": { json: async () => ({ username: "testuser", user_id: "u123" }) },
      "/billing/balance": { json: async () => mockBalance },
      "/billing/transactions": { json: async () => ({ user_id: "u123", transactions: [] }) },
    });

    render(<BillingPage />);

    await waitFor(() => {
      expect(screen.getByText("Logout")).toBeTruthy();
    });
  });

  it("handles cost estimate frame change", async () => {
    global.fetch = jest.fn().mockImplementation((url: string) => {
      const urlStr = typeof url === "string" ? url : "";
      if (urlStr.includes("/billing/cost-estimate")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            generation_cost: 1,
            num_frames: urlStr.includes("num_frames=4") ? 4 : 1,
            total_cost: urlStr.includes("num_frames=4") ? 4 : 1,
          }),
        });
      }
      if (urlStr.includes("/billing/packages")) {
        return Promise.resolve({ ok: true, json: async () => mockPackages });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<BillingPage />);

    await waitFor(() => {
      expect(screen.getByText("Cost Estimate")).toBeTruthy();
    });

    const input = screen.getByDisplayValue("1") as HTMLInputElement;
    fireEvent.change(input, { target: { value: "4" } });

    await waitFor(() => {
      expect(screen.getByText((content) => content.includes("4") && content.includes("credit"))).toBeTruthy();
    });
  });
});