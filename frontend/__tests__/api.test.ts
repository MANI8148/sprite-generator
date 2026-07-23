import {
  checkHealth,
  generateAsset,
  getHistory,
  getDownloadUrl,
  register,
  login,
  getMe,
  getBillingBalance,
  getBillingPackages,
  getBillingTransactions,
  topupCredits,
  getCostEstimate,
  getAuthToken,
  setAuthToken,
  clearAuthToken,
} from "../lib/api";

beforeEach(() => {
  global.fetch = jest.fn();
  localStorage.clear();
});

afterEach(() => {
  jest.restoreAllMocks();
  localStorage.clear();
});

describe("checkHealth", () => {
  it("returns health data on success", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ status: "ok", generator_loaded: true }),
    });

    const result = await checkHealth();
    expect(result.status).toBe("ok");
    expect(result.generator_loaded).toBe(true);
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 503,
    });

    await expect(checkHealth()).rejects.toThrow("Health check failed: 503");
  });
});

describe("generateAsset", () => {
  it("sends POST and returns response", async () => {
    const mockResponse = {
      job_id: "abc123",
      prompt: "test prompt",
      quality_tier: "clean",
      validation: {},
      zip_path: "/tmp/test.zip",
      output_paths: ["/tmp/test.png"],
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const result = await generateAsset({ asset_type: "character" });
    expect(result.job_id).toBe("abc123");
    expect(result.prompt).toBe("test prompt");
  });
});

describe("getHistory", () => {
  it("returns history list", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => [{ job_id: "1", prompt: "test", quality_tier: "clean", outputs: [], zip_path: null }],
    });

    const result = await getHistory();
    expect(result).toHaveLength(1);
    expect(result[0].job_id).toBe("1");
  });
});

describe("getDownloadUrl", () => {
  it("returns correct URL", () => {
    const url = getDownloadUrl("test-job");
    expect(url).toContain("/download/test-job");
  });
});

describe("auth token helpers", () => {
  it("stores and retrieves token", () => {
    setAuthToken("test-token-123");
    expect(getAuthToken()).toBe("test-token-123");
  });

  it("clears token", () => {
    setAuthToken("test-token-123");
    clearAuthToken();
    expect(getAuthToken()).toBeNull();
  });

  it("returns null when no token stored", () => {
    expect(getAuthToken()).toBeNull();
  });
});

describe("register", () => {
  it("sends POST and stores token", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        access_token: "jwt-token",
        token_type: "bearer",
        username: "testuser",
        user_id: "u123",
      }),
    });

    const result = await register("testuser", "password123");
    expect(result.access_token).toBe("jwt-token");
    expect(result.username).toBe("testuser");
    expect(getAuthToken()).toBe("jwt-token");
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 409,
      text: async () => "Username already exists",
    });

    await expect(register("existing", "password123")).rejects.toThrow(
      "Register failed: 409 Username already exists"
    );
  });
});

describe("login", () => {
  it("sends POST and stores token", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        access_token: "jwt-token",
        token_type: "bearer",
        username: "testuser",
        user_id: "u123",
      }),
    });

    const result = await login("testuser", "password123");
    expect(result.access_token).toBe("jwt-token");
    expect(getAuthToken()).toBe("jwt-token");
  });
});

describe("getMe", () => {
  it("sends GET with auth header", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ username: "testuser", user_id: "u123" }),
    });

    const result = await getMe();
    expect(result.username).toBe("testuser");
    expect(result.user_id).toBe("u123");
  });
});

describe("getBillingBalance", () => {
  it("sends GET with auth header", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        user_id: "u123",
        balance: 500,
        generation_cost: 1,
      }),
    });

    const result = await getBillingBalance();
    expect(result.balance).toBe(500);
    expect(result.generation_cost).toBe(1);
  });

  it("throws on failure", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 401,
      text: async () => "Not authenticated",
    });

    await expect(getBillingBalance()).rejects.toThrow(
      "Balance fetch failed: 401 Not authenticated"
    );
  });
});

describe("getBillingPackages", () => {
  it("returns packages without auth", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        packages: [
          { key: "starter", credits: 100, amount_cents: 499, description: "100 credits" },
        ],
      }),
    });

    const result = await getBillingPackages();
    expect(result.packages).toHaveLength(1);
    expect(result.packages[0].key).toBe("starter");
  });
});

describe("getBillingTransactions", () => {
  it("returns transactions with auth", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        user_id: "u123",
        transactions: [
          {
            transaction_id: "tx1",
            amount: 100,
            reason: "topup",
            timestamp: "2024-01-01T00:00:00Z",
          },
        ],
      }),
    });

    const result = await getBillingTransactions();
    expect(result.transactions).toHaveLength(1);
    expect(result.transactions[0].amount).toBe(100);
  });
});

describe("topupCredits", () => {
  it("sends POST with auth header", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        user_id: "u123",
        balance: 600,
        amount_added: 100,
      }),
    });

    const result = await topupCredits(100);
    expect(result.balance).toBe(600);
    expect(result.amount_added).toBe(100);
  });

  it("throws on failure", async () => {
    setAuthToken("test-token");
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 422,
      text: async () => "Amount must be positive",
    });

    await expect(topupCredits(0)).rejects.toThrow(
      "Topup failed: 422 Amount must be positive"
    );
  });
});

describe("getCostEstimate", () => {
  it("returns cost estimate", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        generation_cost: 1,
        num_frames: 4,
        total_cost: 4,
      }),
    });

    const result = await getCostEstimate(4);
    expect(result.total_cost).toBe(4);
  });
});
