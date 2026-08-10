import {
  checkHealth,
  generateAsset,
  getJobStatus,
  generateAndWait,
  getHistory,
  getDownloadUrl,
  getPreviewUrl,
  register,
  login,
  getMe,
  getBillingBalance,
  getBillingPackages,
  getBillingTransactions,
  topupCredits,
  getCostEstimate,
  getLibrary,
  getLibraryAsset,
  getLibraryTags,
  deleteLibraryAsset,
  updateLibraryAsset,
  addAssetTags,
  removeAssetTags,
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
  it("sends POST and returns the queued job", async () => {
    const mockResponse = {
      job_id: "abc123",
      status: "pending",
    };

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    const result = await generateAsset({ asset_type: "character" });
    expect(result.job_id).toBe("abc123");
    expect(result.status).toBe("pending");
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/generate");
    expect(init.method).toBe("POST");
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 503,
      text: async () => "Generator not set",
    });

    await expect(generateAsset({ asset_type: "character" })).rejects.toThrow(
      "Generate failed: 503 Generator not set"
    );
  });
});

describe("getJobStatus", () => {
  it("returns the job status", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({
        job_id: "abc123",
        status: "done",
        prompt: "test prompt",
        quality_tier: "clean",
      }),
    });

    const result = await getJobStatus("abc123");
    expect(result.status).toBe("done");
    expect(result.prompt).toBe("test prompt");
    const [url] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/status/abc123");
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 404,
      text: async () => "Job not found",
    });

    await expect(getJobStatus("missing")).rejects.toThrow(
      "Status fetch failed: 404 Job not found"
    );
  });
});

describe("generateAndWait", () => {
  it("polls until the job is done", async () => {
    const calls = jest.fn();
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ job_id: "abc123", status: "pending" }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ job_id: "abc123", status: "running" }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          job_id: "abc123",
          status: "done",
          prompt: "test prompt",
          quality_tier: "clean",
          output_paths: ["/tmp/test.png"],
        }),
      });

    const result = await generateAndWait(
      { asset_type: "character" },
      { pollIntervalMs: 5, onStatus: (s) => calls(s.status) }
    );

    expect(result.status).toBe("done");
    expect(result.prompt).toBe("test prompt");
    expect(calls).toHaveBeenCalledWith("pending");
    expect(calls).toHaveBeenCalledWith("running");
    expect(calls).toHaveBeenCalledWith("done");
  });

  it("throws when the job fails", async () => {
    (global.fetch as jest.Mock)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ job_id: "abc123", status: "pending" }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ job_id: "abc123", status: "failed", error: "boom" }),
      });

    await expect(
      generateAndWait({ asset_type: "character" }, { pollIntervalMs: 5 })
    ).rejects.toThrow("boom");
  });

  it("times out when the job never completes", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ job_id: "abc123", status: "pending" }),
    });

    await expect(
      generateAndWait({ asset_type: "character" }, { pollIntervalMs: 5, timeoutMs: 20 })
    ).rejects.toThrow("Timed out waiting for job abc123");
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

describe("getPreviewUrl", () => {
  it("returns preview URL with default index", () => {
    const url = getPreviewUrl("test-job");
    expect(url).toContain("/preview/test-job");
    expect(url).toContain("index=0");
  });

  it("returns preview URL with explicit index", () => {
    const url = getPreviewUrl("test-job", 3);
    expect(url).toContain("/preview/test-job");
    expect(url).toContain("index=3");
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

describe("getLibrary", () => {
  const mockAsset = {
    asset_id: "abc123",
    job_id: "abc123",
    asset_type: "character",
    prompt: "a pixel art hero sprite",
    quality_tier: "clean",
    tags: ["hero"],
    category: "",
    thumbnail_path: null,
    zip_path: null,
    output_paths: ["/tmp/abc123.png"],
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
  };

  it("returns library list", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ assets: [mockAsset], total: 1 }),
    });

    const result = await getLibrary();
    expect(result.total).toBe(1);
    expect(result.assets[0].asset_id).toBe("abc123");
  });

  it("sends filter query params", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ assets: [], total: 0 }),
    });

    await getLibrary({ asset_type: "character", search: "hero", limit: 20 });

    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library?");
    expect(url).toContain("asset_type=character");
    expect(url).toContain("search=hero");
    expect(url).toContain("limit=20");
    expect(init.method).toBeUndefined();
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 500,
      text: async () => "boom",
    });

    await expect(getLibrary()).rejects.toThrow("Library fetch failed: 500 boom");
  });
});

describe("getLibraryAsset", () => {
  it("returns a single asset", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ asset_id: "abc123", job_id: "abc123", asset_type: "character" }),
    });

    const result = await getLibraryAsset("abc123");
    expect(result.asset_id).toBe("abc123");
    const [url] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library/abc123");
  });
});

describe("getLibraryTags", () => {
  it("returns tags", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ tags: ["hero", "green"] }),
    });

    const result = await getLibraryTags();
    expect(result.tags).toEqual(["hero", "green"]);
  });
});

describe("deleteLibraryAsset", () => {
  it("sends DELETE and returns confirmation", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ status: "deleted", asset_id: "abc123" }),
    });

    const result = await deleteLibraryAsset("abc123");
    expect(result.status).toBe("deleted");
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library/abc123");
    expect(init.method).toBe("DELETE");
  });

  it("throws on failure", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: false,
      status: 404,
    });

    await expect(deleteLibraryAsset("missing")).rejects.toThrow("Library delete failed: 404");
  });
});

describe("updateLibraryAsset", () => {
  it("sends PATCH with updates", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ asset_id: "abc123", category: "rpg", tags: ["hero"] }),
    });

    const result = await updateLibraryAsset("abc123", { category: "rpg", tags: ["hero"] });
    expect(result.category).toBe("rpg");
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library/abc123");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(init.body)).toEqual({ category: "rpg", tags: ["hero"] });
  });
});

describe("addAssetTags", () => {
  it("sends POST with tags", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ asset_id: "abc123", tags: ["hero", "newtag"] }),
    });

    const result = await addAssetTags("abc123", ["newtag"]);
    expect(result.tags).toEqual(["hero", "newtag"]);
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library/abc123/tags");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body)).toEqual({ tags: ["newtag"] });
  });
});

describe("removeAssetTags", () => {
  it("sends DELETE with tags", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ asset_id: "abc123", tags: [] }),
    });

    const result = await removeAssetTags("abc123", ["hero"]);
    expect(result.tags).toEqual([]);
    const [url, init] = (global.fetch as jest.Mock).mock.calls[0];
    expect(url).toContain("/library/abc123/tags");
    expect(init.method).toBe("DELETE");
    expect(JSON.parse(init.body)).toEqual({ tags: ["hero"] });
  });
});
