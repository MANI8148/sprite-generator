import { checkHealth, generateAsset, getHistory, getDownloadUrl } from "../lib/api";

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.restoreAllMocks();
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
