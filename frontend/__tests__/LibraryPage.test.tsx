import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import LibraryPage from "../pages/library";

jest.mock("next/router", () => ({
  useRouter: () => ({ pathname: "/library" }),
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

const mockAssets = {
  assets: [
    {
      asset_id: "abc123",
      job_id: "abc123",
      asset_type: "character",
      prompt: "a pixel art hero sprite",
      quality_tier: "clean",
      tags: ["hero", "green"],
      category: "",
      thumbnail_path: null,
      zip_path: "/tmp/abc123.zip",
      output_paths: ["/tmp/abc123.png"],
      created_at: "2026-01-01T00:00:00Z",
      updated_at: "2026-01-01T00:00:00Z",
    },
    {
      asset_id: "def456",
      job_id: "def456",
      asset_type: "prop",
      prompt: "a pixel art chest",
      quality_tier: "clean",
      tags: [],
      category: "",
      thumbnail_path: null,
      zip_path: null,
      output_paths: ["/tmp/def456.png"],
      created_at: "2026-01-02T00:00:00Z",
      updated_at: "2026-01-02T00:00:00Z",
    },
  ],
  total: 2,
};

function mockFetch(overrides: Record<string, unknown> = {}) {
  const defaultOk = true;
  const defaultJson = async () => ({});

  const handlers: Record<string, { ok?: boolean; json?: () => Promise<unknown>; text?: () => Promise<string> }> = {
    "/library": { json: async () => mockAssets },
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

describe("LibraryPage", () => {
  it("renders heading and asset rows", async () => {
    mockFetch();
    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText("Asset Library")).toBeTruthy();
      expect(screen.getByText("abc123")).toBeTruthy();
      expect(screen.getByText("def456")).toBeTruthy();
    });

    expect(screen.getByText("a pixel art hero sprite")).toBeTruthy();
    expect(screen.getByText("character")).toBeTruthy();
  });

  it("shows empty state when no assets", async () => {
    mockFetch({ "/library": { json: async () => ({ assets: [], total: 0 }) } });
    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText(/No assets yet/)).toBeTruthy();
    });
  });

  it("displays tags and renders download link for zipped assets", async () => {
    mockFetch();
    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText("hero")).toBeTruthy();
      expect(screen.getByText("green")).toBeTruthy();
    });

    const downloadLinks = screen.getAllByText("Download");
    expect(downloadLinks).toHaveLength(1);
    expect(downloadLinks[0].getAttribute("href")).toContain("/download/abc123");
  });

  it("deletes an asset", async () => {
    mockFetch({
      "/library/abc123": { json: async () => ({ status: "deleted", asset_id: "abc123" }) },
    });

    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText("abc123")).toBeTruthy();
    });

    const deleteButtons = screen.getAllByText("Delete");
    fireEvent.click(deleteButtons[0]);

    await waitFor(() => {
      expect(screen.queryByText("abc123")).toBeNull();
      expect(screen.getByText("def456")).toBeTruthy();
    });
  });

  it("adds a tag to an asset", async () => {
    mockFetch({
      "/library/abc123/tags": {
        json: async () => ({
          ...mockAssets.assets[0],
          tags: ["hero", "green", "newtag"],
        }),
      },
    });

    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText("abc123")).toBeTruthy();
    });

    const input = screen.getAllByPlaceholderText("add tag(s)")[0];
    fireEvent.change(input, { target: { value: "newtag" } });
    fireEvent.click(screen.getAllByText("Add")[0]);

    await waitFor(() => {
      expect(screen.getByText("newtag")).toBeTruthy();
    });
  });

  it("filters assets via search input", async () => {
    global.fetch = jest.fn().mockImplementation((url: string) => {
      const urlStr = typeof url === "string" ? url : "";
      if (urlStr.includes("search=chest")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            assets: [mockAssets.assets[1]],
            total: 1,
          }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => mockAssets });
    });

    render(<LibraryPage />);

    await waitFor(() => {
      expect(screen.getByText("abc123")).toBeTruthy();
    });

    const search = screen.getByPlaceholderText("Search by prompt...");
    fireEvent.change(search, { target: { value: "chest" } });

    await waitFor(() => {
      expect(screen.getByText("a pixel art chest")).toBeTruthy();
      expect(screen.queryByText("a pixel art hero sprite")).toBeNull();
    });
  });
});
