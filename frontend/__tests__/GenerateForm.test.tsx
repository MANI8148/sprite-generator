import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import GenerateForm from "../components/GenerateForm";
import { generateAndWait } from "../lib/api";

jest.mock("../lib/api", () => ({
  generateAndWait: jest.fn(),
  getDownloadUrl: (jobId: string) => `http://localhost:8000/download/${jobId}`,
  getPreviewUrl: (jobId: string) => `http://localhost:8000/preview/${jobId}`,
}));

const doneResult = {
  job_id: "abc123",
  status: "done",
  prompt: "a pixel art hero sprite",
  quality_tier: "clean",
  validation: {},
  zip_path: "/tmp/abc123.zip",
  output_paths: ["/tmp/abc123.png"],
};

describe("GenerateForm", () => {
  beforeEach(() => {
    (generateAndWait as jest.Mock).mockReset();
  });

  it("submits the form and shows the generated result", async () => {
    (generateAndWait as jest.Mock).mockResolvedValue(doneResult);

    render(<GenerateForm />);
    fireEvent.click(screen.getByText("Generate Asset"));

    await waitFor(() => {
      expect(generateAndWait).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByText("a pixel art hero sprite")).toBeTruthy();
    });
    expect(screen.getByText("clean")).toBeTruthy();

    const img = screen.getByAltText("Generated sprite preview") as HTMLImageElement;
    expect(img).toBeTruthy();
    expect(img.getAttribute("src")).toContain("/preview/abc123");

    const link = screen.getByText("Download ZIP");
    expect(link.getAttribute("href")).toContain("/download/abc123");
  });

  it("shows progress updates while the job is queued/running", async () => {
    (generateAndWait as jest.Mock).mockImplementation(
      async (_req: unknown, opts: { onStatus?: (s: unknown) => void } = {}) => {
        opts.onStatus?.({ job_id: "abc123", status: "running" });
        return doneResult;
      }
    );

    render(<GenerateForm />);
    fireEvent.click(screen.getByText("Generate Asset"));

    await waitFor(() => {
      expect(screen.getByText("Job abc123: running...")).toBeTruthy();
    });
    await waitFor(() => {
      expect(screen.getByText("a pixel art hero sprite")).toBeTruthy();
    });
  });

  it("shows an error when generation fails", async () => {
    (generateAndWait as jest.Mock).mockRejectedValue(
      new Error("Generate failed: 503 Generator not set")
    );

    render(<GenerateForm />);
    fireEvent.click(screen.getByText("Generate Asset"));

    await waitFor(() => {
      expect(screen.getByText("Generate failed: 503 Generator not set")).toBeTruthy();
    });
  });

  it("renders all asset controls", () => {
    render(<GenerateForm />);
    expect(screen.getByText("Asset Type:")).toBeTruthy();
    expect(screen.getByText("View:")).toBeTruthy();
    expect(screen.getByText("Animation:")).toBeTruthy();
    expect(screen.getByText("Palette:")).toBeTruthy();
    expect(screen.getByText("Sprite Size:")).toBeTruthy();
    expect(screen.getByText("Export Engine:")).toBeTruthy();
  });
});
