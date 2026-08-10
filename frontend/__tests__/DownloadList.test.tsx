import { render, screen, waitFor } from "@testing-library/react";
import DownloadList from "../components/DownloadList";
import HistoryList from "../components/HistoryList";

jest.mock("../lib/api", () => ({
  getHistory: jest.fn(),
  getDownloadUrl: (jobId: string) => `http://localhost:8000/download/${jobId}`,
  getPreviewUrl: (jobId: string) => `http://localhost:8000/preview/${jobId}`,
}));

import { getHistory } from "../lib/api";
import { HistoryEntry } from "../lib/api";

const entries: HistoryEntry[] = [
  {
    job_id: "job1",
    prompt: "a pixel art hero sprite",
    quality_tier: "clean",
    outputs: ["/tmp/job1.png"],
    zip_path: "/tmp/job1.zip",
  },
];

describe("DownloadList and HistoryList image previews", () => {
  beforeEach(() => {
    (getHistory as jest.Mock).mockReset();
    (getHistory as jest.Mock).mockResolvedValue(entries);
  });

  it("DownloadList renders an image preview per asset", async () => {
    render(<DownloadList />);

    await waitFor(() => {
      const img = screen.getByAltText("Asset preview") as HTMLImageElement;
      expect(img).toBeTruthy();
      expect(img.getAttribute("src")).toContain("/preview/job1");
    });
  });

  it("HistoryList renders an image preview per row", async () => {
    render(<HistoryList />);

    await waitFor(() => {
      const img = screen.getByAltText("Asset preview") as HTMLImageElement;
      expect(img).toBeTruthy();
      expect(img.getAttribute("src")).toContain("/preview/job1");
    });
  });

  it("DownloadList shows an empty state when nothing is downloadable", async () => {
    (getHistory as jest.Mock).mockResolvedValue([]);

    render(<DownloadList />);

    await waitFor(() => {
      expect(
        screen.getByText("No downloadable assets yet. Generate one first.")
      ).toBeTruthy();
    });
  });
});