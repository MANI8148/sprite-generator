const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface GenerateRequest {
  asset_type?: string;
  view?: string;
  animation?: string;
  palette?: string;
  sprite_size?: string;
  theme?: string;
  seed?: number;
  remove_bg?: boolean;
  reduce_palette?: boolean;
  max_colors?: number;
  pixel_cleanup?: boolean;
  auto_center?: boolean;
  upscale?: number;
  engine?: string;
  num_frames?: number;
}

export interface GenerateResponse {
  job_id: string;
  prompt: string;
  quality_tier: string;
  validation: Record<string, unknown>;
  zip_path: string | null;
  output_paths: string[];
}

export interface HealthResponse {
  status: string;
  generator_loaded: boolean;
}

export interface HistoryEntry {
  job_id: string;
  prompt: string;
  quality_tier: string;
  outputs: string[];
  zip_path: string | null;
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function generateAsset(
  req: GenerateRequest
): Promise<GenerateResponse> {
  const res = await fetch(`${API_BASE}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Generate failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const res = await fetch(`${API_BASE}/history`);
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}
