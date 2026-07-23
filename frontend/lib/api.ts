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

export interface BalanceResponse {
  user_id: string;
  balance: number;
  generation_cost: number;
}

export interface PackageInfo {
  key: string;
  credits: number;
  amount_cents: number;
  description: string;
}

export interface TransactionEntry {
  transaction_id: string;
  amount: number;
  reason: string;
  timestamp: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  username: string;
  user_id: string;
}

export interface UserInfo {
  username: string;
  user_id: string;
}

const TOKEN_KEY = "sprite_gen_token";

export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setAuthToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
}

export function clearAuthToken(): void {
  localStorage.removeItem(TOKEN_KEY);
}

function authHeaders(): Record<string, string> {
  const token = getAuthToken();
  return token ? { Authorization: `Bearer ${token}` } : {};
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
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify(req),
  });
  if (!res.ok) throw new Error(`Generate failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getHistory(): Promise<HistoryEntry[]> {
  const res = await fetch(`${API_BASE}/history`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`History fetch failed: ${res.status}`);
  return res.json();
}

export function getDownloadUrl(jobId: string): string {
  return `${API_BASE}/download/${jobId}`;
}

export async function register(
  username: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(`Register failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  setAuthToken(data.access_token);
  return data;
}

export async function login(
  username: string,
  password: string
): Promise<AuthResponse> {
  const res = await fetch(`${API_BASE}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(`Login failed: ${res.status} ${await res.text()}`);
  const data = await res.json();
  setAuthToken(data.access_token);
  return data;
}

export async function getMe(): Promise<UserInfo> {
  const res = await fetch(`${API_BASE}/auth/me`, { headers: authHeaders() });
  if (!res.ok) throw new Error(`Get user failed: ${res.status}`);
  return res.json();
}

export async function getBillingBalance(): Promise<BalanceResponse> {
  const res = await fetch(`${API_BASE}/billing/balance`, {
    headers: authHeaders(),
  });
  if (!res.ok) throw new Error(`Balance fetch failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getBillingTransactions(): Promise<{
  user_id: string;
  transactions: TransactionEntry[];
}> {
  const res = await fetch(`${API_BASE}/billing/transactions`, {
    headers: authHeaders(),
  });
  if (!res.ok)
    throw new Error(`Transactions fetch failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getBillingPackages(): Promise<{
  packages: PackageInfo[];
}> {
  const res = await fetch(`${API_BASE}/billing/packages`);
  if (!res.ok) throw new Error(`Packages fetch failed: ${res.status}`);
  return res.json();
}

export async function topupCredits(
  amount: number,
  reason?: string
): Promise<{ user_id: string; balance: number; amount_added: number }> {
  const res = await fetch(`${API_BASE}/billing/topup`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders() },
    body: JSON.stringify({ amount, reason: reason || "topup" }),
  });
  if (!res.ok) throw new Error(`Topup failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getCostEstimate(
  numFrames?: number
): Promise<{ generation_cost: number; num_frames: number; total_cost: number }> {
  const params = numFrames ? `?num_frames=${numFrames}` : "";
  const res = await fetch(`${API_BASE}/billing/cost-estimate${params}`);
  if (!res.ok) throw new Error(`Cost estimate failed: ${res.status}`);
  return res.json();
}
