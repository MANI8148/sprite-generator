import { useState, useEffect } from "react";
import {
  getBillingBalance,
  getBillingPackages,
  getBillingTransactions,
  getCostEstimate,
  topupCredits,
  register,
  login,
  getMe,
  getAuthToken,
  clearAuthToken,
  BalanceResponse,
  PackageInfo,
  TransactionEntry,
  UserInfo,
} from "../lib/api";

export default function BillingPage() {
  const [user, setUser] = useState<UserInfo | null>(null);
  const [balance, setBalance] = useState<BalanceResponse | null>(null);
  const [packages, setPackages] = useState<PackageInfo[]>([]);
  const [transactions, setTransactions] = useState<TransactionEntry[]>([]);
  const [costEstimate, setCostEstimate] = useState<{
    generation_cost: number;
    num_frames: number;
    total_cost: number;
  } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");

  const [topupAmount, setTopupAmount] = useState(100);
  const [numFrames, setNumFrames] = useState(1);

  useEffect(() => {
    getBillingPackages()
      .then((res) => setPackages(res.packages))
      .catch(() => {});
    getCostEstimate(1)
      .then(setCostEstimate)
      .catch(() => {});
    const token = getAuthToken();
    if (token) {
      loadAuthedData();
    }
  }, []);

  async function loadAuthedData() {
    setLoading(true);
    setError("");
    try {
      const [u, b, txs] = await Promise.all([
        getMe(),
        getBillingBalance(),
        getBillingTransactions(),
      ]);
      setUser(u);
      setBalance(b);
      setTransactions(txs.transactions);
    } catch (err: unknown) {
      if (err instanceof Error) {
        if (err.message.includes("401") || err.message.includes("422")) {
          clearAuthToken();
          setUser(null);
          setBalance(null);
        }
        setError(err.message);
      }
    } finally {
      setLoading(false);
    }
  }

  async function handleAuth(e: React.FormEvent) {
    e.preventDefault();
    setAuthError("");
    setLoading(true);
    try {
      if (authMode === "register") {
        await register(authUsername, authPassword);
      } else {
        await login(authUsername, authPassword);
      }
      setAuthUsername("");
      setAuthPassword("");
      await loadAuthedData();
    } catch (err: unknown) {
      setAuthError(err instanceof Error ? err.message : "Auth failed");
    } finally {
      setLoading(false);
    }
  }

  function handleLogout() {
    clearAuthToken();
    setUser(null);
    setBalance(null);
    setTransactions([]);
  }

  async function handleTopup() {
    setLoading(true);
    setError("");
    try {
      const result = await topupCredits(topupAmount);
      setBalance((prev) =>
        prev ? { ...prev, balance: result.balance } : null
      );
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Topup failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleEstimateChange(frames: number) {
    setNumFrames(frames);
    try {
      const est = await getCostEstimate(frames);
      setCostEstimate(est);
    } catch {
      // ignore
    }
  }

  return (
    <div>
      <h1>Billing</h1>

      {!user && (
        <section
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            background: "#1a1a2e",
            borderRadius: "6px",
          }}
        >
          <h2>{authMode === "login" ? "Login" : "Register"}</h2>
          <form onSubmit={handleAuth}>
            <div style={{ marginBottom: "0.5rem" }}>
              <input
                placeholder="Username"
                value={authUsername}
                onChange={(e) => setAuthUsername(e.target.value)}
                required
                minLength={3}
                style={{
                  padding: "0.5rem",
                  background: "#16213e",
                  color: "#fff",
                  border: "1px solid #333",
                  borderRadius: "4px",
                  width: "200px",
                }}
              />
            </div>
            <div style={{ marginBottom: "0.5rem" }}>
              <input
                type="password"
                placeholder="Password"
                value={authPassword}
                onChange={(e) => setAuthPassword(e.target.value)}
                required
                minLength={6}
                style={{
                  padding: "0.5rem",
                  background: "#16213e",
                  color: "#fff",
                  border: "1px solid #333",
                  borderRadius: "4px",
                  width: "200px",
                }}
              />
            </div>
            {authError && (
              <p style={{ color: "#ff6b6b", fontSize: "0.875rem" }}>
                {authError}
              </p>
            )}
            <button
              type="submit"
              disabled={loading}
              style={{
                padding: "0.5rem 1rem",
                background: "#7c7cff",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
                marginRight: "0.5rem",
              }}
            >
              {loading ? "..." : authMode === "login" ? "Login" : "Register"}
            </button>
            <button
              type="button"
              onClick={() =>
                setAuthMode(authMode === "login" ? "register" : "login")
              }
              style={{
                padding: "0.5rem 1rem",
                background: "transparent",
                color: "#7c7cff",
                border: "1px solid #7c7cff",
                borderRadius: "6px",
              }}
            >
              Switch to {authMode === "login" ? "Register" : "Login"}
            </button>
          </form>
        </section>
      )}

      {user && (
        <section
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            background: "#1a1a2e",
            borderRadius: "6px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <div>
              <h2>
                Welcome, {user.username}
              </h2>
              <p>
                <strong>Balance:</strong>{" "}
                <span
                  style={{
                    color: "#4caf50",
                    fontSize: "1.25rem",
                  }}
                >
                  {balance?.balance ?? "—"}
                </span>{" "}
                credits
              </p>
              <p>
                <strong>Generation Cost:</strong> {balance?.generation_cost ?? "—"}{" "}
                credit(s) per frame
              </p>
            </div>
            <button
              onClick={handleLogout}
              style={{
                padding: "0.5rem 1rem",
                background: "#ff6b6b",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
              }}
            >
              Logout
            </button>
          </div>
        </section>
      )}

      {user && (
        <section
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            background: "#1a1a2e",
            borderRadius: "6px",
          }}
        >
          <h2>Top Up Credits</h2>
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            <input
              type="number"
              min={1}
              value={topupAmount}
              onChange={(e) => setTopupAmount(Math.max(1, Number(e.target.value)))}
              style={{
                padding: "0.5rem",
                background: "#16213e",
                color: "#fff",
                border: "1px solid #333",
                borderRadius: "4px",
                width: "120px",
              }}
            />
            <button
              onClick={handleTopup}
              disabled={loading}
              style={{
                padding: "0.5rem 1rem",
                background: "#4caf50",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
              }}
            >
              {loading ? "..." : "Add Credits"}
            </button>
          </div>
          {error && (
            <p style={{ color: "#ff6b6b", fontSize: "0.875rem", marginTop: "0.5rem" }}>
              {error}
            </p>
          )}
        </section>
      )}

      <section
        style={{
          marginTop: "1.5rem",
          padding: "1rem",
          background: "#1a1a2e",
          borderRadius: "6px",
        }}
      >
        <h2>Credit Packages</h2>
        {packages.length === 0 && <p>Loading packages...</p>}
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          {packages.map((pkg) => (
            <div
              key={pkg.key}
              style={{
                padding: "1rem",
                background: "#16213e",
                borderRadius: "6px",
                minWidth: "180px",
                border: "1px solid #333",
              }}
            >
              <h3 style={{ margin: 0, textTransform: "capitalize" }}>
                {pkg.key}
              </h3>
              <p style={{ fontSize: "0.875rem", color: "#999" }}>
                {pkg.description}
              </p>
              <p>
                <strong>{pkg.credits}</strong> credits
              </p>
            </div>
          ))}
        </div>
      </section>

      <section
        style={{
          marginTop: "1.5rem",
          padding: "1rem",
          background: "#1a1a2e",
          borderRadius: "6px",
        }}
      >
        <h2>Cost Estimate</h2>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <label>Frames:</label>
          <input
            type="number"
            min={1}
            max={100}
            value={numFrames}
            onChange={(e) => handleEstimateChange(Math.max(1, Number(e.target.value)))}
            style={{
              padding: "0.5rem",
              background: "#16213e",
              color: "#fff",
              border: "1px solid #333",
              borderRadius: "4px",
              width: "80px",
            }}
          />
        </div>
        {costEstimate && (
          <p style={{ marginTop: "0.5rem" }}>
            <strong>
              {costEstimate.num_frames} frame(s)
            </strong>{" "}
            x {costEstimate.generation_cost} credit(s) ={" "}
            <span style={{ color: "#4caf50" }}>{costEstimate.total_cost}</span>{" "}
            credit(s)
          </p>
        )}
      </section>

      {user && transactions.length > 0 && (
        <section
          style={{
            marginTop: "1.5rem",
            padding: "1rem",
            background: "#1a1a2e",
            borderRadius: "6px",
          }}
        >
          <h2>Transaction History</h2>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontSize: "0.875rem",
            }}
          >
            <thead>
              <tr style={{ borderBottom: "1px solid #333" }}>
                <th style={{ textAlign: "left", padding: "0.5rem" }}>ID</th>
                <th style={{ textAlign: "right", padding: "0.5rem" }}>Amount</th>
                <th style={{ textAlign: "left", padding: "0.5rem" }}>Reason</th>
                <th style={{ textAlign: "left", padding: "0.5rem" }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {transactions.map((tx) => (
                <tr key={tx.transaction_id} style={{ borderBottom: "1px solid #222" }}>
                  <td style={{ padding: "0.5rem", color: "#999" }}>
                    {tx.transaction_id}
                  </td>
                  <td
                    style={{
                      padding: "0.5rem",
                      textAlign: "right",
                      color: tx.amount > 0 ? "#4caf50" : "#ff6b6b",
                    }}
                  >
                    {tx.amount > 0 ? "+" : ""}
                    {tx.amount}
                  </td>
                  <td style={{ padding: "0.5rem" }}>{tx.reason}</td>
                  <td style={{ padding: "0.5rem", color: "#999" }}>
                    {new Date(tx.timestamp).toLocaleDateString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}