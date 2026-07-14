import { useState, useEffect } from "react";
import { checkHealth, HealthResponse } from "../lib/api";

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState(
    process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
  );
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    setApiUrl(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000");
  }, []);

  const handleCheckHealth = async () => {
    setChecking(true);
    try {
      const result = await checkHealth();
      setHealth(result);
    } catch {
      setHealth({ status: "unreachable", generator_loaded: false });
    } finally {
      setChecking(false);
    }
  };

  return (
    <div>
      <h1>Settings</h1>

      <section style={{ marginTop: "1.5rem" }}>
        <h2>API Connection</h2>
        <p>
          <strong>API URL:</strong> {apiUrl}
        </p>
        <button
          onClick={handleCheckHealth}
          disabled={checking}
          style={{
            padding: "0.5rem 1rem",
            background: "#7c7cff",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            marginTop: "0.5rem",
          }}
        >
          {checking ? "Checking..." : "Check API Health"}
        </button>

        {health && (
          <div
            style={{
              marginTop: "1rem",
              padding: "0.75rem",
              background: "#1a1a2e",
              borderRadius: "6px",
            }}
          >
            <p>
              <strong>Status:</strong>{" "}
              <span style={{ color: health.status === "ok" ? "#4caf50" : "#ff6b6b" }}>
                {health.status}
              </span>
            </p>
            <p>
              <strong>Generator Loaded:</strong>{" "}
              {health.generator_loaded ? "Yes" : "No"}
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
