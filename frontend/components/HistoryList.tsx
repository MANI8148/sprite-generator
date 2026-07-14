import { useEffect, useState } from "react";
import { getHistory, HistoryEntry } from "../lib/api";

export default function HistoryList() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    getHistory()
      .then(setEntries)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading history...</p>;
  if (error) return <p style={{ color: "#ff6b6b" }}>{error}</p>;
  if (entries.length === 0) return <p>No generations yet.</p>;

  return (
    <div>
      <h1>Generation History</h1>
      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #333", textAlign: "left" }}>
            <th style={{ padding: "0.5rem" }}>Job ID</th>
            <th style={{ padding: "0.5rem" }}>Prompt</th>
            <th style={{ padding: "0.5rem" }}>Quality</th>
            <th style={{ padding: "0.5rem" }}>Download</th>
          </tr>
        </thead>
        <tbody>
          {entries.map((entry) => (
            <tr key={entry.job_id} style={{ borderBottom: "1px solid #222" }}>
              <td style={{ padding: "0.5rem" }}>{entry.job_id}</td>
              <td style={{ padding: "0.5rem", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis" }}>
                {entry.prompt}
              </td>
              <td style={{ padding: "0.5rem" }}>{entry.quality_tier}</td>
              <td style={{ padding: "0.5rem" }}>
                {entry.zip_path ? (
                  <a href={`/api/download/${entry.job_id}`} download>
                    Download ZIP
                  </a>
                ) : (
                  "N/A"
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
