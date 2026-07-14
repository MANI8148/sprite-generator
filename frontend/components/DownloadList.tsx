import { useEffect, useState } from "react";
import { getHistory, HistoryEntry, getDownloadUrl } from "../lib/api";

export default function DownloadList() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getHistory()
      .then((data) => setEntries(data.filter((e) => e.zip_path)))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <p>Loading downloads...</p>;

  const downloadable = entries.filter((e) => e.zip_path);

  if (downloadable.length === 0) return <p>No downloadable assets yet. Generate one first.</p>;

  return (
    <div>
      <h1>Downloads</h1>
      <ul style={{ marginTop: "1rem" }}>
        {downloadable.map((entry) => (
          <li
            key={entry.job_id}
            style={{
              padding: "0.75rem",
              marginBottom: "0.5rem",
              background: "#1a1a2e",
              borderRadius: "6px",
              listStyle: "none",
            }}
          >
            <strong>{entry.job_id}</strong> — {entry.prompt.slice(0, 60)}...
            <br />
            <span style={{ color: "#999", fontSize: "0.9rem" }}>
              Quality: {entry.quality_tier}
            </span>
            <br />
            <a
              href={getDownloadUrl(entry.job_id)}
              download
              style={{ color: "#7c7cff" }}
            >
              Download ZIP
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
