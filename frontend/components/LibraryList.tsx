import { useCallback, useEffect, useState } from "react";
import {
  getLibrary,
  deleteLibraryAsset,
  addAssetTags,
  removeAssetTags,
  getDownloadUrl,
  LibraryAsset,
} from "../lib/api";

export default function LibraryList() {
  const [assets, setAssets] = useState<LibraryAsset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [assetType, setAssetType] = useState("");
  const [tagInput, setTagInput] = useState<Record<string, string>>({});

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    getLibrary({ search: search || undefined, asset_type: assetType || undefined })
      .then((data) => setAssets(data.assets))
      .catch((err) => setError(err instanceof Error ? err.message : "Library load failed"))
      .finally(() => setLoading(false));
  }, [search, assetType]);

  useEffect(() => {
    load();
  }, [load]);

  const handleDelete = async (assetId: string) => {
    try {
      await deleteLibraryAsset(assetId);
      setAssets((prev) => prev.filter((a) => a.asset_id !== assetId));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Delete failed");
    }
  };

  const handleAddTag = async (assetId: string) => {
    const raw = tagInput[assetId]?.trim() ?? "";
    if (!raw) return;
    const tags = raw.split(",").map((t) => t.trim()).filter(Boolean);
    try {
      const updated = await addAssetTags(assetId, tags);
      setAssets((prev) => prev.map((a) => (a.asset_id === assetId ? updated : a)));
      setTagInput((prev) => ({ ...prev, [assetId]: "" }));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Add tag failed");
    }
  };

  const handleRemoveTag = async (assetId: string, tag: string) => {
    try {
      const updated = await removeAssetTags(assetId, [tag]);
      setAssets((prev) => prev.map((a) => (a.asset_id === assetId ? updated : a)));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Remove tag failed");
    }
  };

  if (loading && assets.length === 0) return <p>Loading asset library...</p>;
  if (error && assets.length === 0) return <p style={{ color: "#ff6b6b" }}>{error}</p>;
  if (assets.length === 0)
    return (
      <div>
        <h1>Asset Library</h1>
        <p>No assets yet. Generate one first.</p>
      </div>
    );

  return (
    <div>
      <h1>Asset Library</h1>

      <div style={{ display: "flex", gap: "1rem", margin: "1rem 0" }}>
        <input
          type="text"
          placeholder="Search by prompt..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ padding: "0.5rem", background: "#1a1a2e", color: "#fff", border: "1px solid #333", borderRadius: "4px" }}
        />
        <select
          value={assetType}
          onChange={(e) => setAssetType(e.target.value)}
          style={{ padding: "0.5rem", background: "#1a1a2e", color: "#fff", border: "1px solid #333", borderRadius: "4px" }}
        >
          <option value="">All types</option>
          <option value="character">Character</option>
          <option value="building">Building</option>
          <option value="vehicle">Vehicle</option>
          <option value="enemy">Enemy</option>
          <option value="prop">Prop</option>
          <option value="tileset">Tileset</option>
          <option value="ui">UI</option>
          <option value="portrait">Portrait</option>
        </select>
        <button
          onClick={load}
          style={{ padding: "0.5rem 1rem", background: "#7c7cff", color: "#fff", border: "none", borderRadius: "4px" }}
        >
          Refresh
        </button>
      </div>

      {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}

      <table style={{ width: "100%", borderCollapse: "collapse", marginTop: "1rem" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid #333", textAlign: "left" }}>
            <th style={{ padding: "0.5rem" }}>Asset ID</th>
            <th style={{ padding: "0.5rem" }}>Type</th>
            <th style={{ padding: "0.5rem" }}>Prompt</th>
            <th style={{ padding: "0.5rem" }}>Quality</th>
            <th style={{ padding: "0.5rem" }}>Tags</th>
            <th style={{ padding: "0.5rem" }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {assets.map((asset) => (
            <tr key={asset.asset_id} style={{ borderBottom: "1px solid #222", verticalAlign: "top" }}>
              <td style={{ padding: "0.5rem" }}>{asset.asset_id}</td>
              <td style={{ padding: "0.5rem" }}>{asset.asset_type}</td>
              <td style={{ padding: "0.5rem", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis" }}>
                {asset.prompt}
              </td>
              <td style={{ padding: "0.5rem" }}>{asset.quality_tier}</td>
              <td style={{ padding: "0.5rem" }}>
                {asset.tags.map((tag) => (
                  <span
                    key={tag}
                    style={{
                      display: "inline-block",
                      background: "#2a2a4a",
                      color: "#9f9fff",
                      borderRadius: "999px",
                      padding: "0.1rem 0.6rem",
                      margin: "0.1rem 0.25rem 0.1rem 0",
                      fontSize: "0.85rem",
                    }}
                  >
                    {tag}
                    <button
                      onClick={() => handleRemoveTag(asset.asset_id, tag)}
                      aria-label={`Remove tag ${tag}`}
                      style={{
                        background: "none",
                        border: "none",
                        color: "#ff6b6b",
                        marginLeft: "0.35rem",
                        cursor: "pointer",
                      }}
                    >
                      x
                    </button>
                  </span>
                ))}
                <div style={{ marginTop: "0.35rem" }}>
                  <input
                    type="text"
                    placeholder="add tag(s)"
                    value={tagInput[asset.asset_id] ?? ""}
                    onChange={(e) =>
                      setTagInput((prev) => ({ ...prev, [asset.asset_id]: e.target.value }))
                    }
                    style={{ padding: "0.25rem", background: "#1a1a2e", color: "#fff", border: "1px solid #333", borderRadius: "4px", width: "110px" }}
                  />
                  <button
                    onClick={() => handleAddTag(asset.asset_id)}
                    style={{ marginLeft: "0.35rem", padding: "0.25rem 0.6rem", background: "#2a2a4a", color: "#fff", border: "none", borderRadius: "4px", cursor: "pointer" }}
                  >
                    Add
                  </button>
                </div>
              </td>
              <td style={{ padding: "0.5rem" }}>
                {asset.zip_path ? (
                  <a href={getDownloadUrl(asset.job_id)} download style={{ color: "#7c7cff" }}>
                    Download
                  </a>
                ) : (
                  "N/A"
                )}
                <span style={{ margin: "0 0.4rem", color: "#555" }}>|</span>
                <button
                  onClick={() => handleDelete(asset.asset_id)}
                  style={{ background: "none", border: "none", color: "#ff6b6b", cursor: "pointer" }}
                >
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
