import { useState } from "react";
import { generateAsset, GenerateRequest, GenerateResponse } from "../lib/api";

export default function GenerateForm() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<GenerateResponse | null>(null);
  const [error, setError] = useState("");

  const [form, setForm] = useState<GenerateRequest>({
    asset_type: "character",
    view: "front",
    animation: "idle",
    palette: "auto",
    sprite_size: "32x32",
    theme: "",
    seed: -1,
    remove_bg: true,
    reduce_palette: true,
    max_colors: 32,
    pixel_cleanup: true,
    auto_center: true,
    upscale: 1,
    engine: "godot",
    num_frames: 1,
  });

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const target = e.target;
    const name = target.name;
    const value =
      target.type === "checkbox"
        ? (target as HTMLInputElement).checked
        : target.value;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const data = await generateAsset(form);
      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1>Generate Sprite</h1>
      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", flexDirection: "column", gap: "1rem", maxWidth: "500px" }}
      >
        <label>
          Asset Type:
          <select name="asset_type" value={form.asset_type} onChange={handleChange}>
            <option value="character">Character</option>
            <option value="building">Building</option>
            <option value="vehicle">Vehicle</option>
            <option value="enemy">Enemy</option>
            <option value="prop">Prop</option>
          </select>
        </label>

        <label>
          View:
          <select name="view" value={form.view} onChange={handleChange}>
            <option value="front">Front</option>
            <option value="side">Side</option>
            <option value="top">Top</option>
            <option value="isometric">Isometric</option>
            <option value="back">Back</option>
          </select>
        </label>

        <label>
          Animation:
          <select name="animation" value={form.animation} onChange={handleChange}>
            <option value="idle">Idle</option>
            <option value="walk">Walk</option>
            <option value="run">Run</option>
            <option value="attack">Attack</option>
            <option value="jump">Jump</option>
          </select>
        </label>

        <label>
          Palette:
          <select name="palette" value={form.palette} onChange={handleChange}>
            <option value="auto">Auto</option>
            <option value="retro_16">Retro 16</option>
            <option value="retro_32">Retro 32</option>
            <option value="monochrome">Monochrome</option>
            <option value="vivid">Vivid</option>
          </select>
        </label>

        <label>
          Sprite Size:
          <select name="sprite_size" value={form.sprite_size} onChange={handleChange}>
            <option value="16x16">16x16</option>
            <option value="32x32">32x32</option>
            <option value="64x64">64x64</option>
            <option value="128x128">128x128</option>
          </select>
        </label>

        <label>
          Theme:
          <input
            type="text"
            name="theme"
            value={form.theme}
            onChange={handleChange}
            placeholder="fantasy, sci-fi, etc."
          />
        </label>

        <label>
          <input
            type="checkbox"
            name="remove_bg"
            checked={form.remove_bg}
            onChange={handleChange}
          />
          Remove Background
        </label>

        <label>
          Frames:
          <input
            type="number"
            name="num_frames"
            value={form.num_frames}
            onChange={handleChange}
            min={1}
            max={8}
          />
        </label>

        <button
          type="submit"
          disabled={loading}
          style={{
            padding: "0.75rem",
            background: "#7c7cff",
            color: "#fff",
            border: "none",
            borderRadius: "6px",
            fontSize: "1rem",
          }}
        >
          {loading ? "Generating..." : "Generate Asset"}
        </button>
      </form>

      {error && (
        <div style={{ color: "#ff6b6b", marginTop: "1rem" }}>{error}</div>
      )}

      {result && (
        <div
          style={{
            marginTop: "2rem",
            padding: "1rem",
            background: "#1a1a2e",
            borderRadius: "8px",
          }}
        >
          <h2>Result</h2>
          <p>
            <strong>Job ID:</strong> {result.job_id}
          </p>
          <p>
            <strong>Prompt:</strong> {result.prompt}
          </p>
          <p>
            <strong>Quality:</strong> {result.quality_tier}
          </p>
          <p>
            <strong>Outputs:</strong> {result.output_paths.join(", ")}
          </p>
          {result.zip_path && (
            <a href={`/api/download/${result.job_id}`} download>
              Download ZIP
            </a>
          )}
        </div>
      )}
    </div>
  );
}
