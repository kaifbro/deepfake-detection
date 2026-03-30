import React, { useState } from "react";

const API = "http://127.0.0.1:8000";

function App() {
  const [activeTab, setActiveTab] = useState("image");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setResult(null);
    setError(null);
    if (selected && selected.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(selected));
    } else {
      setPreview(null);
    }
  };

  const handleDetect = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API}/detect/${activeTab}`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Something went wrong. Make sure the backend is running.");
    }
    setLoading(false);
  };

  const acceptTypes = {
    image: "image/*",
    video: "video/*",
    audio: "audio/*",
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>🔍 Deepfake Detector</h1>
      <p style={styles.subtitle}>Detect deepfakes in images, videos, and audio</p>

      {/* Tabs */}
      <div style={styles.tabs}>
        {["image", "video", "audio"].map((tab) => (
          <button
            key={tab}
            style={{
              ...styles.tab,
              ...(activeTab === tab ? styles.activeTab : {}),
            }}
            onClick={() => {
              setActiveTab(tab);
              setFile(null);
              setPreview(null);
              setResult(null);
              setError(null);
            }}
          >
            {tab === "image" ? "🖼️ Image" : tab === "video" ? "🎥 Video" : "🎵 Audio"}
          </button>
        ))}
      </div>

      {/* Upload Box */}
      <div style={styles.uploadBox}>
        <input
          type="file"
          accept={acceptTypes[activeTab]}
          onChange={handleFileChange}
          style={styles.fileInput}
          id="fileInput"
        />
        <label htmlFor="fileInput" style={styles.fileLabel}>
          {file ? file.name : `Click to upload ${activeTab}`}
        </label>

        {preview && (
          <img src={preview} alt="preview" style={styles.preview} />
        )}
      </div>

      {/* Detect Button */}
      <button
        style={{
          ...styles.button,
          ...(loading ? styles.buttonDisabled : {}),
        }}
        onClick={handleDetect}
        disabled={!file || loading}
      >
        {loading ? "Analyzing..." : "Detect"}
      </button>

      {/* Error */}
      {error && <p style={styles.error}>{error}</p>}

      {/* Result */}
      {result && (
        <div style={{
          ...styles.resultBox,
          borderColor: result.verdict === "REAL" ? "#22c55e" : "#ef4444",
        }}>
          <h2 style={{
            ...styles.verdict,
            color: result.verdict === "REAL" ? "#22c55e" : "#ef4444",
          }}>
            {result.verdict === "REAL" ? "✅ REAL" : "❌ FAKE"}
          </h2>
          <div style={styles.probRow}>
            <span>Real</span>
            <div style={styles.barBg}>
              <div style={{
                ...styles.bar,
                width: `${result.real}%`,
                backgroundColor: "#22c55e",
              }} />
            </div>
            <span>{result.real}%</span>
          </div>
          <div style={styles.probRow}>
            <span>Fake</span>
            <div style={styles.barBg}>
              <div style={{
                ...styles.bar,
                width: `${result.fake}%`,
                backgroundColor: "#ef4444",
              }} />
            </div>
            <span>{result.fake}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    minHeight: "100vh",
    backgroundColor: "#0f172a",
    color: "#f1f5f9",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    padding: "40px 20px",
    fontFamily: "sans-serif",
  },
  title: {
    fontSize: "2.5rem",
    fontWeight: "bold",
    marginBottom: "8px",
  },
  subtitle: {
    color: "#94a3b8",
    marginBottom: "32px",
  },
  tabs: {
    display: "flex",
    gap: "12px",
    marginBottom: "32px",
  },
  tab: {
    padding: "10px 24px",
    borderRadius: "8px",
    border: "none",
    cursor: "pointer",
    backgroundColor: "#1e293b",
    color: "#94a3b8",
    fontSize: "1rem",
  },
  activeTab: {
    backgroundColor: "#6366f1",
    color: "#fff",
  },
  uploadBox: {
    width: "100%",
    maxWidth: "500px",
    backgroundColor: "#1e293b",
    borderRadius: "12px",
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: "16px",
    marginBottom: "24px",
  },
  fileInput: {
    display: "none",
  },
  fileLabel: {
    cursor: "pointer",
    padding: "12px 24px",
    backgroundColor: "#334155",
    borderRadius: "8px",
    color: "#f1f5f9",
    textAlign: "center",
    width: "100%",
  },
  preview: {
    maxWidth: "100%",
    maxHeight: "250px",
    borderRadius: "8px",
  },
  button: {
    padding: "12px 48px",
    backgroundColor: "#6366f1",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    fontSize: "1.1rem",
    cursor: "pointer",
    marginBottom: "24px",
  },
  buttonDisabled: {
    backgroundColor: "#475569",
    cursor: "not-allowed",
  },
  error: {
    color: "#ef4444",
  },
  resultBox: {
    width: "100%",
    maxWidth: "500px",
    backgroundColor: "#1e293b",
    borderRadius: "12px",
    padding: "24px",
    border: "2px solid",
  },
  verdict: {
    fontSize: "2rem",
    textAlign: "center",
    marginBottom: "16px",
  },
  probRow: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
    marginBottom: "12px",
  },
  barBg: {
    flex: 1,
    backgroundColor: "#334155",
    borderRadius: "4px",
    height: "12px",
    overflow: "hidden",
  },
  bar: {
    height: "100%",
    borderRadius: "4px",
    transition: "width 0.5s ease",
  },
};

export default App;