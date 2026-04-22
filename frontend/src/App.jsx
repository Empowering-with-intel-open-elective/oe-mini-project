import { useState } from "react";
import axios from "axios";
import "./index.css";

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const onFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);
      setResult(null);
    }
  };

  const removeImage = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5001/predict", formData);
      setResult(res.data);
    } catch (err) {
      console.error("Upload failed", err);
      alert("Error connecting to the backend. Is the server running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <div className="header">
          <h2>Cancer Detection</h2>

        </div>

        {!preview ? (
          <div className="upload-area">
            <div className="upload-icon">+</div>
            <div className="upload-text">Upload Skin Image</div>
            <div className="upload-subtext">Clear, well-lit photos work best</div>
            <input type="file" accept="image/*" onChange={onFileChange} />
          </div>
        ) : (
          <div className="preview-container">
            <img src={preview} alt="Preview" className="preview-image" />
            <button className="remove-btn" onClick={removeImage}>✕</button>
          </div>
        )}

        <button
          className="predict-btn"
          onClick={handleUpload}
          disabled={!file || loading}
        >
          {loading ? <span className="loading-spinner"></span> : "Analysis"}
        </button>

        {result && (
          <div className="result-container">
            <div className="result-header">
              <div className="result-label">Analysis Outcome</div>
              <div className={`result-value ${result.is_cancerous ? 'cancerous' : ''}`}>
                {result.status}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;