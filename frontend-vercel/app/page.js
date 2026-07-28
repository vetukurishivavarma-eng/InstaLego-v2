"use client";

import { useEffect, useRef, useState } from "react";

const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const STORAGE_KEY = "landtitle_backend_url";

function normalizeBaseUrl(url) {
  return url.trim().replace(/\/+$/, "");
}

export default function Home() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [savedNotice, setSavedNotice] = useState(false);

  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [errorMessage, setErrorMessage] = useState("");
  const [pdfUrl, setPdfUrl] = useState(null);
  const [pdfFilename, setPdfFilename] = useState("legal-opinion.pdf");

  const saleDeedRef = useRef(null);
  const revenueRecordRef = useRef(null);
  const ecRef = useRef(null);

  // Load any previously saved backend URL override from localStorage on mount.
  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored) {
        setBackendUrl(stored);
      }
    } catch {
      // localStorage unavailable (e.g. private browsing) — fall back to env default silently.
    }
  }, []);

  // Clean up the object URL for the PDF preview when it changes or the component unmounts.
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  function handleBackendUrlChange(e) {
    const value = e.target.value;
    setBackendUrl(value);
    try {
      window.localStorage.setItem(STORAGE_KEY, value);
      setSavedNotice(true);
      setTimeout(() => setSavedNotice(false), 1500);
    } catch {
      // ignore
    }
  }

  function handleResetBackendUrl() {
    setBackendUrl(DEFAULT_BACKEND_URL);
    try {
      window.localStorage.removeItem(STORAGE_KEY);
    } catch {
      // ignore
    }
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMessage("");

    const saleDeedFiles = saleDeedRef.current?.files;
    if (!saleDeedFiles || saleDeedFiles.length === 0) {
      setStatus("error");
      setErrorMessage("Please select at least one Sale Deed file.");
      return;
    }

    const base = normalizeBaseUrl(backendUrl || "");
    if (!base) {
      setStatus("error");
      setErrorMessage(
        "No backend URL is configured. Set NEXT_PUBLIC_API_BASE_URL at build time, or fill in the Backend URL field below."
      );
      return;
    }

    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl);
      setPdfUrl(null);
    }

    setStatus("loading");

    const formData = new FormData();
    for (const file of saleDeedFiles) {
      formData.append("sale_deed", file);
    }
    const revenueRecordFile = revenueRecordRef.current?.files?.[0];
    if (revenueRecordFile) {
      formData.append("revenue_record", revenueRecordFile);
    }
    const ecFile = ecRef.current?.files?.[0];
    if (ecFile) {
      formData.append("ec", ecFile);
    }

    try {
      // This fetch goes straight from the browser to the ngrok-exposed backend.
      // It must never be routed through a Next.js API route / serverless function,
      // since Vercel's free-tier function timeout (~10s) is far shorter than the
      // several minutes this pipeline can take.
      const response = await fetch(`${base}/generate-opinion`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let detail = `Request failed with status ${response.status}.`;
        try {
          const contentType = response.headers.get("content-type") || "";
          if (contentType.includes("application/json")) {
            const body = await response.json();
            if (body && body.detail) {
              detail = body.detail;
            }
          } else {
            const text = await response.text();
            if (text) detail = text;
          }
        } catch {
          // fall back to the generic status-based message above
        }
        setStatus("error");
        setErrorMessage(detail);
        return;
      }

      const blob = await response.blob();
      const objectUrl = URL.createObjectURL(blob);
      setPdfUrl(objectUrl);

      const disposition = response.headers.get("content-disposition") || "";
      const match = disposition.match(/filename="?([^"]+)"?/i);
      if (match && match[1]) {
        setPdfFilename(match[1]);
      }

      setStatus("success");
    } catch (err) {
      setStatus("error");
      setErrorMessage(
        `Could not reach the backend at ${base}. ${
          err instanceof Error ? err.message : String(err)
        }`
      );
    }
  }

  const isLoading = status === "loading";

  return (
    <main className="page">
      <h1>Land Title Diligence — Generate Opinion</h1>
      <p className="subtitle">
        Upload title documents to generate a Legal Opinion PDF. This calls your
        backend directly from the browser (no Vercel proxy), so it works even
        though the backend is running elsewhere and can take several minutes.
      </p>

      <form onSubmit={handleSubmit}>
        <div className="field">
          <label htmlFor="sale_deed">
            Sale Deed <span className="hint">required, multiple files allowed</span>
          </label>
          <input
            id="sale_deed"
            ref={saleDeedRef}
            type="file"
            multiple
            required
            disabled={isLoading}
          />
        </div>

        <div className="field">
          <label htmlFor="revenue_record">
            Revenue Record <span className="hint">optional</span>
          </label>
          <input
            id="revenue_record"
            ref={revenueRecordRef}
            type="file"
            disabled={isLoading}
          />
        </div>

        <div className="field">
          <label htmlFor="ec">
            Encumbrance Certificate <span className="hint">optional</span>
          </label>
          <input id="ec" ref={ecRef} type="file" disabled={isLoading} />
        </div>

        <details
          className="settings"
          open={settingsOpen}
          onToggle={(e) => setSettingsOpen(e.target.open)}
        >
          <summary>Backend URL settings</summary>
          <div className="backendRow">
            <input
              type="text"
              value={backendUrl}
              onChange={handleBackendUrlChange}
              placeholder="https://your-tunnel.ngrok-free.app"
              spellCheck={false}
            />
            <button type="button" onClick={handleResetBackendUrl}>
              Reset to default
            </button>
          </div>
          <div className="saved">
            {savedNotice
              ? "Saved to this browser."
              : "Overrides the build-time NEXT_PUBLIC_API_BASE_URL, saved in this browser's localStorage. Update this if your ngrok tunnel restarts with a new URL — no redeploy needed."}
          </div>
        </details>

        <button type="submit" className="submitBtn" disabled={isLoading}>
          {isLoading ? "Processing…" : "Submit"}
        </button>
      </form>

      {status === "loading" && (
        <div className="statusBox loading">
          Processing… this can take several minutes (OCR plus several sequential
          LLM calls on the backend). Please keep this tab open.
        </div>
      )}

      {status === "error" && (
        <div className="statusBox error">{errorMessage}</div>
      )}

      {status === "success" && pdfUrl && (
        <div className="statusBox success">
          <div>Legal opinion generated successfully.</div>
          <div className="resultActions">
            <a className="downloadLink" href={pdfUrl} download={pdfFilename}>
              Download {pdfFilename}
            </a>
          </div>
          <iframe className="previewFrame" src={pdfUrl} title="Legal opinion PDF preview" />
        </div>
      )}
    </main>
  );
}
