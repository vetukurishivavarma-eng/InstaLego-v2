"use client";

import { useEffect, useRef, useState } from "react";

const DEFAULT_BACKEND_URL = process.env.NEXT_PUBLIC_API_BASE_URL || "";
const STORAGE_KEY = "landtitle_backend_url";

function normalizeBaseUrl(url) {
  return url.trim().replace(/\/+$/, "");
}

// Reads a failed fetch Response and pulls out the best available error
// message: FastAPI's {"detail": "..."} shape if present, otherwise raw text,
// otherwise a generic status-based fallback.
async function extractErrorDetail(response, fallbackPrefix) {
  let detail = `${fallbackPrefix} (status ${response.status}).`;
  try {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) {
      const body = await response.json();
      if (body && body.detail) detail = body.detail;
    } else {
      const text = await response.text();
      if (text) detail = text;
    }
  } catch {
    // fall back to the generic status-based message above
  }
  return detail;
}

const POLL_INTERVAL_MS = 4000;
const MAX_POLL_MS = 10 * 60 * 1000; // 10 minutes -- generous vs. observed multi-deed run times
const MAX_CONSECUTIVE_NETWORK_FAILURES = 5; // ~20s of tolerance for a brief hiccup mid-poll

// Polls GET /opinions/{jobId} until the job reaches "done" or "failed".
// `onStatus` is called with each intermediate status ("pending"/"running")
// so the UI can show live progress instead of a single static message.
// Tolerates a bounded number of consecutive NETWORK-level failures (the
// fetch itself throwing, e.g. a momentary tunnel/router blip) without
// aborting -- the whole point of polling instead of one long-held
// connection is that a brief hiccup shouldn't lose the result, so one
// failed poll must not be fatal the way one failed long-lived request was.
// An actual HTTP error response (the backend reachable but returning
// non-2xx) is still treated as fatal immediately, same as before.
async function pollJobUntilDone(base, jobId, onStatus) {
  const start = Date.now();
  let consecutiveNetworkFailures = 0;
  for (;;) {
    if (Date.now() - start > MAX_POLL_MS) {
      throw new Error(
        `Still not finished after ${Math.round(MAX_POLL_MS / 60000)} minutes. The job may still complete on ` +
          `the backend -- its id is ${jobId}, and results are retained for a while after finishing.`
      );
    }

    let res;
    try {
      res = await fetch(`${base}/opinions/${jobId}`);
    } catch {
      consecutiveNetworkFailures += 1;
      if (consecutiveNetworkFailures > MAX_CONSECUTIVE_NETWORK_FAILURES) {
        throw new Error(
          `Lost contact with the backend while checking on this job (id ${jobId}). It may still finish -- ` +
            `check your connection and try submitting again, or wait and check back.`
        );
      }
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
      continue;
    }
    consecutiveNetworkFailures = 0;

    if (!res.ok) {
      throw new Error(await extractErrorDetail(res, "Could not check job status"));
    }
    const record = await res.json();
    if (record.status === "done") return record;
    if (record.status === "failed") {
      throw new Error(record.error || "Pipeline failed for an unknown reason.");
    }
    onStatus?.(record.status);
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

export default function Home() {
  const [backendUrl, setBackendUrl] = useState(DEFAULT_BACKEND_URL);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [savedNotice, setSavedNotice] = useState(false);

  const [status, setStatus] = useState("idle"); // idle | loading | success | error
  const [jobStatus, setJobStatus] = useState(""); // pending | running, shown while loading
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
    setJobStatus("");

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
      // Submit + poll + download, all going straight from the browser to the
      // ngrok-exposed backend (never through a Next.js API route / serverless
      // function -- Vercel's free-tier function timeout (~10s) is far shorter
      // than this pipeline can take). Deliberately NOT the synchronous
      // /generate-opinion endpoint: holding one HTTP connection open for the
      // full multi-minute run proved unreliable in practice (a free ngrok
      // tunnel or an intermediate router/NAT can drop a long-idle connection
      // well before the backend actually finishes, even though the backend
      // itself completes the pipeline correctly) -- confirmed live, the
      // backend's own logs showed successful completion for a run the
      // browser had already reported as "Failed to fetch". Polling a short
      // status endpoint every few seconds never requires any single
      // connection to survive more than a few seconds.
      const createResponse = await fetch(`${base}/opinions`, {
        method: "POST",
        body: formData,
      });

      if (!createResponse.ok) {
        setStatus("error");
        setErrorMessage(await extractErrorDetail(createResponse, "Could not submit the job"));
        return;
      }

      const { job_id: jobId } = await createResponse.json();
      setJobStatus("pending");
      await pollJobUntilDone(base, jobId, setJobStatus);

      const downloadResponse = await fetch(`${base}/opinions/${jobId}/download`);
      if (!downloadResponse.ok) {
        setStatus("error");
        setErrorMessage(await extractErrorDetail(downloadResponse, "Job finished but the PDF could not be downloaded"));
        return;
      }

      const blob = await downloadResponse.blob();
      const objectUrl = URL.createObjectURL(blob);
      setPdfUrl(objectUrl);

      const disposition = downloadResponse.headers.get("content-disposition") || "";
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
          Processing{jobStatus ? ` (${jobStatus})` : "…"} — this can take several minutes (OCR
          plus several sequential LLM calls on the backend). Keep this tab open; it checks in
          with the backend every few seconds rather than holding one long connection open, so a
          brief network hiccup won&apos;t lose your result the way it used to.
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
