"use client";

import { useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Where the FastAPI backend lives. Local default; on Vercel set NEXT_PUBLIC_API_URL.
const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const EXAMPLES = [
  "Compare LangGraph and CrewAI for building multi-agent systems",
  "What are the tradeoffs of RAG vs fine-tuning for domain QA?",
  "What is the current state of AI agent frameworks in 2026?",
];

export default function Home() {
  const [goal, setGoal] = useState("");
  const [running, setRunning] = useState(false);
  const [steps, setSteps] = useState([]);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const doneRef = useRef(false);

  function research(q) {
    const query = (q ?? goal).trim();
    if (!query || running) return;
    setGoal(query);
    setRunning(true);
    setSteps([]);
    setResult(null);
    setError("");
    doneRef.current = false;

    const es = new EventSource(`${API}/research?goal=${encodeURIComponent(query)}`);

    es.addEventListener("step", (e) => {
      const data = JSON.parse(e.data);
      setSteps((prev) => [...prev, data.message]);
    });

    es.addEventListener("done", (e) => {
      doneRef.current = true;
      setResult(JSON.parse(e.data));
      setRunning(false);
      es.close();
    });

    // Fires for our server-sent 'error' events (have .data) AND for connection
    // problems (no .data). Ignore once we've already finished.
    es.addEventListener("error", (e) => {
      if (doneRef.current) return;
      if (e.data) {
        try {
          setError(JSON.parse(e.data).message || "Something went wrong.");
        } catch {
          setError("Something went wrong.");
        }
      } else {
        setError(`Could not reach the backend at ${API}. Is it running?`);
      }
      setRunning(false);
      es.close();
    });
  }

  return (
    <main className="wrap">
      <h1>🧭 Atlas</h1>
      <p className="sub">
        A multi-agent researcher. It plans, searches the web, reads sources,
        synthesizes, writes a <strong>cited</strong> brief, and critiques itself
        until it&apos;s good enough.
      </p>

      <div className="bar">
        <input
          value={goal}
          onChange={(e) => setGoal(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && research()}
          placeholder="Ask Atlas to research something…"
          disabled={running}
        />
        <button onClick={() => research()} disabled={running || !goal.trim()}>
          {running ? "Researching…" : "Research"}
        </button>
      </div>

      {!running && !result && (
        <div className="examples">
          {EXAMPLES.map((ex) => (
            <button key={ex} className="chip" onClick={() => research(ex)}>
              {ex}
            </button>
          ))}
        </div>
      )}

      {error && <div className="error">{error}</div>}

      {(running || steps.length > 0) && (
        <section className="card">
          <h3>Agents at work</h3>
          {steps.map((s, i) => (
            <div key={i} className="step">
              {s}
            </div>
          ))}
          {running && <div className="step pulse">working…</div>}
        </section>
      )}

      {result && (
        <section className="card report">
          <h3>Research brief</h3>
          <div className="stats">
            <span>{result.sources} sources</span>
            <span>{result.evidence_count} facts</span>
            <span>{result.revisions} revisions</span>
            <span>{result.steps} steps</span>
          </div>
          <article className="md">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {result.draft || "_(no brief produced)_"}
            </ReactMarkdown>
          </article>
          {result.trace_url && (
            <p className="trace">
              🔗{" "}
              <a href={result.trace_url} target="_blank" rel="noreferrer">
                View the full agent trace in Langfuse
              </a>
            </p>
          )}
        </section>
      )}
    </main>
  );
}
