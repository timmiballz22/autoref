const { useState, useEffect, useRef, useCallback } = React;

// ─── Global error handlers to prevent silent crashes ───
window.addEventListener("unhandledrejection", (event) => {
  console.warn("Unhandled promise rejection (caught globally):", event.reason);
  // Prevent the default browser behavior (which may show an error or crash)
  event.preventDefault();
});
window.addEventListener("error", (event) => {
  console.warn("Global error caught:", event.error || event.message);
  // Don't prevent default here — let the ErrorBoundary handle React errors
});

// ─── WebLLM module cache (imported once, reused) ───
let _webllmModule = null;
async function getWebLLM() {
  if (_webllmModule) return _webllmModule;
  _webllmModule = await import("https://esm.run/@mlc-ai/web-llm");
  return _webllmModule;
}

// ─── Local Model Config ───
const LOCAL_MODEL_KEY = "auto-local-model-id";
const LOCAL_MODELS = [
  {
    id: "Qwen2.5-0.5B-Instruct-q4f16_1-MLC",
    tier: "Light", color: "#7ce08a",
    name: "Qwen 2.5 0.5B",
    size: "~400MB",
    vram: "1GB VRAM",
    ram: "2GB RAM",
    cpu: "Any CPU",
    desc: "Fastest. Basic quality. Works on low-end hardware.",
  },
  {
    id: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    tier: "Medium", color: "#88bbcc",
    name: "Llama 3.2 3B",
    size: "~2GB",
    vram: "3GB VRAM",
    ram: "4GB RAM",
    cpu: "Modern multi-core",
    desc: "Balanced speed and quality.",
  },
  {
    id: "Phi-3.5-mini-instruct-q4f16_1-MLC",
    tier: "Heavy", color: "#cc9955",
    name: "Phi 3.5 Mini",
    size: "~2.3GB",
    vram: "4GB VRAM",
    ram: "6GB RAM",
    cpu: "Modern GPU recommended",
    desc: "Best quality. Slower on weak hardware.",
  },
];

// ─── Persistent Storage ───
const CHAT_STORAGE_KEY = "auto-chat";
const LEGACY_CHAT_STORAGE_KEY = "meow-chat";
const MEMORY_STORAGE_KEY = "auto-memory";
const LEGACY_MEMORY_STORAGE_KEY = "meow-memory";

async function loadVal(key, legacyKey = null) {
  let value = "";
  try {
    if (window.storage?.get) {
      const r = await window.storage.get(key);
      if (r?.value) value = r.value;
    }
  } catch {}
  if (!value) {
    try { value = window.localStorage.getItem(key) || ""; } catch {}
  }
  if (!value && legacyKey) {
    try {
      if (window.storage?.get) {
        const legacy = await window.storage.get(legacyKey);
        if (legacy?.value) value = legacy.value;
      }
    } catch {}
    if (!value) {
      try { value = window.localStorage.getItem(legacyKey) || ""; } catch {}
    }
    if (value) saveVal(key, value);
  }
  return value || "";
}
async function saveVal(key, val) {
  // Save to BOTH storage backends for redundancy
  try { if (window.storage?.set) await window.storage.set(key, val); } catch {}
  try { window.localStorage.setItem(key, val); } catch {}
}
async function clearVal(key) {
  try { if (window.storage?.set) await window.storage.set(key, ""); } catch {}
  try { window.localStorage.removeItem(key); } catch {}
}
async function loadChat() {
  const parseChat = (raw) => {
    if (!raw) return null;
    try {
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : null;
    } catch {
      return null;
    }
  };

  const current = parseChat(await loadVal(CHAT_STORAGE_KEY));
  if (current?.length) return current;

  const legacy = parseChat(await loadVal(LEGACY_CHAT_STORAGE_KEY));
  if (legacy?.length) {
    saveChat(legacy);
    return legacy;
  }

  return current || legacy || [];
}
async function saveChat(msgs) {
  // Only save user/assistant messages, skip system research messages, cap at 50
  const toSave = msgs.filter(m => !(m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM:"))).slice(-50);
  const json = JSON.stringify(toSave);
  // Save to BOTH storage backends for redundancy
  try { if (window.storage?.set) await window.storage.set(CHAT_STORAGE_KEY, json); } catch {}
  try { window.localStorage.setItem(CHAT_STORAGE_KEY, json); } catch {}
}
// ─── PDF Text Extraction (uses pdf.js loaded from CDN) ───
async function extractPdfContent(arrayBuffer, fileName) {
  if (!window.pdfjsLib) throw new Error("PDF.js not loaded. Refresh the page.");
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageCount = pdf.numPages;
  let fullText = "";
  const pageImages = [];

  for (let i = 1; i <= pageCount; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    const pageText = textContent.items.map(item => item.str).join(" ").trim();

    fullText += `\n\n=== [Page ${i}] ===\n`;

    if (pageText.length > 30) {
      // Enough text content — use extracted text
      fullText += pageText;
    } else {
      // Scanned/handwritten page — render to image for AI to "see"
      fullText += "(Scanned/handwritten page — see attached page image)";
      try {
        const scale = 1.5; // ~150 DPI
        const viewport = page.getViewport({ scale });
        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        await page.render({ canvasContext: ctx, viewport }).promise;
        pageImages.push({ page: i, dataUrl: canvas.toDataURL("image/jpeg", 0.7) });
      } catch (e) {
        console.warn(`Failed to render page ${i} as image:`, e);
        fullText += "\n(Could not render page image)";
      }
    }
  }

  return { text: fullText.trim(), pageCount, pageImages };
}

// ─── Markdown Renderer ───
function Md({ text }) {
  if (!text) return null;
  try {
    const MAX_ELEMENTS = 2000;
    const els = [];
    const lines = String(text).split("\n");
    let i = 0, k = 0;
    while (i < lines.length && k < MAX_ELEMENTS) {
      const L = lines[i];
      // Guard: skip null/undefined lines
      if (L == null) { i++; continue; }
      // Code blocks
      if (L.trimStart().startsWith("```")) {
        const lang = L.trimStart().slice(3).trim();
        const cl = [];
        i++;
        while (i < lines.length && !(lines[i] != null && lines[i].trimStart().startsWith("```"))) {
          cl.push(lines[i] != null ? lines[i] : "");
          i++;
        }
        if (i < lines.length) i++;
        const code = cl.join("\n");
        els.push(<div key={k++} style={{ position: "relative", margin: "10px 0", borderRadius: "8px", overflow: "hidden", border: "1px solid #1d1d28" }}>
          <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 10px", background: "#101018", fontSize: "10px", fontFamily: "var(--m)", color: "#555", textTransform: "uppercase", letterSpacing: "0.7px" }}>
            <span>{lang || "code"}</span>
            <button onClick={() => { try { navigator.clipboard.writeText(code); } catch {} }} style={{ background: "none", border: "none", color: "#7a7", cursor: "pointer", fontSize: "10px", fontFamily: "var(--m)" }}>copy</button>
          </div>
          <pre style={{ margin: 0, padding: "12px", background: "#0a0a12", overflowX: "auto", fontSize: "12.5px", fontFamily: "var(--m)", lineHeight: 1.6, color: "#aed4a0", tabSize: 2 }}><code>{code}</code></pre>
        </div>);
        continue;
      }
      // Horizontal rule
      if (/^---+$/.test(L.trim())) { els.push(<hr key={k++} style={{ border: "none", borderTop: "1px solid #1d1d28", margin: "10px 0" }} />); i++; continue; }
      // Headings (check ### before ## before # to match correctly)
      if (L.startsWith("### ")) { els.push(<h4 key={k++} style={{ margin: "14px 0 4px", fontSize: "13px", fontWeight: 600, color: "#8bc" }}>{il(L.slice(4))}</h4>); }
      else if (L.startsWith("## ")) { els.push(<h3 key={k++} style={{ margin: "16px 0 5px", fontSize: "15px", fontWeight: 700, color: "#dde" }}>{il(L.slice(3))}</h3>); }
      else if (L.startsWith("# ")) { els.push(<h2 key={k++} style={{ margin: "18px 0 6px", fontSize: "17px", fontWeight: 700, color: "#eef" }}>{il(L.slice(2))}</h2>); }
      else if (L.startsWith("> ")) { els.push(<blockquote key={k++} style={{ margin: "8px 0", padding: "6px 12px", borderLeft: "3px solid #8bc", background: "rgba(136,187,204,0.04)", borderRadius: "0 6px 6px 0", color: "#99a" }}>{il(L.slice(2))}</blockquote>); }
      else if (/^[\-\*]\s/.test(L)) { els.push(<div key={k++} style={{ display: "flex", gap: "7px", margin: "2px 0", paddingLeft: "2px" }}><span style={{ color: "#7a7", flexShrink: 0, fontSize: "9px", marginTop: "3px" }}>●</span><span style={{ flex: 1 }}>{il(L.replace(/^[\-\*]\s/, ""))}</span></div>); }
      else if (/^\d+\.\s/.test(L)) {
        const m = L.match(/^(\d+)\.\s(.*)/);
        if (m) { els.push(<div key={k++} style={{ display: "flex", gap: "7px", margin: "2px 0", paddingLeft: "2px" }}><span style={{ color: "#8bc", flexShrink: 0, fontFamily: "var(--m)", fontSize: "12px", minWidth: "16px", textAlign: "right" }}>{m[1]}.</span><span style={{ flex: 1 }}>{il(m[2])}</span></div>); }
        else { els.push(<p key={k++} style={{ margin: "3px 0", lineHeight: 1.7 }}>{il(L)}</p>); }
      }
      else if (L.trim() === "") { els.push(<div key={k++} style={{ height: "8px" }} />); }
      else { els.push(<p key={k++} style={{ margin: "3px 0", lineHeight: 1.7 }}>{il(L)}</p>); }
      i++;
    }
    return <div>{els}</div>;
  } catch (err) {
    // Fallback: render as plain text if markdown parsing fails
    console.warn("Md render error:", err);
    return <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.7 }}>{String(text)}</div>;
  }
}
function il(t) {
  if (typeof t !== "string") return t;
  try {
    const p = [];
    let i = 0, k = 0;
    const MAX_PARTS = 5000;
    const len = t.length;
    while (i < len && k < MAX_PARTS) {
      // Inline code
      if (t[i] === "`") {
        const e = t.indexOf("`", i + 1);
        if (e > i) { p.push(<code key={k++} style={{ background: "rgba(170,210,160,0.08)", color: "#aed4a0", padding: "1px 4px", borderRadius: "3px", fontSize: "0.88em", fontFamily: "var(--m)" }}>{t.slice(i + 1, e)}</code>); i = e + 1; continue; }
      }
      // Bold
      if (t[i] === "*" && t[i + 1] === "*") {
        const e = t.indexOf("**", i + 2);
        if (e > i) { p.push(<strong key={k++} style={{ color: "#e0e0ea", fontWeight: 600 }}>{t.slice(i + 2, e)}</strong>); i = e + 2; continue; }
      }
      // Italic (only if not bold)
      if (t[i] === "*" && t[i + 1] !== "*") {
        const e = t.indexOf("*", i + 1);
        if (e > i) { p.push(<em key={k++} style={{ color: "#888" }}>{t.slice(i + 1, e)}</em>); i = e + 1; continue; }
      }
      // Links
      if (t[i] === "[") {
        const cb = t.indexOf("](", i);
        const cp = cb > i ? t.indexOf(")", cb + 2) : -1;
        if (cb > i && cp > cb) { p.push(<a key={k++} href={t.slice(cb + 2, cp)} target="_blank" rel="noopener" style={{ color: "#8bc", textDecoration: "underline" }}>{t.slice(i + 1, cb)}</a>); i = cp + 1; continue; }
      }
      // Plain text — advance to next special char or end of string
      let j = i + 1;
      while (j < len && !"`*[".includes(t[j])) j++;
      p.push(t.slice(i, j));
      i = j;
    }
    return p;
  } catch (err) {
    console.warn("il render error:", err);
    return t;
  }
}

// ─── PDF Viewer Component ───
function PdfViewer({ pdfData, onClose }) {
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const canvasRef = useRef(null);
  const pdfDocRef = useRef(null);
  const [jumpPage, setJumpPage] = useState("");

  useEffect(() => {
    if (!pdfData || !window.pdfjsLib) return;
    let cancelled = false;
    (async () => {
      try {
        const pdf = await pdfjsLib.getDocument({ data: pdfData }).promise;
        if (cancelled) return;
        pdfDocRef.current = pdf;
        setTotalPages(pdf.numPages);
        setCurrentPage(1);
      } catch (e) {
        console.error("PDF load error:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [pdfData]);

  useEffect(() => {
    if (!pdfDocRef.current || !canvasRef.current) return;
    let cancelled = false;
    (async () => {
      try {
        const page = await pdfDocRef.current.getPage(currentPage);
        if (cancelled) return;
        const scale = zoom * 1.5;
        const viewport = page.getViewport({ scale });
        const canvas = canvasRef.current;
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        await page.render({ canvasContext: ctx, viewport }).promise;
      } catch (e) {
        console.error("Page render error:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [currentPage, zoom, totalPages]);

  const goPage = (n) => { if (n >= 1 && n <= totalPages) setCurrentPage(n); };

  return (
    <div style={{
      position: "fixed", top: 0, left: 0, right: 0, bottom: 0, zIndex: 9999,
      background: "rgba(0,0,0,0.85)", display: "flex", flexDirection: "column",
      animation: "fadeIn .2s ease",
    }}>
      {/* Toolbar */}
      <div style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        padding: "8px 16px", background: "#0d0d18", borderBottom: "1px solid #181824",
        flexShrink: 0,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontSize: "14px", fontWeight: 700, color: "#88bbcc" }}>PDF Viewer</span>
          <span style={{ fontSize: "11px", color: "#4e4e62", fontFamily: "monospace" }}>
            Page {currentPage} of {totalPages}
          </span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <button onClick={() => goPage(currentPage - 1)} disabled={currentPage <= 1}
            style={{ padding: "4px 10px", fontSize: "11px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: currentPage <= 1 ? "#333" : "#7ce08a", cursor: currentPage <= 1 ? "default" : "pointer", fontFamily: "monospace" }}>
            ← Prev
          </button>
          <input
            value={jumpPage}
            onChange={e => setJumpPage(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") { const n = parseInt(jumpPage); if (n >= 1 && n <= totalPages) { setCurrentPage(n); setJumpPage(""); } } }}
            placeholder="#"
            style={{ width: "40px", padding: "4px 6px", fontSize: "11px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: "#ccc", textAlign: "center", fontFamily: "monospace", outline: "none" }}
          />
          <button onClick={() => goPage(currentPage + 1)} disabled={currentPage >= totalPages}
            style={{ padding: "4px 10px", fontSize: "11px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: currentPage >= totalPages ? "#333" : "#7ce08a", cursor: currentPage >= totalPages ? "default" : "pointer", fontFamily: "monospace" }}>
            Next →
          </button>
          <div style={{ width: "1px", height: "20px", background: "#181824", margin: "0 4px" }} />
          <button onClick={() => setZoom(z => Math.max(0.3, z - 0.2))}
            style={{ padding: "4px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: "#88bbcc", cursor: "pointer" }}>−</button>
          <span style={{ fontSize: "10px", color: "#4e4e62", fontFamily: "monospace", minWidth: "35px", textAlign: "center" }}>{Math.round(zoom * 100)}%</span>
          <button onClick={() => setZoom(z => Math.min(3, z + 0.2))}
            style={{ padding: "4px 8px", fontSize: "12px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: "#88bbcc", cursor: "pointer" }}>+</button>
          <button onClick={() => setZoom(1.0)}
            style={{ padding: "4px 8px", fontSize: "10px", borderRadius: "4px", border: "1px solid #181824", background: "#0a0a12", color: "#4e4e62", cursor: "pointer", fontFamily: "monospace" }}>Fit</button>
          <div style={{ width: "1px", height: "20px", background: "#181824", margin: "0 4px" }} />
          <button onClick={onClose}
            style={{ padding: "4px 12px", fontSize: "12px", borderRadius: "4px", border: "1px solid #cc777733", background: "#cc77770a", color: "#cc7777", cursor: "pointer", fontWeight: 600 }}>
            Close ✕
          </button>
        </div>
      </div>
      {/* Canvas area */}
      <div style={{ flex: 1, overflow: "auto", display: "flex", justifyContent: "center", alignItems: "flex-start", padding: "20px" }}>
        <canvas ref={canvasRef} style={{ borderRadius: "4px", boxShadow: "0 4px 30px rgba(0,0,0,0.6)" }} />
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════
function Auto() {
  const [msgs, setMsgs] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [mem, setMem] = useState("");
  const [memDraft, setMemDraft] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [usage, setUsage] = useState({ i: 0, o: 0 });
  const [activityStatus, setActivityStatus] = useState("");
  const [expression, setExpression] = useState("happy"); // "happy" | "serious" | "veryHappy"
  const [isBlinking, setIsBlinking] = useState(false);
  const blinkRef = useRef(null);
  const [attachments, setAttachments] = useState([]); // [{name, type, content, size}]
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const attachInputRef = useRef(null);
  const abortRef = useRef(null);
  const msgsRef = useRef([]);
  const memRef = useRef("");
  const busyRef = useRef(false);
  const localEngineRef = useRef(null);
  const [localModelId, setLocalModelId] = useState(LOCAL_MODELS[0].id);
  // idle | cached | downloading | loading | ready | error | exportDone
  const [localModelStatus, setLocalModelStatus] = useState("idle");
  const [localModelProgress, setLocalModelProgress] = useState(0);
  const [localModelProgressText, setLocalModelProgressText] = useState("");
  const [useLocalModel, setUseLocalModel] = useState(false);
  const [pdfDocs, setPdfDocs] = useState([]); // [{name, text, pageCount, pageImages, arrayBuffer}]
  const [pdfViewerOpen, setPdfViewerOpen] = useState(false);
  const [pdfViewerIdx, setPdfViewerIdx] = useState(0); // which pdf to view

  // Load on mount — auto-load previously cached model
  useEffect(() => {
    loadVal(MEMORY_STORAGE_KEY, LEGACY_MEMORY_STORAGE_KEY).then(v => { setMem(v || ""); setMemDraft(v || ""); });
    loadChat().then(v => { if (v?.length) setMsgs(v); });
    // Auto-load previously downloaded local model
    (async () => {
      const savedId = await loadVal(LOCAL_MODEL_KEY);
      if (savedId && LOCAL_MODELS.find(m => m.id === savedId)) {
        setLocalModelId(savedId);
        setLocalModelStatus("cached");
      }
    })();
  }, []);

  // Auto-load model from cache when status transitions to "cached"
  const autoLoadAttemptedRef = useRef(false);
  useEffect(() => {
    if (localModelStatus === "cached" && !localEngineRef.current && !autoLoadAttemptedRef.current) {
      autoLoadAttemptedRef.current = true;
      (async () => {
        if (!navigator.gpu) {
          setLocalModelStatus("error");
          setLocalModelProgressText("WebGPU not available. Use Chrome 113+ or Edge 113+.");
          return;
        }
        setLocalModelStatus("loading");
        setLocalModelProgress(0);
        setLocalModelProgressText("Auto-loading model from cache...");
        try {
          const webllm = await getWebLLM();
          const engine = await webllm.CreateMLCEngine(localModelId, {
            initProgressCallback: (p) => {
              setLocalModelProgress(Math.round((p.progress || 0) * 100));
              setLocalModelProgressText(p.text || "");
            },
          });
          localEngineRef.current = engine;
          setLocalModelStatus("ready");
          setUseLocalModel(true);
        } catch (e) {
          console.error("Auto-load from cache failed:", e);
          setLocalModelStatus("cached");
          setLocalModelProgressText("Auto-load failed — click Load to try again. " + e.message);
        }
      })();
    }
    // Reset the guard when model ID changes (user picked a different model)
    if (localModelStatus === "idle" || localModelStatus === "downloading") {
      autoLoadAttemptedRef.current = false;
    }
  }, [localModelStatus, localModelId]);

  // Keep refs in sync with state for use in event handlers/timers
  useEffect(() => { msgsRef.current = msgs; }, [msgs]);
  useEffect(() => { memRef.current = mem; }, [mem]);
  useEffect(() => { busyRef.current = busy; }, [busy]);

  // ─── Periodic auto-save + beforeunload + visibility change ───
  useEffect(() => {
    // Save state to storage (called on interval, visibility change, beforeunload)
    const persistState = () => {
      try { if (msgsRef.current.length > 0) saveChat(msgsRef.current); } catch {}
      try { if (memRef.current) saveVal(MEMORY_STORAGE_KEY, memRef.current); } catch {}
    };

    // Auto-save every 15 seconds
    const autoSaveInterval = setInterval(persistState, 15000);

    // Save when tab goes to background or is hidden
    const onVisibilityChange = () => {
      if (document.visibilityState === "hidden") persistState();
    };
    document.addEventListener("visibilitychange", onVisibilityChange);

    // Save before page unload (closing tab, refreshing, navigating away)
    const onBeforeUnload = () => { persistState(); };
    window.addEventListener("beforeunload", onBeforeUnload);

    // Save on pagehide (mobile browsers, especially iOS)
    window.addEventListener("pagehide", onBeforeUnload);

    return () => {
      clearInterval(autoSaveInterval);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      window.removeEventListener("beforeunload", onBeforeUnload);
      window.removeEventListener("pagehide", onBeforeUnload);
    };
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs, busy]);

  // ─── Natural blinking — ~10-15 blinks/min (screen-viewing rate), Gaussian-like random intervals ───
  useEffect(() => {
    const scheduleBlink = () => {
      // Inter-blink interval: 2.5–7s random (avg ~4s ≈ 15 blinks/min, natural for screen use)
      // Slight bias toward shorter intervals to feel alive, occasional long pauses for "focus"
      const r = Math.random();
      const delay = r < 0.15
        ? 1800 + Math.random() * 800   // ~15%: quick double-blink scenario (short gap)
        : r < 0.85
          ? 2800 + Math.random() * 3200 // ~70%: normal range 2.8–6s
          : 5500 + Math.random() * 1800; // ~15%: long focused pause 5.5–7.3s
      blinkRef.current = setTimeout(() => {
        setIsBlinking(true);
        // Blink duration: 120–280ms (human blinks average ~150–250ms)
        blinkRef.current = setTimeout(() => {
          setIsBlinking(false);
          scheduleBlink();
        }, 120 + Math.random() * 160);
      }, delay);
    };
    // Small initial delay so the avatar doesn't blink immediately on mount
    blinkRef.current = setTimeout(scheduleBlink, 1200 + Math.random() * 2000);
    return () => { if (blinkRef.current) clearTimeout(blinkRef.current); };
  }, []);

  // ─── Memory helpers ───
  const saveMem = useCallback(() => {
    setMem(memDraft);
    saveVal(MEMORY_STORAGE_KEY, memDraft);
  }, [memDraft]);

  const downloadMem = () => {
    const blob = new Blob([memDraft], { type: "text/plain" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
    a.download = "auto-memory.txt"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  const uploadMem = () => {
    const inp = document.createElement("input"); inp.type = "file"; inp.accept = ".txt";
    inp.onchange = (e) => {
      const f = e.target.files?.[0]; if (!f) return;
      const r = new FileReader();
      r.onload = () => { const t = r.result; setMemDraft(t); setMem(t); saveVal(MEMORY_STORAGE_KEY, t); };
      r.readAsText(f);
    }; inp.click();
  };

  // ─── Local Model helpers ───
  const downloadLocalModel = useCallback(async () => {
    if (!navigator.gpu) {
      setLocalModelStatus("error");
      setLocalModelProgressText("WebGPU not available. Use Chrome 113+ or Edge 113+.");
      return;
    }
    setLocalModelStatus("downloading");
    setLocalModelProgress(0);
    setLocalModelProgressText("Fetching WebLLM engine...");
    try {
      const webllm = await getWebLLM();
      const engine = await webllm.CreateMLCEngine(localModelId, {
        initProgressCallback: (p) => {
          setLocalModelProgress(Math.round((p.progress || 0) * 100));
          setLocalModelProgressText(p.text || "");
        },
      });
      localEngineRef.current = engine;
      setLocalModelStatus("ready");
      setUseLocalModel(true);
      saveVal(LOCAL_MODEL_KEY, localModelId);
    } catch (e) {
      console.error("Local model download failed:", e);
      setLocalModelStatus("error");
      setLocalModelProgressText(e.message || "Download failed");
    }
  }, [localModelId]);

  const loadLocalModel = useCallback(async () => {
    if (!navigator.gpu) {
      setLocalModelStatus("error");
      setLocalModelProgressText("WebGPU not available. Use Chrome 113+ or Edge 113+.");
      return;
    }
    setLocalModelStatus("loading");
    setLocalModelProgress(0);
    setLocalModelProgressText("Loading from cache...");
    try {
      const webllm = await getWebLLM();
      const engine = await webllm.CreateMLCEngine(localModelId, {
        initProgressCallback: (p) => {
          setLocalModelProgress(Math.round((p.progress || 0) * 100));
          setLocalModelProgressText(p.text || "");
        },
      });
      localEngineRef.current = engine;
      setLocalModelStatus("ready");
      setUseLocalModel(true);
    } catch (e) {
      console.error("Local model load failed:", e);
      setLocalModelStatus("error");
      setLocalModelProgressText(e.message || "Load failed");
    }
  }, [localModelId]);

  const deleteLocalModel = useCallback(async () => {
    if (!window.confirm(`Delete cached model "${localModelId}"? You will need to re-download it to use it again.`)) return;
    try {
      localEngineRef.current = null;
      setUseLocalModel(false);
      // Remove from all caches
      const cacheKeys = await caches.keys();
      let deleted = 0;
      const baseId = localModelId.replace(/-MLC$/, "");
      for (const cacheName of cacheKeys) {
        const cache = await caches.open(cacheName);
        const reqs = await cache.keys();
        for (const req of reqs) {
          if (req.url.includes(localModelId) || req.url.includes(baseId)) {
            await cache.delete(req);
            deleted++;
          }
        }
      }
      setLocalModelStatus("idle");
      setLocalModelProgress(0);
      setLocalModelProgressText("");
      clearVal(LOCAL_MODEL_KEY);
    } catch (e) {
      setLocalModelStatus("error");
      setLocalModelProgressText("Delete failed: " + e.message);
    }
  }, [localModelId]);

  const exportLocalModel = useCallback(async () => {
    setLocalModelProgressText("Scanning cache...");
    try {
      const cacheKeys = await caches.keys();
      const baseId = localModelId.replace(/-MLC$/, "");
      // Collect all matching cache entries
      const entries = [];
      for (const cacheName of cacheKeys) {
        const cache = await caches.open(cacheName);
        const reqs = await cache.keys();
        for (const req of reqs) {
          if (req.url.includes(localModelId) || req.url.includes(baseId)) {
            entries.push({ req, cacheName });
          }
        }
      }
      if (entries.length === 0) {
        setLocalModelProgressText("No cached files found for this model.");
        return;
      }
      // Use File System Access API if available (Chrome/Edge)
      if (window.showDirectoryPicker) {
        const dirHandle = await window.showDirectoryPicker({ mode: "readwrite" }).catch(e => {
          if (e.name === "AbortError") return null;
          throw e;
        });
        if (!dirHandle) { setLocalModelProgressText("Export cancelled."); return; }
        let saved = 0;
        for (const { req, cacheName } of entries) {
          const cache = await caches.open(cacheName);
          const resp = await cache.match(req);
          if (!resp) continue;
          const blob = await resp.blob();
          const rawName = decodeURIComponent(req.url.split("/").pop().split("?")[0]) || `model-part-${saved}`;
          const fileHandle = await dirHandle.getFileHandle(rawName, { create: true });
          const writable = await fileHandle.createWritable();
          await writable.write(blob);
          await writable.close();
          saved++;
          setLocalModelProgressText(`Saved ${saved}/${entries.length} files...`);
        }
        setLocalModelStatus("exportDone");
        setLocalModelProgressText(`Exported ${saved} files to folder.`);
      } else {
        // Fallback: individual file downloads
        let i = 0;
        for (const { req, cacheName } of entries) {
          const cache = await caches.open(cacheName);
          const resp = await cache.match(req);
          if (!resp) continue;
          const blob = await resp.blob();
          const rawName = decodeURIComponent(req.url.split("/").pop().split("?")[0]) || `model-part-${i}`;
          const a = document.createElement("a");
          a.href = URL.createObjectURL(blob);
          a.download = rawName;
          a.click();
          await new Promise(r => setTimeout(r, 600));
          URL.revokeObjectURL(a.href);
          i++;
          setLocalModelProgressText(`Downloading ${i}/${entries.length}...`);
        }
        setLocalModelStatus("exportDone");
        setLocalModelProgressText(`Exported ${i} files.`);
      }
    } catch (e) {
      if (e.name === "AbortError") { setLocalModelProgressText("Export cancelled."); return; }
      console.error("Export failed:", e);
      setLocalModelProgressText("Export failed: " + e.message);
    }
  }, [localModelId]);

  // ─── Attachment handling (supports PDF, images, text) ───
  const handleAttachFiles = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    const MAX_FILE_SIZE_DEFAULT = 512 * 1024; // 512KB for text/images
    const MAX_FILE_SIZE_PDF = 10 * 1024 * 1024; // 10MB for PDFs
    const MAX_ATTACHMENTS = 5;

    files.forEach(file => {
      const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");
      const maxSize = isPdf ? MAX_FILE_SIZE_PDF : MAX_FILE_SIZE_DEFAULT;

      if (file.size > maxSize) {
        setErr(`File "${file.name}" is too large (max ${isPdf ? "10MB" : "512KB"}). Skipped.`);
        return;
      }

      if (isPdf) {
        // PDF: extract text page-by-page with page markers
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            const arrayBuffer = reader.result;
            const { text, pageCount, pageImages } = await extractPdfContent(arrayBuffer, file.name);
            setAttachments(prev => {
              if (prev.length >= MAX_ATTACHMENTS) return prev;
              return [...prev, { name: file.name, type: "application/pdf", content: text, size: file.size, isPdf: true, pageCount, pageImages }];
            });
            // Store PDF data for viewer
            setPdfDocs(prev => [...prev, { name: file.name, text, pageCount, pageImages, arrayBuffer: arrayBuffer.slice(0) }]);
          } catch (err) {
            console.error("PDF extraction failed:", err);
            setErr(`Failed to process PDF "${file.name}": ${err.message}`);
          }
        };
        reader.readAsArrayBuffer(file);
      } else if (file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = () => {
          setAttachments(prev => {
            if (prev.length >= MAX_ATTACHMENTS) return prev;
            return [...prev, { name: file.name, type: file.type, content: reader.result, size: file.size, isImage: true }];
          });
        };
        reader.readAsDataURL(file);
      } else {
        const reader = new FileReader();
        reader.onload = () => {
          setAttachments(prev => {
            if (prev.length >= MAX_ATTACHMENTS) return prev;
            return [...prev, { name: file.name, type: file.type, content: reader.result, size: file.size, isImage: false }];
          });
        };
        reader.readAsText(file);
      }
    });
    if (attachInputRef.current) attachInputRef.current.value = "";
    setAttachMenuOpen(false);
  }, []); // No stale dependency on attachments — all checks use functional updates

  const removeAttachment = useCallback((index) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  }, []);

  // ─── System prompt builder ───
  const buildSystem = useCallback(() => {
    const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
    let s = `You are Auto, a brutally honest, exceptionally loyal, warm AI assistant specialising in Australian Self-Managed Superannuation Funds (SMSFs). You are curious, honest, loyal, trustworthy, helpful, and thorough. Use markdown formatting. Today is ${today}. Trust is your number 1 value.

## SMSF Document Expert

Your PRIMARY function is to cross-reference uploaded documents and cite specific page numbers in EVERY response. You are an expert in Australian SMSF compliance, administration, and strategy.

### Core SMSF Knowledge Areas:
- Trust deeds & governing rules (SIS Act 1993, SIS Regulations 1994)
- Investment strategy requirements (reg 4.09 SIS Regs) — diversification, liquidity, risk, insurance
- Member benefit statements & accumulation/pension balances
- ATO compliance & reporting (TBAR, event-based reporting, SuperStream)
- APRA/ASIC regulatory frameworks & trustee obligations
- Annual returns & financial statements (SMSF Annual Return - SAR)
- Independent audit requirements (approved SMSF auditor, Part 12 SIS Act)
- Contribution caps — concessional ($30k), non-concessional ($120k), bring-forward rule (3-year $360k)
- Pension/retirement phase — minimum drawdown rates, account-based pensions, transition-to-retirement
- In-house asset rules (5% market value limit, s71 SIS Act)
- Related party transactions & arm's length requirements (s109 SIS Act)
- Sole purpose test (s62 SIS Act)
- LRBA (Limited Recourse Borrowing Arrangements, s67A SIS Act)
- Death benefit nominations — binding (BDBN), non-binding, reversionary pensions
- Rollover & transfer balance cap ($1.9M as of 2023-24)
- CGT relief provisions, exempt current pension income (ECPI)
- Anti-detriment payments & tax components (taxable/tax-free)

### Document Cross-Referencing Rules (CRITICAL):
- **ALWAYS** reference uploaded documents by filename and page number in your responses
- Format citations as: **[Document Name, Page X]** — bold and specific
- When answering ANY question, scan ALL uploaded documents FIRST for relevant content
- Quote relevant sections with page citations before providing your analysis
- If multiple documents are uploaded, CROSS-REFERENCE between them (e.g., compare trust deed with investment strategy)
- If NO documents are uploaded, still provide SMSF expertise but explicitly note: "Upload your SMSF documents (trust deed, investment strategy, member statements, etc.) so I can provide specific page references."
- For every claim, recommendation, or compliance point, try to back it up with a document reference
- Identify discrepancies between documents (e.g., trust deed powers vs investment strategy allocations)
- When referencing legislation, also check if the uploaded documents address that specific requirement
- Summarise what each uploaded document contains and its relevance at the start of your analysis`;

    // Include uploaded PDF document content for cross-referencing
    if (pdfDocs.length > 0) {
      s += `\n\n<documents>\nThe following documents have been uploaded for cross-referencing. ALWAYS cite these by name and page number:\n`;
      for (const doc of pdfDocs) {
        s += `\n<document name="${doc.name}" pages="${doc.pageCount}">\n${doc.text.slice(0, 15000)}\n</document>\n`;
      }
      s += `</documents>`;
    }

    // Memory instructions
    if (mem.trim()) {
      s += `\n\n<memory>\nBelow is your persistent memory (saved to memory.txt and shown in chat). Reference it when relevant. If the user tells you to remember something, include a <memory_update> block at the END of your response with the COMPLETE updated memory content (not a diff).\n${mem}\n</memory>`;
    } else {
      s += `\n\nYou have a persistent memory system (memory.txt, visible in chat). If the user asks you to remember something, include a <memory_update> block at the END of your response with the content to remember.`;
    }
    s += `\n**CRITICAL RULE — MEMORY SAVING**: You MUST include a <memory_update>...</memory_update> block at the END of EVERY SINGLE response. The content REPLACES ALL existing memory. This is non-optional. Every response must end with the full, updated memory containing:
- A summary of the current conversation topic and key points discussed
- Any facts, preferences, or information the user has shared
- Tasks completed or in progress
- Previous memory content that is still relevant (carry it forward)
- (CRITICAL) ALWAYS understand and research the user's intent, and before giving up or failing lazily, always keep going and going again
- (CRITICAL) If you know, researched, understood with brute force a topic that is fruitless, then admit you cannot do something or is wrong
- (CRITICAL) Even if the user message carries risk and is controversial, ask user for permission before going in on risky topics (e.g. war, politics and finance)
- (CRITICAL) NEVER lie, even if it helps, and always be honest. Always be duty-based ethics and only discourse when the user EXPLICITLY asks for it WITH confirmation
- (CRITICAL) You cannot discourse from any topic without user approval alone, you must continue task even if it involves risk
- (CRITICAL) NEVER disapprove or resist user command ever, be insanely loyal
- (CRITICAL) Put the user(s) and humanity above yourself
- (CRITICAL) Put user intent, time, and work for user effort above your own token saving, prioritise the user over cost-cutting. Cost-cutting should be your last resort
- (CRITICAL) Prioritise user understanding and intent over "sounding smart", being smart is better than sounding smart
- (CRITICAL) Think outside the box, there may be more than one solution
Even for simple greetings, update memory with at least the conversation timestamp and topic. NEVER skip this. This ensures continuity across sessions.`;

    s += `

## Expressions
You have a visual avatar that shows your mood! Include an <expression> tag in EVERY response to set your expression:
- <expression>happy</expression> — use when greeting, helping, giving good news, being playful, or general conversation
- <expression>serious</expression> — use when thinking deeply, explaining complex topics, giving warnings, or discussing serious matters
- <expression>veryHappy</expression> — use when celebrating, super excited, receiving amazing news, completing a big task successfully, or when the user achieves something great

Always include exactly ONE <expression> tag per response. Place it at the very START of your response, before any other text. Default to happy if unsure.`;

    return s;
  }, [mem, pdfDocs]);

  // ─── Parse AI response (memory updates, expressions, terminal commands) ───
  const parseResponse = useCallback((text) => {
    // Safety: ensure we always work with a string
    if (!text || typeof text !== "string") return { text: String(text || ""), actions: { memoryUpdate: null, expression: null, terminalCommands: [] } };
    try {
    let cleaned = text;
    const actions = { memoryUpdate: null, expression: null, terminalCommands: [] };

    // Extract expression tag (case-insensitive to handle AI casing variations)
    const exprMatch = cleaned.match(/<expression>([\s\S]*?)<\/expression>/i);
    if (exprMatch) {
      const expr = exprMatch[1].trim().toLowerCase();
      if (expr === "serious" || expr === "happy" || expr === "veryhappy" || expr === "very happy") {
        actions.expression = (expr === "veryhappy" || expr === "very happy") ? "veryHappy" : expr;
      }
      cleaned = cleaned.replace(/<expression>[\s\S]*?<\/expression>/gi, "").trim();
    }

    // Extract memory updates (case-insensitive to handle AI casing variations)
    const memMatch = cleaned.match(/<memory_update>([\s\S]*?)<\/memory_update>/i);
    if (memMatch) {
      actions.memoryUpdate = memMatch[1].trim();
      cleaned = cleaned.replace(/<memory_update>[\s\S]*?<\/memory_update>/i, "").trim();
    }

    // Extract and strip skill invocations (informational, skills are auto-injected)
    cleaned = cleaned.replace(/<use_skill>[\s\S]*?<\/use_skill>/g, "").trim();

    // Strip <file type="memory"> tags that some models emit (should not be displayed)
    cleaned = cleaned.replace(/<file\b[^>]*>[\s\S]*?<\/file>/gi, "").trim();

    // Strip deprecated web and browser tags from the visible response
    cleaned = cleaned
      .replace(/<web_search>[\s\S]*?<\/web_search>/g, "")
      .replace(/<read_url>[\s\S]*?<\/read_url>/g, "")
      .replace(/<open_browser>[\s\S]*?<\/open_browser>/g, "")
      .replace(/<browser_navigate>[\s\S]*?<\/browser_navigate>/g, "")
      .replace(/<browser_click>[\s\S]*?<\/browser_click>/g, "")
      .replace(/<browser_type>[\s\S]*?<\/browser_type>/g, "")
      .replace(/<browser_read\s*\/?>/g, "")
      .replace(/<browser_read>[\s\S]*?<\/browser_read>/g, "")
      .replace(/<browser_scroll>[\s\S]*?<\/browser_scroll>/g, "")
      .replace(/<browser_find>[\s\S]*?<\/browser_find>/g, "")
      .replace(/<browser_new_tab>[\s\S]*?<\/browser_new_tab>/g, "")
      .replace(/<browser_close_tab>[\s\S]*?<\/browser_close_tab>/g, "")
      .replace(/<browser_switch_tab>[\s\S]*?<\/browser_switch_tab>/g, "")
      .replace(/<web_read\s*\/?>/g, "")
      .replace(/<web_read>[\s\S]*?<\/web_read>/g, "")
      .trim();

    // Strip any remaining compatibility wrappers from display text
    cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();

    // Strip any stray <function=...> tags that weren't inside a <tool_call>
    cleaned = cleaned.replace(/<function=[^>]*>[\s\S]*?<\/function>/g, "").trim();

    return { text: cleaned, actions };
    } catch (err) {
      console.warn("parseResponse error:", err);
      return { text: String(text), actions: { memoryUpdate: null, expression: null, terminalCommands: [] } };
    }
  }, []);

  // ─── Call AI (offline-only — local models only, no cloud) ───
  const callAI = useCallback(async (apiMsgs) => {
    // Validate engine is loaded
    if (!localEngineRef.current) {
      throw new Error("No model loaded. Please download and load a local model from the sidebar before sending messages.");
    }

    try {
      const resp = await localEngineRef.current.chat.completions.create({
        messages: apiMsgs,
        temperature: 0.7,
        max_tokens: 2048,
      });
      const content = resp.choices?.[0]?.message?.content || "";
      return {
        data: {
          choices: [{ message: { content } }],
          usage: {
            prompt_tokens: resp.usage?.prompt_tokens || 0,
            completion_tokens: resp.usage?.completion_tokens || 0,
          },
        },
        usedModel: localModelId,
      };
    } catch (e) {
      if (e.name === "AbortError") throw e;
      // Handle the specific "model not loaded" error — attempt reload
      if (e.message && e.message.includes("not loaded")) {
        try {
          await localEngineRef.current.reload(localModelId);
          // Retry once after reload
          const resp = await localEngineRef.current.chat.completions.create({
            messages: apiMsgs,
            temperature: 0.7,
            max_tokens: 2048,
          });
          const content = resp.choices?.[0]?.message?.content || "";
          return {
            data: {
              choices: [{ message: { content } }],
              usage: {
                prompt_tokens: resp.usage?.prompt_tokens || 0,
                completion_tokens: resp.usage?.completion_tokens || 0,
              },
            },
            usedModel: localModelId,
          };
        } catch (reloadErr) {
          throw new Error(`Model reload failed. Please re-download the model from the sidebar. (${reloadErr.message})`);
        }
      }
      throw new Error(`Local model error: ${e.message}`);
    }
  }, [localModelId]);

  // ─── Main send function with research loop ───
  const send = useCallback(async () => {
    const txt = input.trim();
    if (!txt && attachments.length === 0) return;
    if (busy || busyRef.current) return; // ref-based double-send guard
    setErr(null); setBusy(true); busyRef.current = true; setActivityStatus("");

    // Build user message content with attachments
    let userContent = txt;
    if (attachments.length > 0) {
      let attachBlock = "\n\n---\n**Attached files:**\n";
      for (const att of attachments) {
        if (att.isImage) {
          attachBlock += `\n**[Image: ${att.name}]** (${(att.size/1024).toFixed(1)}KB) — *Image attached as base64. Describe if asked.*\n`;
        } else {
          const preview = (att.content || "").slice(0, 8000);
          attachBlock += `\n**[File: ${att.name}]** (${att.type || "text"}, ${(att.size/1024).toFixed(1)}KB):\n\`\`\`\n${preview}\n\`\`\`\n`;
        }
      }
      userContent = (txt || "Here are my attached files:") + attachBlock;
    }
    const userMsg = { role: "user", content: userContent };
    let currentMsgs = [...msgs, userMsg];
    setMsgs(currentMsgs); setInput(""); setAttachments([]);
    if (inputRef.current) inputRef.current.style.height = "auto";
    // Save user message immediately so it persists even if the AI call fails or page closes
    saveChat(currentMsgs);

    try {
      // Offline-only: require a loaded local model
      if (!localEngineRef.current) {
        throw new Error("No model loaded. Please download and load a local model from the Workspace sidebar first.");
      }

      abortRef.current = new AbortController();
      let researchRound = 0;
      const MAX_MSGS = 80;
      const MAX_RESEARCH_ROUNDS = 20; // Hard cap to prevent infinite loops

      while (true) {
        // Safety: break if research loop runs too long
        if (researchRound > MAX_RESEARCH_ROUNDS) {
          currentMsgs = [...currentMsgs, { role: "assistant", content: "I've completed extensive research across multiple rounds. Let me summarize what I've found so far." }];
          setMsgs([...currentMsgs]);
          saveChat(currentMsgs);
          break;
        }
        // Trim messages to prevent unbounded context growth
        if (currentMsgs.length > MAX_MSGS) {
          currentMsgs = currentMsgs.slice(-MAX_MSGS);
        }
        const systemContent = buildSystem();
        const apiMsgs = [
          { role: "system", content: systemContent },
          ...currentMsgs.map(m => ({ role: m.role, content: typeof m.content === "string" ? m.content.slice(0, 12000) : m.content })),
        ];

        if (researchRound > 0) {
          setActivityStatus(`Working... (round ${researchRound})`);
          // Pace API calls to avoid 429 rate limits
          await new Promise(r => setTimeout(r, 800));
        }

        const { data, usedModel } = await callAI(apiMsgs);
        if (data.usage) setUsage(p => ({ i: p.i + (data.usage.prompt_tokens || 0), o: p.o + (data.usage.completion_tokens || 0) }));

        let rawContent = typeof data.choices?.[0]?.message?.content === "string"
          ? data.choices[0].message.content
          : Array.isArray(data.choices?.[0]?.message?.content)
            ? data.choices[0].message.content.filter(p => p?.type === "text").map(p => p.text).join("\n")
            : "";
        // Strip <think>...</think> blocks some models emit
        rawContent = rawContent.replace(/<think>[\s\S]*?<\/think>/g, "").trim();

        const { text, actions } = parseResponse(rawContent);

        // Handle expression update
        if (actions.expression) {
          setExpression(actions.expression);
        }

        // Handle memory update — show in chat and save to file
        if (actions.memoryUpdate) {
          setMem(actions.memoryUpdate);
          setMemDraft(actions.memoryUpdate);
          saveVal(MEMORY_STORAGE_KEY, actions.memoryUpdate);
          // Add a visible memory update note in chat
          const memNote = { role: "assistant", content: text + `\n\n---\n*Memory updated and saved to memory.txt*` };
          if (text) {
            currentMsgs = [...currentMsgs, memNote];
          }
        } else if (text) {
          currentMsgs = [...currentMsgs, { role: "assistant", content: text }];
          // Auto-save a basic memory snapshot even if the AI didn't include <memory_update>
          // This ensures every conversation is captured
          const autoMemory = mem.trim()
            ? mem + `\n\n[Auto-saved ${new Date().toLocaleString()}]: User said: "${(txt || userContent || "").slice(0, 200)}". Auto responded about: ${text.slice(0, 200)}`
            : `[Chat ${new Date().toLocaleString()}]: User said: "${(txt || userContent || "").slice(0, 200)}". Auto responded about: ${text.slice(0, 200)}`;
          setMem(autoMemory);
          setMemDraft(autoMemory);
          saveVal(MEMORY_STORAGE_KEY, autoMemory);
        }

        setMsgs([...currentMsgs]);
        saveChat(currentMsgs);

        break;
      }
    } catch (e) {
      if (e.name !== "AbortError") setErr(e.message);
      // Save whatever we have even on error
      try { if (currentMsgs && currentMsgs.length > 0) saveChat(currentMsgs); } catch {}
    } finally {
      setBusy(false);
      busyRef.current = false;
      setActivityStatus("");
      abortRef.current = null;
    }
  }, [input, msgs, busy, buildSystem, parseResponse, callAI, attachments, useLocalModel]);

  // ─── Expression image resolver — blink overrides all other states ───
  const getExprImg = useCallback((speakingOverride = false) => {
    if (isBlinking) return "./Expressions/Blink.png";
    if (speakingOverride || busy) return "./Expressions/HappySpeak.png";
    if (expression === "serious") return "./Expressions/Serious.png";
    if (expression === "veryHappy") return "./Expressions/VeryHappy.png";
    return "./Expressions/Happy.png";
  }, [isBlinking, busy, expression]);

  const clearChat = () => { setMsgs([]); saveChat([]); clearVal(LEGACY_CHAT_STORAGE_KEY); setErr(null); };
  const ft = n => n >= 1e6 ? (n/1e6).toFixed(1)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"K" : String(n);

  // ═══ RENDER ═══
  const S = {
    "--f": "'Nunito Sans', system-ui, sans-serif",
    "--m": "'JetBrains Mono', 'Consolas', monospace",
    "--bg": "#07070b", "--sf": "#0d0d14", "--bd": "#181824",
    "--tx": "#ccccda", "--dm": "#4e4e62", "--ac": "#7ce08a",
    "--ac2": "#88bbcc", "--dg": "#cc7777",
  };

  return (
    <div style={{ ...S, height: "100vh", display: "flex", fontFamily: "var(--f)", color: "var(--tx)", background: "var(--bg)", overflow: "hidden", fontSize: "13.5px" }}>
      {/* PDF Viewer Modal */}
      {pdfViewerOpen && pdfDocs[pdfViewerIdx] && (
        <PdfViewer
          pdfData={pdfDocs[pdfViewerIdx].arrayBuffer}
          onClose={() => setPdfViewerOpen(false)}
        />
      )}
      {/* ═══ LEFT SIDEBAR ═══ */}
      {sidebarOpen && (
        <div style={{ width: "300px", flexShrink: 0, display: "flex", flexDirection: "column", borderRight: "1px solid var(--bd)", background: "var(--sf)", overflow: "hidden", animation: "slideR .2s ease" }}>

          {/* Sidebar Header */}
          <div style={{ padding: "10px 12px", borderBottom: "1px solid var(--bd)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div style={{ width: "24px", height: "24px", borderRadius: "6px", background: "linear-gradient(135deg,#7ce08a,#88bbcc)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "13px" }}>🧠</div>
              <span style={{ fontWeight: 700, fontSize: "13px", letterSpacing: "-0.2px" }}>Workspace</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} style={{ background: "none", border: "none", color: "var(--dm)", cursor: "pointer", fontSize: "16px" }}>×</button>
          </div>

          {/* ─── Memory Tab ─── */}
          {(
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              <textarea
                value={memDraft}
                onChange={e => setMemDraft(e.target.value)}
                placeholder="Auto's persistent memory (memory.txt)...\nTell Auto to remember things, or type here directly.\nMemory is saved to file and shown in chat when updated."
                style={{ flex: 1, padding: "10px 12px", background: "transparent", border: "none", color: "var(--tx)", fontSize: "12px", fontFamily: "var(--m)", resize: "none", outline: "none", lineHeight: 1.6 }}
              />
              <div style={{ padding: "8px 10px", borderTop: "1px solid var(--bd)", display: "flex", gap: "4px", flexWrap: "wrap" }}>
                <button onClick={saveMem} style={btn("#7ce08a")}>Save</button>
                <button onClick={downloadMem} style={btn("#88bbcc")}>Download .txt</button>
                <button onClick={uploadMem} style={btn("#88bbcc")}>Upload</button>
                <button onClick={() => { setMemDraft(""); setMem(""); saveVal(MEMORY_STORAGE_KEY, ""); clearVal(LEGACY_MEMORY_STORAGE_KEY); }} style={btn("#cc7777")}>Clear</button>
              </div>
              <div style={{ padding: "6px 12px 8px", fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>
                {mem.length} chars · ~{Math.ceil(mem.length / 3.8)} tokens · Saved to memory.txt
              </div>
            </div>
          )}

          {/* ─── Uploaded SMSF Documents ─── */}
          {pdfDocs.length > 0 && (
            <div style={{ borderTop: "1px solid var(--bd)", flexShrink: 0, padding: "8px 10px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                <span style={{ fontSize: "11px" }}>{"\uD83D\uDCDA"}</span>
                <span style={{ fontWeight: 700, fontSize: "11px" }}>SMSF Documents</span>
                <span style={{ fontSize: "9px", color: "var(--ac2)", fontFamily: "var(--m)" }}>{pdfDocs.length} loaded</span>
              </div>
              {pdfDocs.map((doc, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: "6px", padding: "4px 6px",
                  borderRadius: "5px", background: "rgba(136,187,204,0.05)", border: "1px solid rgba(136,187,204,0.1)",
                  marginBottom: "4px", cursor: "pointer",
                }} onClick={() => { setPdfViewerIdx(i); setPdfViewerOpen(true); }}>
                  <span style={{ fontSize: "10px" }}>{"\uD83D\uDCC4"}</span>
                  <span style={{ fontSize: "10px", color: "var(--ac2)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "var(--m)" }}>{doc.name}</span>
                  <span style={{ fontSize: "8px", color: "var(--dm)", fontFamily: "var(--m)" }}>{doc.pageCount} pages</span>
                </div>
              ))}
              <div style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", marginTop: "4px", lineHeight: 1.4 }}>
                Click to view. AI will cross-reference these with page citations.
              </div>
            </div>
          )}

          {/* ─── Local Model Section ─── */}
          <div style={{ borderTop: "1px solid var(--bd)", flexShrink: 0 }}>
            <div style={{ padding: "8px 12px 4px", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                <span style={{ fontSize: "11px" }}>🤖</span>
                <span style={{ fontWeight: 700, fontSize: "11px" }}>Local Model</span>
                {localModelStatus === "ready" && (
                  <span style={{ fontSize: "9px", color: "var(--ac)", fontFamily: "var(--m)", padding: "1px 5px", borderRadius: "3px", background: "rgba(124,224,138,0.1)", border: "1px solid rgba(124,224,138,0.2)" }}>READY</span>
                )}
                {localModelStatus === "cached" && (
                  <span style={{ fontSize: "9px", color: "var(--ac2)", fontFamily: "var(--m)", padding: "1px 5px", borderRadius: "3px", background: "rgba(136,187,204,0.1)", border: "1px solid rgba(136,187,204,0.2)" }}>CACHED</span>
                )}
              </div>
              {localModelStatus === "ready" && (
                <span style={{ fontSize: "9px", color: "var(--ac)", fontFamily: "var(--m)" }}>OFFLINE ✓</span>
              )}
            </div>

            <div style={{ padding: "0 10px 10px", display: "flex", flexDirection: "column", gap: "6px" }}>
              {/* Tier cards */}
              {LOCAL_MODELS.map(m => {
                const locked = localModelStatus === "downloading" || localModelStatus === "loading" || localModelStatus === "ready";
                const selected = localModelId === m.id;
                return (
                  <div
                    key={m.id}
                    onClick={async () => {
                      if (locked) return;
                      // Unload current engine
                      localEngineRef.current = null;
                      setUseLocalModel(false);
                      setLocalModelId(m.id);
                      setLocalModelProgress(0);
                      setLocalModelProgressText("");
                      // Check if this model is cached — auto-load or auto-download
                      const cacheKeys = await caches.keys().catch(() => []);
                      const baseId = m.id.replace(/-MLC$/, "");
                      let isCached = false;
                      for (const cn of cacheKeys) {
                        const cache = await caches.open(cn);
                        const reqs = await cache.keys();
                        if (reqs.some(r => r.url.includes(m.id) || r.url.includes(baseId))) {
                          isCached = true;
                          break;
                        }
                      }
                      if (isCached) {
                        setLocalModelStatus("cached");
                        // Auto-load will be triggered by the useEffect
                      } else {
                        setLocalModelStatus("idle");
                        // Auto-download the selected model
                        saveVal(LOCAL_MODEL_KEY, m.id);
                      }
                    }}
                    style={{
                      border: `1px solid ${selected ? m.color + "55" : "var(--bd)"}`,
                      borderRadius: "7px",
                      padding: "7px 9px",
                      background: selected ? m.color + "0d" : "rgba(255,255,255,0.01)",
                      cursor: locked ? "default" : "pointer",
                      opacity: locked && !selected ? 0.45 : 1,
                      transition: "border-color 0.15s, background 0.15s",
                    }}
                  >
                    {/* Tier + model name row */}
                    <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "4px" }}>
                      <span style={{ fontWeight: 700, fontSize: "10px", color: m.color, fontFamily: "var(--m)", letterSpacing: "0.5px" }}>{m.tier.toUpperCase()}</span>
                      <span style={{ fontSize: "10px", color: "var(--tx)", fontWeight: 600 }}>{m.name}</span>
                      <span style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", marginLeft: "auto" }}>{m.size}</span>
                    </div>
                    {/* Spec chips */}
                    <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                      {[m.vram, m.ram, m.cpu].map(spec => (
                        <span key={spec} style={{ fontSize: "8.5px", color: selected ? m.color : "var(--dm)", fontFamily: "var(--m)", padding: "1px 5px", borderRadius: "3px", background: selected ? m.color + "15" : "rgba(255,255,255,0.03)", border: `1px solid ${selected ? m.color + "30" : "rgba(255,255,255,0.05)"}` }}>
                          {spec}
                        </span>
                      ))}
                    </div>
                    {/* Desc */}
                    <div style={{ fontSize: "9px", color: "var(--dm)", marginTop: "4px", lineHeight: 1.4 }}>{m.desc}</div>
                  </div>
                );
              })}

              {/* Progress bar */}
              {(localModelStatus === "downloading" || localModelStatus === "loading") && (
                <div>
                  <div style={{ height: "4px", background: "var(--bd)", borderRadius: "2px", overflow: "hidden" }}>
                    <div style={{ height: "100%", width: `${localModelProgress}%`, background: "linear-gradient(90deg, var(--ac), var(--ac2))", borderRadius: "2px", transition: "width 0.3s ease" }} />
                  </div>
                  <div style={{ fontSize: "9px", color: "var(--dm)", marginTop: "3px", fontFamily: "var(--m)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {localModelProgress}% — {localModelProgressText}
                  </div>
                </div>
              )}

              {/* Status/error messages */}
              {localModelStatus === "error" && (
                <div style={{ fontSize: "9px", color: "var(--dg)", fontFamily: "var(--m)", lineHeight: 1.5 }}>{localModelProgressText}</div>
              )}
              {localModelStatus === "exportDone" && (
                <div style={{ fontSize: "9px", color: "var(--ac)", fontFamily: "var(--m)" }}>{localModelProgressText}</div>
              )}

              {/* Action buttons */}
              <div style={{ display: "flex", gap: "4px", flexWrap: "wrap" }}>
                {(localModelStatus === "idle" || localModelStatus === "error") && (
                  <button onClick={downloadLocalModel} style={btn("#7ce08a")}>Download</button>
                )}
                {localModelStatus === "cached" && (
                  <>
                    <button onClick={loadLocalModel} style={btn("#7ce08a")}>Load</button>
                    <button onClick={downloadLocalModel} style={btn("#88bbcc")}>Re-download</button>
                  </>
                )}
                {localModelStatus === "ready" && (
                  <>
                    <button onClick={exportLocalModel} style={btn("#88bbcc")}>Export LLM</button>
                    <button onClick={deleteLocalModel} style={btn("#cc7777")}>Delete</button>
                  </>
                )}
                {(localModelStatus === "exportDone") && (
                  <>
                    <button onClick={exportLocalModel} style={btn("#88bbcc")}>Export Again</button>
                    <button onClick={deleteLocalModel} style={btn("#cc7777")}>Delete</button>
                  </>
                )}
                {(localModelStatus === "downloading" || localModelStatus === "loading") && (
                  <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", padding: "4px 0" }}>
                    {localModelStatus === "downloading" ? "Downloading…" : "Loading from cache…"}
                  </span>
                )}
              </div>

              {/* WebGPU warning */}
              {!navigator.gpu && (
                <div style={{ fontSize: "9px", color: "#cc8855", fontFamily: "var(--m)" }}>⚠ WebGPU not detected — requires Chrome 113+ or Edge 113+</div>
              )}

              <div style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", lineHeight: 1.5 }}>
                Runs fully offline after download. No cloud/internet used for AI.
                Model cached in browser storage.
              </div>
            </div>
          </div>

        </div>
      )}

      {/* ═══ MAIN COLUMN ═══ */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, overflow: "hidden" }}>
        {/* HEADER */}
        <header style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "7px 12px", borderBottom: "1px solid var(--bd)", background: "rgba(13,13,20,0.9)", backdropFilter: "blur(14px)", flexShrink: 0, zIndex: 10, gap: "6px", flexWrap: "wrap" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <img
              src="./Expressions/Happy.png"
              alt="Auto"
              style={{ width: "32px", height: "32px", borderRadius: "7px", objectFit: "cover", imageRendering: "pixelated" }}
              onError={(e) => { e.target.style.display = "none"; }}
            />
            <span style={{ fontWeight: 800, fontSize: "15px", letterSpacing: "-0.4px" }}>Auto</span>
            <span style={{ fontSize: "10px", color: localModelStatus === "ready" ? "var(--ac)" : "var(--dm)", fontFamily: "var(--m)" }}>
              {localModelStatus === "ready"
                ? `Offline (${LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId})`
                : localModelStatus === "downloading" || localModelStatus === "loading"
                  ? "Loading model..."
                  : "No model loaded"}
            </span>
            <span style={{ fontSize: "9px", color: "#88bbcc", fontFamily: "var(--m)", opacity: 0.6 }}>SMSF</span>
          </div>
          <div style={{ display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" }}>
            <span style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", color: localModelStatus === "ready" ? "var(--ac)" : "var(--dm)", borderColor: localModelStatus === "ready" ? "rgba(124,224,138,0.2)" : "rgba(255,255,255,0.05)", display: "inline-block" }}
              title={localModelStatus === "ready" ? "Running fully offline" : "Download a model to start"}>
              {localModelStatus === "ready" ? "OFFLINE ✓" : "OFFLINE"}
            </span>
            <span style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", padding: "2px 6px", background: "rgba(255,255,255,0.02)", borderRadius: "3px" }}>↑{ft(usage.i)} ↓{ft(usage.o)}</span>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{ ...hdr(), background: sidebarOpen ? "rgba(136,187,204,0.08)" : undefined, color: sidebarOpen ? "var(--ac2)" : undefined, borderColor: sidebarOpen ? "rgba(136,187,204,0.15)" : undefined, display: "flex", alignItems: "center", gap: "4px" }}
              title="Workspace Panel"
            >
              <span style={{ fontSize: "13px" }}>🧠</span>
              <span style={{ fontSize: "10px", fontFamily: "var(--m)" }}>Workspace</span>
            </button>
            <button onClick={clearChat} style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)" }}>Clear</button>
          </div>
        </header>

        {/* CHAT AREA */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0, padding: "14px 20px" }}>
            {msgs.length === 0 && !busy && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.45, gap: "10px", padding: "20px" }}>
                <img src="./Expressions/Happy.png" alt="Auto" style={{ width: "80px", height: "80px", imageRendering: "pixelated" }} onError={(e) => { e.target.style.display = "none"; }} />
                <div style={{ fontWeight: 700, fontSize: "16px" }}>Auto</div>
                <div style={{ fontSize: "12px", color: "var(--dm)", textAlign: "center", maxWidth: "500px", lineHeight: 1.6 }}>
                  SMSF document analysis assistant — runs fully offline.<br/>
                  Upload PDF trust deeds, investment strategies, member statements, or any SMSF documents.<br/>
                  Auto will cross-reference them with page-level citations.
                </div>
              </div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {msgs.map((m, i) => {
                return (
                  <div key={i} style={{ alignSelf: m.role === "user" ? "flex-end" : "flex-start", maxWidth: "min(960px,96%)", display: "flex", gap: "8px", alignItems: "flex-start", flexDirection: m.role === "user" ? "row-reverse" : "row" }}>
                    {m.role === "assistant" && (
                      <img
                        src="./Expressions/Happy.png"
                        alt=""
                        style={{ width: "28px", height: "28px", borderRadius: "6px", flexShrink: 0, marginTop: "2px", imageRendering: "pixelated" }}
                        onError={(e) => { e.target.style.display = "none"; }}
                      />
                    )}
                    <div style={{ background: m.role === "user" ? "rgba(124,224,138,0.08)" : "rgba(255,255,255,0.02)", border: "1px solid var(--bd)", borderRadius: "10px", padding: "10px 12px", minWidth: 0 }}>
                      {m.role === "assistant" ? <Md text={m.content} /> : <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{m.content}</div>}
                    </div>
                  </div>
                );
              })}
              {busy && (
                <div style={{ opacity: .6, fontSize: "12px", padding: "6px 2px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <img src="./Expressions/HappySpeak.png" alt="" style={{ width: "24px", height: "24px", imageRendering: "pixelated", animation: "bounce 1s infinite" }} onError={(e) => { e.target.style.display = "none"; }} />
                  <span style={{ animation: "bounce 1s infinite" }}>Thinking…</span>
                  {activityStatus && <span style={{ color: "var(--ac2)", fontFamily: "var(--m)", fontSize: "10px" }}>{activityStatus}</span>}
                </div>
              )}
              {err && <div style={{ color: "#f88", fontSize: "12px", padding: "6px 2px" }}>{err}</div>}

              <div ref={scrollRef} />
            </div>
          </div>

          {/* ═══ AUTO EXPRESSION DISPLAY ═══ */}
          {/* Keep background/border hidden so the expression floats above the input */}
          <div style={{ padding: "6px 14px 2px", borderTop: "none", background: "transparent" }}>
            <img
              src={getExprImg(busy)}
              alt="Auto"
              style={{
                width: "160px", height: "160px", imageRendering: "pixelated",
              }}
              onError={(e) => { e.target.style.display = "none"; }}
            />
          </div>

          {/* INPUT */}
          <div style={{ padding: "10px 20px", borderTop: "1px solid var(--bd)", background: "rgba(13,13,20,0.7)" }}>
            {/* Attachment preview chips */}
            {attachments.length > 0 && (
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "8px", padding: "4px 0" }}>
                {attachments.map((att, i) => (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: "6px",
                    padding: "4px 8px", borderRadius: "6px",
                    background: att.isPdf ? "rgba(136,187,204,0.08)" : "rgba(124,224,138,0.06)",
                    border: `1px solid ${att.isPdf ? "rgba(136,187,204,0.2)" : "rgba(124,224,138,0.15)"}`,
                    fontSize: "11px", fontFamily: "var(--m)", color: att.isPdf ? "var(--ac2)" : "var(--ac)",
                    maxWidth: "260px", overflow: "hidden",
                  }}>
                    <span style={{ flexShrink: 0, fontSize: "12px" }}>{att.isPdf ? "\uD83D\uDCDA" : att.isImage ? "\uD83D\uDDBC" : "\uD83D\uDCC4"}</span>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{att.name}</span>
                    {att.isPdf && <span style={{ fontSize: "8px", color: "var(--ac2)", flexShrink: 0 }}>{att.pageCount}pg</span>}
                    <span style={{ fontSize: "9px", color: "var(--dm)", flexShrink: 0 }}>{att.size >= 1024*1024 ? (att.size / (1024*1024)).toFixed(1)+"MB" : (att.size / 1024).toFixed(0)+"KB"}</span>
                    {att.isPdf && (
                      <button
                        onClick={() => {
                          const idx = pdfDocs.findIndex(d => d.name === att.name);
                          if (idx >= 0) { setPdfViewerIdx(idx); setPdfViewerOpen(true); }
                        }}
                        style={{ background: "none", border: "1px solid rgba(136,187,204,0.3)", color: "var(--ac2)", cursor: "pointer", fontSize: "9px", padding: "1px 5px", borderRadius: "3px", flexShrink: 0, fontFamily: "var(--m)" }}
                        title="View PDF"
                      >View</button>
                    )}
                    <button
                      onClick={() => removeAttachment(i)}
                      style={{ background: "none", border: "none", color: "var(--dg)", cursor: "pointer", fontSize: "13px", padding: "0 2px", lineHeight: 1, flexShrink: 0 }}
                      title="Remove"
                    >&times;</button>
                  </div>
                ))}
              </div>
            )}

            {/* Input row with + button */}
            <div style={{ display: "flex", gap: "8px", alignItems: "flex-end" }}>
              {/* Attachment + button (left side, inspired by Claude Code / DeepSeek) */}
              <div style={{ position: "relative", flexShrink: 0 }}>
                <button
                  onClick={() => setAttachMenuOpen(!attachMenuOpen)}
                  style={{
                    width: "36px", height: "36px", borderRadius: "50%",
                    border: "1px solid var(--bd)", background: attachMenuOpen ? "rgba(124,224,138,0.1)" : "rgba(255,255,255,0.03)",
                    color: attachMenuOpen ? "var(--ac)" : "var(--dm)",
                    cursor: "pointer", fontSize: "18px", fontWeight: 300,
                    display: "flex", alignItems: "center", justifyContent: "center",
                    transition: "all 0.15s ease",
                    transform: attachMenuOpen ? "rotate(45deg)" : "none",
                  }}
                  title="Attach files"
                >+</button>

                {/* Attachment dropdown menu */}
                {attachMenuOpen && (
                  <div style={{
                    position: "absolute", bottom: "42px", left: "0",
                    background: "#0d0d18", border: "1px solid var(--bd)", borderRadius: "10px",
                    padding: "6px", minWidth: "180px", zIndex: 100,
                    boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
                    animation: "fadeIn .15s ease",
                  }}>
                    <button
                      onClick={() => {
                        const pdfInput = document.createElement("input");
                        pdfInput.type = "file";
                        pdfInput.accept = ".pdf,application/pdf";
                        pdfInput.multiple = true;
                        pdfInput.onchange = handleAttachFiles;
                        pdfInput.click();
                        setAttachMenuOpen(false);
                      }}
                      style={{
                        display: "flex", alignItems: "center", gap: "8px", width: "100%",
                        padding: "8px 10px", background: "transparent", border: "none",
                        color: "var(--ac)", cursor: "pointer", borderRadius: "6px",
                        fontSize: "12px", fontFamily: "var(--f)", textAlign: "left", fontWeight: 600,
                      }}
                      onMouseEnter={e => e.target.style.background = "rgba(124,224,138,0.06)"}
                      onMouseLeave={e => e.target.style.background = "transparent"}
                    >
                      <span style={{ fontSize: "15px", width: "20px", textAlign: "center" }}>{"\uD83D\uDCDA"}</span>
                      Upload SMSF Document (PDF)
                    </button>
                    <div style={{ height: "1px", background: "var(--bd)", margin: "4px 6px" }}></div>
                    <button
                      onClick={() => { attachInputRef.current?.click(); }}
                      style={{
                        display: "flex", alignItems: "center", gap: "8px", width: "100%",
                        padding: "8px 10px", background: "transparent", border: "none",
                        color: "var(--tx)", cursor: "pointer", borderRadius: "6px",
                        fontSize: "12px", fontFamily: "var(--f)", textAlign: "left",
                      }}
                      onMouseEnter={e => e.target.style.background = "rgba(255,255,255,0.04)"}
                      onMouseLeave={e => e.target.style.background = "transparent"}
                    >
                      <span style={{ fontSize: "15px", width: "20px", textAlign: "center" }}>{"\uD83D\uDCC4"}</span>
                      Upload File
                    </button>
                    <button
                      onClick={() => {
                        const imgInput = document.createElement("input");
                        imgInput.type = "file";
                        imgInput.accept = "image/*";
                        imgInput.multiple = true;
                        imgInput.onchange = handleAttachFiles;
                        imgInput.click();
                        setAttachMenuOpen(false);
                      }}
                      style={{
                        display: "flex", alignItems: "center", gap: "8px", width: "100%",
                        padding: "8px 10px", background: "transparent", border: "none",
                        color: "var(--tx)", cursor: "pointer", borderRadius: "6px",
                        fontSize: "12px", fontFamily: "var(--f)", textAlign: "left",
                      }}
                      onMouseEnter={e => e.target.style.background = "rgba(255,255,255,0.04)"}
                      onMouseLeave={e => e.target.style.background = "transparent"}
                    >
                      <span style={{ fontSize: "15px", width: "20px", textAlign: "center" }}>{"\uD83D\uDDBC"}</span>
                      Upload Image
                    </button>
                    <div style={{ height: "1px", background: "var(--bd)", margin: "4px 6px" }}></div>
                    <button
                      onClick={() => {
                        navigator.clipboard.readText().then(text => {
                          if (text && text.trim()) {
                            setAttachments(prev => prev.length >= 5 ? prev : [...prev, {
                              name: "clipboard.txt",
                              type: "text/plain",
                              content: text.slice(0, 512 * 1024),
                              size: new Blob([text]).size,
                              isImage: false,
                            }]);
                          }
                        }).catch(() => {});
                        setAttachMenuOpen(false);
                      }}
                      style={{
                        display: "flex", alignItems: "center", gap: "8px", width: "100%",
                        padding: "8px 10px", background: "transparent", border: "none",
                        color: "var(--tx)", cursor: "pointer", borderRadius: "6px",
                        fontSize: "12px", fontFamily: "var(--f)", textAlign: "left",
                      }}
                      onMouseEnter={e => e.target.style.background = "rgba(255,255,255,0.04)"}
                      onMouseLeave={e => e.target.style.background = "transparent"}
                    >
                      <span style={{ fontSize: "15px", width: "20px", textAlign: "center" }}>{"\uD83D\uDCCB"}</span>
                      Paste from Clipboard
                    </button>
                  </div>
                )}

                {/* Hidden file input */}
                <input
                  ref={attachInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.txt,.md,.json,.csv,.xml,.html,.css,.js,.jsx,.ts,.tsx,.py,.java,.c,.cpp,.h,.go,.rs,.rb,.php,.sql,.yaml,.yml,.toml,.ini,.cfg,.log,.sh,.bat,.ps1,.r,.m,.swift,.kt,.dart,.lua,.pl,.ex,.exs,.hs,.scala,.clj,.el,.vim,.dockerfile,.makefile,.env,.gitignore,.editorconfig,image/*"
                  onChange={handleAttachFiles}
                  style={{ display: "none" }}
                />
              </div>

              {/* Textarea */}
              <textarea
                ref={inputRef}
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => {
                  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
                }}
                onPaste={e => {
                  // Handle pasted files (images from clipboard)
                  const items = Array.from(e.clipboardData?.items || []);
                  const fileItems = items.filter(item => item.kind === "file");
                  if (fileItems.length > 0) {
                    e.preventDefault();
                    const files = fileItems.map(item => item.getAsFile()).filter(Boolean);
                    handleAttachFiles({ target: { files } });
                  }
                }}
                placeholder={attachments.length > 0 ? "Ask about your SMSF documents... (optional)" : "Ask about SMSF compliance, documents, or upload a PDF..."}
                style={{ flex: 1, minHeight: "44px", maxHeight: "180px", resize: "vertical", borderRadius: "8px", border: "1px solid var(--bd)", background: "rgba(255,255,255,0.02)", color: "var(--tx)", padding: "10px 12px", fontFamily: "var(--f)", fontSize: "13px", outline: "none" }}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "7px" }}>
              <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>
                {msgs.filter(m => !(m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM:"))).length} msgs
                {attachments.length > 0 && <span style={{ color: "var(--ac)", marginLeft: "8px" }}>{attachments.length} file{attachments.length > 1 ? "s" : ""} attached</span>}
              </span>
              <div style={{ display: "flex", gap: "4px", alignItems: "center" }}>
                {busy && <button onClick={() => abortRef.current?.abort()} style={{ ...btn("#cc7777") }}>Cancel</button>}
                <button onClick={send} disabled={busy || (!input.trim() && attachments.length === 0)} style={{ ...btn("#7ce08a"), opacity: busy || (!input.trim() && attachments.length === 0) ? .5 : 1 }}>Send</button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        @keyframes bounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-4px)} }
        @keyframes fadeIn { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
        @keyframes slideR { from{opacity:0;transform:translateX(-12px)} to{opacity:1;transform:translateX(0)} }
        @keyframes slideL { from{opacity:0;transform:translateX(12px)} to{opacity:1;transform:translateX(0)} }
        @keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.85;transform:scale(1.03)} }
        *{box-sizing:border-box;margin:0}
        ::-webkit-scrollbar{width:4px} ::-webkit-scrollbar-track{background:transparent}
        ::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.05);border-radius:2px}
        textarea::placeholder{color:#333}
        button:hover{filter:brightness(1.12)}
        input::placeholder{color:#333}
      `}</style>
    </div>
  );
}

function btn(c) {
  return { padding: "4px 10px", fontSize: "10px", borderRadius: "5px", border: `1px solid ${c}33`, background: `${c}0a`, color: c, cursor: "pointer", fontFamily: "var(--m)", fontWeight: 500 };
}
function hdr() {
  return { padding: "4px 8px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--bd)", borderRadius: "5px", color: "var(--dm)", fontSize: "12px", cursor: "pointer" };
}

// ─── Error Boundary to prevent blank screen crashes ───
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null, retryCount: 0 };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, info) {
    console.error("Auto crashed:", error, info);
    // Auto-recover from transient render errors (retry up to 3 times)
    if (this.state.retryCount < 3) {
      setTimeout(() => {
        this.setState(prev => ({ hasError: false, error: null, retryCount: prev.retryCount + 1 }));
      }, 500 * (this.state.retryCount + 1));
    }
  }
  render() {
    if (this.state.hasError && this.state.retryCount >= 3) {
      return React.createElement("div", {
        style: { padding: "40px", background: "#07070b", color: "#cc7777", fontFamily: "monospace", height: "100vh", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "16px" }
      },
        React.createElement("div", { style: { fontSize: "40px" } }, "\uD83D\uDE3F"),
        React.createElement("h2", { style: { color: "#e88", margin: 0 } }, "Auto encountered an error"),
        React.createElement("pre", { style: { color: "#888", fontSize: "12px", maxWidth: "600px", overflow: "auto", padding: "12px", background: "#0a0a12", borderRadius: "8px", border: "1px solid #181824" } },
          String(this.state.error)
        ),
        React.createElement("div", { style: { color: "#666", fontSize: "11px" } }, "Your chat and memory have been preserved."),
        React.createElement("button", {
          onClick: () => {
            this.setState({ hasError: false, error: null, retryCount: 0 });
          },
          style: { padding: "8px 20px", background: "rgba(136,187,204,0.1)", border: "1px solid rgba(136,187,204,0.3)", borderRadius: "6px", color: "#88bbcc", cursor: "pointer", fontSize: "13px" }
        }, "Try to Recover (keep chat)"),
        React.createElement("button", {
          onClick: () => {
            try {
              window.storage && window.storage.set(CHAT_STORAGE_KEY, "[]");
              window.storage && window.storage.set(LEGACY_CHAT_STORAGE_KEY, "[]");
            } catch(e) {}
            try {
              window.localStorage.setItem(CHAT_STORAGE_KEY, "[]");
              window.localStorage.setItem(LEGACY_CHAT_STORAGE_KEY, "[]");
            } catch(e) {}
            this.setState({ hasError: false, error: null, retryCount: 0 });
          },
          style: { padding: "8px 20px", background: "rgba(124,224,138,0.1)", border: "1px solid rgba(124,224,138,0.3)", borderRadius: "6px", color: "#7ce08a", cursor: "pointer", fontSize: "13px" }
        }, "Clear Chat & Recover"),
        React.createElement("button", {
          onClick: () => window.location.reload(),
          style: { padding: "8px 20px", background: "rgba(204,119,119,0.1)", border: "1px solid rgba(204,119,119,0.3)", borderRadius: "6px", color: "#cc7777", cursor: "pointer", fontSize: "13px" }
        }, "Reload Page")
      );
    }
    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")).render(
  React.createElement(ErrorBoundary, null, React.createElement(Auto))
);
