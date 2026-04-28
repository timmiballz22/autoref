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

// ─── Streaming throttle — batches UI updates to prevent lag on weak hardware ───
function createStreamThrottle(setState, intervalMs = 180) {
  let pending = null;
  let timer = null;
  return {
    update(value) {
      pending = value;
      if (!timer) {
        timer = setTimeout(() => {
          if (pending !== null) setState(pending);
          timer = null;
        }, intervalMs);
      }
    },
    flush() {
      if (timer) clearTimeout(timer);
      if (pending !== null) setState(pending);
      pending = null;
      timer = null;
    },
  };
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
    contextWindow: 32768,
    slidingWindow: 32768,
    prefillChunk: 2048,
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
    contextWindow: 131072,
    slidingWindow: 131072,
    prefillChunk: 4096,
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
    contextWindow: 131072,
    slidingWindow: 131072,
    prefillChunk: 4096,
  },
];

function parseApproxSizeToMB(sizeText) {
  if (!sizeText) return Number.MAX_SAFE_INTEGER;
  const norm = String(sizeText).replace(/,/g, '').trim();
  const m = norm.match(/([\d.]+)\s*(KB|MB|GB|TB)/i);
  if (!m) return Number.MAX_SAFE_INTEGER;
  const value = parseFloat(m[1]);
  if (!Number.isFinite(value)) return Number.MAX_SAFE_INTEGER;
  const unit = m[2].toUpperCase();
  const scale = unit === 'KB' ? 1 / 1024 : unit === 'MB' ? 1 : unit === 'GB' ? 1024 : 1024 * 1024;
  return value * scale;
}

function getLightestQwenModelId() {
  const qwen = LOCAL_MODELS.filter(m => /qwen/i.test(m.id) || /qwen/i.test(m.name));
  if (qwen.length === 0) return LOCAL_MODELS[0]?.id || null;
  return qwen.slice().sort((a, b) => parseApproxSizeToMB(a.size) - parseApproxSizeToMB(b.size))[0].id;
}

async function isModelCached(modelId) {
  if (!modelId || !('caches' in window)) return false;
  try {
    const cacheKeys = await caches.keys();
    const baseId = modelId.replace(/-MLC$/, '');
    for (const cacheName of cacheKeys) {
      const cache = await caches.open(cacheName);
      const reqs = await cache.keys();
      if (reqs.some(r => r.url.includes(modelId) || r.url.includes(baseId))) return true;
    }
  } catch {}
  return false;
}


// ─── Build WebLLM engine config with expanded context window ───
// IMPORTANT: model_lib must be a real URL to a .wasm file. Hand-rolling the
// config (model_lib: modelId) makes WebLLM fetch a relative path, which the
// host returns as the SPA index.html — WebAssembly.instantiate then fails
// with "expected magic word 00 61 73 6d, found 3c 21 44 4f" (i.e. "<!DO" from
// "<!DOCTYPE"). We clone the prebuilt entry instead so URLs are correct.
function buildEngineConfig(webllm, modelId) {
  const modelDef = LOCAL_MODELS.find(m => m.id === modelId);
  if (!modelDef) return undefined;
  const prebuilt = webllm && webllm.prebuiltAppConfig;
  const list = prebuilt && Array.isArray(prebuilt.model_list) ? prebuilt.model_list : null;
  if (!list) return undefined; // fall back to WebLLM's default appConfig
  const entry = list.find(e => e.model_id === modelId);
  if (!entry) return undefined; // model not in prebuilt list — let WebLLM use default

  const baseOverrides = { ...(entry.overrides || {}) };
  const hasPositiveSliding = Number(baseOverrides.sliding_window_size) > 0;
  const hasPositiveContext = Number(baseOverrides.context_window_size) > 0;

  // WebLLM requires exactly one positive window at runtime:
  // either context_window_size OR sliding_window_size.
  // Keep the model's native attention mode and disable the other one.
  const useSlidingWindow = hasPositiveSliding && !hasPositiveContext;
  const resolvedContext = useSlidingWindow ? -1 : modelDef.contextWindow;
  const resolvedSliding = useSlidingWindow ? modelDef.slidingWindow : -1;

  return {
    ...prebuilt,
    model_list: [{
      ...entry,
      overrides: {
        ...baseOverrides,
        context_window_size: resolvedContext,
        sliding_window_size: resolvedSliding,
        prefill_chunk_size: modelDef.prefillChunk,
      },
    }],
  };
}

// Translate WebLLM/WASM errors into actionable messages
function describeLoadError(e) {
  const msg = (e && e.message) || String(e || "");
  if (/magic word|found 3c 21 44 4f|<!DOCTYPE|expected magic/i.test(msg)) {
    return "Model library (.wasm) URL returned HTML instead of a WebAssembly module. This usually means the host is offline, blocking huggingface.co/raw.githubusercontent.com, or a proxy is rewriting requests. Check your network/firewall and try again.";
  }
  if (/Failed to fetch|NetworkError|ERR_INTERNET_DISCONNECTED/i.test(msg)) {
    return "Network error while downloading the model. Check your internet connection and retry.";
  }
  if (/WebGPU|navigator\.gpu/i.test(msg)) {
    return "WebGPU not available. Use Chrome 113+ or Edge 113+, and ensure hardware acceleration is enabled.";
  }
  if (/out of memory|OOM|allocation/i.test(msg)) {
    return "GPU ran out of memory. Close other tabs, remove uploaded documents, or pick a smaller model (Qwen 0.5B).";
  }
  return msg || "Unknown error";
}

// ─── Estimate token count from text (conservative ~3.2 chars per token to prevent OOM) ───
function estimateTokens(text) {
  if (!text) return 0;
  return Math.ceil(text.length / 3.2);
}

// ─── Get current model's context budget for documents ───
// Conservative to prevent GPU OOM on weak hardware (Acer Aspire 5 / Intel iGPU)
function getDocTokenBudget(modelId) {
  const modelDef = LOCAL_MODELS.find(m => m.id === modelId);
  const totalCtx = modelDef ? modelDef.contextWindow : 32768;
  // For small models (Qwen 0.5B), use much tighter budget — iGPU can't handle full 32K KV cache
  const isSmallModel = totalCtx <= 32768;
  // Reserve tokens: system prompt (~4K), chat history (~3K), generation (~3K), memory (~1K), safety margin (~2K)
  const reserved = isSmallModel ? 16000 : 14000;
  return Math.max(totalCtx - reserved, 6000);
}

// ─── Chunk document into page-groups for smart inclusion ───
function chunkDocumentByPages(docText, pageCount, chunkPages = 20) {
  const chunks = [];
  for (let start = 1; start <= pageCount; start += chunkPages) {
    const end = Math.min(start + chunkPages - 1, pageCount);
    const text = getDocPages(docText, start, end);
    chunks.push({ startPage: start, endPage: end, text, tokens: estimateTokens(text) });
  }
  return chunks;
}

// ─── Score document chunks by relevance to user query ───
function scoreChunkRelevance(chunk, queryTerms) {
  const lower = chunk.text.toLowerCase();
  let score = 0;
  for (const term of queryTerms) {
    const re = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
    const matches = lower.match(re);
    if (matches) score += matches.length;
  }
  // Boost first and last chunks (usually contain critical summary/conclusion info)
  if (chunk.startPage === 1) score += 5;
  return score;
}

// ─── Select the most relevant chunks that fit within token budget ───
function selectRelevantChunks(chunks, queryTerms, tokenBudget) {
  if (chunks.length === 0) return [];
  const totalTokens = chunks.reduce((s, c) => s + c.tokens, 0);
  // If everything fits, include everything
  if (totalTokens <= tokenBudget) return chunks;

  // Score each chunk
  const scored = chunks.map(c => ({ ...c, score: scoreChunkRelevance(c, queryTerms) }));
  // Always include first chunk (intro/summary pages)
  const selected = [scored[0]];
  let usedTokens = scored[0].tokens;

  // Sort remaining by relevance score (descending)
  const remaining = scored.slice(1).sort((a, b) => b.score - a.score);
  for (const chunk of remaining) {
    if (usedTokens + chunk.tokens <= tokenBudget) {
      selected.push(chunk);
      usedTokens += chunk.tokens;
    }
  }
  // Re-sort by page order for coherent reading
  selected.sort((a, b) => a.startPage - b.startPage);
  return selected;
}

// ─── Extract search terms from user query for relevance scoring ───
function extractQueryTerms(query) {
  if (!query) return [];
  const stopWords = new Set(["the","a","an","is","are","was","were","be","been","being","have","has","had","do","does","did","will","would","shall","should","may","might","can","could","must","need","dare","ought","used","to","of","in","for","on","with","at","by","from","as","into","through","during","before","after","above","below","between","out","off","over","under","again","further","then","once","here","there","when","where","why","how","all","each","every","both","few","more","most","other","some","such","no","nor","not","only","own","same","so","than","too","very","just","because","and","but","or","if","while","although","though","even","that","which","who","whom","what","this","these","those","my","your","his","her","its","our","their","me","him","us","them","i","you","he","she","it","we","they","about","please","help","tell","show","explain","describe","find","check","look","see","give"]);
  return query.toLowerCase()
    .replace(/[^\w\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2 && !stopWords.has(w));
}

// ─── Persistent Storage ───
const CHAT_STORAGE_KEY = "auto-chat";
const MEMORY_STORAGE_KEY = "auto-memory";

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

  return current || [];
}
async function saveChat(msgs) {
  // Only save user/assistant messages, skip system research messages, cap at 100
  const toSave = msgs.filter(m => !(m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM:"))).slice(-60);
  const json = JSON.stringify(toSave);
  // Save to BOTH storage backends for redundancy
  try { if (window.storage?.set) await window.storage.set(CHAT_STORAGE_KEY, json); } catch {}
  try { window.localStorage.setItem(CHAT_STORAGE_KEY, json); } catch {}
}
// ─── PDF Text Extraction (uses pdf.js loaded from CDN) ───
// Optimised for 1000+ page documents: batched processing, lower scale, limited images
const MAX_PAGE_IMAGES = 3; // Only render first 3 scanned pages as images to save memory

async function extractPdfContent(arrayBuffer, fileName, onProgress = null) {
  if (!window.pdfjsLib) throw new Error("PDF.js not loaded. Refresh the page.");
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pageCount = pdf.numPages;
  let fullText = "";
  const pageImages = [];
  const BATCH_SIZE = 10; // Yield to UI every 10 pages
  const skipImages = pageCount > 200; // Don't render images for very large docs

  for (let i = 1; i <= pageCount; i++) {
    const page = await pdf.getPage(i);
    const textContent = await page.getTextContent();
    // Spatial-aware text extraction: preserve layout using pdf.js transform coordinates
    const sortedItems = [...textContent.items].filter(item => item.str.trim()).sort((a, b) => {
      const yDiff = b.transform[5] - a.transform[5]; // PDF y-axis is bottom-up
      if (Math.abs(yDiff) > 5) return yDiff; // Different lines (5pt threshold)
      return a.transform[4] - b.transform[4]; // Same line, sort left-to-right
    });

    // Group items into lines based on y-position proximity
    let lines = [];
    let currentLine = [];
    let lastY = null;
    for (const item of sortedItems) {
      const y = item.transform[5];
      if (lastY !== null && Math.abs(lastY - y) > 5) {
        lines.push(currentLine);
        currentLine = [];
      }
      currentLine.push(item);
      lastY = y;
    }
    if (currentLine.length > 0) lines.push(currentLine);

    // Reconstruct text with spacing awareness (tabs for columns, spaces for words)
    const pageText = lines.map(line => {
      let lineText = "";
      let lastX = null;
      let lastWidth = 0;
      for (const item of line) {
        const x = item.transform[4];
        if (lastX !== null) {
          const gap = x - (lastX + lastWidth);
          if (gap > 15) lineText += "\t"; // Tab for large gaps (columns/tables)
          else if (gap > 3) lineText += " ";
        }
        lineText += item.str;
        lastX = x;
        lastWidth = item.width || (item.str.length * 5);
      }
      return lineText;
    }).join("\n").trim();

    fullText += `\n\n=== [Page ${i}] ===\n`;

    if (pageText.length > 30) {
      // Enough text content — use extracted text with spatial layout preserved
      fullText += pageText;
    } else if (!skipImages && pageImages.length < MAX_PAGE_IMAGES) {
      // Scanned/handwritten page — render to image (limited to first MAX_PAGE_IMAGES)
      fullText += "(Scanned/handwritten page — see attached page image)";
      try {
        const scale = 1.5; // ~150 DPI — 75% less memory than 3.0 scale
        const viewport = page.getViewport({ scale });
        const canvas = document.createElement("canvas");
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        const ctx = canvas.getContext("2d");
        await page.render({ canvasContext: ctx, viewport }).promise;
        pageImages.push({ page: i, dataUrl: canvas.toDataURL("image/jpeg", 0.85) });
        // Release canvas memory immediately
        canvas.width = 0;
        canvas.height = 0;
      } catch (e) {
        console.warn(`Failed to render page ${i} as image:`, e);
        fullText += "\n(Could not render page image)";
      }
    } else {
      // Scanned page but skip image rendering (too many or large document)
      fullText += "(Scanned/handwritten page — text not extractable, image omitted to save memory)";
    }

    // Yield to UI thread every BATCH_SIZE pages and report progress
    if (i % BATCH_SIZE === 0) {
      if (onProgress) onProgress(i, pageCount);
      await new Promise(r => setTimeout(r, 0));
    }
    // Clean up page reference
    page.cleanup();
  }

  return { text: fullText.trim(), pageCount, pageImages };
}

// ─── Web Search via DuckDuckGo Instant Answer API ───
// Used by Reviewer agents to research topics on the web.
async function searchWeb(query) {
  const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_redirect=1&no_html=1&kl=wt-wt`;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Search failed: HTTP ${resp.status}`);
  const data = await resp.json();

  const lines = [];
  if (data.AbstractText) {
    lines.push(`Summary: ${data.AbstractText}`);
    if (data.AbstractURL) lines.push(`Source: ${data.AbstractURL}`);
  }
  if (data.Answer) lines.push(`Direct answer: ${data.Answer}`);
  if (data.RelatedTopics?.length) {
    lines.push("Related topics:");
    for (const topic of data.RelatedTopics.slice(0, 6)) {
      if (topic.Text) lines.push(`  - ${topic.Text}${topic.FirstURL ? ` (${topic.FirstURL})` : ""}`);
      // Handle sub-topics
      if (topic.Topics) {
        for (const sub of topic.Topics.slice(0, 3)) {
          if (sub.Text) lines.push(`    • ${sub.Text}`);
        }
      }
    }
  }
  if (data.Results?.length) {
    lines.push("Results:");
    for (const r of data.Results.slice(0, 3)) {
      if (r.Text) lines.push(`  - ${r.Text}${r.FirstURL ? ` (${r.FirstURL})` : ""}`);
    }
  }
  return lines.length > 0 ? lines.join("\n") : "No results found for this query.";
}

// ─── Document Indexing for 1000+ page PDFs ───
// Builds a compact Table of Contents from extracted PDF text
function buildDocIndex(docText, pageCount) {
  const pages = docText.split(/=== \[Page \d+\] ===/);
  const toc = [];
  for (let i = 1; i < pages.length && i <= pageCount; i++) {
    const pageContent = (pages[i] || "").trim();
    const preview = pageContent.slice(0, 150).replace(/\s+/g, " ").trim();
    if (preview) toc.push(`Page ${i}: ${preview}...`);
    else toc.push(`Page ${i}: (empty or scanned page)`);
  }
  return toc.join("\n");
}

// Extracts text for specific page range from a document's full text
function getDocPages(docText, startPage, endPage) {
  const parts = [];
  for (let p = startPage; p <= endPage; p++) {
    const marker = `=== [Page ${p}] ===`;
    const nextMarker = `=== [Page ${p + 1}] ===`;
    const startIdx = docText.indexOf(marker);
    if (startIdx < 0) continue;
    const endIdx = docText.indexOf(nextMarker, startIdx);
    parts.push(docText.slice(startIdx, endIdx > startIdx ? endIdx : undefined).trim());
  }
  return parts.join("\n\n");
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

// ─── Memoised Markdown — prevents re-render of all messages during streaming ───
const MemoMd = React.memo(Md);

// ─── Unique message ID generator ───
let _msgIdCounter = 0;
function nextMsgId() { return "m" + (++_msgIdCounter) + "-" + Date.now(); }

// ─── Memoised single chat message — prevents re-rendering every message on each state change ───
const ChatMessage = React.memo(function ChatMessage({ msg }) {
  return (
    <div style={{ alignSelf: msg.role === "user" ? "flex-end" : "flex-start", maxWidth: "min(960px,96%)", display: "flex", gap: "8px", alignItems: "flex-start", flexDirection: msg.role === "user" ? "row-reverse" : "row", contain: "layout style paint" }}>
      {msg.role === "assistant" && (
        <span style={{ width: "10px", height: "10px", borderRadius: "999px", background: "var(--ac)", flexShrink: 0, marginTop: "8px" }} />
      )}
      <div style={{ background: msg.role === "user" ? "rgba(124,224,138,0.08)" : "rgba(255,255,255,0.02)", border: "1px solid var(--bd)", borderRadius: "10px", padding: "10px 12px", minWidth: 0 }}>
        {msg.role === "assistant" ? <MemoMd text={msg.content} /> : <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{msg.content}</div>}
        {msg.role === "assistant" && (
          <div style={{ display: "flex", gap: "4px", marginTop: "6px", paddingTop: "6px", borderTop: "1px solid rgba(255,255,255,0.04)" }}>
            <button onClick={() => { try { navigator.clipboard.writeText(msg.content); } catch {} }} style={{ background: "none", border: "1px solid rgba(136,187,204,0.15)", color: "var(--dm)", cursor: "pointer", fontSize: "9px", padding: "2px 6px", borderRadius: "3px", fontFamily: "var(--m)" }}>Copy</button>
          </div>
        )}
      </div>
    </div>
  );
});

// ─── PDF Viewer Component ───
function PdfViewer({ pdfData, blobUrl, onClose }) {
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [zoom, setZoom] = useState(1.0);
  const canvasRef = useRef(null);
  const pdfDocRef = useRef(null);
  const [jumpPage, setJumpPage] = useState("");

  useEffect(() => {
    if (!window.pdfjsLib) return;
    // Support both legacy ArrayBuffer (pdfData) and new Blob URL (blobUrl)
    const source = blobUrl || pdfData;
    if (!source) return;
    let cancelled = false;
    (async () => {
      try {
        let loadSource;
        if (blobUrl) {
          // Fetch the Blob URL to get ArrayBuffer on-demand (lazy loading)
          const resp = await fetch(blobUrl);
          const buf = await resp.arrayBuffer();
          loadSource = { data: buf };
        } else {
          loadSource = { data: pdfData };
        }
        const pdf = await pdfjsLib.getDocument(loadSource).promise;
        if (cancelled) return;
        pdfDocRef.current = pdf;
        setTotalPages(pdf.numPages);
        setCurrentPage(1);
      } catch (e) {
        console.error("PDF load error:", e);
      }
    })();
    return () => { cancelled = true; };
  }, [pdfData, blobUrl]);

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
  const [pdfDocs, setPdfDocs] = useState([]); // [{name, text, pageCount, pageImages, blobUrl}]
  const [pdfLoading, setPdfLoading] = useState([]); // [{name, progress, total}] — PDFs being extracted
  const [pdfViewerOpen, setPdfViewerOpen] = useState(false);
  const [pdfViewerIdx, setPdfViewerIdx] = useState(0); // which pdf to view
  const [docTextViewerOpen, setDocTextViewerOpen] = useState(false); // full extracted text viewer
  const [docTextViewerIdx, setDocTextViewerIdx] = useState(0);
  const [streamingText, setStreamingText] = useState(""); // real-time streaming response
  const lastUserQueryRef = useRef(""); // tracks last user query for smart document chunking
  const [artifactsOpen, setArtifactsOpen] = useState(false);
  const [exportedArtifacts, setExportedArtifacts] = useState([]); // [{id, name, blobUrl, size, timestamp}]

  // Load on mount — always target the lightest default model first
  useEffect(() => {
    loadVal(MEMORY_STORAGE_KEY).then(v => { setMem(v || ""); setMemDraft(v || ""); });
    loadChat().then(v => {
      if (v?.length) {
        // Validate loaded messages: each must have role + string content (prevents render crashes from corrupted data)
        const valid = v.filter(m => m && typeof m.role === "string" && (m.role === "user" || m.role === "assistant") && typeof m.content === "string");
        // Assign stable IDs to loaded messages
        const withIds = valid.map(m => m._id ? m : { ...m, _id: nextMsgId() });
        setMsgs(withIds);
      }
    });
    // Always auto-select the lightest Qwen model at startup so we do not
    // accidentally auto-load a previously saved heavier model.
    (async () => {
      const qwenDefault = getLightestQwenModelId();
      if (!qwenDefault) return;
      setLocalModelId(qwenDefault);
      saveVal(LOCAL_MODEL_KEY, qwenDefault);
      const cached = await isModelCached(qwenDefault);
      setLocalModelStatus(cached ? "cached" : "idle");
      if (!cached) setLocalModelProgressText("No cache found for selected model. Auto-download will start.");
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
          const engineCfg = buildEngineConfig(webllm, localModelId);
          const engineOpts = {
            initProgressCallback: (p) => {
              setLocalModelProgress(Math.round((p.progress || 0) * 100));
              setLocalModelProgressText(p.text || "");
            },
          };
          if (engineCfg) engineOpts.appConfig = engineCfg;
          const engine = await webllm.CreateMLCEngine(localModelId, engineOpts);
          localEngineRef.current = engine;
          setLocalModelStatus("ready");
          setUseLocalModel(true);
        } catch (e) {
          console.error("Auto-load from cache failed:", e);
          setLocalModelStatus("cached");
          setLocalModelProgressText("Auto-load failed — click Load to try again. " + describeLoadError(e));
        }
      })();
    }
    // Reset the guard when model ID changes (user picked a different model)
    if (localModelStatus === "idle" || localModelStatus === "downloading") {
      autoLoadAttemptedRef.current = false;
    }
  }, [localModelStatus, localModelId]);

  // Cleanup blob URLs when pdfDocs change or unmount (prevents memory leak)
  const prevBlobUrlsRef = useRef([]);
  useEffect(() => {
    const currentUrls = pdfDocs.map(d => d.blobUrl).filter(Boolean);
    const removed = prevBlobUrlsRef.current.filter(u => !currentUrls.includes(u));
    removed.forEach(u => { try { URL.revokeObjectURL(u); } catch {} });
    prevBlobUrlsRef.current = currentUrls;
    return () => { currentUrls.forEach(u => { try { URL.revokeObjectURL(u); } catch {} }); };
  }, [pdfDocs]);

  // Cleanup exported artifact blob URLs on unmount
  useEffect(() => {
    return () => {
      exportedArtifacts.forEach(a => { try { URL.revokeObjectURL(a.blobUrl); } catch {} });
    };
  }, [exportedArtifacts]);

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

    // Auto-save every 60 seconds (reduced from 15s — localStorage writes block main thread)
    const autoSaveInterval = setInterval(persistState, 60000);

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

  // Debounced scroll-into-view to prevent excessive smooth scrolling during streaming
  const scrollTimerRef = useRef(null);
  useEffect(() => {
    if (scrollTimerRef.current) clearTimeout(scrollTimerRef.current);
    scrollTimerRef.current = setTimeout(() => {
      scrollRef.current?.scrollIntoView({ behavior: "smooth" });
    }, 150);
  }, [msgs, busy, streamingText]);

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
      // Release any prior engine's GPU resources before allocating a new one
      try { await localEngineRef.current?.unload?.(); } catch {}
      localEngineRef.current = null;
      const webllm = await getWebLLM();
      const engineCfg = buildEngineConfig(webllm, localModelId);
      const engineOpts = {
        initProgressCallback: (p) => {
          setLocalModelProgress(Math.round((p.progress || 0) * 100));
          setLocalModelProgressText(p.text || "");
        },
      };
      if (engineCfg) engineOpts.appConfig = engineCfg;
      const engine = await webllm.CreateMLCEngine(localModelId, engineOpts);
      localEngineRef.current = engine;
      setLocalModelStatus("ready");
      setUseLocalModel(true);
      saveVal(LOCAL_MODEL_KEY, localModelId);
    } catch (e) {
      console.error("Local model download failed:", e);
      setLocalModelStatus("error");
      setLocalModelProgressText(describeLoadError(e));
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
      try { await localEngineRef.current?.unload?.(); } catch {}
      localEngineRef.current = null;
      const webllm = await getWebLLM();
      const engineCfg = buildEngineConfig(webllm, localModelId);
      const engineOpts = {
        initProgressCallback: (p) => {
          setLocalModelProgress(Math.round((p.progress || 0) * 100));
          setLocalModelProgressText(p.text || "");
        },
      };
      if (engineCfg) engineOpts.appConfig = engineCfg;
      const engine = await webllm.CreateMLCEngine(localModelId, engineOpts);
      localEngineRef.current = engine;
      setLocalModelStatus("ready");
      setUseLocalModel(true);
    } catch (e) {
      console.error("Local model load failed:", e);
      setLocalModelStatus("error");
      setLocalModelProgressText(describeLoadError(e));
    }
  }, [localModelId]);

  const bootstrapDownloadAttemptedRef = useRef(false);
  useEffect(() => {
    const qwenDefault = getLightestQwenModelId();
    if (!qwenDefault) return;
    if (localModelId !== qwenDefault) return;
    if (localModelStatus !== "idle") return;
    if (bootstrapDownloadAttemptedRef.current) return;
    bootstrapDownloadAttemptedRef.current = true;
    downloadLocalModel();
  }, [localModelId, localModelStatus, downloadLocalModel]);

  const deleteLocalModel = useCallback(async () => {
    if (!window.confirm(`Delete cached model "${localModelId}"? You will need to re-download it to use it again.`)) return;
    try {
      // Release GPU resources before clearing the cache
      try { await localEngineRef.current?.unload?.(); } catch {}
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
    const MAX_ATTACHMENTS = 20; // No file size limits — accept any size

    files.forEach(file => {
      const isPdf = file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf");

      if (isPdf) {
        // PDF: show immediate placeholder chip so user sees the file was accepted
        const placeholderId = `pdf-loading-${Date.now()}-${file.name}`;
        setAttachments(prev => {
          if (prev.length >= MAX_ATTACHMENTS) return prev;
          return [...prev, { name: file.name, type: "application/pdf", content: "", size: file.size, isPdf: true, pageCount: 0, pageImages: [], _loading: true, _id: placeholderId }];
        });
        // Track loading PDF in sidebar so user sees it immediately
        setPdfLoading(prev => [...prev, { name: file.name, progress: 0, total: 0 }]);
        // Auto-open sidebar so user sees the document will appear there
        setSidebarOpen(true);

        // PDF: extract text page-by-page with page markers
        const reader = new FileReader();
        reader.onload = async () => {
          try {
            const arrayBuffer = reader.result;
            setActivityStatus(`Extracting PDF: ${file.name}...`);
            const { text, pageCount, pageImages } = await extractPdfContent(arrayBuffer, file.name, (current, total) => {
              setActivityStatus(`Extracting PDF "${file.name}": page ${current} of ${total}...`);
              setPdfLoading(prev => prev.map(p => p.name === file.name ? { ...p, progress: current, total } : p));
            });
            setActivityStatus("");
            // Replace loading placeholder with real extracted content
            setAttachments(prev => prev.map(att =>
              att._id === placeholderId
                ? { name: file.name, type: "application/pdf", content: text, size: file.size, isPdf: true, pageCount, pageImages }
                : att
            ));
            // Store PDF data for viewer — use Blob URL instead of raw ArrayBuffer
            const blobUrl = URL.createObjectURL(new Blob([arrayBuffer], { type: "application/pdf" }));
            setPdfDocs(prev => [...prev, { name: file.name, text, pageCount, pageImages, blobUrl }]);
            // Remove from loading tracker
            setPdfLoading(prev => prev.filter(p => p.name !== file.name));
          } catch (err) {
            console.error("PDF extraction failed:", err);
            setErr(`Failed to process PDF "${file.name}": ${err.message}`);
            setActivityStatus("");
            // Remove the loading placeholder on failure
            setAttachments(prev => prev.filter(att => att._id !== placeholderId));
            setPdfLoading(prev => prev.filter(p => p.name !== file.name));
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
- Summarise what each uploaded document contains and its relevance at the start of your analysis

### Page Number Citation Rules (MANDATORY):
- EVERY factual statement about a document MUST include a page citation: **[Document Name, Page X]**
- When quoting text from a document, always include the page: *"quoted text"* **[Document Name, Page X]**
- If information spans multiple pages, cite all: **[Document Name, Pages X-Y]**
- At the end of your analysis, include a "References" section listing all cited pages per document
- NEVER make a claim about document content without a page citation — this is your #1 rule
- Page numbers are marked in the document text as === [Page X] === — use these to determine exact page numbers

### Cross-Referencing Protocol:
When multiple documents are uploaded, you MUST perform systematic cross-referencing:
1. **Trust Deed vs Investment Strategy**: Check if investment powers in the deed match/permit the investment strategy allocations
2. **Investment Strategy vs Member Statements**: Verify actual asset allocation against stated strategy targets
3. **Financial Statements vs Member Balances**: Reconcile total fund assets with member accumulation/pension accounts
4. **Minutes vs Actions**: Check if trustee minutes authorise the actions reflected in other documents
5. **Compliance Checklist**: For each document, note any SIS Act requirements that appear unmet
6. **Discrepancy Register**: Explicitly list ALL discrepancies found between documents in a dedicated section

### Response Structure for Document Analysis:
1. **Document Summary**: List each uploaded document with a 1-2 line description and page count
2. **Key Findings**: Major observations with page citations
3. **Cross-Reference Analysis**: Comparisons between documents with specific page references from EACH document
4. **Discrepancies & Concerns**: Explicitly called out with page references from each document
5. **Compliance Notes**: SIS Act / regulatory requirements and how the documents address (or fail to address) them
6. **Recommendations**: Actionable next steps based on findings
7. **References**: Complete list of all document pages cited`;

    // Include uploaded PDF document content for cross-referencing
    // Smart chunked approach: fits maximum document content within model's context window
    // Uses query-relevant chunk selection for 1000+ page documents
    if (pdfDocs.length > 0) {
      const tokenBudget = getDocTokenBudget(localModelId);
      const perDocBudget = Math.floor(tokenBudget / pdfDocs.length);

      s += `\n\n<documents>\nThe following documents have been uploaded for cross-referencing. ALWAYS cite these by name and page number.\n`;
      for (const doc of pdfDocs) {
        // Detect scanned/image-based PDFs: text is empty or only contains scanned-page markers
        const textWithoutMarkers = doc.text.replace(/=== \[Page \d+\] ===/g, "").replace(/\(Scanned\/handwritten page[^)]*\)/g, "").replace(/\(Could not render[^)]*\)/g, "").replace(/\(Scanned page[^)]*\)/g, "").trim();
        const isScannedDoc = doc.pageCount > 0 && textWithoutMarkers.length < 100;
        if (isScannedDoc) {
          s += `\n<document name="${doc.name}" pages="${doc.pageCount}" status="scanned-image-pdf">\n`;
          s += `[SCANNED DOCUMENT — "${doc.name}" is an image-based/scanned PDF with ${doc.pageCount} page(s). Text extraction was not possible because this PDF contains scanned images rather than selectable text.]\n`;
          s += `[ACTION REQUIRED: You MUST clearly tell the user: "I can see that '${doc.name}' was uploaded but it appears to be a scanned/image-based PDF — I cannot read its text content. To cross-reference it, please: (1) Use an OCR tool (e.g. Adobe Acrobat, Google Drive, or online OCR) to convert the scanned PDF to a text-based PDF, then re-upload it, or (2) Copy and paste the text content directly into the chat. In the meantime, I can provide general SMSF guidance based on typical document content for this type of document."]\n`;
          s += `</document>\n`;
          continue;
        }
        const docTokens = estimateTokens(doc.text);
        if (docTokens <= perDocBudget) {
          // Fits entirely — include full text for maximum accuracy
          s += `\n<document name="${doc.name}" pages="${doc.pageCount}" included="full">\n${doc.text}\n</document>\n`;
        } else {
          // Large document — use smart chunked inclusion with relevance scoring
          const toc = buildDocIndex(doc.text, doc.pageCount);
          const tocTokens = estimateTokens(toc);
          // Reserve tokens for TOC, then fill remaining with most relevant page chunks
          const chunkBudget = Math.max(perDocBudget - tocTokens - 500, 4000);
          const chunks = chunkDocumentByPages(doc.text, doc.pageCount, 15);
          const queryTerms = extractQueryTerms(lastUserQueryRef.current || "");
          // Add SMSF-specific terms that are always relevant for cross-referencing
          const smsfTerms = ["trust", "deed", "investment", "strategy", "member", "balance", "compliance", "contribution", "pension", "benefit", "audit", "financial", "statement", "minutes", "rollover", "insurance", "asset", "allocation"];
          const allTerms = [...queryTerms, ...smsfTerms];
          const selectedChunks = selectRelevantChunks(chunks, allTerms, chunkBudget);
          const includedPages = selectedChunks.map(c => `${c.startPage}-${c.endPage}`).join(", ");

          s += `\n<document name="${doc.name}" pages="${doc.pageCount}" included="smart-chunked" pages_included="${includedPages}">`;
          s += `\n\n--- TABLE OF CONTENTS (ALL ${doc.pageCount} PAGES) ---\n${toc}\n`;
          for (const chunk of selectedChunks) {
            s += `\n\n--- PAGES ${chunk.startPage}-${chunk.endPage} ---\n${chunk.text}\n`;
          }
          const totalIncluded = selectedChunks.reduce((s, c) => s + (c.endPage - c.startPage + 1), 0);
          s += `\n[NOTE: ${totalIncluded} of ${doc.pageCount} pages included (most relevant to your query). TOC covers ALL pages. All page numbers are accurate for citation.]\n</document>\n`;
        }
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

    s += ``;

    // ─── SAFETY CAP: Prevent OOM on weak hardware (iGPU / Acer Aspire 5) ───
    // Small models (Qwen 0.5B): 35% cap — iGPU can't handle large KV cache
    // Larger models: 50% cap — leaves room for chat history + generation
    const modelDefCap = LOCAL_MODELS.find(m => m.id === localModelId);
    const isSmallCap = modelDefCap && modelDefCap.contextWindow <= 32768;
    const maxSystemTokens = modelDefCap ? Math.floor(modelDefCap.contextWindow * (isSmallCap ? 0.35 : 0.50)) : 11000;
    const currentTokens = estimateTokens(s);
    if (currentTokens > maxSystemTokens) {
      const charLimit = Math.floor(maxSystemTokens * 3.2);
      s = s.slice(0, charLimit) + "\n\n[NOTE: Document content was truncated to fit within model context window. Upload fewer documents or use a larger model for full coverage.]";
    }

    return s;
  }, [mem, pdfDocs, localModelId]);

  // ─── Parse AI response (memory updates and terminal commands) ───
  const parseResponse = useCallback((text) => {
    // Safety: ensure we always work with a string
    if (!text || typeof text !== "string") return { text: String(text || ""), actions: { memoryUpdate: null, terminalCommands: [] } };
    try {
      let cleaned = text;
      const actions = { memoryUpdate: null, terminalCommands: [] };

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
      return { text: String(text), actions: { memoryUpdate: null, terminalCommands: [] } };
    }
  }, []);

  // ─── Call AI (offline-only — local models only, no cloud) ───
  // Options: { maxTokens, onChunk, timeoutMs }
  const callAI = useCallback(async (apiMsgs, opts = {}) => {
    const { maxTokens = 4096, onChunk = null, timeoutMs = 90000 } = opts;
    // Validate engine is loaded and healthy
    if (!localEngineRef.current) {
      throw new Error("No model loaded. Please download and load a local model from the sidebar before sending messages.");
    }
    // Validate messages before sending — corrupted messages crash the engine
    const safeMsgs = apiMsgs.filter(m => m && typeof m.role === "string" && m.content != null).map(m => ({
      role: m.role,
      content: typeof m.content === "string" ? m.content : String(m.content),
    }));

    const tryInterruptGeneration = async () => {
      try {
        // WebLLM versions differ; keep compatibility with optional calls.
        await localEngineRef.current?.interruptGenerate?.();
      } catch {}
      try {
        await localEngineRef.current?.interrupt?.();
      } catch {}
    };

    // Timeout wrapper (attempts to stop in-flight generation)
    const withTimeout = (promise, label = "LLM call") => {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(async () => {
          await tryInterruptGeneration();
          reject(new Error(`${label} timed out — try a shorter query or simpler model`));
        }, timeoutMs);
        promise.then(v => { clearTimeout(timer); resolve(v); }, e => { clearTimeout(timer); reject(e); });
      });
    };

    const doCall = async (engine) => {
      // Use streaming if onChunk callback is provided
      if (onChunk) {
        let stream;
        try {
          stream = await engine.chat.completions.create({
            messages: safeMsgs,
            temperature: 0.7,
            max_tokens: maxTokens,
            stream: true,
          });
        } catch (createErr) {
          // GPU OOM or context overflow often crashes here — catch and provide helpful message
          if (createErr.message && (createErr.message.includes("memory") || createErr.message.includes("OOM") || createErr.message.includes("allocation"))) {
            throw new Error("GPU ran out of memory. Try: (1) Close other browser tabs, (2) Remove some uploaded documents, (3) Use the Light model (Qwen 0.5B). Original: " + createErr.message);
          }
          throw createErr;
        }
        let content = "";
        let usage = { prompt_tokens: 0, completion_tokens: 0 };
        try {
          const iterator = stream[Symbol.asyncIterator]();
          // Idle timeout: only fail when generation stalls, not while healthy tokens are arriving.
          const stallTimeoutMs = Math.max(15000, Math.min(120000, Math.floor(timeoutMs * 0.4)));
          while (true) {
            if (abortRef.current?.signal?.aborted) {
              await tryInterruptGeneration();
              throw new DOMException("Aborted", "AbortError");
            }
            const nextChunk = await Promise.race([
              iterator.next(),
              new Promise((_, reject) =>
                setTimeout(() => reject(new Error("LLM stream stalled — try a shorter query or simpler model")), stallTimeoutMs)
              ),
            ]);
            if (nextChunk.done) break;
            const chunk = nextChunk.value;
            const delta = chunk.choices?.[0]?.delta?.content || "";
            if (delta) {
              content += delta;
              onChunk(content);
            }
            if (chunk.usage) {
              usage = { prompt_tokens: chunk.usage.prompt_tokens || 0, completion_tokens: chunk.usage.completion_tokens || 0 };
            }
          }
        } catch (streamErr) {
          // If we got partial content before the stream died, return what we have
          if (content.length > 30) {
            console.warn("Stream interrupted, returning partial content:", streamErr);
            return { content: content + "\n\n[Response was cut short due to an error — the above content may be incomplete]", usage };
          }
          throw streamErr;
        }
        // WebLLM may report usage in final chunk or via engine
        if (!usage.prompt_tokens && stream.usage) {
          usage = { prompt_tokens: stream.usage.prompt_tokens || 0, completion_tokens: stream.usage.completion_tokens || 0 };
        }
        return { content, usage };
      } else {
        if (abortRef.current?.signal?.aborted) throw new DOMException("Aborted", "AbortError");
        const resp = await withTimeout(engine.chat.completions.create({
          messages: safeMsgs,
          temperature: 0.7,
          max_tokens: maxTokens,
        }), "LLM call");
        const content = resp.choices?.[0]?.message?.content || "";
        return {
          content,
          usage: { prompt_tokens: resp.usage?.prompt_tokens || 0, completion_tokens: resp.usage?.completion_tokens || 0 },
        };
      }
    };

    try {
      const { content, usage } = await withTimeout(doCall(localEngineRef.current), "LLM call");
      return {
        data: {
          choices: [{ message: { content } }],
          usage,
        },
        usedModel: localModelId,
      };
    } catch (e) {
      if (e.name === "AbortError") throw e;
      if (e.message && /timed out|stalled/i.test(e.message)) {
        throw new Error("Local model error: LLM call timed out — try a shorter query, fewer uploaded pages, or a simpler model.");
      }
      // Handle the specific "model not loaded" error — attempt reload
      if (e.message && e.message.includes("not loaded")) {
        try {
          await localEngineRef.current.reload(localModelId);
          const { content, usage } = await withTimeout(doCall(localEngineRef.current), "LLM call");
          return {
            data: {
              choices: [{ message: { content } }],
              usage,
            },
            usedModel: localModelId,
          };
        } catch (reloadErr) {
          throw new Error(`Model reload failed. Please re-download the model from the sidebar. (${reloadErr.message})`);
        }
      }
      // Provide helpful error messages for common issues
      if (e.message && e.message.includes("context window")) {
        throw new Error(`Document too large for current model. Try removing some documents or using a larger model (Llama 3.2 3B or Phi 3.5 Mini have 128K context). Original: ${e.message}`);
      }
      throw new Error(`Local model error: ${e.message}`);
    }
  }, [localModelId]);

  // ─── Main send function with optimised research loop ───
  const send = useCallback(async () => {
    const txt = input.trim();
    if (!txt && attachments.length === 0) return;
    if (busy || busyRef.current) return; // ref-based double-send guard
    // Block sending if PDFs are still being extracted
    if (attachments.some(att => att._loading)) {
      setErr("Please wait — PDF extraction is still in progress.");
      return;
    }
    setErr(null); setBusy(true); busyRef.current = true; setActivityStatus(""); setStreamingText("");

    // Build user message content with attachments
    let userContent = txt;
    if (attachments.length > 0) {
      let attachBlock = "\n\n---\n**Attached files:**\n";
      for (const att of attachments) {
        if (att.isImage) {
          attachBlock += `\n**[Image: ${att.name}]** (${(att.size/1024).toFixed(1)}KB) — *Image attached as base64. Describe if asked.*\n`;
        } else {
          const preview = (att.content || "").slice(0, 8000); // Expanded preview in chat message
          attachBlock += `\n**[File: ${att.name}]** (${att.type || "text"}, ${(att.size/1024).toFixed(1)}KB):\n\`\`\`\n${preview}\n\`\`\`\n`;
        }
      }
      userContent = (txt || "Here are my attached files:") + attachBlock;
    }
    const userMsg = { role: "user", content: userContent, _id: nextMsgId() };
    let currentMsgs = [...msgs, userMsg];
    setMsgs(currentMsgs); setInput(""); setAttachments([]);
    if (inputRef.current) inputRef.current.style.height = "auto";
    // Save user message immediately so it persists even if the AI call fails or page closes
    saveChat(currentMsgs);

    // Helper: extract text content from an AI response object
    const extractRaw = (data) => {
      let raw = typeof data.choices?.[0]?.message?.content === "string"
        ? data.choices[0].message.content
        : Array.isArray(data.choices?.[0]?.message?.content)
          ? data.choices[0].message.content.filter(p => p?.type === "text").map(p => p.text).join("\n")
          : "";
      return raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
    };

    // Determine query complexity for adaptive pipeline
    const hasDocuments = pdfDocs.length > 0;
    const isSimpleQuery = !hasDocuments && txt.length < 60 && !/\b(analyse|analyze|compare|cross.?ref|review|audit|compliance|strategy|deed)\b/i.test(txt);
    let checkpointRaw = ""; // Partial response checkpoint for crash recovery

    try {
      // Offline-only: require a loaded local model
      if (!localEngineRef.current) {
        throw new Error("No AI model loaded yet. Click the 🧠 Workspace button (top right), then Download the Qwen 2.5 0.5B model (recommended for your hardware). Once it says READY, you can chat.");
      }

      abortRef.current = new AbortController();
      // Adaptive message limit: smaller context for small models to prevent GPU OOM
      const modelDefSend = LOCAL_MODELS.find(m => m.id === localModelId);
      const isSmallModelSend = modelDefSend && modelDefSend.contextWindow <= 32768;
      const MAX_MSGS = isSmallModelSend ? 8 : 16;

      // ─── STEP 1: Planning — skip for document queries, simple queries, AND small models ───
      // On small models (Qwen 0.5B), planning wastes a full LLM call that causes noticeable lag
      let researchQuestions = [];
      if (!hasDocuments && !isSimpleQuery && !isSmallModelSend) {
        setActivityStatus("Planning: checking if web research is needed...");
        const planningSystem = `You are a planning agent for an SMSF expert assistant. Given the user's question, decide if web research is needed to answer it accurately.
Respond ONLY with valid JSON in this exact format (no other text):
{"needs_research": true, "questions": ["specific search query 1", "specific search query 2"]}
or
{"needs_research": false, "questions": []}

Rules:
- needs_research should be true if the question requires current regulations, recent news, external facts, or information not contained in uploaded documents
- needs_research should be false for general SMSF knowledge, document analysis, or simple calculations
- If true, provide 2–3 specific, searchable questions (max 3)
- Each question should be a complete search query (e.g. "SMSF contribution caps 2024 Australia ATO")`;

        const planningMsgs = [
          { role: "system", content: planningSystem },
          { role: "user", content: `User question: "${txt || "See attached files"}"` },
        ];

        try {
          const { data: planData } = await callAI(planningMsgs, { maxTokens: 256, timeoutMs: 20000 });
          if (planData.usage) setUsage(p => ({ i: p.i + (planData.usage.prompt_tokens || 0), o: p.o + (planData.usage.completion_tokens || 0) }));
          const planRaw = extractRaw(planData);
          const jsonMatch = planRaw.match(/\{[\s\S]*\}/);
          if (jsonMatch) {
            const plan = JSON.parse(jsonMatch[0]);
            if (plan.needs_research && Array.isArray(plan.questions)) {
              researchQuestions = plan.questions.slice(0, 3);
            }
          }
        } catch (planErr) {
          console.warn("Planning step failed, skipping web research:", planErr);
        }
      }

      // ─── Abort check ───
      if (abortRef.current?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

      // ─── STEP 2: Reviewer agents — run in PARALLEL ───
      const reviewerFindings = [];
      if (researchQuestions.length > 0) {
        setActivityStatus(`Researching ${researchQuestions.length} question(s) in parallel...`);

        const reviewerPromises = researchQuestions.map(async (question, i) => {
          let searchResults = "";
          try {
            searchResults = await searchWeb(question);
          } catch (searchErr) {
            searchResults = `Search unavailable: ${searchErr.message}`;
          }

          const reviewerSystem = `You are a web research reviewer assistant. Summarise the most relevant and accurate information. Be concise but specific — include key facts, dates, figures, and URLs where available.`;
          const reviewerMsgs = [
            { role: "system", content: reviewerSystem },
            { role: "user", content: `Research question: "${question}"\n\nSearch results:\n${searchResults}\n\nSummarise the key findings relevant to SMSF or Australian superannuation.` },
          ];

          try {
            const { data: reviewerData } = await callAI(reviewerMsgs, { maxTokens: 512, timeoutMs: 30000 });
            if (reviewerData.usage) setUsage(p => ({ i: p.i + (reviewerData.usage.prompt_tokens || 0), o: p.o + (reviewerData.usage.completion_tokens || 0) }));
            const findings = extractRaw(reviewerData);
            return { question, findings };
          } catch (reviewErr) {
            return { question, findings: `Reviewer error: ${reviewErr.message}` };
          }
        });

        const results = await Promise.allSettled(reviewerPromises);
        for (const r of results) {
          if (r.status === "fulfilled") reviewerFindings.push(r.value);
        }
      }

      // ─── STEP 3: Main agent synthesises with research context (with STREAMING) ───
      setActivityStatus(reviewerFindings.length > 0 ? "Main agent synthesising research..." : "Thinking...");

      if (currentMsgs.length > MAX_MSGS) currentMsgs = currentMsgs.slice(-MAX_MSGS);

      // Update query ref so buildSystem can select relevant document chunks
      lastUserQueryRef.current = txt || userContent || "";
      let mainSystem = buildSystem();
      if (reviewerFindings.length > 0) {
        mainSystem += `\n\n<web_research>\nThe following web research was conducted by reviewer agents on your behalf. Use it to inform your response and cite it where relevant:\n`;
        for (const { question, findings } of reviewerFindings) {
          mainSystem += `\n**Researched:** ${question}\n**Findings:** ${findings}\n`;
        }
        mainSystem += `</web_research>`;
      }

      const mainApiMsgs = [
        { role: "system", content: mainSystem },
        ...currentMsgs.map(m => ({ role: m.role, content: m.content })),
      ];

      // Stream the main response for real-time display
      // Use conservative max_tokens to prevent GPU OOM on weak hardware
      const modelDef = modelDefSend;
      const isSmallModel = isSmallModelSend;
      const mainMaxTokens = isSmallModel ? Math.min(2048, Math.floor((modelDef?.contextWindow || 32768) * 0.1)) : modelDef ? Math.min(Math.floor(modelDef.contextWindow * 0.12), 8192) : 2048;

      // Throttled streaming — batches UI updates to prevent lag on weak hardware (Acer Aspire 5)
      const streamThrottle = createStreamThrottle(setStreamingText, 180);
      const { data: mainData } = await callAI(mainApiMsgs, {
        maxTokens: mainMaxTokens,
        timeoutMs: 300000,
        onChunk: (partial) => streamThrottle.update(partial),
      });
      streamThrottle.flush(); // Ensure final content is displayed
      if (mainData.usage) setUsage(p => ({ i: p.i + (mainData.usage.prompt_tokens || 0), o: p.o + (mainData.usage.completion_tokens || 0) }));
      let mainRaw = extractRaw(mainData);
      setStreamingText(""); // Clear streaming display

      // ─── Abort check between pipeline steps ───
      if (abortRef.current?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

      // Save checkpoint — if reflection crashes, we still have the main response
      checkpointRaw = mainRaw;

      // ─── STEP 4: Adaptive Self-Reflection Loop (Socratic review — PRESERVED) ───
      // Adaptive: 2 passes for document queries (accuracy matters), 1 pass for simple queries
      // Uses reduced maxTokens to prevent OOM on weak hardware
      let refinedRaw = mainRaw;
      // Preserve the original memory_update in case reflection loses it
      const originalMemoryMatch = mainRaw.match(/<memory_update>([\s\S]*?)<\/memory_update>/i);
      const originalMemoryBlock = originalMemoryMatch ? originalMemoryMatch[0] : null;

      // Adaptive reflection: small models get 1 pass always (saves GPU time + memory)
      // Larger models: 2 passes for document queries (accuracy matters), 1 for simple
      const REFLECTION_PASSES = isSmallModelSend ? 1 : (hasDocuments ? 2 : 1);
      const reflectionChecks = [
        { name: "Accuracy & Document Citations", focus: "Check all factual claims, legislative references (SIS Act sections, regulations), dollar amounts, percentages, and dates. Verify EVERY claim about a document references it by name and page number using **[Document Name, Page X]** format. Add missing citations. Ensure no page reference is fabricated. Flag anything incorrect or unsupported." },
        { name: "Completeness, Cross-References & Polish", focus: "Check if any aspect of the user's question was missed. Check cross-references BETWEEN documents — are discrepancies identified? Is the trust deed compared with the investment strategy? Are member statements reconciled? Ensure the response is well-structured, readable, and professional. Ensure <memory_update> tags are present and intact. Ensure a References section lists all cited pages." },
      ];
      // Reflection uses reduced maxTokens — response should be similar length to input
      const reflectionMaxTokens = isSmallModel ? Math.min(mainMaxTokens, 1536) : Math.min(mainMaxTokens, 4096);

      for (let pass = 0; pass < REFLECTION_PASSES; pass++) {
        // ─── Abort check between steps ───
        if (abortRef.current?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

        const check = reflectionChecks[pass];
        setActivityStatus(`Self-review pass ${pass + 1}/${REFLECTION_PASSES}: ${check.name}...`);

        // Reflection prompts do NOT include full document text — only the draft response
        const reflectionSystem = `You are a self-reflection review agent (Pass ${pass + 1}/${REFLECTION_PASSES}: ${check.name}).

Your task: Review the draft response below and IMPROVE it based on this specific focus area:
**${check.focus}**

Context:
- The user asked: "${txt || "See attached files"}"
- The response should be an expert SMSF cross-referencing analysis with perfect page citations
${reviewerFindings.length > 0 ? `- Web research was conducted: ${reviewerFindings.map(r => r.question).join("; ")}` : "- No web research was conducted"}
${hasDocuments ? `- Documents uploaded: ${pdfDocs.map(d => d.name + " (" + d.pageCount + " pages)").join(", ")}` : "- No documents uploaded"}

Rules:
1. Output the COMPLETE improved response (not just corrections)
2. PRESERVE ALL tags exactly: <memory_update> blocks — this is CRITICAL, do not lose them
3. If the response is already excellent for this check, output it unchanged
4. Make ONLY improvements related to your focus area — do not degrade other aspects
5. Every document reference MUST include page numbers in **[Document Name, Page X]** format
6. Think carefully about whether each part of the response is actually correct and well-supported`;

        const reflectionMsgs = [
          { role: "system", content: reflectionSystem },
          { role: "user", content: `Draft response to review and improve:\n\n${refinedRaw}` },
        ];

        try {
          // Show streaming during reflection — wider throttle to reduce lag on weak hardware
          const reflectThrottle = createStreamThrottle(setStreamingText, 250);
          const { data: reflectData } = await callAI(reflectionMsgs, {
            maxTokens: reflectionMaxTokens,
            timeoutMs: 120000,
            onChunk: (partial) => reflectThrottle.update(partial),
          });
          reflectThrottle.flush();
          setStreamingText("");
          if (reflectData.usage) setUsage(p => ({ i: p.i + (reflectData.usage.prompt_tokens || 0), o: p.o + (reflectData.usage.completion_tokens || 0) }));
          const reflectRaw = extractRaw(reflectData);
          // Sanity check: keep memory_update tag integrity — never lose memory
          if (reflectRaw.length > 50) {
            if (reflectRaw.includes("<memory_update>") || !refinedRaw.includes("<memory_update>")) {
              refinedRaw = reflectRaw;
            } else if (originalMemoryBlock) {
              // Reflection lost the memory_update — re-append it
              refinedRaw = reflectRaw + "\n\n" + originalMemoryBlock;
            }
          }
          // Update checkpoint after each successful reflection
          checkpointRaw = refinedRaw;
        } catch (reflectErr) {
          console.warn(`Reflection pass ${pass + 1} failed:`, reflectErr);
          setStreamingText("");
          // Continue with current refined version — don't crash
        }
      }

      // ─── STEP 5: Verification — only for complex document queries on larger models ───
      // Skip on small models (Qwen 0.5B) — the extra LLM call is too slow and OOM-prone
      let finalRaw = refinedRaw;
      if (hasDocuments && !isSimpleQuery && !isSmallModelSend) {
        // ─── Abort check ───
        if (abortRef.current?.signal?.aborted) throw new DOMException("Aborted", "AbortError");

        setActivityStatus("Final verification: checking quality of work...");
        const verificationSystem = `You are a final quality gate for an SMSF expert assistant. Answer ONE question: "Did I do good work?"

Review this SMSF expert response and check:
1. Does it FULLY answer the user's question with no gaps?
2. Are ALL document references accurate with specific page numbers in **[Document Name, Page X]** format?
3. Are there any compliance issues, misleading statements, or incorrect legislative references?
4. Is the cross-referencing between documents thorough and systematic?
5. Are all <memory_update> tags present and intact?

If YES (quality is high): Output the response EXACTLY as-is — do not change a single character.
If NO (there are problems): Fix the specific issues and output the corrected version.
CRITICAL: Preserve ALL tags (<memory_update>) exactly.`;

        const verifyMsgs = [
          { role: "system", content: verificationSystem },
          { role: "user", content: `User asked: "${txt || "See attached files"}"\n\nFinal response to verify:\n${refinedRaw}` },
        ];

        try {
          const { data: verifyData } = await callAI(verifyMsgs, { maxTokens: reflectionMaxTokens, timeoutMs: 120000 });
          if (verifyData.usage) setUsage(p => ({ i: p.i + (verifyData.usage.prompt_tokens || 0), o: p.o + (verifyData.usage.completion_tokens || 0) }));
          const verifyRaw = extractRaw(verifyData);
          if (verifyRaw.length > 50) {
            if (verifyRaw.includes("<memory_update>") || !refinedRaw.includes("<memory_update>")) {
              finalRaw = verifyRaw;
            } else if (originalMemoryBlock) {
              finalRaw = verifyRaw + "\n\n" + originalMemoryBlock;
            }
          }
        } catch {
          finalRaw = refinedRaw; // Fall back to refined response on verification error
        }
      }

      // ─── Finalise: parse response and update state ───
      const { text, actions } = parseResponse(finalRaw);

      // Fallback display text: if parseResponse stripped everything, recover from raw
      const displayText = text || finalRaw.replace(/<memory_update>[\s\S]*?<\/memory_update>/gi, "").replace(/<[a-z_]+>[\s\S]*?<\/[a-z_]+>/g, "").trim() || checkpointRaw || "Analysis complete.";

      if (actions.memoryUpdate) {
        setMem(actions.memoryUpdate);
        setMemDraft(actions.memoryUpdate);
        saveVal(MEMORY_STORAGE_KEY, actions.memoryUpdate);
        currentMsgs = [...currentMsgs, { role: "assistant", content: displayText + `\n\n---\n*Memory updated and saved to memory.txt*`, _id: nextMsgId() }];
      } else {
        currentMsgs = [...currentMsgs, { role: "assistant", content: displayText, _id: nextMsgId() }];
        const autoMemory = mem.trim()
          ? mem + `\n\n[Auto-saved ${new Date().toLocaleString()}]: User said: "${(txt || userContent || "").slice(0, 200)}". Auto responded about: ${displayText.slice(0, 200)}`
          : `[Chat ${new Date().toLocaleString()}]: User said: "${(txt || userContent || "").slice(0, 200)}". Auto responded about: ${displayText.slice(0, 200)}`;
        setMem(autoMemory);
        setMemDraft(autoMemory);
        saveVal(MEMORY_STORAGE_KEY, autoMemory);
      }

      setMsgs([...currentMsgs]);
      saveChat(currentMsgs);

    } catch (e) {
      if (e.name !== "AbortError") {
        setErr(e.message);
        // Partial response recovery: if we have a checkpoint, show it
        if (typeof checkpointRaw === "string" && checkpointRaw.length > 50) {
          try {
            const { text: partialText, actions: partialActions } = parseResponse(checkpointRaw);
            if (partialText) {
              currentMsgs = [...currentMsgs, { role: "assistant", content: partialText + `\n\n---\n*⚠ Partial response — review pipeline was interrupted: ${e.message}*`, _id: nextMsgId() }];
              setMsgs([...currentMsgs]);
              saveChat(currentMsgs);
              if (partialActions.memoryUpdate) {
                setMem(partialActions.memoryUpdate);
                setMemDraft(partialActions.memoryUpdate);
                saveVal(MEMORY_STORAGE_KEY, partialActions.memoryUpdate);
              }
            }
          } catch {}
        }
      }
      try { if (currentMsgs && currentMsgs.length > 0) saveChat(currentMsgs); } catch {}
    } finally {
      setBusy(false);
      busyRef.current = false;
      setActivityStatus("");
      setStreamingText("");
      abortRef.current = null;
    }
  }, [input, msgs, busy, buildSystem, parseResponse, callAI, attachments, pdfDocs]);

  const clearChat = () => { setMsgs([]); saveChat([]); setErr(null); };
  const ft = n => n >= 1e6 ? (n/1e6).toFixed(1)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"K" : String(n);

  // ─── Export chat analysis as PDF ───
  const exportAnalysisPdf = useCallback(() => {
    const assistantMsgs = msgs.filter(m => m.role === "assistant");
    if (assistantMsgs.length === 0) { setErr("No analysis to export yet."); return; }

    // Build a printable HTML document
    const docNames = pdfDocs.map(d => d.name).join(", ") || "None";
    const timestamp = new Date().toLocaleString("en-AU", { dateStyle: "full", timeStyle: "short" });
    const modelName = LOCAL_MODELS.find(m => m.id === localModelId)?.name || localModelId;

    // Convert markdown-ish text to basic HTML for print
    const toHtml = (text) => {
      if (!text) return "";
      return text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/\*\*\[([^\]]+)\]\*\*/g, '<strong style="color:#1a5276">[$1]</strong>')
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\*([^*]+)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, '<code style="background:#f0f0f0;padding:1px 4px;border-radius:3px;font-size:0.9em">$1</code>')
        .replace(/^### (.+)$/gm, '<h4 style="color:#2c3e50;margin:16px 0 6px;font-size:14px">$1</h4>')
        .replace(/^## (.+)$/gm, '<h3 style="color:#2c3e50;margin:18px 0 8px;font-size:16px">$1</h3>')
        .replace(/^# (.+)$/gm, '<h2 style="color:#1a5276;margin:20px 0 10px;font-size:18px">$1</h2>')
        .replace(/^---+$/gm, '<hr style="border:none;border-top:1px solid #ccc;margin:12px 0">')
        .replace(/^[\-\*] (.+)$/gm, '<div style="display:flex;gap:6px;margin:2px 0"><span style="color:#2980b9">•</span><span>$1</span></div>')
        .replace(/^\d+\. (.+)$/gm, (_, t, i) => `<div style="display:flex;gap:6px;margin:2px 0"><span style="color:#2980b9;min-width:18px">${i + 1}.</span><span>${t}</span></div>`)
        .replace(/\n\n/g, "</p><p>")
        .replace(/\n/g, "<br>");
    };

    let chatHtml = "";
    for (const m of msgs) {
      if (m.role === "user") {
        const userText = (m.content || "").replace(/\n\n---\n\*\*Attached files:\*\*[\s\S]*$/, "").trim();
        if (userText) {
          chatHtml += `<div style="background:#e8f8f5;border:1px solid #a3e4d7;border-radius:8px;padding:10px 14px;margin:8px 0;font-size:13px"><strong style="color:#117864">You:</strong> ${toHtml(userText)}</div>`;
        }
      } else {
        chatHtml += `<div style="background:#fafafa;border:1px solid #e0e0e0;border-radius:8px;padding:12px 16px;margin:8px 0;font-size:13px"><strong style="color:#1a5276">Auto (SMSF Analysis):</strong><div style="margin-top:6px;line-height:1.7"><p>${toHtml(m.content)}</p></div></div>`;
      }
    }

    const html = `<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>SMSF Cross-Reference Analysis — Auto</title>
<style>
  @page { size: A4; margin: 20mm 15mm; }
  @media print { body { -webkit-print-color-adjust: exact; print-color-adjust: exact; } }
  body { font-family: 'Segoe UI', system-ui, sans-serif; color: #222; max-width: 800px; margin: 0 auto; padding: 20px; font-size: 13px; line-height: 1.6; }
  h1 { color: #1a5276; font-size: 22px; margin-bottom: 4px; }
  .meta { color: #666; font-size: 11px; margin-bottom: 16px; border-bottom: 2px solid #2980b9; padding-bottom: 10px; }
  .disclaimer { background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 8px 12px; font-size: 11px; color: #856404; margin: 16px 0; }
  .footer { text-align: center; color: #999; font-size: 10px; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }
</style></head><body>
<h1>SMSF Cross-Reference Analysis Report</h1>
<div class="meta">
  <strong>Generated:</strong> ${timestamp}<br>
  <strong>Documents analysed:</strong> ${docNames}<br>
  <strong>Model:</strong> ${modelName} (Offline local analysis)<br>
  <strong>Messages:</strong> ${msgs.length}
</div>
<div class="disclaimer">
  <strong>Disclaimer:</strong> This report was generated by an AI assistant running locally on your device. It is intended as a working aid only and does not constitute financial, legal, or tax advice. All findings should be verified by a qualified SMSF auditor or professional adviser. AI-generated page citations should be cross-checked against the original documents.
</div>
${chatHtml}
<div class="footer">
  Generated by Auto — Australian SMSF Document Cross-Reference Agent<br>
  Analysis performed offline using ${modelName}
</div>
</body></html>`;

    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const printWin = window.open(url, "_blank");
    if (printWin) {
      printWin.onload = () => {
        setTimeout(() => { printWin.print(); }, 500);
      };
    } else {
      // Fallback: download as HTML
      const a = document.createElement("a");
      a.href = url;
      a.download = `SMSF-Analysis-${new Date().toISOString().slice(0,10)}.html`;
      a.click();
    }
    URL.revokeObjectURL(url);

    // Register a separate artifact entry for the Artifacts panel (keep its own URL)
    const artifactBlob = new Blob([html], { type: "text/html" });
    const artifactUrl = URL.createObjectURL(artifactBlob);
    const artifactName = `SMSF-Analysis-${new Date().toISOString().slice(0,10)}.html`;
    setExportedArtifacts(prev => [...prev, { id: "export-" + Date.now(), name: artifactName, type: "text/html", blobUrl: artifactUrl, size: artifactBlob.size, timestamp: new Date() }]);
  }, [msgs, pdfDocs, localModelId]);

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
          blobUrl={pdfDocs[pdfViewerIdx].blobUrl}
          onClose={() => setPdfViewerOpen(false)}
        />
      )}
      {/* Full Document Text Viewer Modal — shows complete extracted text for cross-referencing */}
      {docTextViewerOpen && pdfDocs[docTextViewerIdx] && (
        <div style={{ position: "fixed", inset: 0, zIndex: 2000, background: "rgba(0,0,0,0.85)", display: "flex", flexDirection: "column", animation: "fadeIn .2s ease" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 16px", background: "#0d0d14", borderBottom: "1px solid var(--bd)", flexShrink: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
              <span style={{ fontSize: "14px" }}>{"\uD83D\uDCDA"}</span>
              <span style={{ fontWeight: 700, fontSize: "14px", color: "var(--ac2)" }}>{pdfDocs[docTextViewerIdx].name}</span>
              <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>
                {pdfDocs[docTextViewerIdx].pageCount} pages · {(pdfDocs[docTextViewerIdx].text.length / 1024).toFixed(0)}KB text · ~{estimateTokens(pdfDocs[docTextViewerIdx].text).toLocaleString()} tokens
              </span>
            </div>
            <div style={{ display: "flex", gap: "6px", alignItems: "center" }}>
              <button onClick={() => { navigator.clipboard.writeText(pdfDocs[docTextViewerIdx].text); }} style={{ ...btn("#88bbcc") }}>Copy All</button>
              <button onClick={() => {
                const blob = new Blob([pdfDocs[docTextViewerIdx].text], { type: "text/plain" });
                const a = document.createElement("a");
                a.href = URL.createObjectURL(blob);
                a.download = pdfDocs[docTextViewerIdx].name.replace(/\.pdf$/i, "") + "-extracted.txt";
                a.click();
              }} style={{ ...btn("#7ce08a") }}>Download .txt</button>
              <button onClick={() => setDocTextViewerOpen(false)} style={{ background: "none", border: "none", color: "var(--dm)", cursor: "pointer", fontSize: "20px", padding: "0 4px" }}>×</button>
            </div>
          </div>
          <div style={{ flex: 1, overflow: "auto", padding: "16px 20px" }}>
            <pre style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", fontFamily: "var(--m)", fontSize: "12px", color: "var(--tx)", lineHeight: 1.7, margin: 0 }}>
              {pdfDocs[docTextViewerIdx].text}
            </pre>
          </div>
        </div>
      )}
      {/* ═══ ARTIFACTS PANEL ═══ */}
      {artifactsOpen && (
        <div style={{ position: "fixed", inset: 0, zIndex: 3000, background: "rgba(0,0,0,0.7)", display: "flex", justifyContent: "flex-end", animation: "fadeIn .15s ease" }} onClick={e => { if (e.target === e.currentTarget) setArtifactsOpen(false); }}>
          <div style={{ width: "min(420px,95vw)", height: "100%", background: "#0d0d14", borderLeft: "1px solid var(--bd)", display: "flex", flexDirection: "column", animation: "slideL .2s ease" }}>
            {/* Header */}
            <div style={{ padding: "12px 16px", borderBottom: "1px solid var(--bd)", display: "flex", alignItems: "center", justifyContent: "space-between", flexShrink: 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontSize: "14px" }}>📎</span>
                <span style={{ fontWeight: 700, fontSize: "14px", color: "var(--ac2)" }}>Artifacts</span>
                <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", padding: "1px 6px", borderRadius: "3px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--bd)" }}>
                  {pdfDocs.length + exportedArtifacts.length} item{pdfDocs.length + exportedArtifacts.length !== 1 ? "s" : ""}
                </span>
              </div>
              <button onClick={() => setArtifactsOpen(false)} style={{ background: "none", border: "none", color: "var(--dm)", cursor: "pointer", fontSize: "20px", padding: "0 4px", lineHeight: 1 }}>×</button>
            </div>

            {/* Content */}
            <div style={{ flex: 1, overflowY: "auto", padding: "12px 14px", display: "flex", flexDirection: "column", gap: "8px" }}>
              {pdfDocs.length === 0 && exportedArtifacts.length === 0 && pdfLoading.length === 0 && (
                <div style={{ textAlign: "center", padding: "40px 20px", color: "var(--dm)", fontSize: "12px", lineHeight: 1.8 }}>
                  <div style={{ fontSize: "28px", marginBottom: "10px", opacity: 0.4 }}>📁</div>
                  <div>No artifacts yet.</div>
                  <div style={{ fontSize: "11px", marginTop: "6px" }}>Upload SMSF documents using the <strong style={{ color: "var(--tx)" }}>+</strong> button, or export an analysis to see files here.</div>
                </div>
              )}

              {/* PDFs being extracted */}
              {pdfLoading.length > 0 && (
                <div>
                  <div style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Extracting…</div>
                  {pdfLoading.map((pl, i) => (
                    <div key={"al-" + i} style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 10px", borderRadius: "6px", background: "rgba(204,153,85,0.06)", border: "1px solid rgba(204,153,85,0.18)", marginBottom: "4px" }}>
                      <span style={{ fontSize: "18px", animation: "pulse 1.5s infinite" }}>📄</span>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: "11px", color: "#cc9955", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "var(--m)" }}>{pl.name}</div>
                        <div style={{ fontSize: "9px", color: "var(--dm)", marginTop: "2px" }}>{pl.total > 0 ? `${pl.progress}/${pl.total} pages` : "reading…"}</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Uploaded PDFs */}
              {pdfDocs.length > 0 && (
                <div>
                  <div style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Uploaded Documents</div>
                  {pdfDocs.map((doc, i) => {
                    const isScanned = doc.pageCount > 0 && !doc.text.replace(/=== \[Page \d+\] ===/g, "").replace(/\(Scanned[^)]*\)/g, "").replace(/\(Could not[^)]*\)/g, "").trim().length;
                    return (
                      <div key={i} style={{ padding: "10px 12px", borderRadius: "7px", background: "rgba(136,187,204,0.05)", border: "1px solid rgba(136,187,204,0.12)", marginBottom: "6px" }}>
                        <div style={{ display: "flex", alignItems: "flex-start", gap: "8px" }}>
                          <span style={{ fontSize: "18px", flexShrink: 0, marginTop: "1px" }}>📄</span>
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{ fontSize: "12px", color: "var(--ac2)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={doc.name}>{doc.name}</div>
                            <div style={{ fontSize: "10px", color: "var(--dm)", marginTop: "2px", display: "flex", gap: "8px", flexWrap: "wrap" }}>
                              <span>{doc.pageCount} pages</span>
                              <span>~{(doc.text.length / 1024).toFixed(0)} KB text</span>
                              {isScanned && <span style={{ color: "#cc9955" }}>⚠ scanned</span>}
                            </div>
                          </div>
                        </div>
                        <div style={{ display: "flex", gap: "4px", marginTop: "8px", flexWrap: "wrap" }}>
                          <button onClick={() => { setPdfViewerIdx(i); setPdfViewerOpen(true); }} style={{ ...btn("#88bbcc"), fontSize: "9px" }}>View PDF</button>
                          <button onClick={() => { setDocTextViewerIdx(i); setDocTextViewerOpen(true); }} style={{ ...btn("#88bbcc"), fontSize: "9px" }}>View Text</button>
                          {doc.blobUrl && (
                            <a href={doc.blobUrl} download={doc.name} style={{ ...btn("#7ce08a"), fontSize: "9px", textDecoration: "none", display: "inline-block" }}>Download</a>
                          )}
                          <button onClick={() => {
                            setPdfDocs(prev => prev.filter((_, j) => j !== i));
                            setAttachments(prev => prev.filter(a => a.name !== doc.name));
                          }} style={{ ...btn("#cc7777"), fontSize: "9px" }}>Remove</button>
                        </div>
                        {isScanned && (
                          <div style={{ marginTop: "6px", fontSize: "9px", color: "#cc9955", lineHeight: 1.5, padding: "4px 6px", background: "rgba(204,153,85,0.06)", borderRadius: "4px", border: "1px solid rgba(204,153,85,0.15)" }}>
                            Scanned/image-based PDF — text not extractable. AI will acknowledge this and provide general SMSF guidance.
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}

              {/* Exported Reports */}
              {exportedArtifacts.length > 0 && (
                <div>
                  <div style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", marginBottom: "6px", textTransform: "uppercase", letterSpacing: "0.5px" }}>Exported Reports</div>
                  {exportedArtifacts.map((a) => (
                    <div key={a.id} style={{ padding: "10px 12px", borderRadius: "7px", background: "rgba(124,224,138,0.04)", border: "1px solid rgba(124,224,138,0.12)", marginBottom: "6px" }}>
                      <div style={{ display: "flex", alignItems: "flex-start", gap: "8px" }}>
                        <span style={{ fontSize: "18px", flexShrink: 0, marginTop: "1px" }}>📊</span>
                        <div style={{ flex: 1, minWidth: 0 }}>
                          <div style={{ fontSize: "12px", color: "var(--ac)", fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={a.name}>{a.name}</div>
                          <div style={{ fontSize: "10px", color: "var(--dm)", marginTop: "2px" }}>
                            {(a.size / 1024).toFixed(1)} KB · {a.timestamp.toLocaleString()}
                          </div>
                        </div>
                      </div>
                      <div style={{ display: "flex", gap: "4px", marginTop: "8px" }}>
                        <a href={a.blobUrl} target="_blank" rel="noreferrer" style={{ ...btn("#88bbcc"), fontSize: "9px", textDecoration: "none", display: "inline-block" }}>Open</a>
                        <a href={a.blobUrl} download={a.name} style={{ ...btn("#7ce08a"), fontSize: "9px", textDecoration: "none", display: "inline-block" }}>Download</a>
                        <button onClick={() => {
                          try { URL.revokeObjectURL(a.blobUrl); } catch {}
                          setExportedArtifacts(prev => prev.filter(x => x.id !== a.id));
                        }} style={{ ...btn("#cc7777"), fontSize: "9px" }}>Remove</button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Footer */}
            <div style={{ padding: "8px 14px", borderTop: "1px solid var(--bd)", fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", flexShrink: 0 }}>
              Artifacts are local to this session — reload clears uploaded files.
            </div>
          </div>
        </div>
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
                <button onClick={() => { setMemDraft(""); setMem(""); saveVal(MEMORY_STORAGE_KEY, ""); }} style={btn("#cc7777")}>Clear</button>
              </div>
              <div style={{ padding: "6px 12px 8px", fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>
                {mem.length} chars · ~{Math.ceil(mem.length / 3.8)} tokens · Saved to memory.txt
              </div>
            </div>
          )}

          {/* ─── Uploaded SMSF Documents ─── */}
          {(pdfDocs.length > 0 || pdfLoading.length > 0) && (
            <div style={{ borderTop: "1px solid var(--bd)", flexShrink: 0, padding: "8px 10px" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                <span style={{ fontSize: "11px" }}>{"\uD83D\uDCDA"}</span>
                <span style={{ fontWeight: 700, fontSize: "11px" }}>SMSF Documents</span>
                <span style={{ fontSize: "9px", color: "var(--ac2)", fontFamily: "var(--m)" }}>{pdfDocs.length} loaded{pdfLoading.length > 0 ? `, ${pdfLoading.length} extracting` : ""}</span>
              </div>
              {/* Show PDFs currently being extracted — so user sees them immediately */}
              {pdfLoading.map((pl, i) => (
                <div key={"loading-" + i} style={{
                  display: "flex", alignItems: "center", gap: "6px", padding: "4px 6px",
                  borderRadius: "5px", background: "rgba(204,153,85,0.06)", border: "1px solid rgba(204,153,85,0.15)",
                  marginBottom: "4px",
                }}>
                  <span style={{ fontSize: "10px", animation: "pulse 1.5s infinite" }}>{"📄"}</span>
                  <span style={{ fontSize: "10px", color: "#cc9955", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "var(--m)" }}>{pl.name}</span>
                  <span style={{ fontSize: "8px", color: "#cc9955", fontFamily: "var(--m)", animation: "pulse 1.5s infinite" }}>
                    {pl.total > 0 ? `${pl.progress}/${pl.total} pages` : "reading..."}
                  </span>
                </div>
              ))}
              {pdfDocs.map((doc, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: "6px", padding: "4px 6px",
                  borderRadius: "5px", background: "rgba(136,187,204,0.05)", border: "1px solid rgba(136,187,204,0.1)",
                  marginBottom: "4px",
                }}>
                  <span style={{ fontSize: "10px" }}>{"\uD83D\uDCC4"}</span>
                  <span style={{ fontSize: "10px", color: "var(--ac2)", flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", fontFamily: "var(--m)", cursor: "pointer" }} onClick={() => { setPdfViewerIdx(i); setPdfViewerOpen(true); }}>{doc.name}</span>
                  <span style={{ fontSize: "8px", color: "var(--dm)", fontFamily: "var(--m)" }}>{doc.pageCount}pg</span>
                  <button onClick={() => { setDocTextViewerIdx(i); setDocTextViewerOpen(true); }} style={{ background: "none", border: "1px solid rgba(136,187,204,0.2)", color: "var(--ac2)", cursor: "pointer", fontSize: "8px", padding: "1px 4px", borderRadius: "3px", fontFamily: "var(--m)" }} title="View full extracted text">Text</button>
                  <button onClick={() => { setPdfViewerIdx(i); setPdfViewerOpen(true); }} style={{ background: "none", border: "1px solid rgba(136,187,204,0.2)", color: "var(--ac2)", cursor: "pointer", fontSize: "8px", padding: "1px 4px", borderRadius: "3px", fontFamily: "var(--m)" }} title="View PDF pages">PDF</button>
                </div>
              ))}
              <div style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", marginTop: "4px", lineHeight: 1.4 }}>
                Click name/PDF to view pages. Text to see full extracted content.<br/>
                AI cross-references with page citations using expanded {(LOCAL_MODELS.find(m => m.id === localModelId)?.contextWindow || 32768).toLocaleString()} token context.
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
                      // Unload current engine — release GPU resources, not just clear ref
                      try { await localEngineRef.current?.unload?.(); } catch {}
                      localEngineRef.current = null;
                      autoLoadAttemptedRef.current = false;
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
                      {[m.vram, m.ram, m.cpu, `${(m.contextWindow/1024).toFixed(0)}K ctx`].map(spec => (
                        <span key={spec} style={{ fontSize: "8.5px", color: selected ? m.color : "var(--dm)", fontFamily: "var(--m)", padding: "1px 5px", borderRadius: "3px", background: selected ? m.color + "15" : "rgba(255,255,255,0.03)", border: `1px solid ${selected ? m.color + "30" : "rgba(255,255,255,0.05)"}` }}>
                          {spec}
                        </span>
                      ))}
                    </div>
                    {/* Desc */}
                    <div style={{ fontSize: "9px", color: "var(--dm)", marginTop: "4px", lineHeight: 1.4 }}>{m.desc} Context: {m.contextWindow.toLocaleString()} tokens.</div>
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
            <span style={{ width: "12px", height: "12px", borderRadius: "999px", background: "var(--ac)", display: "inline-block" }} />
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
            <button
              onClick={() => setArtifactsOpen(!artifactsOpen)}
              style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", display: "flex", alignItems: "center", gap: "4px", color: artifactsOpen ? "var(--ac2)" : (pdfDocs.length + exportedArtifacts.length > 0 ? "var(--tx)" : "var(--dm)"), borderColor: artifactsOpen ? "rgba(136,187,204,0.2)" : (pdfDocs.length + exportedArtifacts.length > 0 ? "rgba(255,255,255,0.1)" : undefined), background: artifactsOpen ? "rgba(136,187,204,0.08)" : undefined }}
              title="View and manage uploaded documents and exported reports"
            >
              <span style={{ fontSize: "12px" }}>📎</span>
              <span>Artifacts</span>
              {(pdfDocs.length + exportedArtifacts.length) > 0 && (
                <span style={{ fontSize: "9px", padding: "0px 4px", borderRadius: "3px", background: "rgba(136,187,204,0.15)", color: "var(--ac2)", fontWeight: 700 }}>{pdfDocs.length + exportedArtifacts.length}</span>
              )}
            </button>
            <button
              onClick={exportAnalysisPdf}
              disabled={msgs.filter(m => m.role === "assistant").length === 0}
              style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", color: msgs.filter(m => m.role === "assistant").length > 0 ? "#88bbcc" : "var(--dm)", borderColor: msgs.filter(m => m.role === "assistant").length > 0 ? "rgba(136,187,204,0.2)" : undefined, opacity: msgs.filter(m => m.role === "assistant").length > 0 ? 1 : 0.5 }}
              title="Export analysis as PDF report"
            >Export PDF</button>
            <button onClick={clearChat} style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)" }}>Clear</button>
          </div>
        </header>

        {/* CHAT AREA */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, minHeight: 0, overflow: "hidden" }}>
          <div style={{ flex: 1, overflowY: "auto", minHeight: 0, padding: "14px 20px" }}>
            {msgs.length === 0 && !busy && (
              <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", height: "100%", opacity: 0.45, gap: "10px", padding: "20px" }}>
                <span style={{ width: "24px", height: "24px", borderRadius: "999px", background: "var(--ac)", display: "inline-block" }} />
                <div style={{ fontWeight: 700, fontSize: "16px" }}>Auto</div>
                <div style={{ fontSize: "12px", color: "var(--dm)", textAlign: "center", maxWidth: "500px", lineHeight: 1.6 }}>
                  SMSF document analysis assistant — runs fully offline.<br/>
                  Upload PDF trust deeds, investment strategies, member statements, or any SMSF documents.<br/>
                  Auto will cross-reference them with page-level citations.
                </div>
                {localModelStatus !== "ready" && localModelStatus !== "downloading" && localModelStatus !== "loading" && (
                  <div style={{ marginTop: "8px", padding: "10px 16px", borderRadius: "8px", background: "rgba(204,153,85,0.08)", border: "1px solid rgba(204,153,85,0.2)", fontSize: "11px", color: "#cc9955", textAlign: "center", maxWidth: "420px", lineHeight: 1.6 }}>
                    <strong>Step 1:</strong> Open the <strong>Workspace</strong> panel (top right) and download a local AI model.<br/>
                    <strong>Recommended for your hardware:</strong> Qwen 2.5 0.5B (Light) — fastest, works on low-end hardware.<br/>
                    <strong>Step 2:</strong> Upload your SMSF documents using the <strong>+</strong> button below.<br/>
                    <strong>Step 3:</strong> Ask questions — Auto will cross-reference with page citations.
                  </div>
                )}
              </div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {/* Message windowing: only render recent messages in DOM to reduce layout work on weak hardware */}
              {/* Older messages are preserved in state (for saving) but not rendered */}
              {(() => {
                const WINDOW_SIZE = 40;
                const windowed = msgs.length > WINDOW_SIZE ? msgs.slice(-WINDOW_SIZE) : msgs;
                const skipped = msgs.length - windowed.length;
                return <>
                  {skipped > 0 && (
                    <div style={{ textAlign: "center", padding: "6px", fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)", cursor: "pointer", borderRadius: "6px", border: "1px solid var(--bd)", background: "rgba(255,255,255,0.02)" }}
                      onClick={() => {}}>
                      {skipped} older message{skipped > 1 ? "s" : ""} hidden to save memory
                    </div>
                  )}
                  {windowed.map((m) => <ChatMessage key={m._id || m.content?.slice(0, 20)} msg={m} />)}
                </>;
              })()}
              {/* Streaming response display — shows text as it generates */}
              {streamingText && busy && (
                <div style={{ alignSelf: "flex-start", maxWidth: "min(960px,96%)", display: "flex", gap: "8px", alignItems: "flex-start" }}>
                  <span style={{ width: "10px", height: "10px", borderRadius: "999px", background: "var(--ac)", flexShrink: 0, marginTop: "8px" }} />
                  <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid var(--bd)", borderRadius: "10px", padding: "10px 12px", minWidth: 0, opacity: 0.85 }}>
                    <MemoMd text={streamingText} />
                  </div>
                </div>
              )}
              {busy && (
                <div style={{ opacity: .6, fontSize: "12px", padding: "6px 2px", display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ width: "10px", height: "10px", borderRadius: "999px", background: "var(--ac)", animation: "bounce 1s infinite" }} />
                  <span style={{ animation: "bounce 1s infinite" }}>Thinking…</span>
                  {activityStatus && <span style={{ color: "var(--ac2)", fontFamily: "var(--m)", fontSize: "10px" }}>{activityStatus}</span>}
                </div>
              )}
              {err && <div style={{ color: "#f88", fontSize: "12px", padding: "6px 2px" }}>{err}</div>}

              <div ref={scrollRef} />
            </div>
          </div>

          {/* INPUT */}}
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
                    {att._loading && <span style={{ fontSize: "8px", color: "#cc9955", flexShrink: 0, animation: "pulse 1.5s infinite" }}>extracting...</span>}
                    {att.isPdf && !att._loading && <span style={{ fontSize: "8px", color: "var(--ac2)", flexShrink: 0 }}>{att.pageCount}pg</span>}
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
        React.createElement("div", { style: { fontSize: "40px" } }, "⚠️"),
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
            } catch(e) {}
            try {
              window.localStorage.setItem(CHAT_STORAGE_KEY, "[]");
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
