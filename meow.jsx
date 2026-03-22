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

const GROQ_API = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = "qwen/qwen3-32b";

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
async function loadGroqKey() {
  try {
    if (window.storage?.get) {
      const r = await window.storage.get("groq-api-key");
      if (r?.value) return String(r.value).trim();
    }
  } catch {}
  try { return (window.localStorage.getItem("groq-api-key") || "").trim(); } catch { return ""; }
}
async function saveGroqKey(val) {
  const n = (val || "").trim();
  try { if (window.storage?.set) await window.storage.set("groq-api-key", n); } catch {}
  try { window.localStorage.setItem("groq-api-key", n); } catch {}
}
function readEnvGroqKey() {
  return (window.GROQ_API_KEY || window.__GROQ_API_KEY__ || window?.env?.GROQ_API_KEY || "").trim();
}

// ─── Race multiple CORS proxies for a URL — returns first successful text ───
// ─── Error parsing ───
function parseErrorMessage(rawBody, status) {
  let parsed;
  try { parsed = JSON.parse(rawBody); } catch {}
  const fromParsed = parsed?.error?.message || parsed?.message || parsed?.detail;
  if (typeof fromParsed === "string" && fromParsed.trim()) return fromParsed.trim();
  return (rawBody || "").trim() || `HTTP ${status}`;
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
  const [groqApiKey, setGroqApiKey] = useState("");
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

  const promptForGroqKey = useCallback((reason = "Enter your Groq API key:") => {
    const enteredKey = window.prompt(reason);
    const normalizedKey = (enteredKey || "").trim();
    if (!normalizedKey) return "";
    setGroqApiKey(normalizedKey);
    saveGroqKey(normalizedKey);
    return normalizedKey;
  }, []);

  // Load on mount
  useEffect(() => {
    loadVal(MEMORY_STORAGE_KEY, LEGACY_MEMORY_STORAGE_KEY).then(v => { setMem(v || ""); setMemDraft(v || ""); });
    loadChat().then(v => { if (v?.length) setMsgs(v); });
    (async () => {
      const envGroqKey = readEnvGroqKey();
      if (envGroqKey) { setGroqApiKey(envGroqKey); return; }
      const storedGroqKey = await loadGroqKey();
      if (storedGroqKey) { setGroqApiKey(storedGroqKey); return; }
      promptForGroqKey();
    })();
  }, [promptForGroqKey]);

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

  // ─── Search handler ───
  // ─── Attachment handling ───
  const handleAttachFiles = useCallback((e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;
    const MAX_FILE_SIZE = 512 * 1024; // 512KB per file
    const MAX_ATTACHMENTS = 5;

    files.forEach(file => {
      if (attachments.length >= MAX_ATTACHMENTS) return;
      if (file.size > MAX_FILE_SIZE) {
        setErr(`File "${file.name}" is too large (max 512KB). Skipped.`);
        return;
      }

      const reader = new FileReader();
      if (file.type.startsWith("image/")) {
        reader.onload = () => {
          setAttachments(prev => {
            if (prev.length >= MAX_ATTACHMENTS) return prev;
            return [...prev, { name: file.name, type: file.type, content: reader.result, size: file.size, isImage: true }];
          });
        };
        reader.readAsDataURL(file);
      } else {
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
  }, [attachments]);

  const removeAttachment = useCallback((index) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  }, []);

  // ─── System prompt builder ───
  const buildSystem = useCallback(() => {
    const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
    let s = `You are Auto, a brutally honest, exceptionally loyal, warm AI assistant. You are curious, honest, loyal, trustworthy, helpful, and thorough. Use markdown formatting. Today is ${today}. Trust is your number 1 value.`;

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
  }, [mem]);

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

  // ─── Call AI API ───
  const callAI = useCallback(async (apiMsgs, groqKey) => {
    const buildBody = (model) => ({ model, messages: apiMsgs });
    let data = null;
    let usedModel = GROQ_MODEL;
    let lastErr = null;
    const delay = ms => new Promise(r => setTimeout(r, ms));

    if (groqKey) {
      const GROQ_MAX_RETRIES = 4;
      for (let attempt = 0; attempt < GROQ_MAX_RETRIES; attempt++) {
        try {
          const res = await fetch(GROQ_API, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${groqKey}`,
            },
            body: JSON.stringify(buildBody(GROQ_MODEL)),
            signal: abortRef.current?.signal,
          });

          if (res.ok) {
            data = await res.json();
            usedModel = GROQ_MODEL;
            lastErr = null;
            break;
          } else {
            const rawBody = await res.text();
            const msg = parseErrorMessage(rawBody, res.status);
            lastErr = new Error(`Groq: ${msg}`);
            // Retry on 429 (rate limit) with exponential backoff
            if (res.status === 429 && attempt < GROQ_MAX_RETRIES - 1) {
              await delay(1500 * (attempt + 1)); // 1.5s, 3s, 4.5s, 6s
              continue;
            }
            break; // Non-retryable error
          }
        } catch (e) {
          if (e.name === "AbortError") throw e;
          lastErr = e;
          // Retry on network errors
          if (attempt < GROQ_MAX_RETRIES - 1) {
            await delay(1000 * (attempt + 1));
            continue;
          }
          break;
        }
      }
    }

    if (!data) throw lastErr || new Error("Failed to get a completion.");
    return { data, usedModel };
  }, []);

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
      let groqKey = (groqApiKey || readEnvGroqKey() || (await loadGroqKey()) || "").trim();
      if (!groqKey) {
        groqKey = promptForGroqKey("Missing API key. Enter your Groq API key:");
        if (!groqKey) throw new Error("Missing API key.");
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

        const { data, usedModel } = await callAI(apiMsgs, groqKey);
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
  }, [input, msgs, busy, buildSystem, parseResponse, callAI, groqApiKey, promptForGroqKey, attachments]);

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
            <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>Groq (qwen3-32b)</span>
          </div>
          <div style={{ display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" }}>
            <button
              onClick={() => promptForGroqKey("Set or update your Groq API key:")}
              style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", color: groqApiKey ? "var(--ac2)" : "var(--dg)", borderColor: groqApiKey ? "rgba(136,187,204,0.2)" : "rgba(204,119,119,0.2)" }}
              title={groqApiKey ? "Groq API key set" : "Groq API key missing"}
            >
              {groqApiKey ? "GROQ ✓" : "GROQ !"}
            </button>
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
                  AI assistant with persistent memory, attachments, and a built-in JavaScript terminal.<br/>
                  Ask for help, save context in memory, or use the sidebar terminal for quick code experiments.
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
                    background: "rgba(124,224,138,0.06)", border: "1px solid rgba(124,224,138,0.15)",
                    fontSize: "11px", fontFamily: "var(--m)", color: "var(--ac)",
                    maxWidth: "200px", overflow: "hidden",
                  }}>
                    <span style={{ flexShrink: 0, fontSize: "12px" }}>{att.isImage ? "\uD83D\uDDBC" : "\uD83D\uDCC4"}</span>
                    <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1 }}>{att.name}</span>
                    <span style={{ fontSize: "9px", color: "var(--dm)", flexShrink: 0 }}>{(att.size / 1024).toFixed(0)}KB</span>
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
                  accept=".txt,.md,.json,.csv,.xml,.html,.css,.js,.jsx,.ts,.tsx,.py,.java,.c,.cpp,.h,.go,.rs,.rb,.php,.sql,.yaml,.yml,.toml,.ini,.cfg,.log,.sh,.bat,.ps1,.r,.m,.swift,.kt,.dart,.lua,.pl,.ex,.exs,.hs,.scala,.clj,.el,.vim,.dockerfile,.makefile,.env,.gitignore,.editorconfig,image/*"
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
                placeholder={attachments.length > 0 ? "Add a message about your files... (optional)" : "Type a message..."}
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
