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

const API = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL = "stepfun/step-3.5-flash:free";
const MODEL_FALLBACKS = [
  DEFAULT_MODEL,
  "qwen/qwen3-coder:free",
];
const GROQ_API = "https://api.groq.com/openai/v1/chat/completions";
const GROQ_MODEL = "qwen/qwen3-32b"; // Default model for Groq key
const CORS_PROXIES = [
  { base: "https://api.allorigins.win/raw?url=", encode: true },
  { base: "https://api.codetabs.com/v1/proxy?quest=", encode: true },
  { base: "https://corsproxy.org/?", encode: true },
  { base: "https://corsproxy.io/?url=", encode: true },
  { base: "https://thingproxy.freeboard.io/fetch/", encode: false },
  { base: "https://api.cors.lol/?url=", encode: true },
  { base: "https://corsproxy.garmeeh.workers.dev/", encode: false },
];

// ─── Proxy health tracking — deprioritize failing proxies ───
const _proxyHealth = {};
function _recordProxyResult(proxyBase, success) {
  if (!_proxyHealth[proxyBase]) _proxyHealth[proxyBase] = { ok: 0, fail: 0, lastFail: 0 };
  if (success) { _proxyHealth[proxyBase].ok++; }
  else { _proxyHealth[proxyBase].fail++; _proxyHealth[proxyBase].lastFail = Date.now(); }
}
function _getSortedProxies() {
  // Sort proxies: working ones first, recently-failed ones last
  return [...CORS_PROXIES].sort((a, b) => {
    const ha = _proxyHealth[a.base] || { ok: 0, fail: 0, lastFail: 0 };
    const hb = _proxyHealth[b.base] || { ok: 0, fail: 0, lastFail: 0 };
    const recentA = (Date.now() - ha.lastFail) < 60000 ? ha.fail : 0;
    const recentB = (Date.now() - hb.lastFail) < 60000 ? hb.fail : 0;
    return recentA - recentB;
  });
}

// ─── Agent Skills Registry ───
// Pre-listed skills the AI can invoke during processing. Organized by category.
const SKILLS_REGISTRY = {
  // ── Data & Information ──
  "data-analysis": {
    id: "data-analysis", category: "data", name: "Data Analysis",
    description: "Analyze datasets, compute statistics, identify patterns, and provide data-driven insights.",
    triggers: ["analyze", "statistics", "dataset", "csv data", "json data", "mean", "median", "average", "trend", "correlation", "outlier"],
  },
  "data-visualization": {
    id: "data-visualization", category: "data", name: "Data Visualization",
    description: "Create text-based charts, graphs, and visual data representations.",
    triggers: ["chart", "graph", "plot", "visualize", "bar chart", "histogram", "pie chart", "visualization"],
  },
  "fact-checker": {
    id: "fact-checker", category: "data", name: "Fact Checker",
    description: "Verify claims, cross-reference information across multiple sources.",
    triggers: ["is this true", "verify", "fact check", "is it true", "confirm", "debunk", "misinformation"],
  },
  "summarizer": {
    id: "summarizer", category: "data", name: "Summarizer",
    description: "Condense long texts, articles, and documents into clear summaries.",
    triggers: ["summarize", "summary", "tldr", "tl;dr", "key points", "gist", "condense", "brief"],
  },
  "json-csv-parser": {
    id: "json-csv-parser", category: "data", name: "JSON & CSV Parser",
    description: "Parse, transform, query, and convert between JSON, CSV, and structured data formats.",
    triggers: ["parse json", "parse csv", "convert json", "convert csv", "json to", "csv to", "tsv", "transform data"],
  },

  // ── Website Loading & Scraping ──
  "web-scraper": {
    id: "web-scraper", category: "web", name: "Web Scraper",
    description: "Extract structured data from websites including tables, lists, prices, and more.",
    triggers: ["scrape", "extract data from", "pull data", "get data from website", "scraping"],
  },
  "site-mapper": {
    id: "site-mapper", category: "web", name: "Site Mapper",
    description: "Map website structure, discover pages, and analyze navigation hierarchy.",
    triggers: ["sitemap", "map website", "site structure", "website architecture", "all pages on"],
  },
  "content-extractor": {
    id: "content-extractor", category: "web", name: "Content Extractor",
    description: "Extract clean article text, main content, and metadata from web pages.",
    triggers: ["extract article", "get text from", "read this page", "clean content", "article text"],
  },
  "link-checker": {
    id: "link-checker", category: "web", name: "Link Checker",
    description: "Verify links on a webpage, check for broken URLs and link health.",
    triggers: ["check links", "broken links", "dead links", "link audit", "verify urls"],
  },
  "search-researcher": {
    id: "search-researcher", category: "web", name: "Search & Research",
    description: "Conduct deep multi-query web research, synthesizing information from multiple sources.",
    triggers: ["research", "investigate", "find out about", "deep dive", "comprehensive search", "look into"],
  },

  // ── Tool Usage ──
  "file-converter": {
    id: "file-converter", category: "tools", name: "File Converter",
    description: "Convert between file formats: JSON, CSV, XML, YAML, HTML, Markdown, and more.",
    triggers: ["convert to", "export as", "save as", "transform to", "json to csv", "csv to json", "xml to", "yaml to"],
  },
  "regex-builder": {
    id: "regex-builder", category: "tools", name: "Regex Builder",
    description: "Build, test, explain, and debug regular expressions.",
    triggers: ["regex", "regular expression", "pattern matching", "match pattern", "validate format", "regexp"],
  },
  "api-tester": {
    id: "api-tester", category: "tools", name: "API Tester",
    description: "Test REST API endpoints, construct requests, and analyze responses.",
    triggers: ["test api", "api request", "http request", "fetch endpoint", "curl", "rest api", "status code"],
  },
  "text-tools": {
    id: "text-tools", category: "tools", name: "Text Tools",
    description: "Encode/decode, case conversion, word counts, text transformation, and manipulation.",
    triggers: ["encode", "decode", "base64", "url encode", "word count", "character count", "uppercase", "lowercase", "camelcase", "snake_case"],
  },
  "timestamp-converter": {
    id: "timestamp-converter", category: "tools", name: "Timestamp Converter",
    description: "Convert between date formats, timezones, Unix timestamps, and human-readable dates.",
    triggers: ["timestamp", "unix time", "epoch", "convert date", "timezone", "iso 8601", "date format"],
  },

  // ── Math & Science ──
  "calculator": {
    id: "calculator", category: "math", name: "Advanced Calculator",
    description: "Perform calculations from basic arithmetic to financial math, algebra, and number theory.",
    triggers: ["calculate", "compute", "solve", "what is", "how much", "percentage", "interest", "mortgage", "factorial", "prime"],
  },
  "unit-converter": {
    id: "unit-converter", category: "math", name: "Unit Converter",
    description: "Convert between units: length, weight, temperature, volume, speed, data, and more.",
    triggers: ["convert", "how many", "miles to", "kg to", "fahrenheit", "celsius", "liters to", "bytes to"],
  },
  "statistics": {
    id: "statistics", category: "math", name: "Statistics",
    description: "Descriptive stats, probability, distributions, correlation, and regression analysis.",
    triggers: ["standard deviation", "probability", "distribution", "regression", "correlation", "p-value", "confidence interval", "hypothesis"],
  },
  "science-helper": {
    id: "science-helper", category: "math", name: "Science Helper",
    description: "Physics, chemistry, and biology formulas, constants, and calculations.",
    triggers: ["physics", "chemistry", "biology", "formula", "element", "molecule", "force", "energy", "velocity", "acceleration", "periodic table"],
  },
  "geometry": {
    id: "geometry", category: "math", name: "Geometry",
    description: "Areas, volumes, perimeters, angles, trigonometry, and coordinate geometry.",
    triggers: ["area", "volume", "perimeter", "angle", "triangle", "circle", "sphere", "pythagorean", "trigonometry", "distance between"],
  },

  // ── Coding ──
  "code-reviewer": {
    id: "code-reviewer", category: "coding", name: "Code Reviewer",
    description: "Review code for bugs, performance, security, and style issues.",
    triggers: ["review code", "code review", "what's wrong", "check my code", "any bugs", "improve code", "code quality"],
  },
  "code-generator": {
    id: "code-generator", category: "coding", name: "Code Generator",
    description: "Generate code snippets, functions, classes, and modules in any language.",
    triggers: ["write code", "create function", "generate", "build a", "implement", "make a function", "code for"],
  },
  "debugger": {
    id: "debugger", category: "coding", name: "Debugger",
    description: "Debug code: identify bugs, trace errors, explain stack traces, and provide fixes.",
    triggers: ["debug", "error", "bug", "doesn't work", "crashes", "stack trace", "exception", "fix this", "broken code"],
  },
  "explainer": {
    id: "explainer", category: "coding", name: "Code Explainer",
    description: "Explain code line-by-line, break down algorithms, and teach programming concepts.",
    triggers: ["explain", "what does this do", "how does this work", "break down", "walk through", "teach me", "understand"],
  },
  "refactorer": {
    id: "refactorer", category: "coding", name: "Refactorer",
    description: "Refactor code for readability, performance, and maintainability.",
    triggers: ["refactor", "clean up", "improve", "optimize", "modernize", "simplify code", "make readable"],
  },
  "snippet-library": {
    id: "snippet-library", category: "coding", name: "Snippet Library",
    description: "Ready-to-use code snippets for common programming tasks.",
    triggers: ["snippet", "how do i", "example of", "template for", "boilerplate", "starter code"],
  },

  // ── Accessibility ──
  "a11y-checker": {
    id: "a11y-checker", category: "accessibility", name: "Accessibility Checker",
    description: "Audit web content for WCAG 2.1 compliance, identify issues, and suggest fixes.",
    triggers: ["accessibility", "a11y", "wcag", "ada compliance", "accessible", "audit accessibility"],
  },
  "alt-text-generator": {
    id: "alt-text-generator", category: "accessibility", name: "Alt Text Generator",
    description: "Generate descriptive alt text for images following accessibility best practices.",
    triggers: ["alt text", "image description", "describe image", "alt attribute", "screen reader image"],
  },
  "color-contrast": {
    id: "color-contrast", category: "accessibility", name: "Color Contrast Checker",
    description: "Check color contrast ratios for WCAG compliance and suggest accessible colors.",
    triggers: ["contrast ratio", "color contrast", "readable colors", "wcag contrast", "color accessibility"],
  },
  "screen-reader-guide": {
    id: "screen-reader-guide", category: "accessibility", name: "Screen Reader Guide",
    description: "Optimize content for screen readers, write ARIA attributes, and ensure AT compatibility.",
    triggers: ["screen reader", "aria", "aria-label", "assistive technology", "voiceover", "nvda", "jaws"],
  },
  "keyboard-navigation": {
    id: "keyboard-navigation", category: "accessibility", name: "Keyboard Navigation",
    description: "Ensure interfaces are fully keyboard-operable with proper focus management.",
    triggers: ["keyboard", "tab order", "focus", "keyboard navigation", "keyboard accessible", "focus trap", "skip nav"],
  },
};

// Build skills summary for system prompt
function buildSkillsSummary() {
  const categories = {};
  for (const skill of Object.values(SKILLS_REGISTRY)) {
    if (!categories[skill.category]) categories[skill.category] = [];
    categories[skill.category].push(skill);
  }
  const labels = { data: "Data & Information", web: "Website & Scraping", tools: "Tool Usage", math: "Math & Science", coding: "Coding", accessibility: "Accessibility" };
  let summary = "";
  for (const [cat, skills] of Object.entries(categories)) {
    summary += `\n### ${labels[cat] || cat}\n`;
    for (const s of skills) {
      summary += `- **${s.name}** (\`${s.id}\`): ${s.description}\n`;
    }
  }
  return summary;
}

// Detect which skills are relevant to a user message
function detectRelevantSkills(message) {
  if (!message || typeof message !== "string") return [];
  const lower = message.toLowerCase();
  const matched = [];
  for (const skill of Object.values(SKILLS_REGISTRY)) {
    for (const trigger of skill.triggers) {
      if (lower.includes(trigger)) {
        matched.push(skill);
        break;
      }
    }
  }
  return matched;
}

// ─── Known SPA domains that need direct mode (proxy returns empty shell) ───
const SPA_DOMAINS = [
  "x.com", "twitter.com", "facebook.com", "instagram.com", "threads.net",
  "linkedin.com", "reddit.com", "tiktok.com", "discord.com", "twitch.tv",
  "netflix.com", "spotify.com", "youtube.com", "gmail.com", "docs.google.com",
  "drive.google.com", "maps.google.com", "web.whatsapp.com", "telegram.org",
  "app.slack.com", "figma.com", "canva.com", "notion.so",
  "google.com", "amazon.com", "ebay.com", "pinterest.com",
  "tumblr.com", "medium.com", "dev.to", "stackoverflow.com",
  "github.com", "gitlab.com", "bitbucket.org",
  "airbnb.com", "booking.com", "yelp.com",
  "trello.com", "asana.com", "jira.atlassian.com",
];

// ─── Sites known to aggressively block proxies (route straight to Jina Reader) ───
const PROXY_HOSTILE_DOMAINS = [
  "coinmarketcap.com", "finance.yahoo.com", "bloomberg.com", "wsj.com",
  "ft.com", "nytimes.com", "washingtonpost.com",
  "cloudflare.com", "indeed.com", "zillow.com", "glassdoor.com",
  "linkedin.com", "paypal.com", "chase.com", "bankofamerica.com",
  "wellsfargo.com", "target.com", "walmart.com", "bestbuy.com",
  "homedepot.com", "lowes.com", "costco.com", "kroger.com",
  "reuters.com", "economist.com", "barrons.com", "cnbc.com",
  "cnn.com", "foxnews.com", "bbc.com", "theguardian.com",
];

function _getDomain(url) {
  try { return new URL(url).hostname.replace(/^www\./, ""); } catch { return ""; }
}
function _isDomainInList(url, list) {
  const domain = _getDomain(url);
  return list.some(d => domain === d || domain.endsWith("." + d));
}

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
async function loadApiKey() {
  try {
    if (window.storage?.get) {
      const r = await window.storage.get("openrouter-api-key");
      if (r?.value) return String(r.value).trim();
    }
  } catch {}
  try { return (window.localStorage.getItem("openrouter-api-key") || "").trim(); } catch { return ""; }
}
async function saveApiKey(val) {
  const n = (val || "").trim();
  try { if (window.storage?.set) await window.storage.set("openrouter-api-key", n); } catch {}
  try { window.localStorage.setItem("openrouter-api-key", n); } catch {}
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
function readEnvApiKey() {
  return (window.OPENROUTER_API_KEY || window.__OPENROUTER_API_KEY__ || window?.env?.OPENROUTER_API_KEY || "").trim();
}
function readEnvGroqKey() {
  return (window.GROQ_API_KEY || window.__GROQ_API_KEY__ || window?.env?.GROQ_API_KEY || "").trim();
}

// ─── Race multiple CORS proxies for a URL — returns first successful text ───
async function fetchWithProxyRace(targetUrl, timeoutMs = 12000) {
  const sortedProxies = _getSortedProxies();
  return new Promise((resolve) => {
    let settled = false;
    const controllers = sortedProxies.map(() => new AbortController());
    let pending = sortedProxies.length;

    const globalTid = setTimeout(() => {
      if (!settled) {
        settled = true;
        controllers.forEach(c => { try { c.abort(); } catch {} });
        resolve(null);
      }
    }, timeoutMs + 2000);

    function onDone(html, proxyBase) {
      if (settled) return;
      if (!html || html.length < 50) { onFail(null, proxyBase); return; }
      settled = true;
      clearTimeout(globalTid);
      controllers.forEach(c => { try { c.abort(); } catch {} });
      _recordProxyResult(proxyBase, true);
      resolve(html);
    }
    function onFail(err, proxyBase) {
      if (settled) return;
      if (proxyBase) _recordProxyResult(proxyBase, false);
      pending--;
      if (pending <= 0) { settled = true; clearTimeout(globalTid); resolve(null); }
    }

    sortedProxies.forEach((proxy, i) => {
      const tid = setTimeout(() => { try { controllers[i].abort(); } catch {} }, timeoutMs);
      const proxyUrl = proxy.base + (proxy.encode ? encodeURIComponent(targetUrl) : targetUrl);
      fetch(proxyUrl, { signal: controllers[i].signal, cache: "no-store" })
        .then(r => { clearTimeout(tid); if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
        .then(html => { onDone(html, proxy.base); })
        .catch(() => { clearTimeout(tid); onFail(null, proxy.base); });
    });
  });
}

// ─── Web Search via DuckDuckGo ───
async function performSearch(query) {
  const results = [];

  // Primary: DuckDuckGo HTML via CORS proxy race (real search results)
  try {
    const ddgUrl = `https://html.duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
    const html = await fetchWithProxyRace(ddgUrl, 15000);
    if (html) {
      const doc = new DOMParser().parseFromString(html, "text/html");
      doc.querySelectorAll(".result, .web-result").forEach(item => {
        const a = item.querySelector(".result__a, .result-link");
        const snip = item.querySelector(".result__snippet, .result-snippet");
        if (a) {
          const href = a.getAttribute("href") || "";
          const urlMatch = href.match(/uddg=([^&]+)/);
          const url = urlMatch ? decodeURIComponent(urlMatch[1]) : href;
          if (url.startsWith("http")) {
            results.push({
              title: a.textContent.trim(),
              snippet: snip?.textContent?.trim() || "",
              url,
            });
          }
        }
      });
    }
  } catch (e) {
    // AbortError is expected when proxy race times out — don't spam console
    if (e?.name !== "AbortError") console.warn("DDG HTML search failed:", e);
  }

  // Fallback 2: Google search via CORS proxy
  if (results.length === 0) {
    try {
      const googleUrl = `https://www.google.com/search?q=${encodeURIComponent(query)}&num=10&hl=en`;
      const gHtml = await fetchWithProxyRace(googleUrl, 12000);
      if (gHtml) {
        const gDoc = new DOMParser().parseFromString(gHtml, "text/html");
        // Google wraps results in <div class="g"> blocks
        gDoc.querySelectorAll("div.g, div.tF2Cxc, div.MjjYud div[data-hveid]").forEach(item => {
          const a = item.querySelector("a[href^='http']");
          const h3 = item.querySelector("h3");
          const snip = item.querySelector(".VwiC3b, .IsZvec, .s3v9rd, span.st");
          if (a && h3) {
            const href = a.getAttribute("href") || "";
            if (href.startsWith("http") && !href.includes("google.com/search")) {
              results.push({
                title: h3.textContent.trim(),
                snippet: snip?.textContent?.trim() || "",
                url: href,
              });
            }
          }
        });
      }
    } catch (e) {
      if (e?.name !== "AbortError") console.warn("Google search failed:", e);
    }
  }

  // Fallback 3: Brave Search via CORS proxy
  if (results.length === 0) {
    try {
      const braveUrl = `https://search.brave.com/search?q=${encodeURIComponent(query)}&source=web`;
      const bHtml = await fetchWithProxyRace(braveUrl, 12000);
      if (bHtml) {
        const bDoc = new DOMParser().parseFromString(bHtml, "text/html");
        bDoc.querySelectorAll("#results .snippet, .result").forEach(item => {
          const a = item.querySelector("a[href^='http']");
          const title = item.querySelector(".snippet-title, .title, h2, h3");
          const snip = item.querySelector(".snippet-description, .snippet-content, .description");
          if (a && title) {
            const href = a.getAttribute("href") || "";
            if (href.startsWith("http") && !href.includes("brave.com")) {
              results.push({
                title: title.textContent.trim(),
                snippet: snip?.textContent?.trim() || "",
                url: href,
              });
            }
          }
        });
      }
    } catch (e) {
      if (e?.name !== "AbortError") console.warn("Brave search failed:", e);
    }
  }

  // Fallback 4: DuckDuckGo JSON API (instant answers — no proxy needed, direct CORS)
  if (results.length === 0) {
    try {
      const url = `https://api.duckduckgo.com/?q=${encodeURIComponent(query)}&format=json&no_html=1&skip_disambig=1`;
      const ctrl2 = new AbortController();
      const tid2 = setTimeout(() => ctrl2.abort(), 8000);
      const res = await fetch(url, { signal: ctrl2.signal });
      clearTimeout(tid2);
      if (res.ok) {
        const data = await res.json();
        if (data.AbstractText) {
          results.push({ title: data.Heading || "Summary", snippet: data.AbstractText, url: data.AbstractURL || "" });
        }
        for (const t of (data.RelatedTopics || [])) {
          if (t.Text && t.FirstURL) results.push({ title: t.Text.slice(0, 100), snippet: t.Text, url: t.FirstURL });
          for (const sub of (t.Topics || [])) {
            if (sub.Text && sub.FirstURL) results.push({ title: sub.Text.slice(0, 100), snippet: sub.Text, url: sub.FirstURL });
          }
        }
      }
    } catch (e) { console.warn("DDG API failed:", e); }
  }

  return results.filter(r => r.url).slice(0, 12);
}

// ─── Fetch via Jina Reader API (renders JS via headless Chrome) ───
// format: "text" (default, returns markdown/text), "html" (returns full rendered HTML)
async function fetchWithJinaReader(targetUrl, timeoutMs = 20000, format) {
  const jinaUrl = "https://r.jina.ai/" + targetUrl;
  // Try direct fetch first with proper headers (Jina supports CORS)
  try {
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), timeoutMs);
    const headers = {};
    if (format === "html") {
      headers["x-respond-with"] = "html";
      headers["Accept"] = "text/html";
    } else {
      headers["Accept"] = "text/plain";
    }
    const res = await fetch(jinaUrl, {
      signal: ctrl.signal,
      headers: headers,
    });
    clearTimeout(tid);
    if (res.ok) {
      const text = await res.text();
      if (text && text.length > 100) return text;
    }
  } catch {}
  // Fallback: route through CORS proxies (can't send custom Jina headers this way)
  try {
    const text = await fetchWithProxyRace(jinaUrl, timeoutMs);
    if (text && text.length > 100) return text;
  } catch {}
  return null;
}

// ─── Fetch page text for AI reading (with fallbacks for dynamic sites) ───
async function fetchPageText(url) {
  // Helper to extract text from HTML string
  function extractText(html) {
    const doc = new DOMParser().parseFromString(html, "text/html");
    doc.querySelectorAll("script,style,nav,footer,header,aside,iframe,noscript,svg").forEach(el => el.remove());
    return (doc.body?.textContent || "").replace(/\s+/g, " ").trim();
  }

  // 1. PRIMARY: Jina Reader API — renders JavaScript via headless Chrome, returns clean text
  //    Works on SPAs (x.com, reddit, etc.), paywalled sites, and dynamic content
  try {
    const jinaText = await fetchWithJinaReader(url, 20000);
    if (jinaText && jinaText.length > 100) return jinaText.slice(0, 12000);
  } catch {}

  // 1b. Try Jina HTML mode and extract text from rendered HTML (catches content markdown misses)
  try {
    const jinaHtml = await fetchWithJinaReader(url, 15000, "html");
    if (jinaHtml && jinaHtml.length > 200) {
      const text = extractText(jinaHtml);
      if (text.length > 200) return text.slice(0, 12000);
    }
  } catch {}

  // 2. Fallback: Direct CORS proxy fetch (for simple static sites — faster than Jina)
  try {
    const html = await fetchWithProxyRace(url, 12000);
    if (html) {
      const text = extractText(html);
      if (text.length > 200) return text.slice(0, 8000);
    }
  } catch {}

  // 3. Fallback: Wayback Machine archived version
  try {
    const wbUrl = "https://web.archive.org/web/2if_/" + url;
    const wbHtml = await fetchWithProxyRace(wbUrl, 10000);
    if (wbHtml) {
      const text = extractText(wbHtml);
      if (text.length > 100) return text.slice(0, 8000);
    }
  } catch {}

  // 4. Fallback: Wayback Machine CDX API for latest snapshot
  try {
    const cdxUrl = "https://web.archive.org/web/timemap/link/" + url;
    const cdxHtml = await fetchWithProxyRace(cdxUrl, 8000);
    if (cdxHtml) {
      const matches = cdxHtml.match(/https:\/\/web\.archive\.org\/web\/\d+\/[^\s<>"]+/g);
      if (matches && matches.length > 0) {
        const latestUrl = matches[matches.length - 1].replace(/\/web\/(\d+)\//, "/web/$1if_/");
        const snapHtml = await fetchWithProxyRace(latestUrl, 10000);
        if (snapHtml) {
          const text = extractText(snapHtml);
          if (text.length > 100) return text.slice(0, 8000);
        }
      }
    }
  } catch {}

  // 5. Fallback: 12ft.io for paywalls
  try {
    const ftUrl = "https://12ft.io/api/proxy?q=" + encodeURIComponent(url);
    const ftHtml = await fetchWithProxyRace(ftUrl, 10000);
    if (ftHtml) {
      const text = extractText(ftHtml);
      if (text.length > 100) return text.slice(0, 8000);
    }
  } catch {}

  // 6. Fallback: Google Cache (often has content even when site blocks direct access)
  try {
    const cacheUrl = "https://webcache.googleusercontent.com/search?q=cache:" + encodeURIComponent(url) + "&strip=1";
    const cacheHtml = await fetchWithProxyRace(cacheUrl, 10000);
    if (cacheHtml) {
      const text = extractText(cacheHtml);
      if (text.length > 100) return text.slice(0, 8000);
    }
  } catch {}

  return null;
}

// ─── iframe control script (injected into fetched pages) ───
// Written in ES5 so serialization via .toString() works predictably
function _iframeCtrl() {
  if (window.__autoCtrlLoaded) return;
  window.__autoCtrlLoaded = true;

  function reply(e, id, payload) {
    try { e.source.postMessage({ autoBrowser: true, type: "cmdReply", id: id, payload: payload }, "*"); } catch(ex) {}
  }
  // Also support replying to parent (for direct mode injection)
  function replyParent(id, payload) {
    try { window.parent.postMessage({ autoBrowser: true, type: "cmdReply", id: id, payload: payload }, "*"); } catch(ex) {}
  }

  window.addEventListener("message", function(e) {
    var d = e.data;
    if (!d || !d.autoBrowserCmd) return;
    var id = d.id;
    var sendReply = function(payload) {
      try { reply(e, id, payload); } catch(ex) {}
      try { replyParent(id, payload); } catch(ex) {}
    };
    if (d.cmd === "read") {
      // Wait a tick for any pending renders
      setTimeout(function() {
        var clone = document.body ? document.body.cloneNode(true) : null;
        if (clone) { var rm = clone.querySelectorAll("script,style,noscript,iframe,svg,link[rel=stylesheet]"); for (var i=0;i<rm.length;i++) rm[i].parentNode && rm[i].parentNode.removeChild(rm[i]); }
        var text = ((clone && clone.textContent) || "").replace(/\s+/g, " ").trim().slice(0, 8000);
        var aEls = document.querySelectorAll("a[href]");
        var links = [];
        for (var i = 0; i < Math.min(aEls.length, 25); i++) {
          var aEl = aEls[i];
          if (aEl.offsetParent !== null || aEl.closest("nav,header,footer")) {
            links.push({ text: (aEl.textContent || "").trim().slice(0, 80), href: aEl.href });
          }
        }
        var inpEls = document.querySelectorAll("input,textarea,select,button,[role=button],[contenteditable=true]");
        var inputs = [];
        for (var i = 0; i < Math.min(inpEls.length, 25); i++) {
          var el = inpEls[i];
          inputs.push({ tag: el.tagName.toLowerCase(), type: el.type || "", id: el.id || "", name: el.name || "", placeholder: el.placeholder || "", text: (el.textContent || "").trim().slice(0, 60), ariaLabel: el.getAttribute("aria-label") || "" });
        }
        // Get page metadata
        var meta = {};
        try { meta.description = (document.querySelector('meta[name="description"]') || {}).content || ""; } catch(ex) {}
        try { meta.ogTitle = (document.querySelector('meta[property="og:title"]') || {}).content || ""; } catch(ex) {}
        sendReply({ text: text, title: document.title || "", url: window.location.href, links: links, inputs: inputs, meta: meta });
      }, 50);
    } else if (d.cmd === "click") {
      var sel = d.selector, el = null;
      var selLower = sel.toLowerCase();
      // Strategy 1: CSS selector
      try { el = document.querySelector(sel); } catch(ex) {}
      // Strategy 2: Search clickable elements by textContent
      if (!el) {
        var cands = document.querySelectorAll("a,button,input[type=submit],input[type=button],[onclick],[role=button],[role=link],summary,[tabindex],label");
        // Prefer exact match first
        for (var i = 0; i < cands.length; i++) { if ((cands[i].textContent || "").trim().toLowerCase() === selLower) { el = cands[i]; break; } }
        // Then contains match
        if (!el) { for (var i = 0; i < cands.length; i++) { if ((cands[i].textContent || "").trim().toLowerCase().indexOf(selLower) >= 0) { el = cands[i]; break; } } }
      }
      // Strategy 3: aria-label, title, value, data-testid
      if (!el) {
        var allAttr = document.querySelectorAll("[aria-label],[title],[value],[data-testid],[data-test]");
        for (var i = 0; i < allAttr.length; i++) {
          var a = ((allAttr[i].getAttribute("aria-label") || "") + (allAttr[i].getAttribute("title") || "") + (allAttr[i].getAttribute("value") || "") + (allAttr[i].getAttribute("data-testid") || "")).toLowerCase();
          if (a.indexOf(selLower) >= 0) { el = allAttr[i]; break; }
        }
      }
      // Strategy 4: Search any visible element
      if (!el) {
        var all = document.querySelectorAll("*");
        for (var i = 0; i < all.length; i++) {
          var txt = (all[i].textContent || "").trim().toLowerCase();
          if ((txt === selLower || (txt.length < selLower.length * 3 && txt.indexOf(selLower) >= 0)) && all[i].offsetParent !== null) { el = all[i]; break; }
        }
      }
      // Strategy 5: partial match on visible elements
      if (!el) {
        var all2 = document.querySelectorAll("*");
        for (var i = 0; i < all2.length; i++) { if ((all2[i].textContent || "").trim().toLowerCase().indexOf(selLower) >= 0 && all2[i].offsetParent !== null) { el = all2[i]; break; } }
      }
      if (el) {
        try { el.scrollIntoView({ behavior: "smooth", block: "center" }); } catch(ex) {}
        var rect = el.getBoundingClientRect();
        var prevOutline = el.style.outline, prevOffset = el.style.outlineOffset;
        el.style.outline = "2px solid #7ce08a"; el.style.outlineOffset = "2px";
        setTimeout(function() { try { el.style.outline = prevOutline; el.style.outlineOffset = prevOffset; } catch(ex){} }, 1400);
        try { el.focus(); } catch(ex) {}
        try { el.click(); } catch(ex) {}
        try { el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true, view: window })); } catch(ex) {}
        // For checkboxes/radios, toggle checked state
        if (el.type === "checkbox" || el.type === "radio") { try { el.checked = !el.checked; el.dispatchEvent(new Event("change", { bubbles: true })); } catch(ex) {} }
        sendReply({ success: true, element: el.tagName, text: (el.textContent || "").trim().slice(0, 60), x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 });
      } else { sendReply({ success: false, error: "Element not found: " + sel }); }
    } else if (d.cmd === "type") {
      var sel = d.selector, text = d.text, el = null;
      try { el = document.querySelector(sel); } catch(ex) {}
      if (!el) {
        var inps = document.querySelectorAll("input,textarea,[contenteditable=true]");
        for (var i = 0; i < inps.length; i++) {
          var inp = inps[i];
          var searchStr = ((inp.placeholder || "") + (inp.name || "") + (inp.id || "") + (inp.getAttribute("aria-label") || "") + (inp.type || "")).toLowerCase();
          if (searchStr.indexOf(sel.toLowerCase()) >= 0) { el = inp; break; }
        }
      }
      // Fallback: first visible input/textarea
      if (!el) {
        var inps2 = document.querySelectorAll("input:not([type=hidden]):not([type=submit]):not([type=button]),textarea,[contenteditable=true]");
        for (var i = 0; i < inps2.length; i++) { if (inps2[i].offsetParent !== null) { el = inps2[i]; break; } }
      }
      if (el) {
        try {
          el.focus();
          // For contenteditable
          if (el.getAttribute("contenteditable") === "true") {
            el.textContent = text;
            el.dispatchEvent(new Event("input", { bubbles: true }));
          } else {
            // Clear and set value, with native input setter for React compatibility
            var nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value");
            if (!nativeSetter) nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value");
            if (nativeSetter && nativeSetter.set) { nativeSetter.set.call(el, text); } else { el.value = text; }
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            // Fire keydown/keyup for frameworks that listen to these
            el.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true }));
            el.dispatchEvent(new KeyboardEvent("keyup", { bubbles: true }));
          }
        } catch(ex) {}
        sendReply({ success: true });
      } else { sendReply({ success: false, error: "Input not found: " + sel }); }
    } else if (d.cmd === "scroll") {
      var dir = d.direction, amt = d.amount || 400;
      if (dir === "up") window.scrollBy({ top: -amt, behavior: "smooth" });
      else if (dir === "down") window.scrollBy({ top: amt, behavior: "smooth" });
      else if (dir === "top") window.scrollTo({ top: 0, behavior: "smooth" });
      else if (dir === "bottom") window.scrollTo({ top: document.body ? document.body.scrollHeight : 0, behavior: "smooth" });
      sendReply({ success: true, scrollY: window.scrollY, scrollHeight: document.body ? document.body.scrollHeight : 0 });
    } else if (d.cmd === "find") {
      var q = (d.query || "").toLowerCase();
      var fEls = document.querySelectorAll("a,button,input,textarea,select,[onclick],[role=button],[role=link],summary,[tabindex],[aria-label],label");
      var matches = [];
      for (var i = 0; i < fEls.length && matches.length < 20; i++) {
        var el = fEls[i];
        var t = ((el.textContent || "") + (el.id || "") + (el.name || "") + (el.placeholder || "") + (el.className || "") + (el.getAttribute("aria-label") || "") + (el.getAttribute("title") || "") + (el.getAttribute("value") || "")).toLowerCase();
        if (t.indexOf(q) >= 0) matches.push({ tag: el.tagName, id: el.id || "", text: (el.textContent || "").trim().slice(0, 60), href: el.href || "", type: el.type || "", name: el.name || "" });
      }
      if (matches.length < 5) {
        var allEls = document.querySelectorAll("div,span,p,h1,h2,h3,h4,h5,h6,li,td,th,label,section,article,main");
        for (var i = 0; i < allEls.length && matches.length < 20; i++) {
          var el = allEls[i];
          var txt = (el.textContent || "").trim().toLowerCase();
          if (txt.length < 200 && txt.indexOf(q) >= 0 && el.offsetParent !== null) {
            var already = false;
            for (var j = 0; j < matches.length; j++) { if (matches[j].text === (el.textContent || "").trim().slice(0, 60)) { already = true; break; } }
            if (!already) matches.push({ tag: el.tagName, id: el.id || "", text: (el.textContent || "").trim().slice(0, 60), href: "" });
          }
        }
      }
      sendReply({ matches: matches });
    } else if (d.cmd === "screenshot") {
      // Return a structured description of visible elements for AI understanding
      var visible = [];
      var allVis = document.querySelectorAll("h1,h2,h3,h4,p,a,button,input,textarea,img,table,ul,ol,nav,main,article,section,form");
      for (var i = 0; i < Math.min(allVis.length, 50); i++) {
        var el = allVis[i];
        if (el.offsetParent === null && el.tagName !== "BODY" && el.tagName !== "HTML") continue;
        var r = el.getBoundingClientRect();
        if (r.width === 0 || r.height === 0) continue;
        var info = { tag: el.tagName.toLowerCase(), text: (el.textContent || "").trim().slice(0, 80) };
        if (el.id) info.id = el.id;
        if (el.href) info.href = el.href;
        if (el.src) info.src = el.src;
        if (el.alt) info.alt = el.alt;
        if (el.type) info.type = el.type;
        if (el.placeholder) info.placeholder = el.placeholder;
        visible.push(info);
      }
      sendReply({ elements: visible, title: document.title, url: window.location.href });
    }
  });

  // Intercept link clicks so the popup can fetch and load the new page via proxy
  document.addEventListener("click", function(ev) {
    var el = ev.target;
    while (el && el.tagName !== "A") el = el.parentNode;
    if (!el || el.tagName !== "A") return;
    var href = el.href || "";
    if (!/^https?:\/\//i.test(href)) return;
    // Allow target=_blank to work normally in direct mode
    if (el.target === "_blank" && window.__autoDirectMode) return;
    ev.preventDefault();
    ev.stopPropagation();
    try { window.parent.postMessage({ autoBrowser: true, type: "iframeNavigate", url: href }, "*"); } catch(ex) {}
  }, true);

  // Intercept form submissions
  document.addEventListener("submit", function(ev) {
    var form = ev.target;
    if (!form || form.tagName !== "FORM") return;
    var method = (form.method || "get").toLowerCase();
    if (method !== "get") return;
    var action = form.action || window.location.href;
    if (!/^https?:\/\//i.test(action)) return;
    ev.preventDefault();
    var params = new URLSearchParams();
    var els = form.elements;
    for (var i = 0; i < els.length; i++) {
      if (els[i].name && !els[i].disabled && els[i].type !== "submit" && els[i].type !== "button") {
        params.set(els[i].name, els[i].value || "");
      }
    }
    var qs = params.toString();
    var url = action + (qs ? (action.indexOf("?") >= 0 ? "&" : "?") + qs : "");
    try { window.parent.postMessage({ autoBrowser: true, type: "iframeNavigate", url: url }, "*"); } catch(ex) {}
  }, true);
}

// ─── Popup window script (runs in popup, serialized via .toString()) ───
function _popupScript(cfg) {
  var PROXY = cfg.proxy;
  var IFRAME_CTRL = cfg.iframeCtrl;
  var SPA_DOMAINS = cfg.spaDomains || [];
  var PROXY_HOSTILE_DOMAINS = cfg.proxyHostileDomains || [];

  function getDomain(url) {
    try { return new URL(url).hostname.replace(/^www\./, ""); } catch(e) { return ""; }
  }
  function isDomainInList(url, list) {
    var domain = getDomain(url);
    return list.some(function(d) { return domain === d || domain.indexOf("." + d) === domain.length - d.length - 1; });
  }
  // ═══ TAB STATE ═══
  var tabs = []; // Array of { id, url, title, history, histIdx, srcdoc, iframeSrc, directMode }
  var activeTabId = null;
  var tabIdCounter = 0;
  var iframe, urlInput, loadingOverlay, loadingText, agentLog, statusText, statusMode, agentBadge, agentBadgeText, takeoverBtn, clickIndicator, agentPanel, panelToggle, tabBar;
  var agentMode = true, panelCollapsed = false;

  function createTab(url, switchToIt) {
    var tab = { id: ++tabIdCounter, url: url || "", title: "New Tab", history: [], histIdx: -1, srcdoc: null, iframeSrc: null, directMode: false };
    tabs.push(tab);
    if (switchToIt !== false) switchTab(tab.id);
    renderTabs();
    notifyParent("tabsChanged", getTabsSummary());
    return tab;
  }

  function closeTab(tabId) {
    if (tabs.length <= 1) return; // Always keep at least one tab
    var idx = tabs.findIndex(function(t) { return t.id === tabId; });
    if (idx < 0) return;
    revokeTabBlobUrl(tabId); // Clean up blob URL memory
    tabs.splice(idx, 1);
    if (activeTabId === tabId) {
      var newIdx = Math.min(idx, tabs.length - 1);
      switchTab(tabs[newIdx].id);
    }
    renderTabs();
    notifyParent("tabsChanged", getTabsSummary());
  }

  function switchTab(tabId) {
    // Save current tab state
    var curTab = getActiveTab();
    if (curTab && iframe) {
      try { curTab.srcdoc = iframe.srcdoc || null; } catch(e) {}
      try { curTab.iframeSrc = iframe.src || null; } catch(e) {}
    }
    activeTabId = tabId;
    var tab = getActiveTab();
    if (!tab) return;
    // Restore tab state into iframe
    if (iframe) {
      if (tab.srcdoc) {
        iframe.removeAttribute("src");
        iframe.srcdoc = tab.srcdoc;
      } else if (tab.iframeSrc && tab.iframeSrc !== "about:blank") {
        iframe.removeAttribute("srcdoc");
        iframe.src = tab.iframeSrc;
      } else {
        iframe.removeAttribute("src");
        iframe.srcdoc = getWelcomeHtml();
      }
    }
    if (urlInput) urlInput.value = tab.url || "";
    updateStatus();
    renderTabs();
    notifyParent("tabSwitched", { tabId: tabId, url: tab.url });
  }

  function getActiveTab() {
    return tabs.find(function(t) { return t.id === activeTabId; }) || tabs[0] || null;
  }

  function getTabsSummary() {
    return tabs.map(function(t) { return { id: t.id, url: t.url, title: t.title, active: t.id === activeTabId }; });
  }

  function getWelcomeHtml() {
    return "<!DOCTYPE html><html><body style='background:#07070b;color:#555;font-family:system-ui;display:flex;align-items:center;justify-content:center;height:100vh;margin:0'><div style='text-align:center'><div style='font-size:40px;margin-bottom:12px;opacity:0.3'>\uD83D\uDC31</div><div style='font-size:13px;color:#444'>Auto Browser</div><div style='font-size:11px;color:#333;margin-top:6px'>Navigate to a URL or let the AI browse for you</div></div></body></html>";
  }

  function renderTabs() {
    if (!tabBar) return;
    tabBar.innerHTML = "";
    tabs.forEach(function(tab) {
      var tabEl = document.createElement("div");
      tabEl.className = "tab" + (tab.id === activeTabId ? " active" : "");
      var titleSpan = document.createElement("span");
      titleSpan.className = "tab-title";
      titleSpan.textContent = tab.title || tab.url.replace(/^https?:\/\/(www\.)?/, "").slice(0, 25) || "New Tab";
      titleSpan.title = tab.url || "New Tab";
      titleSpan.onclick = function() { switchTab(tab.id); };
      tabEl.appendChild(titleSpan);
      if (tabs.length > 1) {
        var closeBtn = document.createElement("span");
        closeBtn.className = "tab-close";
        closeBtn.textContent = "\u00D7";
        closeBtn.onclick = function(e) { e.stopPropagation(); closeTab(tab.id); };
        tabEl.appendChild(closeBtn);
      }
      tabBar.appendChild(tabEl);
    });
    // Add "+" button
    var addBtn = document.createElement("div");
    addBtn.className = "tab-add";
    addBtn.textContent = "+";
    addBtn.title = "New Tab";
    addBtn.onclick = function() { createTab(""); };
    tabBar.appendChild(addBtn);
  }

  function init() {
    iframe = document.getElementById("pf");
    urlInput = document.getElementById("ui");
    loadingOverlay = document.getElementById("lo");
    loadingText = document.getElementById("lt");
    agentLog = document.getElementById("al");
    statusText = document.getElementById("st");
    statusMode = document.getElementById("sm");
    agentBadge = document.getElementById("ab");
    agentBadgeText = document.getElementById("abt");
    takeoverBtn = document.getElementById("tb");
    clickIndicator = document.getElementById("ci");
    agentPanel = document.getElementById("ap");
    panelToggle = document.getElementById("pt");
    tabBar = document.getElementById("tab-bar");

    document.getElementById("back-btn").onclick = goBack;
    document.getElementById("fwd-btn").onclick = goForward;
    document.getElementById("reload-btn").onclick = doReload;
    document.getElementById("go-btn").onclick = doGo;
    takeoverBtn.onclick = toggleTakeover;
    document.getElementById("dm").onclick = toggleDirect;
    urlInput.onkeydown = function(e) { if (e.key === "Enter") doGo(); };
    document.getElementById("ph").onclick = togglePanel;
    window.addEventListener("message", onMessage);

    // Create initial tab
    createTab("");

    hideLoading();
    addLog("Browser ready \u2014 AI agent mode active (with tabs!) \u2014 JavaScript enabled", "ok");
    updateDirectBtn();
    notifyParent("ready", {});
    notifyParent("directModeChanged", { direct: false, tabId: activeTabId });
  }

  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  function addLog(msg, type) {
    if (!agentLog) return;
    var ts = new Date().toLocaleTimeString("en", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" });
    var d = document.createElement("div");
    var safeType = (type || "").replace(/[^a-zA-Z0-9_-]/g, "");
    d.className = "le " + safeType;
    d.innerHTML = "<span class=\"ts\">" + ts + "</span><span>" + esc(msg) + "</span>";
    agentLog.appendChild(d);
    agentLog.scrollTop = agentLog.scrollHeight;
    if (agentLog.children.length > 100) agentLog.firstChild && agentLog.firstChild.parentNode && agentLog.firstChild.parentNode.removeChild(agentLog.firstChild);
  }

  function showLoading(url) { loadingOverlay.style.display = "flex"; loadingText.textContent = "Loading " + (url || "").slice(0, 55) + "..."; }
  function hideLoading() { if (loadingOverlay) loadingOverlay.style.display = "none"; }

  function updateUrl(url) {
    var tab = getActiveTab();
    if (tab) { tab.url = url; }
    if (urlInput) urlInput.value = url;
    updateStatus();
    renderTabs();
    notifyParent("urlChange", { url: url, tabId: activeTabId });
  }

  function updateStatus() {
    var tab = getActiveTab();
    var url = tab ? tab.url : "";
    if (statusText) statusText.textContent = url || "about:blank";
    if (document.getElementById("back-btn")) document.getElementById("back-btn").disabled = !tab || tab.histIdx <= 0;
    if (document.getElementById("fwd-btn")) document.getElementById("fwd-btn").disabled = !tab || tab.histIdx >= tab.history.length - 1;
  }

  function addToHistory(url) {
    var tab = getActiveTab();
    if (!tab) return;
    if (tab.history[tab.histIdx] !== url) {
      tab.history = tab.history.slice(0, tab.histIdx + 1);
      tab.history.push(url);
      tab.histIdx = tab.history.length - 1;
    }
  }

  function notifyParent(type, payload) {
    try { var target = window.parent !== window ? window.parent : window.opener; if (target) target.postMessage({ autoBrowser: true, type: type, payload: payload }, "*"); } catch(e) {}
  }

  // ─── Blob URL loading — enables JavaScript execution in proxy-fetched pages ───
  var _tabBlobUrls = {}; // tabId → blobUrl, for cleanup
  function loadHtmlAsBlobUrl(html, tabId) {
    // Revoke previous blob URL for this tab to prevent memory leaks
    if (_tabBlobUrls[tabId]) {
      try { URL.revokeObjectURL(_tabBlobUrls[tabId]); } catch(e) {}
      delete _tabBlobUrls[tabId];
    }
    var blob = new Blob([html], { type: "text/html;charset=utf-8" });
    var blobUrl = URL.createObjectURL(blob);
    _tabBlobUrls[tabId] = blobUrl;
    iframe.removeAttribute("srcdoc");
    iframe.src = blobUrl;
  }
  function revokeTabBlobUrl(tabId) {
    if (_tabBlobUrls[tabId]) {
      try { URL.revokeObjectURL(_tabBlobUrls[tabId]); } catch(e) {}
      delete _tabBlobUrls[tabId];
    }
  }

  function navigateTo(url, replyId, targetTabId, _archiveFallback) {
    if (!url) return;
    if (!/^https?:\/\//i.test(url)) url = "https://" + url;

    // If a specific tab is targeted, switch to it
    if (targetTabId && targetTabId !== activeTabId) {
      var targetTab = tabs.find(function(t) { return t.id === targetTabId; });
      if (targetTab) switchTab(targetTabId);
    }

    var tab = getActiveTab();
    var isDirectMode = tab ? tab.directMode : false;

    // ─── Auto-detect SPA sites → use Jina Reader (direct mode fails due to X-Frame-Options) ───
    if (!isDirectMode && !_archiveFallback && isDomainInList(url, SPA_DOMAINS)) {
      addLog("SPA detected (" + getDomain(url) + ") → loading via Jina Reader (headless Chrome)", "nav");
      tryJinaReaderFallback(url, replyId);
      return;
    }

    // ─── Direct mode: load URL directly in iframe (full JS support) ───
    if (isDirectMode) {
      showLoading(url);
      addLog("[Tab " + activeTabId + "] Direct: " + url.slice(0, 55), "nav");
      var directErrorTimer = null;
      var dtid = setTimeout(function() {
        iframe.onload = null; iframe.onerror = null;
        hideLoading();
        addLog("Direct load timed out — falling back to proxy mode", "err");
        // Timeout likely means the page was blocked or hung — try proxy mode
        if (!_archiveFallback) {
          var tab = getActiveTab();
          if (tab) { tab.directMode = false; updateDirectBtn(); }
          navigateTo(url, replyId, null, false);
        } else {
          updateUrl(url);
          addToHistory(url);
          if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: false, url: url, error: "Direct load timed out" } });
        }
      }, 15000);

      function onDirectLoadDone() {
        clearTimeout(dtid);
        hideLoading(); updateUrl(url);
        addToHistory(url);
        // Check if the page actually loaded (CSP/X-Frame-Options may block it silently)
        var pageBlocked = false;
        try {
          var doc = iframe.contentDocument;
          if (doc && doc.title) { getActiveTab().title = doc.title; renderTabs(); }
          // If we can access the document and it's essentially empty, the load was likely blocked
          if (doc && doc.body && doc.body.innerHTML.length < 10 && !doc.title) pageBlocked = true;
          // Also detect browser error pages (about:blank, chrome-error://, etc.)
          if (doc && doc.body) {
            var bodyText = doc.body.innerText || "";
            if (/refused to connect|blocked|ERR_BLOCKED_BY_RESPONSE/i.test(bodyText)) pageBlocked = true;
          }
        } catch(e) {
          // Cross-origin — could be a real page OR an X-Frame-Options block.
          // Try to detect by checking iframe dimensions and visibility heuristics.
          pageBlocked = false;
          try {
            // If the iframe loaded but shows the browser's built-in error page,
            // the contentWindow will exist but contentDocument will throw.
            // Check if we can communicate with the page via postMessage probe.
            if (iframe.contentWindow) {
              // Additional heuristic: check if the iframe has no visible content
              // by attempting to get its computed dimensions
              var rect = iframe.getBoundingClientRect();
              if (rect.width > 0 && rect.height > 0) {
                // Looks like it loaded something — try probing for content
                pageBlocked = false;
              }
            }
          } catch(e2) { /* ignore */ }
        }

        if (pageBlocked && !_archiveFallback) {
          addLog("Direct load blocked (X-Frame-Options/CSP) — falling back to proxy mode", "err");
          // Fall back to PROXY MODE first (not Jina) — proxy mode works in embedded iframes
          var tab = getActiveTab();
          if (tab) { tab.directMode = false; updateDirectBtn(); }
          navigateTo(url, replyId, null, false);
          return;
        }

        tryInjectCtrlDirect();
        addLog("Loaded (direct): " + url.slice(0, 55), "ok");
        if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, direct: true } });
      }

      iframe.onload = onDirectLoadDone;
      iframe.onerror = function() {
        clearTimeout(dtid);
        hideLoading();
        addLog("Direct load error — falling back to proxy mode", "err");
        if (!_archiveFallback) {
          // Fall back to PROXY MODE first instead of Jina Reader
          var tab = getActiveTab();
          if (tab) { tab.directMode = false; updateDirectBtn(); }
          navigateTo(url, replyId, null, false);
        }
        else { showErrorPage(url, "Failed to load in direct mode", replyId); }
      };
      iframe.removeAttribute("srcdoc");
      iframe.src = url;
      return;
    }

    // ─── Proxy mode: fetch via CORS proxy, rewrite assets, inject into iframe ───
    showLoading(url);
    addLog("[Tab " + activeTabId + "] Navigate: " + url.slice(0, 55), "nav");
    var rawProxies = (cfg.proxies && cfg.proxies.length) ? cfg.proxies : [{ base: cfg.proxy, encode: true }];
    var proxies = rawProxies.map(function(p) { return typeof p === "string" ? { base: p, encode: true } : p; });

    var settled = false;
    var controllers = proxies.map(function() { return new AbortController(); });
    var pending = proxies.length;
    var lastErr = new Error("All CORS proxies failed");

    function onSuccess(html) {
      if (settled) return;
      if (!html || html.length < 50) { onFail(new Error("Empty response")); return; }
      settled = true;
      clearTimeout(navTimeout);
      controllers.forEach(function(c) { try { c.abort(); } catch(e) {} });
      html = rewriteHtml(html, url);

      // Async: fetch and inline external CSS for complete visual rendering
      addLog("Inlining stylesheets for full rendering...", "nav");
      fetchAndInlineCss(html, url, function(processedHtml) {
        processedHtml = injectCtrl(processedHtml);
        processedHtml = injectResourceHelper(processedHtml);

        var ltid = setTimeout(function() {
          iframe.onload = null;
          hideLoading();
          addLog("Timeout \u2014 page rendering stalled", "err");
          if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: false, error: "Page render timeout" } });
        }, 18000);
        iframe.onload = function() {
          clearTimeout(ltid);
          hideLoading(); updateUrl(url);
          addToHistory(url);
          // Extract title from loaded content
          try {
            var doc = iframe.contentDocument;
            if (doc && doc.title) { getActiveTab().title = doc.title; renderTabs(); }
          } catch(e) {}
          addLog("Loaded: " + url.slice(0, 55), "ok");
          if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url } });
        };
        // Use blob URL instead of srcdoc — this enables JavaScript execution
        // because blob URLs get the parent's origin (not null like srcdoc)
        loadHtmlAsBlobUrl(processedHtml, activeTabId);
      });
    }

    function onFail(err) {
      if (settled) return;
      if (err && err.name !== "AbortError") lastErr = err;
      pending--;
      if (pending > 0) return;
      settled = true;
      clearTimeout(navTimeout);
      var msg = lastErr.name === "AbortError" ? "Request timed out" : (lastErr.message || "Unknown error");
      hideLoading(); addLog("Proxy error: " + msg, "err");

      // All proxy requests failed — try Jina Reader (headless Chrome renders the page,
      // bypasses most anti-bot protections since it uses real browser fingerprints)
      addLog("All proxies failed → trying Jina Reader (headless Chrome)...", "nav");
      tryJinaReaderFallback(url, replyId);
    }

    var navTimeout = setTimeout(function() {
      if (!settled) {
        settled = true;
        controllers.forEach(function(c) { try { c.abort(); } catch(e) {} });
        hideLoading();

        if (isDomainInList(url, PROXY_HOSTILE_DOMAINS)) {
          addLog("Proxy timeout on hostile domain → trying Jina Reader...", "err");
          tryJinaReaderFallback(url, replyId);
          return;
        }

        addLog("Navigation timeout — trying Jina Reader fallback...", "err");
        tryJinaReaderFallback(url, replyId);
      }
    }, 20000);

    proxies.forEach(function(proxy, i) {
      var tid = setTimeout(function() { try { controllers[i].abort(); } catch(e) {} }, 12000);
      var proxyUrl = proxy.base + (proxy.encode ? encodeURIComponent(url) : url);
      fetch(proxyUrl, { signal: controllers[i].signal, cache: "no-store" })
        .then(function(r) {
          clearTimeout(tid);
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.text();
        })
        .then(function(html) { onSuccess(html); })
        .catch(function(e) { clearTimeout(tid); onFail(e); });
    });
  }

  // ─── Try loading an archived version via Wayback Machine ───
  function tryArchiveFallback(url, replyId) {
    addLog("Trying Wayback Machine archive for: " + url.slice(0, 50), "nav");
    var archiveUrl = "https://web.archive.org/web/2024if_/" + url;
    showLoading("archive of " + url);
    var tid = setTimeout(function() {
      iframe.onload = null;
      hideLoading();
      addLog("Archive load timed out — trying Jina Reader...", "err");
      tryJinaReaderFallback(url, replyId);
    }, 15000);
    iframe.onload = function() {
      clearTimeout(tid);
      hideLoading();
      updateUrl(url);
      addToHistory(url);
      addLog("Loaded archived version: " + url.slice(0, 50), "ok");
      if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, archived: true } });
    };
    iframe.removeAttribute("srcdoc");
    iframe.src = archiveUrl;
  }

  // ─── Jina Reader fallback — renders any page via headless Chrome ───
  // Strategy: First try to get REAL HTML from Jina (x-respond-with: html), render like proxy mode.
  // If that fails, fall back to markdown text and render as styled readable page.
  function tryJinaReaderFallback(url, replyId) {
    addLog("Trying Jina Reader (headless Chrome) for: " + url.slice(0, 50), "nav");
    showLoading("Jina Reader: " + url);
    var jinaUrl = "https://r.jina.ai/" + url;

    var settled = false;
    var jinaTimeout = setTimeout(function() {
      if (!settled) {
        settled = true;
        hideLoading();
        addLog("Jina Reader timed out", "err");
        // Last resort: try Google Cache before giving up
        tryGoogleCacheFallback(url, replyId);
      }
    }, 30000);

    // ── Phase 1: Try direct fetch to Jina with x-respond-with: html header ──
    // This returns the FULL rendered HTML of the page (like a real browser would see)
    function tryJinaHtml() {
      var ctrl = new AbortController();
      var tid = setTimeout(function() { ctrl.abort(); }, 18000);
      fetch(jinaUrl, {
        signal: ctrl.signal,
        headers: {
          "x-respond-with": "html",
          "Accept": "text/html",
          "x-timeout": "15",
        },
        cache: "no-store",
      })
      .then(function(r) {
        clearTimeout(tid);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function(html) {
        if (settled) return;
        // Validate it's actual HTML (not an error page or empty)
        if (!html || html.length < 100 || !/<(?:html|head|body|div|main|article)/i.test(html)) {
          addLog("Jina HTML response too small or not HTML, trying markdown fallback...", "nav");
          tryJinaMarkdown();
          return;
        }
        settled = true;
        clearTimeout(jinaTimeout);
        hideLoading();
        updateUrl(url);
        addToHistory(url);

        // Process the real HTML just like proxy mode — rewrite URLs, inline CSS, inject controls
        addLog("Got real HTML from Jina Reader, rendering site...", "nav");
        html = rewriteHtml(html, url);
        fetchAndInlineCss(html, url, function(processedHtml) {
          processedHtml = injectCtrl(processedHtml);
          processedHtml = injectResourceHelper(processedHtml);

          iframe.onload = function() {
            try {
              var doc = iframe.contentDocument;
              if (doc && doc.title) { getActiveTab().title = doc.title; renderTabs(); }
            } catch(e) {}
            getActiveTab().title = getActiveTab().title || getDomain(url);
            renderTabs();
          };
          // Use blob URL for Jina HTML too — enables JavaScript execution
          loadHtmlAsBlobUrl(processedHtml, activeTabId);
          addLog("Loaded real HTML via Jina Reader: " + url.slice(0, 50), "ok");
          if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, jinaReader: true, htmlMode: true } });
        });
      })
      .catch(function(e) {
        clearTimeout(tid);
        if (settled) return;
        addLog("Jina HTML direct fetch failed (" + (e.message || "error") + "), trying markdown...", "nav");
        tryJinaMarkdown();
      });
    }

    // ── Phase 2: Fall back to markdown via CORS proxies (original behavior) ──
    function tryJinaMarkdown() {
      if (settled) return;
      var proxies = (cfg.proxies && cfg.proxies.length) ? cfg.proxies : [{ base: cfg.proxy, encode: true }];
      proxies = proxies.map(function(p) { return typeof p === "string" ? { base: p, encode: true } : p; });
      var controllers = proxies.map(function() { return new AbortController(); });
      var pending = proxies.length;

      // Also try direct Jina fetch for markdown (no CORS proxy needed)
      pending++;
      var directCtrl = new AbortController();
      var directTid = setTimeout(function() { directCtrl.abort(); }, 20000);
      fetch(jinaUrl, {
        signal: directCtrl.signal,
        headers: { "Accept": "text/plain" },
        cache: "no-store",
      })
      .then(function(r) {
        clearTimeout(directTid);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .then(function(text) { onMarkdownSuccess(text); })
      .catch(function() { clearTimeout(directTid); onMarkdownFail(); });

      function onMarkdownSuccess(text) {
        if (settled) return;
        if (!text || text.length < 50) { onMarkdownFail(); return; }
        settled = true;
        clearTimeout(jinaTimeout);
        controllers.forEach(function(c) { try { c.abort(); } catch(e) {} });
        try { directCtrl.abort(); } catch(e) {}
        hideLoading();
        updateUrl(url);
        addToHistory(url);

        // Render the Jina markdown/text as a rich readable page in the iframe
        var renderedContent = renderMarkdownToHtml(text);

        var readerHtml = "<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><style>"
          + "*{box-sizing:border-box}"
          + "body{background:#07070b;color:#bbc;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:20px 30px 40px;font-size:14px;line-height:1.7;max-width:900px;margin:0 auto}"
          + "a{color:#88bbcc;text-decoration:underline} a:hover{color:#aaddee}"
          + "img{max-width:100%;height:auto;border-radius:8px;margin:8px 0;display:block}"
          + "pre{white-space:pre-wrap;word-break:break-word}"
          + "table{border-collapse:collapse} th,td{border:1px solid rgba(136,187,204,0.15);padding:8px 12px}"
          + "blockquote{border-left:3px solid rgba(136,187,204,0.3);padding:8px 16px;margin:8px 0;color:#99a}"
          + "hr{border:none;border-top:1px solid rgba(136,187,204,0.15);margin:16px 0}"
          + ".jina-badge{position:fixed;top:8px;right:12px;background:rgba(136,187,204,0.12);border:1px solid rgba(136,187,204,0.25);border-radius:6px;padding:3px 10px;font-size:10px;color:#88bbcc;font-family:monospace;z-index:100}"
          + "::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-thumb{background:rgba(136,187,204,0.2);border-radius:3px}::-webkit-scrollbar-track{background:transparent}"
          + "</style></head><body>"
          + "<div class='jina-badge'>Jina Reader (text)</div>"
          + renderedContent
          + "<" + "script>" + IFRAME_CTRL + "</" + "script>"
          + "</body></html>";

        iframe.onload = function() {
          getActiveTab().title = "Jina: " + getDomain(url);
          renderTabs();
        };
        iframe.removeAttribute("src");
        iframe.srcdoc = readerHtml;
        addLog("Loaded via Jina Reader (markdown): " + url.slice(0, 50), "ok");
        if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, jinaReader: true } });
      }

      function onMarkdownFail() {
        if (settled) return;
        pending--;
        if (pending > 0) return;
        settled = true;
        clearTimeout(jinaTimeout);
        hideLoading();
        addLog("Jina Reader failed (all methods)", "err");
        // Try Google Cache as last resort
        tryGoogleCacheFallback(url, replyId);
      }

      proxies.forEach(function(proxy, i) {
        var tid = setTimeout(function() { try { controllers[i].abort(); } catch(e) {} }, 20000);
        var proxyUrl = proxy.base + (proxy.encode ? encodeURIComponent(jinaUrl) : jinaUrl);
        fetch(proxyUrl, { signal: controllers[i].signal, cache: "no-store" })
          .then(function(r) {
            clearTimeout(tid);
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.text();
          })
          .then(function(text) { onMarkdownSuccess(text); })
          .catch(function(e) { clearTimeout(tid); onMarkdownFail(); });
      });
    }

    // Start with HTML mode first
    tryJinaHtml();
  }

  // ─── Google Cache fallback — last resort for blocked sites ───
  function tryGoogleCacheFallback(url, replyId) {
    addLog("Trying Google Cache for: " + url.slice(0, 50), "nav");
    showLoading("Google Cache: " + url);
    var cacheUrl = "https://webcache.googleusercontent.com/search?q=cache:" + encodeURIComponent(url) + "&strip=0";

    var settled = false;
    var tid = setTimeout(function() {
      if (!settled) {
        settled = true;
        hideLoading();
        showErrorPage(url, "All loading methods failed (proxy, direct, Jina Reader, Google Cache)", replyId);
      }
    }, 15000);

    // Try loading Google Cache directly in iframe (Google Cache sets no X-Frame-Options)
    iframe.onload = function() {
      if (settled) return;
      settled = true;
      clearTimeout(tid);
      hideLoading();
      updateUrl(url);
      addToHistory(url);
      getActiveTab().title = "Cache: " + getDomain(url);
      renderTabs();
      addLog("Loaded via Google Cache: " + url.slice(0, 50), "ok");
      if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, cached: true } });
    };
    iframe.onerror = function() {
      if (settled) return;
      settled = true;
      clearTimeout(tid);
      hideLoading();

      // Final fallback: try fetching via CORS proxy
      fetchWithProxyRace(cacheUrl, 12000).then(function(html) {
        if (html && html.length > 200) {
          html = rewriteHtml(html, url);
          html = injectCtrl(html);
          // Use blob URL for JS support
          loadHtmlAsBlobUrl(html, activeTabId);
          updateUrl(url);
          addToHistory(url);
          getActiveTab().title = "Cache: " + getDomain(url);
          renderTabs();
          addLog("Loaded via Google Cache (proxy): " + url.slice(0, 50), "ok");
          if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: true, url: url, cached: true } });
        } else {
          showErrorPage(url, "All loading methods failed (proxy, direct, Jina Reader, Google Cache)", replyId);
        }
      }).catch(function() {
        showErrorPage(url, "All loading methods failed (proxy, direct, Jina Reader, Google Cache)", replyId);
      });
    };
    iframe.removeAttribute("srcdoc");
    iframe.src = cacheUrl;
  }

  // ─── Markdown to HTML renderer (extracted for reuse) ───
  function renderMarkdownToHtml(text) {
    var lines = text.split("\n");
    var htmlParts = [];
    var inCodeBlock = false;
    var codeBlockLang = "";
    var codeLines = [];
    var inTable = false;
    var tableRows = [];

    function escHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

    function processInline(line) {
      line = line.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function(m, alt, src) {
        return "<img src='" + src.replace(/'/g, "&#39;") + "' alt='" + escHtml(alt) + "' style='max-width:100%;height:auto;border-radius:8px;margin:8px 0;display:block' onerror=\"this.style.display='none'\">";
      });
      line = line.replace(/\[([^\]]+)\]\(([^)]+)\)/g, "<a href='$2' style='color:#88bbcc;text-decoration:underline' target='_blank'>$1</a>");
      line = line.replace(/\*\*(.+?)\*\*/g, "<strong style='color:#dde'>$1</strong>");
      line = line.replace(/(?:^|[^*])\*([^*]+?)\*(?:[^*]|$)/g, function(m, content) { return m.replace("*" + content + "*", "<em>" + content + "</em>"); });
      line = line.replace(/`([^`]+)`/g, "<code style='background:rgba(136,187,204,0.1);padding:1px 5px;border-radius:3px;font-family:\"JetBrains Mono\",monospace;font-size:0.9em;color:#9cc'>$1</code>");
      line = line.replace(/~~(.+?)~~/g, "<del>$1</del>");
      return line;
    }

    for (var li = 0; li < lines.length; li++) {
      var rawLine = lines[li];
      if (/^```/.test(rawLine)) {
        if (inCodeBlock) {
          htmlParts.push("<pre style='background:#0d0d14;border:1px solid rgba(136,187,204,0.15);border-radius:8px;padding:14px 18px;overflow-x:auto;margin:12px 0;font-family:\"JetBrains Mono\",monospace;font-size:12px;line-height:1.6;color:#bcd'>" + escHtml(codeLines.join("\n")) + "</pre>");
          codeLines = [];
          inCodeBlock = false;
        } else {
          inCodeBlock = true;
          codeBlockLang = rawLine.slice(3).trim();
        }
        continue;
      }
      if (inCodeBlock) { codeLines.push(rawLine); continue; }
      if (/^\|.*\|/.test(rawLine)) {
        if (/^\|[\s\-:|]+\|$/.test(rawLine.trim())) { continue; }
        var cells = rawLine.split("|").filter(function(c, idx, arr) { return idx > 0 && idx < arr.length - 1; });
        if (!inTable) { inTable = true; tableRows = []; }
        tableRows.push(cells);
        var nextLine = li + 1 < lines.length ? lines[li + 1] : "";
        if (!/^\|.*\|/.test(nextLine)) {
          var tableHtml = "<table style='border-collapse:collapse;width:100%;margin:12px 0;font-size:13px'>";
          for (var ri = 0; ri < tableRows.length; ri++) {
            var tag = ri === 0 ? "th" : "td";
            var bgStyle = ri === 0 ? "background:rgba(136,187,204,0.08);" : (ri % 2 === 0 ? "background:rgba(255,255,255,0.02);" : "");
            tableHtml += "<tr>";
            for (var ci = 0; ci < tableRows[ri].length; ci++) {
              tableHtml += "<" + tag + " style='border:1px solid rgba(136,187,204,0.15);padding:8px 12px;text-align:left;" + bgStyle + "'>" + processInline(escHtml(tableRows[ri][ci].trim())) + "</" + tag + ">";
            }
            tableHtml += "</tr>";
          }
          tableHtml += "</table>";
          htmlParts.push(tableHtml);
          inTable = false;
          tableRows = [];
        }
        continue;
      }
      var trimmed = rawLine.trim();
      if (!trimmed) { htmlParts.push("<div style='height:12px'></div>"); continue; }
      if (/^#### (.+)/.test(trimmed)) { htmlParts.push("<h4 style='color:#9ab;margin:14px 0 6px;font-size:14px'>" + processInline(escHtml(trimmed.slice(5))) + "</h4>"); continue; }
      if (/^### (.+)/.test(trimmed)) { htmlParts.push("<h3 style='color:#88bbcc;margin:16px 0 8px;font-size:15px'>" + processInline(escHtml(trimmed.slice(4))) + "</h3>"); continue; }
      if (/^## (.+)/.test(trimmed)) { htmlParts.push("<h2 style='color:#9bd;margin:20px 0 10px;font-size:18px'>" + processInline(escHtml(trimmed.slice(3))) + "</h2>"); continue; }
      if (/^# (.+)/.test(trimmed)) { htmlParts.push("<h1 style='color:#ade;margin:24px 0 12px;font-size:22px'>" + processInline(escHtml(trimmed.slice(2))) + "</h1>"); continue; }
      if (/^[-*_]{3,}\s*$/.test(trimmed)) { htmlParts.push("<hr style='border:none;border-top:1px solid rgba(136,187,204,0.15);margin:16px 0'>"); continue; }
      if (/^>\s?(.*)/.test(trimmed)) {
        var quoteText = trimmed.replace(/^>\s?/, "");
        htmlParts.push("<blockquote style='border-left:3px solid rgba(136,187,204,0.3);padding:8px 16px;margin:8px 0;color:#99a;background:rgba(136,187,204,0.04);border-radius:0 6px 6px 0'>" + processInline(escHtml(quoteText)) + "</blockquote>");
        continue;
      }
      if (/^\d+\.\s+(.+)/.test(trimmed)) {
        var olContent = trimmed.replace(/^\d+\.\s+/, "");
        htmlParts.push("<div style='padding-left:20px;margin:3px 0;display:flex;gap:6px'><span style='color:#667;flex-shrink:0'>" + trimmed.match(/^\d+/)[0] + ".</span><span>" + processInline(escHtml(olContent)) + "</span></div>");
        continue;
      }
      if (/^[-*+]\s+(.+)/.test(trimmed)) {
        var ulContent = trimmed.replace(/^[-*+]\s+/, "");
        htmlParts.push("<div style='padding-left:20px;margin:3px 0;display:flex;gap:8px'><span style='color:#88bbcc;flex-shrink:0'>&bull;</span><span>" + processInline(escHtml(ulContent)) + "</span></div>");
        continue;
      }
      htmlParts.push("<p style='margin:4px 0;line-height:1.7'>" + processInline(escHtml(trimmed)) + "</p>");
    }
    if (inCodeBlock && codeLines.length > 0) {
      htmlParts.push("<pre style='background:#0d0d14;border:1px solid rgba(136,187,204,0.15);border-radius:8px;padding:14px 18px;overflow-x:auto;margin:12px 0;font-family:\"JetBrains Mono\",monospace;font-size:12px;line-height:1.6;color:#bcd'>" + escHtml(codeLines.join("\n")) + "</pre>");
    }
    return htmlParts.join("\n");
  }

  function showErrorPage(url, msg, replyId) {
    var domain = getDomain(url);
    var diagnosis = "";
    if (msg.indexOf("403") >= 0) diagnosis = "This site actively blocks proxy/bot requests.";
    else if (msg.indexOf("timeout") >= 0 || msg.indexOf("Timeout") >= 0) diagnosis = "The site took too long to respond through proxies.";
    else if (msg.indexOf("CORS") >= 0 || msg.indexOf("cors") >= 0) diagnosis = "CORS proxies were blocked by the site's security policy.";
    else if (msg.indexOf("frame") >= 0 || msg.indexOf("Frame") >= 0 || msg.indexOf("CSP") >= 0) diagnosis = "The site blocks iframe embedding via Content Security Policy.";
    else diagnosis = "The site could not be reached through any available method.";

    var errHtml = "<!DOCTYPE html><html><body style='background:#07070b;color:#cc7777;font-family:monospace;padding:30px;font-size:13px'>"
      + "<h2 style='margin:0 0 10px;color:#e88'>Failed to load page</h2>"
      + "<p style='color:#888;word-break:break-all;margin-bottom:8px'>" + esc(url) + "</p>"
      + "<p style='color:#cc7777'>" + esc(msg) + "</p>"
      + "<p style='color:#888;margin-top:8px;font-size:11px'>" + esc(diagnosis) + "</p>"
      + "<p style='color:#555;margin-top:12px;font-size:11px'>Fallback chain: CORS proxy → Direct mode → Jina Reader (HTML) → Jina Reader (text) → Google Cache</p>"
      + "<div style='margin-top:16px;display:flex;gap:8px;flex-wrap:wrap'>"
      + "<button onclick=\"window.parent.postMessage({autoBrowserAction:'tryJinaReader',url:'" + esc(url).replace(/'/g, "\\'") + "'},'*')\" style='padding:6px 14px;background:rgba(200,160,255,0.12);border:1px solid rgba(200,160,255,0.3);border-radius:5px;color:#c8a0ff;cursor:pointer;font-size:11px;font-family:monospace'>Try Jina Reader</button>"
      + "<button onclick=\"window.parent.postMessage({autoBrowserAction:'tryArchive',url:'" + esc(url).replace(/'/g, "\\'") + "'},'*')\" style='padding:6px 14px;background:rgba(136,187,204,0.15);border:1px solid rgba(136,187,204,0.3);border-radius:5px;color:#88bbcc;cursor:pointer;font-size:11px;font-family:monospace'>Try Archived Version</button>"
      + "<button onclick=\"window.parent.postMessage({autoBrowserAction:'tryGoogleCache',url:'" + esc(url).replace(/'/g, "\\'") + "'},'*')\" style='padding:6px 14px;background:rgba(255,200,100,0.1);border:1px solid rgba(255,200,100,0.3);border-radius:5px;color:#ddb050;cursor:pointer;font-size:11px;font-family:monospace'>Try Google Cache</button>"
      + "<button onclick=\"window.parent.postMessage({autoBrowserAction:'tryDirect',url:'" + esc(url).replace(/'/g, "\\'") + "'},'*')\" style='padding:6px 14px;background:rgba(124,224,138,0.1);border:1px solid rgba(124,224,138,0.3);border-radius:5px;color:#7ce08a;cursor:pointer;font-size:11px;font-family:monospace'>Retry Direct Mode</button>"
      + "</div>"
      + "</body></html>";
    iframe.onload = null;
    iframe.srcdoc = errHtml;
    if (replyId != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: replyId, payload: { success: false, error: msg, diagnosis: diagnosis } });
  }

  // ─── Comprehensive HTML rewriting for proxy mode ───
  function rewriteHtml(html, pageUrl) {
    try {
      var u = new URL(pageUrl);
      var origin = u.origin;
      var basePath = origin + u.pathname.split("/").slice(0, -1).join("/") + "/";

      // ── Strip Content-Security-Policy meta tags (blocks asset loading in proxy mode) ──
      html = html.replace(/<meta[^>]*http-equiv\s*=\s*["']?Content-Security-Policy["']?[^>]*>/gi, "");

      // ── Strip X-Frame-Options meta tags ──
      html = html.replace(/<meta[^>]*http-equiv\s*=\s*["']?X-Frame-Options["']?[^>]*>/gi, "");

      // ── Strip common frame-busting scripts ──
      // Remove inline scripts that check top !== self, window.top, parent.location, etc.
      html = html.replace(/<script[^>]*>[\s\S]*?(?:top\s*!==?\s*(?:self|window)|window\.top\.location|parent\.location|top\.location\s*=|frameElement|self\s*!==?\s*top)[\s\S]*?<\/script>/gi, "<!-- frame-bust removed -->");

      // ── Set referrer policy to allow resource loading ──
      html = html.replace(/<meta[^>]*name\s*=\s*["']?referrer["']?[^>]*>/gi, "");

      // ── Convert <link rel="preload" as="style"> to regular stylesheets ──
      // Modern sites use preload hints that won't auto-apply in srcdoc; convert them
      html = html.replace(/<link([^>]*rel\s*=\s*["']?\s*preload\s*["']?)([^>]*as\s*=\s*["']?\s*style\s*["']?[^>]*)>/gi, function(m, relPart, rest) {
        return '<link rel="stylesheet"' + rest.replace(/\bas\s*=\s*["']?\s*style\s*["']?/gi, '') + '>';
      });

      // ── Also convert <link type="text/css" ...> preloads with missing as="style" ──
      html = html.replace(/<link([^>]*rel\s*=\s*["']?\s*preload\s*["']?[^>]*type\s*=\s*["']text\/css["'][^>]*)>/gi, function(m, attrs) {
        if (!/as\s*=\s*["']style["']/i.test(attrs)) {
          return '<link rel="stylesheet"' + attrs.replace(/\brel\s*=\s*["']?preload["']?/i, '').replace(/\btype\s*=\s*["']text\/css["']/i, '') + '>';
        }
        return m;
      });

      // ── Strip crossorigin attribute from stylesheet <link> tags ──
      // Without crossorigin, CSS loads in no-cors mode (no CORS enforcement, CSS applies visually).
      // With crossorigin, CORS IS enforced and many CDNs will block unfamiliar origins.
      html = html.replace(/<link([^>]*rel\s*=\s*["']?\s*stylesheet\s*["']?[^>]*)>/gi, function(m, attrs) {
        var stripped = attrs.replace(/\bcrossorigin\s*(?:=\s*["']?[^"'\s>]*["']?)?/gi, '');
        if (stripped !== attrs) return '<link' + stripped + '>';
        return m;
      });

      // Remove existing <base> tags
      html = html.replace(/<base\s[^>]*>/gi, "");

      // Add our base tag
      var baseTag = '<base href="' + basePath + '">';
      var headMatch = html.match(/<head[^>]*>/i);
      if (headMatch) {
        html = html.replace(/<head[^>]*>/i, function(m) { return m + baseTag; });
      } else {
        html = '<head>' + baseTag + '</head>' + html;
      }

      // Rewrite relative URLs in src, href, action, poster, data attributes to absolute
      html = html.replace(/((?:src|href|action|poster|data)\s*=\s*["'])(?!data:|javascript:|blob:|#|mailto:|tel:|about:)((?:\/\/|\/|\.\.\/|\.\/|(?!https?:\/\/|\/\/))[^"']*)(["'])/gi, function(match, prefix, relUrl, suffix) {
        try {
          if (/^\/\//.test(relUrl)) return prefix + u.protocol + relUrl + suffix;
          var absUrl = new URL(relUrl, basePath).href;
          return prefix + absUrl + suffix;
        } catch(e) { return match; }
      });

      // Rewrite CSS url() references inside <style> blocks and style attributes
      html = html.replace(/url\(\s*["']?(?!data:|blob:|#|about:)((?:\/\/|\/|\.\.\/|\.\/|(?!https?:\/\/|\/\/))[^"')]+)["']?\s*\)/gi, function(match, relUrl) {
        try {
          relUrl = relUrl.trim();
          if (/^\/\//.test(relUrl)) return "url(" + u.protocol + relUrl + ")";
          var absUrl = new URL(relUrl, basePath).href;
          return "url(" + absUrl + ")";
        } catch(e) { return match; }
      });

      // Rewrite bare @import "..." and @import '...' references (without url())
      html = html.replace(/@import\s+["'](?!data:|blob:|https?:\/\/)(.*?)["']/gi, function(match, relUrl) {
        try {
          relUrl = relUrl.trim();
          if (/^\/\//.test(relUrl)) return '@import "' + u.protocol + relUrl + '"';
          var absUrl = new URL(relUrl, basePath).href;
          return '@import "' + absUrl + '"';
        } catch(e) { return match; }
      });

      // Rewrite srcset attributes (responsive images)
      html = html.replace(/srcset\s*=\s*["']([^"']+)["']/gi, function(match, srcsetVal) {
        try {
          var parts = srcsetVal.split(",").map(function(part) {
            var trimmed = part.trim().split(/\s+/);
            var imgUrl = trimmed[0];
            if (imgUrl && !/^https?:\/\/|^data:|^blob:/.test(imgUrl)) {
              try { imgUrl = new URL(imgUrl, basePath).href; } catch(e) {}
            }
            trimmed[0] = imgUrl;
            return trimmed.join(" ");
          });
          return 'srcset="' + parts.join(", ") + '"';
        } catch(e) { return match; }
      });

      // Rewrite <link rel="preload"> and <link rel="prefetch"> href to absolute
      html = html.replace(/<link([^>]*rel\s*=\s*["'](?:preload|prefetch|icon|shortcut icon|apple-touch-icon)["'][^>]*)>/gi, function(match, attrs) {
        return match.replace(/href\s*=\s*["'](?!data:|blob:|https?:\/\/|\/\/)(.*?)["']/i, function(hm, relUrl) {
          try {
            var absUrl = new URL(relUrl.trim(), basePath).href;
            return 'href="' + absUrl + '"';
          } catch(e) { return hm; }
        });
      });

      // ── Keep <noscript> blocks as-is — blob URL mode runs JS, so noscript should remain hidden ──
      // (No unwrapping needed — JavaScript executes normally in blob URL proxy mode)
      html = html.replace(/<noscript[^>]*>([\s\S]*?)<\/noscript>/gi, function(m, inner) {
        // Only unwrap if it contains critical stylesheets (some sites put styles in noscript)
        if (/<link[^>]*rel\s*=\s*["']?stylesheet/i.test(inner)) {
          return '<!-- noscript stylesheet unwrapped -->' + inner;
        }
        return m;
      });

      // Convert lazy-loaded images (data-src, data-lazy-src) to real src so they display
      html = html.replace(/<img([^>]*)\bdata-src\s*=\s*["']([^"']+)["']([^>]*)>/gi, function(match, before, dataSrc, after) {
        // Only add src if there's no existing real src (or src is a placeholder)
        if (/\bsrc\s*=\s*["'](?!data:)([^"']{10,})["']/i.test(before + after)) return match;
        try {
          var absSrc = /^https?:\/\//i.test(dataSrc) ? dataSrc : new URL(dataSrc, basePath).href;
          return '<img' + before + ' src="' + absSrc + '"' + after + '>';
        } catch(e) { return match; }
      });
      html = html.replace(/<img([^>]*)\bdata-lazy-src\s*=\s*["']([^"']+)["']([^>]*)>/gi, function(match, before, dataSrc, after) {
        if (/\bsrc\s*=\s*["'](?!data:)([^"']{10,})["']/i.test(before + after)) return match;
        try {
          var absSrc = /^https?:\/\//i.test(dataSrc) ? dataSrc : new URL(dataSrc, basePath).href;
          return '<img' + before + ' src="' + absSrc + '"' + after + '>';
        } catch(e) { return match; }
      });
      // Also convert data-srcset to srcset for lazy-loaded responsive images
      html = html.replace(/<img([^>]*)\bdata-srcset\s*=\s*["']([^"']+)["']([^>]*)>/gi, function(match, before, dataSrcset, after) {
        if (/\bsrcset\s*=/i.test(before + after)) return match;
        return '<img' + before + ' srcset="' + dataSrcset + '"' + after + '>';
      });
      // Convert loading="lazy" images — force eager loading since proxy pages don't scroll-trigger
      html = html.replace(/(<img[^>]*)\bloading\s*=\s*["']lazy["']/gi, '$1 loading="eager"');

      // Inject referrer policy — use origin so CDNs don't block but full path isn't leaked
      var metaReferrer = '<meta name="referrer" content="origin">';
      html = html.replace(/<head[^>]*>/i, function(m) { return m + metaReferrer; });

      // Ensure viewport meta exists (many sites assume one is present for responsive CSS)
      if (!/<meta[^>]*name\s*=\s*["']?viewport["']?/i.test(html)) {
        html = html.replace(/<head[^>]*>/i, function(m) { return m + '<meta name="viewport" content="width=device-width,initial-scale=1">'; });
      }

      // Inject helper styles — minimal, don't override site styles
      var helperStyle = '<style data-auto-helper>'
        + 'img[src=""],img:not([src]){display:none!important}'
        + 'img:not([width]):not([style*="width"]){max-width:100%;height:auto}'
        + '</style>';
      html = html.replace(/<\/head>/i, helperStyle + '</head>');

    } catch(e) {
      // Fallback: just add base tag
      try {
        var u2 = new URL(pageUrl);
        var base2 = u2.origin + u2.pathname.split("/").slice(0, -1).join("/") + "/";
        html = html.replace(/<base\s[^>]*>/gi, "");
        html = html.replace(/<head[^>]*>/i, function(m) { return m + '<base href="' + base2 + '">'; });
      } catch(e2) {}
    }
    return html;
  }

  function injectCtrl(html) {
    var scriptTag = "<scr" + "ipt>" + IFRAME_CTRL + "<\/scr" + "ipt>";
    // Inject before </body> for better compatibility (after page content loads)
    var replaced = html.replace(/<\/body>/i, scriptTag + "</body>");
    if (replaced !== html) return replaced;
    // Fallback: inject before </html>
    replaced = html.replace(/<\/html>/i, scriptTag + "</html>");
    if (replaced !== html) return replaced;
    // Last resort: append at end
    return html + scriptTag;
  }

  // ─── Fetch and inline external CSS for complete rendering in proxy mode ───
  // This ensures all stylesheets load even from srcdoc (null origin) iframes
  function fetchAndInlineCss(html, pageUrl, callback) {
    var u, origin, basePath;
    try {
      u = new URL(pageUrl);
      origin = u.origin;
      basePath = origin + u.pathname.split("/").slice(0, -1).join("/") + "/";
    } catch(e) { callback(html); return; }

    // Find all <link rel="stylesheet"> tags
    var linkPattern = /<link[^>]*rel\s*=\s*["']?\s*stylesheet\s*["']?[^>]*>/gi;
    var links = [];
    var match;
    while ((match = linkPattern.exec(html)) !== null) {
      var tag = match[0];
      var hrefMatch = tag.match(/href\s*=\s*["']([^"']+)["']/i);
      if (hrefMatch && hrefMatch[1]) {
        var href = hrefMatch[1].trim();
        // Resolve to absolute URL
        try {
          if (/^\/\//.test(href)) href = u.protocol + href;
          else if (!/^https?:\/\//i.test(href)) href = new URL(href, basePath).href;
        } catch(e) { continue; }
        if (/^https?:\/\//i.test(href)) {
          links.push({ tag: tag, href: href });
        }
      }
    }

    // Also find @import in inline <style> blocks and track those URLs
    var styleImportPattern = /<style[^>]*>([\s\S]*?)<\/style>/gi;
    var importUrls = [];
    while ((match = styleImportPattern.exec(html)) !== null) {
      var styleContent = match[1];
      var importMatch;
      var importRe = /@import\s+(?:url\(\s*["']?|["'])(https?:\/\/[^"')]+)["']?\)?/gi;
      while ((importMatch = importRe.exec(styleContent)) !== null) {
        importUrls.push(importMatch[1]);
      }
    }

    if (links.length === 0 && importUrls.length === 0) { callback(html); return; }

    // Limit to prevent too many fetches — increased for modern sites with many stylesheets
    var maxFetch = 50;
    var toFetch = links.slice(0, maxFetch);
    var remaining = toFetch.length + Math.min(importUrls.length, 20);
    var cssResults = {};
    var importResults = {};
    var finalized = false;

    function resolveCssUrls(cssText, cssUrl) {
      try {
        var cssBase = cssUrl.split("?")[0].split("/").slice(0, -1).join("/") + "/";
        var cssOrigin = new URL(cssUrl).origin;

        // Resolve url() references to absolute URLs
        cssText = cssText.replace(/url\(\s*["']?(?!data:|blob:|#|about:)(.*?)["']?\s*\)/gi, function(m, relUrl) {
          try {
            relUrl = relUrl.trim();
            if (!relUrl) return m;
            if (/^https?:\/\//i.test(relUrl)) return m; // Already absolute
            if (/^\/\//.test(relUrl)) return "url(" + u.protocol + relUrl + ")";
            var abs = new URL(relUrl, cssBase).href;
            return "url(" + abs + ")";
          } catch(e) { return m; }
        });

        // Resolve @import url() to absolute
        cssText = cssText.replace(/@import\s+url\(\s*["']?(?!data:|blob:|https?:\/\/)(.*?)["']?\s*\)/gi, function(m, relUrl) {
          try {
            var abs = new URL(relUrl.trim(), cssBase).href;
            return '@import url("' + abs + '")';
          } catch(e) { return m; }
        });

        // Resolve bare @import "..." to absolute
        cssText = cssText.replace(/@import\s+["'](?!data:|blob:|https?:\/\/)(.*?)["']/gi, function(m, relUrl) {
          try {
            var abs = new URL(relUrl.trim(), cssBase).href;
            return '@import "' + abs + '"';
          } catch(e) { return m; }
        });
      } catch(e) {}
      return cssText;
    }

    // ─── Proxy font & resource URLs inside CSS ───
    // Fonts in @font-face ALWAYS require CORS — proxy them through multiple fallback proxies
    // so if one proxy is down or rate-limited, others are tried automatically by the browser.
    var CSS_FONT_PROXIES = [
      "https://corsproxy.io/?url=",
      "https://api.allorigins.win/raw?url=",
      "https://corsproxy.org/?",
    ];
    function proxyCssResourceUrls(cssText) {
      // Proxy url() inside @font-face blocks using multiple proxy sources as CSS fallback list.
      // The browser tries each src in order, so proxy1 → proxy2 → original URL.
      cssText = cssText.replace(/@font-face\s*\{[^}]*\}/gi, function(fontBlock) {
        return fontBlock.replace(
          /url\(\s*["']?(https?:\/\/[^"')]+)["']?\s*\)(\s*format\s*\([^)]*\))?/gi,
          function(m, absUrl, fmtPart) {
            if (absUrl.indexOf("corsproxy") >= 0 || absUrl.indexOf("allorigins") >= 0 ||
                absUrl.indexOf("codetabs") >= 0 || absUrl.indexOf("cors.lol") >= 0) return m;
            var fmt = fmtPart ? (" " + fmtPart.trim()) : "";
            // Emit proxy sources + original URL as fallback
            return CSS_FONT_PROXIES.map(function(p) {
              return "url(" + p + encodeURIComponent(absUrl) + ")" + fmt;
            }).concat(["url(" + absUrl + ")" + fmt]).join(", ");
          }
        );
      });
      return cssText;
    }

    // Timeout: don't block page loading forever waiting for CSS
    var fetchTimeout = setTimeout(function() { finalize(); }, 15000);

    function finalize() {
      if (finalized) return;
      finalized = true;
      clearTimeout(fetchTimeout);

      // Replace <link> tags with inline <style> blocks (for successfully fetched CSS)
      for (var i = 0; i < toFetch.length; i++) {
        var link = toFetch[i];
        if (cssResults[link.href]) {
          var cssText = resolveCssUrls(cssResults[link.href], link.href);
          // Also inline any @import results we fetched
          cssText = cssText.replace(/@import\s+(?:url\(\s*["']?|["'])(https?:\/\/[^"')]+)["']?\)?\s*;?/gi, function(m, importUrl) {
            if (importResults[importUrl]) {
              return "/* inlined @import: " + importUrl.slice(0, 60) + " */\n" + resolveCssUrls(importResults[importUrl], importUrl);
            }
            return m;
          });
          // Proxy font URLs so they load correctly
          cssText = proxyCssResourceUrls(cssText);
          // Preserve media attribute from original <link> tag
          var mediaMatch = link.tag.match(/media\s*=\s*["']([^"']+)["']/i);
          var mediaAttr = mediaMatch ? ' media="' + mediaMatch[1] + '"' : '';
          var replacement = '<style data-inlined-from="' + link.href.slice(0, 120).replace(/"/g, '&quot;') + '"' + mediaAttr + '>' + cssText + '</style>';
          // Prefer exact string replacement (most reliable), fall back to regex
          var tagIdx = html.indexOf(link.tag);
          if (tagIdx !== -1) {
            html = html.slice(0, tagIdx) + replacement + html.slice(tagIdx + link.tag.length);
          } else {
            var safeTag = link.tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            html = html.replace(new RegExp(safeTag, 'i'), replacement);
          }
        }
        // If fetch failed, leave the <link> tag as-is — browser will try to load it directly
      }

      // Also inline @import results into existing inline <style> blocks
      if (Object.keys(importResults).length > 0) {
        html = html.replace(/@import\s+(?:url\(\s*["']?|["'])(https?:\/\/[^"')]+)["']?\)?\s*;?/gi, function(m, importUrl) {
          if (importResults[importUrl]) {
            return "/* inlined @import */\n" + resolveCssUrls(importResults[importUrl], importUrl);
          }
          return m;
        });
      }

      // Proxy font URLs in any remaining inline <style> blocks (not from inlined links)
      html = html.replace(/<style([^>]*)>([\s\S]*?)<\/style>/gi, function(m, attrs, content) {
        if (attrs.indexOf("data-auto-helper") >= 0) return m; // Skip our helper styles
        var proxied = proxyCssResourceUrls(content);
        if (proxied !== content) return '<style' + attrs + '>' + proxied + '</style>';
        return m;
      });

      callback(html);
    }

    function checkDone() {
      remaining--;
      if (remaining <= 0) finalize();
    }

    // Helper: fetch a URL through CORS proxy race
    function fetchCssViaProxy(cssUrl, onDone) {
      var proxies = (cfg.proxies && cfg.proxies.length) ? cfg.proxies : [{ base: cfg.proxy, encode: true }];
      proxies = proxies.map(function(p) { return typeof p === "string" ? { base: p, encode: true } : p; });

      var settled = false;
      var controllers = proxies.map(function() { return new AbortController(); });
      var pendingP = proxies.length;

      var tid = setTimeout(function() {
        if (!settled) { settled = true; controllers.forEach(function(c) { try { c.abort(); } catch(e) {} }); onDone(null); }
      }, 10000);

      proxies.forEach(function(proxy, pi) {
        var proxyUrl = proxy.base + (proxy.encode ? encodeURIComponent(cssUrl) : cssUrl);
        fetch(proxyUrl, { signal: controllers[pi].signal, cache: "no-store" })
          .then(function(r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.text(); })
          .then(function(css) {
            if (settled) return;
            var cssOk = false;
            if (css && css.length > 10) {
              var _h = css.slice(0, 600).toLowerCase();
              // Reject HTML error pages and JSON error responses
              cssOk = _h.indexOf("<!doctype") === -1
                && _h.indexOf("<html") === -1
                && _h.indexOf("<body") === -1
                && !(_h.trimLeft().charAt(0) === '{' || _h.trimLeft().charAt(0) === '[');
            }
            if (cssOk) {
              settled = true;
              clearTimeout(tid);
              controllers.forEach(function(c) { try { c.abort(); } catch(e) {} });
              onDone(css);
            } else {
              pendingP--;
              if (pendingP <= 0 && !settled) { settled = true; clearTimeout(tid); onDone(null); }
            }
          })
          .catch(function() {
            if (settled) return;
            pendingP--;
            if (pendingP <= 0) { settled = true; clearTimeout(tid); onDone(null); }
          });
      });
    }

    // Fetch all external stylesheets via CORS proxy
    toFetch.forEach(function(link) {
      fetchCssViaProxy(link.href, function(css) {
        if (css) {
          cssResults[link.href] = css;
          // Resolve relative URLs in the fetched CSS BEFORE extracting @imports —
          // this ensures relative @import paths (e.g. @import "../reset.css") are resolved
          // to absolute URLs so they can be fetched.
          var cssForScan = resolveCssUrls(css, link.href);
          // Extract @import URLs from fetched CSS and fetch those too (1 level deep)
          var importRe = /@import\s+(?:url\(\s*["']?|["'])(https?:\/\/[^"')]+)["']?\)?/gi;
          var impMatch;
          var newImports = [];
          while ((impMatch = importRe.exec(cssForScan)) !== null) {
            var impUrl = impMatch[1];
            if (!importResults[impUrl] && newImports.indexOf(impUrl) === -1) {
              newImports.push(impUrl);
            }
          }
          if (newImports.length > 0) {
            remaining += newImports.length;
            newImports.forEach(function(impUrl) {
              fetchCssViaProxy(impUrl, function(impCss) {
                if (impCss) importResults[impUrl] = impCss;
                checkDone();
              });
            });
          }
        }
        checkDone();
      });
    });

    // Fetch @import URLs found in inline <style> blocks
    importUrls.slice(0, 20).forEach(function(impUrl) {
      fetchCssViaProxy(impUrl, function(css) {
        if (css) importResults[impUrl] = css;
        checkDone();
      });
    });

    // If nothing to fetch, finalize immediately
    if (toFetch.length === 0 && importUrls.length === 0) finalize();
  }

  // ─── Resource helper script — handles failed resource loads via CORS proxy retry ───
  function injectResourceHelper(html) {
    var helperScript = "<scr" + "ipt>(function(){"
      + "var PROXIES=['https://corsproxy.io/?url=','https://api.allorigins.win/raw?url=','https://corsproxy.org/?'];"
      + "var retried={};"

      // Retry failed images/media via CORS proxy
      + "function retryViaProxy(el){"
      + "var src=el.getAttribute('data-original-src')||el.src||el.currentSrc;"
      + "if(!src||retried[src]||!/^https?:\\/\\//i.test(src))return;"
      + "retried[src]=1;"
      + "var idx=0;"
      + "function tryNext(){"
      + "if(idx>=PROXIES.length)return;"
      + "el.setAttribute('data-original-src',src);"
      + "el.onerror=tryNext;"
      + "el.src=PROXIES[idx]+encodeURIComponent(src);"
      + "idx++;"
      + "}"
      + "tryNext();"
      + "}"

      // Proxy font URLs in a CSS text block — emit multiple proxy fallbacks per font src
      + "function proxyCssFonts(css){"
      + "return css.replace(/@font-face\\s*\\{[^}]*\\}/gi,function(fb){"
      + "return fb.replace(/url\\(\\s*[\"']?(https?:\\/\\/[^\"')]+)[\"']?\\s*\\)(\\s*format\\s*\\([^)]*\\))?/gi,function(m,u,fmt){"
      + "if(u.indexOf('corsproxy')>=0||u.indexOf('allorigins')>=0||u.indexOf('codetabs')>=0)return m;"
      + "var f=fmt?(' '+fmt.trim()):'';"
      + "return PROXIES.map(function(p){return 'url('+p+encodeURIComponent(u)+')'+f;}).concat(['url('+u+')'+f]).join(', ');"
      + "});"
      + "});"
      + "}"

      // Listen for resource load errors globally
      + "document.addEventListener('error',function(e){"
      + "var el=e.target;"
      + "if(!el||!el.tagName)return;"
      + "var tag=el.tagName;"
      + "if((tag==='IMG'||tag==='VIDEO'||tag==='AUDIO'||tag==='SOURCE')&&!el.getAttribute('data-proxy-retried')){"
      + "el.setAttribute('data-proxy-retried','1');"
      + "retryViaProxy(el);"
      + "}"
      // Retry failed link stylesheets via fetch+inline (with font proxying)
      + "if(tag==='LINK'&&el.rel==='stylesheet'&&el.href&&!el.getAttribute('data-proxy-retried')){"
      + "el.setAttribute('data-proxy-retried','1');"
      + "var cssHref=el.href;"
      + "var idx2=0;"
      + "function tryFetchCss(){"
      + "if(idx2>=PROXIES.length)return;"
      + "fetch(PROXIES[idx2]+encodeURIComponent(cssHref),{cache:'no-store'})"
      + ".then(function(r){if(!r.ok)throw new Error();return r.text()})"
      + ".then(function(css){"
      + "var _h2=css?css.slice(0,400).toLowerCase():'';"
      + "var _ok2=css&&css.length>10&&_h2.indexOf('<!doctype')===-1&&_h2.indexOf('<html')===-1&&_h2.indexOf('<body')===-1&&!(_h2.trimLeft&&_h2.trimLeft().charAt(0)==='{');"
      + "if(_ok2){"
      + "css=proxyCssFonts(css);"
      + "var s=document.createElement('style');"
      + "var mediaAttr=el.getAttribute('media');"
      + "if(mediaAttr)s.setAttribute('media',mediaAttr);"
      + "s.textContent=css;"
      + "el.parentNode.insertBefore(s,el);"
      + "el.parentNode.removeChild(el);"
      + "}else{idx2++;tryFetchCss();}"
      + "})"
      + ".catch(function(){idx2++;tryFetchCss();});"
      + "}"
      + "tryFetchCss();"
      + "}"
      + "},true);"

      // MutationObserver: handle dynamically added elements
      + "if(typeof MutationObserver!=='undefined'){"
      + "var baseUri=document.baseURI||'';"
      + "var obs=new MutationObserver(function(muts){"
      + "muts.forEach(function(m){"
      + "if(m.addedNodes)for(var i=0;i<m.addedNodes.length;i++){"
      + "var n=m.addedNodes[i];"
      + "if(n.nodeType!==1)continue;"
      + "var imgs=n.tagName==='IMG'?[n]:(n.querySelectorAll?Array.prototype.slice.call(n.querySelectorAll('img[src]')):[]); "
      + "imgs.forEach(function(img){"
      + "if(img.src&&!/^https?:\\/\\/|^data:|^blob:/i.test(img.getAttribute('src'))){"
      + "try{img.src=new URL(img.getAttribute('src'),baseUri).href;}catch(e){}"
      + "}"
      + "});"
      + "var links=n.tagName==='LINK'?[n]:(n.querySelectorAll?Array.prototype.slice.call(n.querySelectorAll('link[rel=stylesheet]')):[]);"
      + "links.forEach(function(lnk){"
      + "if(lnk.href&&!/^https?:\\/\\//i.test(lnk.getAttribute('href'))){"
      + "try{lnk.href=new URL(lnk.getAttribute('href'),baseUri).href;}catch(e){}"
      + "}"
      + "});"
      // Proxy fonts in dynamically injected <style> blocks (CSS-in-JS, styled-components, etc.)
      + "var stags=n.tagName==='STYLE'?[n]:(n.querySelectorAll?n.querySelectorAll('style'):[]);"
      + "[].forEach.call(stags,function(st){"
      + "if(st.getAttribute('data-proxy-patched')||st.getAttribute('data-auto-helper'))return;"
      + "st.setAttribute('data-proxy-patched','1');"
      + "var txt=st.textContent||'';"
      + "if(/@font-face/.test(txt)){"
      + "var fixed=proxyCssFonts(txt);"
      + "if(fixed!==txt)st.textContent=fixed;"
      + "}"
      + "});"
      + "}"
      + "});"
      + "});"
      + "if(document.body)obs.observe(document.body,{childList:true,subtree:true});"
      + "else document.addEventListener('DOMContentLoaded',function(){if(document.body)obs.observe(document.body,{childList:true,subtree:true});});"
      + "}"

      // Preload: fetch any remaining link[rel=stylesheet] that hasn't loaded, via proxy
      + "function _retryUnloadedLinks(){"
      + "var styleLinks=document.querySelectorAll('link[rel=stylesheet]');"
      + "for(var i=0;i<styleLinks.length;i++){"
      + "var lnk=styleLinks[i];"
      + "if(lnk.sheet)continue;"
      + "if(!lnk.href||lnk.getAttribute('data-proxy-retried'))continue;"
      + "lnk.setAttribute('data-proxy-retried','1');"
      + "(function(el,href){"
      + "var idx3=0;"
      + "function tryFetch3(){"
      + "if(idx3>=PROXIES.length)return;"
      + "fetch(PROXIES[idx3]+encodeURIComponent(href),{cache:'no-store'})"
      + ".then(function(r){if(!r.ok)throw new Error();return r.text();})"
      + ".then(function(css){"
      + "var _h3=css?css.slice(0,400).toLowerCase():'';"
      + "var _ok3=css&&css.length>10&&_h3.indexOf('<!doctype')===-1&&_h3.indexOf('<html')===-1&&_h3.indexOf('<body')===-1&&!(_h3.trimLeft&&_h3.trimLeft().charAt(0)==='{');"
      + "if(_ok3){"
      + "css=proxyCssFonts(css);"
      + "var s=document.createElement('style');"
      + "var mediaAttr=el.getAttribute('media');"
      + "if(mediaAttr)s.setAttribute('media',mediaAttr);"
      + "s.textContent=css;"
      + "el.parentNode.insertBefore(s,el);"
      + "try{el.parentNode.removeChild(el);}catch(e){}"
      + "}else{idx3++;tryFetch3();}"
      + "})"
      + ".catch(function(){idx3++;tryFetch3();});"
      + "}"
      + "tryFetch3();"
      + "})(lnk,lnk.href);"
      + "}"
      + "}"  // end _retryUnloadedLinks

      // Also scan existing inline styles for un-proxied font URLs and fix them
      + "function _fixInlineFonts(){"
      + "var inlineStyles=document.querySelectorAll('style');"
      + "for(var j=0;j<inlineStyles.length;j++){"
      + "var st=inlineStyles[j];"
      + "if(st.getAttribute('data-auto-helper')||st.getAttribute('data-proxy-patched'))continue;"
      + "st.setAttribute('data-proxy-patched','1');"
      + "var txt=st.textContent||'';"
      + "if(/@font-face/.test(txt)){"
      + "var fixed=proxyCssFonts(txt);"
      + "if(fixed!==txt)st.textContent=fixed;"
      + "}"
      + "}"
      + "}"
      + "document.addEventListener('DOMContentLoaded',function(){"
      + "_retryUnloadedLinks();"
      + "_fixInlineFonts();"
      // Second pass after a short delay to catch late-loaded CSS and dynamic style injections
      + "setTimeout(function(){_retryUnloadedLinks();_fixInlineFonts();},2500);"
      + "});"

      + "})()</" + "script>";

    // Inject before </body> for best compatibility
    var replaced = html.replace(/<\/body>/i, helperScript + "</body>");
    if (replaced !== html) return replaced;
    replaced = html.replace(/<\/html>/i, helperScript + "</html>");
    if (replaced !== html) return replaced;
    return html + helperScript;
  }

  // Try to inject the control script into directly-loaded pages (for AI read in direct mode)
  function tryInjectCtrlDirect() {
    try {
      var doc = iframe.contentDocument;
      if (!doc || !doc.body) return;
      if (doc.querySelector("script[data-auto-ctrl]")) return;
      var s = doc.createElement("script");
      s.setAttribute("data-auto-ctrl", "1");
      s.textContent = IFRAME_CTRL;
      doc.body.appendChild(s);
      addLog("Injected AI read support into direct page", "ok");
    } catch(e) {
      // Cross-origin: can't inject. That's OK — we'll fall back
      addLog("Cannot inject into direct page (cross-origin)", "nav");
    }
  }

  function updateDirectBtn() {
    var tab = getActiveTab();
    var dm = tab ? tab.directMode : false;
    var dmBtn = document.getElementById("dm");
    if (!dmBtn) return;
    if (dm) {
      dmBtn.textContent = "Direct \u2713";
      dmBtn.style.color = "#7ce08a";
      dmBtn.style.borderColor = "rgba(124,224,138,0.3)";
      dmBtn.style.background = "rgba(124,224,138,0.1)";
    } else {
      dmBtn.textContent = "Direct";
      dmBtn.style.color = "#88bbcc";
      dmBtn.style.borderColor = "rgba(136,187,204,0.3)";
      dmBtn.style.background = "rgba(136,187,204,0.1)";
    }
  }

  function doGo() { var u = urlInput.value.trim(); if (u) navigateTo(u); }
  function goBack() {
    var tab = getActiveTab();
    if (tab && tab.histIdx > 0) { tab.histIdx--; navigateTo(tab.history[tab.histIdx]); }
  }
  function goForward() {
    var tab = getActiveTab();
    if (tab && tab.histIdx < tab.history.length - 1) { tab.histIdx++; navigateTo(tab.history[tab.histIdx]); }
  }
  function doReload() {
    var tab = getActiveTab();
    if (tab && tab.url) navigateTo(tab.url);
  }

  function toggleTakeover() {
    agentMode = !agentMode;
    if (!agentMode) {
      agentBadge.className = "badge inactive"; agentBadgeText.textContent = "AI PAUSED";
      takeoverBtn.textContent = "Resume AI"; takeoverBtn.className = "tbtn resume";
      statusMode.textContent = "USER MODE"; statusMode.style.color = "#cc7777";
      addLog("User took control \u2014 AI paused", "err");
      notifyParent("userTookOver", {});
    } else {
      agentBadge.className = "badge"; agentBadgeText.textContent = "AI AGENT";
      takeoverBtn.textContent = "Take Over"; takeoverBtn.className = "tbtn";
      statusMode.textContent = "AI MODE"; statusMode.style.color = "#7ce08a";
      addLog("AI control resumed", "ok");
      notifyParent("aiResumed", {});
    }
  }

  function togglePanel() {
    panelCollapsed = !panelCollapsed;
    agentPanel.classList.toggle("collapsed", panelCollapsed);
    panelToggle.textContent = panelCollapsed ? "\u25c2" : "\u25b8";
  }

  function toggleDirect() {
    var tab = getActiveTab();
    if (!tab) return;
    tab.directMode = !tab.directMode;
    updateDirectBtn();
    if (tab.directMode) {
      addLog("Direct mode ON \u2014 full JavaScript, AI can still read pages", "ok");
    } else {
      addLog("Proxy mode \u2014 AI has full page control", "ok");
    }
    notifyParent("directModeChanged", { direct: tab.directMode, tabId: tab.id });
    // Reload current page in new mode
    if (tab.url) navigateTo(tab.url);
  }

  function showClick(x, y) {
    clickIndicator.style.left = x + "px"; clickIndicator.style.top = y + "px";
    clickIndicator.style.display = "block";
    setTimeout(function() { clickIndicator.style.display = "none"; }, 700);
  }

  function onMessage(e) {
    var d = e.data;
    // Handle error page action buttons
    if (d && d.autoBrowserAction === "tryJinaReader" && d.url) {
      tryJinaReaderFallback(d.url, null);
      return;
    }
    if (d && d.autoBrowserAction === "tryArchive" && d.url) {
      tryArchiveFallback(d.url, null);
      return;
    }
    if (d && d.autoBrowserAction === "tryGoogleCache" && d.url) {
      tryGoogleCacheFallback(d.url, null);
      return;
    }
    if (d && d.autoBrowserAction === "tryDirect" && d.url) {
      var tab = getActiveTab();
      if (tab) { tab.directMode = true; updateDirectBtn(); }
      navigateTo(d.url, null);
      return;
    }
    if (!d || !d.autoBrowser) return;
    // Forward iframe replies to parent
    if (d.type === "cmdReply") {
      notifyParent_raw(d);
      if (d.payload && d.payload.x != null) showClick(d.payload.x, d.payload.y);
      return;
    }
    // Handle navigation from iframe link clicks
    if (d.type === "iframeNavigate" && d.url) {
      addLog("Link click: " + d.url.slice(0, 60), "nav");
      navigateTo(d.url);
      return;
    }
    var id = d.id, data = d.data || {};
    if (d.cmd === "setDirectMode") {
      var tab = getActiveTab();
      if (tab && data.direct !== tab.directMode) toggleDirect();
      return;
    }
    // Tab management commands
    if (d.cmd === "newTab") {
      var newTab = createTab(data.url || "");
      if (data.url) navigateTo(data.url, id);
      else if (id != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { success: true, tabId: newTab.id } });
      return;
    }
    if (d.cmd === "closeTab") {
      closeTab(data.tabId || activeTabId);
      if (id != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { success: true } });
      return;
    }
    if (d.cmd === "switchTab") {
      if (data.tabId) switchTab(data.tabId);
      if (id != null) notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { success: true, tabId: activeTabId } });
      return;
    }
    if (d.cmd === "getTabs") {
      notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { tabs: getTabsSummary() } });
      return;
    }
    if (d.cmd === "navigate") { navigateTo(data.url, id, data.tabId); }
    else if (d.cmd === "click") {
      if (!agentMode) { notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { success: false, error: "User has taken over" } }); return; }
      addLog("Click: " + (data.selector || "").slice(0, 40));
      iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "click", id: id, selector: data.selector }, "*");
    } else if (d.cmd === "type") {
      if (!agentMode) { notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { success: false, error: "User has taken over" } }); return; }
      addLog("Type \u201c" + (data.text || "").slice(0, 30) + "\u201d \u2192 " + (data.selector || "").slice(0, 30));
      iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "type", id: id, selector: data.selector, text: data.text }, "*");
    } else if (d.cmd === "read") {
      addLog("Reading page content...");
      // In direct mode, try to read via injected script; if that fails, return what we know
      var tab = getActiveTab();
      if (tab && tab.directMode) {
        // Try injecting ctrl first
        tryInjectCtrlDirect();
        // Send read command - it will work if injection succeeded
        if (iframe.contentWindow) {
          iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "read", id: id }, "*");
          // Set a timeout fallback in case the message isn't received (cross-origin)
          setTimeout(function() {
            // Check if we already got a reply (resolver was deleted)
            // We can't check from popup, but parent will timeout if no reply
          }, 3000);
        } else {
          notifyParent_raw({ autoBrowser: true, type: "cmdReply", id: id, payload: { text: "(Could not access page content)", title: tab.url, url: tab.url, directMode: true } });
        }
      } else {
        iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "read", id: id }, "*");
      }
    } else if (d.cmd === "scroll") {
      if (!agentMode) return;
      addLog("Scroll: " + data.direction);
      iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "scroll", id: id, direction: data.direction, amount: data.amount || 400 }, "*");
    } else if (d.cmd === "find") {
      iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "find", id: id, query: data.query }, "*");
    } else if (d.cmd === "screenshot") {
      iframe.contentWindow && iframe.contentWindow.postMessage({ autoBrowserCmd: true, cmd: "screenshot", id: id }, "*");
    } else if (d.cmd === "logMsg") {
      addLog(data.msg, data.type || "");
    }
  }

  function notifyParent_raw(msg) {
    try { var target = window.parent !== window ? window.parent : window.opener; if (target) target.postMessage(msg, "*"); } catch(e) {}
  }

  if (document.readyState === "loading") { document.addEventListener("DOMContentLoaded", init); } else { init(); }
}

// ─── Build popup HTML (blob) ───
function buildPopupHtml() {
  var iframeCtrlSrc = "(" + _iframeCtrl.toString() + ")()";
  var popupScriptSrc = "(" + _popupScript.toString() + ")(" + JSON.stringify({ proxy: CORS_PROXIES[0].base, proxies: CORS_PROXIES, iframeCtrl: iframeCtrlSrc, spaDomains: SPA_DOMAINS, proxyHostileDomains: PROXY_HOSTILE_DOMAINS }) + ")";
  var css = [
    "* { box-sizing: border-box; margin: 0; padding: 0; }",
    "body { background: #07070b; color: #ccccda; font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; font-size: 12px; }",
    // ─── Tab Bar ───
    "#tab-bar { display: flex; align-items: stretch; background: #08080f; border-bottom: 1px solid #181824; flex-shrink: 0; overflow-x: auto; min-height: 30px; padding: 0 2px; gap: 1px; }",
    "#tab-bar::-webkit-scrollbar { height: 2px; } #tab-bar::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 1px; }",
    ".tab { display: flex; align-items: center; gap: 4px; padding: 4px 10px; background: rgba(255,255,255,0.02); border: 1px solid transparent; border-bottom: none; border-radius: 6px 6px 0 0; cursor: pointer; max-width: 180px; min-width: 60px; flex-shrink: 0; transition: background 0.15s; margin-top: 2px; }",
    ".tab:hover { background: rgba(255,255,255,0.06); }",
    ".tab.active { background: #0d0d14; border-color: #181824; border-bottom: 1px solid #0d0d14; margin-bottom: -1px; position: relative; z-index: 1; }",
    ".tab-title { font-size: 10px; font-family: 'Segoe UI', system-ui, sans-serif; color: #777; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; user-select: none; }",
    ".tab.active .tab-title { color: #ccc; font-weight: 600; }",
    ".tab-close { font-size: 13px; color: #444; cursor: pointer; padding: 0 2px; line-height: 1; border-radius: 3px; flex-shrink: 0; }",
    ".tab-close:hover { color: #cc7777; background: rgba(204,119,119,0.15); }",
    ".tab-add { display: flex; align-items: center; justify-content: center; padding: 4px 10px; font-size: 16px; color: #444; cursor: pointer; flex-shrink: 0; border-radius: 6px 6px 0 0; margin-top: 2px; }",
    ".tab-add:hover { color: #7ce08a; background: rgba(124,224,138,0.08); }",
    // ─── Toolbar ───
    "#toolbar { display: flex; align-items: center; gap: 6px; padding: 5px 10px; background: #0d0d14; border-bottom: 1px solid #181824; flex-shrink: 0; }",
    ".nav-btn { background: rgba(255,255,255,0.05); border: 1px solid #181824; border-radius: 5px; color: #888; cursor: pointer; padding: 4px 8px; font-size: 13px; line-height: 1; }",
    ".nav-btn:hover { background: rgba(255,255,255,0.1); color: #ccc; } .nav-btn:disabled { opacity: 0.3; cursor: default; }",
    "#ui { flex: 1; background: rgba(255,255,255,0.04); border: 1px solid #181824; border-radius: 6px; color: #ccccda; font-size: 11px; padding: 5px 10px; outline: none; font-family: 'JetBrains Mono', monospace; }",
    "#ui:focus { border-color: rgba(136,187,204,0.4); }",
    "#go-btn { background: rgba(136,187,204,0.1); border: 1px solid rgba(136,187,204,0.3); border-radius: 5px; color: #88bbcc; cursor: pointer; padding: 5px 10px; font-size: 11px; }",
    "#go-btn:hover { background: rgba(136,187,204,0.2); }",
    ".badge { padding: 3px 8px; border-radius: 5px; font-size: 10px; font-weight: 700; background: rgba(124,224,138,0.1); border: 1px solid rgba(124,224,138,0.3); color: #7ce08a; display: flex; align-items: center; gap: 4px; white-space: nowrap; letter-spacing: 0.5px; }",
    ".badge.inactive { background: rgba(255,255,255,0.03); border-color: #181824; color: #555; }",
    ".tbtn { padding: 4px 10px; border-radius: 5px; font-size: 10px; background: rgba(204,119,119,0.1); border: 1px solid rgba(204,119,119,0.3); color: #cc7777; cursor: pointer; white-space: nowrap; }",
    ".tbtn:hover { background: rgba(204,119,119,0.2); } .tbtn.resume { background: rgba(124,224,138,0.1); border-color: rgba(124,224,138,0.3); color: #7ce08a; }",
    // ─── Content Area ───
    "#content-area { flex: 1; position: relative; overflow: hidden; display: flex; flex-direction: row; min-height: 0; }",
    "#pf { flex: 1; border: none; background: #fff; min-height: 0; min-width: 0; height: 100%; }",
    "#lo { position: absolute; left: 0; top: 0; bottom: 0; right: 240px; background: rgba(7,7,11,0.95); display: none; flex-direction: column; align-items: center; justify-content: center; gap: 12px; z-index: 100; }",
    ".spin { width: 30px; height: 30px; border: 3px solid #181824; border-top-color: #7ce08a; border-radius: 50%; animation: spin 0.8s linear infinite; }",
    "@keyframes spin { to { transform: rotate(360deg); } }",
    // ─── Agent Log Panel (right side) ───
    "#ap { background: rgba(7,7,11,0.98); border-left: 1px solid #181824; transition: width 0.25s; overflow: hidden; width: 240px; flex-shrink: 0; display: flex; flex-direction: column; height: 100%; }",
    "#ap.collapsed { width: 28px; }",
    "#ph { display: flex; align-items: center; justify-content: space-between; padding: 6px 8px; cursor: pointer; user-select: none; border-bottom: 1px solid #181824; flex-shrink: 0; }",
    "#pt-label { font-size: 9px; font-weight: 700; color: #7ce08a; font-family: monospace; letter-spacing: 1px; display: flex; align-items: center; gap: 6px; }",
    "#ap.collapsed #pt-label span { display: none; }",
    "#al { padding: 5px 8px 8px; overflow-y: auto; flex: 1; display: flex; flex-direction: column; gap: 2px; min-height: 0; }",
    ".le { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #88bbcc; padding: 2px 4px; border-radius: 3px; background: rgba(136,187,204,0.04); display: flex; flex-direction: column; gap: 1px; animation: fi 0.2s ease; word-break: break-word; }",
    ".le .ts { color: #333; font-size: 8px; flex-shrink: 0; }",
    ".le.err { color: #cc7777; background: rgba(204,119,119,0.04); } .le.ok { color: #7ce08a; background: rgba(124,224,138,0.04); } .le.nav { color: #aaa; }",
    "@keyframes fi { from { opacity: 0; transform: translateY(2px); } to { opacity: 1; transform: translateY(0); } }",
    // ─── Click Indicator ───
    "#ci { position: absolute; width: 22px; height: 22px; border: 2px solid #7ce08a; border-radius: 50%; pointer-events: none; transform: translate(-50%,-50%); z-index: 200; display: none; animation: ca 0.6s forwards; }",
    "@keyframes ca { 0%{opacity:1;transform:translate(-50%,-50%) scale(0.4)} 50%{opacity:1;transform:translate(-50%,-50%) scale(1.3)} 100%{opacity:0;transform:translate(-50%,-50%) scale(1.6)} }",
    // ─── Status Bar ───
    "#sb { display: flex; align-items: center; gap: 8px; padding: 2px 10px; background: #080810; border-top: 1px solid rgba(255,255,255,0.03); font-size: 9px; font-family: monospace; color: #444; flex-shrink: 0; }",
    "#st { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; } #sm { color: #7ce08a; font-weight: 700; }"
  ].join("\n");

  var body = [
    '<div id="tab-bar"></div>',
    '<div id="toolbar">',
    '  <button class="nav-btn" id="back-btn" disabled>&#9664;</button>',
    '  <button class="nav-btn" id="fwd-btn" disabled>&#9654;</button>',
    '  <button class="nav-btn" id="reload-btn">&#8635;</button>',
    '  <input id="ui" type="text" placeholder="Enter URL...">',
    '  <button id="go-btn">Go</button>',
    '  <div class="badge" id="ab"><span>&#9679;</span><span id="abt">AI AGENT</span></div>',
    '  <button class="tbtn" id="tb">Take Over</button>',
    '  <button id="dm" style="padding:4px 10px;border-radius:5px;font-size:10px;background:rgba(136,187,204,0.1);border:1px solid rgba(136,187,204,0.3);color:#88bbcc;cursor:pointer;white-space:nowrap;font-weight:700" title="Direct mode: loads pages with full JavaScript. AI can still read pages.">Direct</button>',
    '</div>',
    '<div id="content-area">',
    '  <iframe id="pf" sandbox="allow-scripts allow-forms allow-popups allow-same-origin allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation"></iframe>',
    '  <div id="lo"><div class="spin"></div><div id="lt" style="font-size:11px;color:#555;font-family:monospace">Loading...</div></div>',
    '  <div id="ci"></div>',
    '  <div id="ap">',
    '    <div id="ph"><span id="pt-label">&#9636; <span>CONSOLE</span></span><span id="pt">&#9658;</span></div>',
    '    <div id="al"></div>',
    '  </div>',
    '</div>',
    '<div id="sb"><span id="st">Ready</span><span id="sm">AI MODE</span></div>'
  ].join("\n");

  return "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Auto Browser</title><style>" + css + "</style></head><body>" + body + "<script>" + popupScriptSrc + "<\/script></body></html>";
}

// ─── Agent Browser Manager (embedded iframe mode) ───
var agentBrowser = (function() {
  var embeddedIframe = null, currentUrl = "", agentMode = true, directMode = false;
  var pendingResolvers = {}, msgId = 0;
  var listenerAdded = false;
  var onUrlChangeCb = null, onUserTookOverCb = null, onPopupBlockedCb = null, onTabsChangedCb = null;
  var pendingInitUrl = null, isReady = false, readyResolvers = [];
  var currentTabs = [];
  var onShowBrowserCb = null; // callback to show the embedded browser in React
  var blobUrl = null;

  function initListener(onUrlChange, onUserTookOver, onPopupBlocked, onTabsChanged) {
    onUrlChangeCb = onUrlChange; onUserTookOverCb = onUserTookOver;
    if (onPopupBlocked) onPopupBlockedCb = onPopupBlocked;
    if (onTabsChanged) onTabsChangedCb = onTabsChanged;
    if (listenerAdded) return;
    listenerAdded = true;
    window.addEventListener("message", function(e) {
      var d = e.data;
      if (!d || !d.autoBrowser) return;
      if (d.type === "cmdReply") {
        var res = pendingResolvers[d.id];
        if (res) { delete pendingResolvers[d.id]; res(d.payload); }
        return;
      }
      if (d.type === "urlChange") { currentUrl = (d.payload && d.payload.url) || ""; onUrlChangeCb && onUrlChangeCb(currentUrl); }
      if (d.type === "userTookOver") { agentMode = false; onUserTookOverCb && onUserTookOverCb(); }
      if (d.type === "aiResumed") { agentMode = true; }
      if (d.type === "directModeChanged") { directMode = !!(d.payload && d.payload.direct); }
      if (d.type === "tabsChanged") { currentTabs = d.payload || []; onTabsChangedCb && onTabsChangedCb(currentTabs); }
      if (d.type === "tabSwitched") { currentUrl = (d.payload && d.payload.url) || ""; onUrlChangeCb && onUrlChangeCb(currentUrl); }
      if (d.type === "ready") {
        isReady = true;
        var rrs = readyResolvers.splice(0);
        rrs.forEach(function(r) { r(); });
        if (pendingInitUrl) { var navUrl = pendingInitUrl; pendingInitUrl = null; _send("navigate", { url: navUrl }, true); }
      }
    });
  }

  function isOpen() { return embeddedIframe && embeddedIframe.contentWindow && document.body.contains(embeddedIframe); }

  function open(url) {
    initListener(onUrlChangeCb, onUserTookOverCb, onPopupBlockedCb, onTabsChangedCb);
    if (!isOpen()) {
      // Build the browser HTML and set it as the iframe src
      var html = buildPopupHtml();
      var blob = new Blob([html], { type: "text/html" });
      if (blobUrl) { try { URL.revokeObjectURL(blobUrl); } catch(e) {} }
      blobUrl = URL.createObjectURL(blob);
      pendingInitUrl = url || null;
      isReady = false;
      // Show the embedded browser via React callback
      if (onShowBrowserCb) onShowBrowserCb(blobUrl);
    } else {
      if (url) _send("navigate", { url: url });
    }
  }

  function _send(cmd, data, waitForReply, customTimeout) {
    if (!isOpen()) return Promise.resolve(null);
    var id = ++msgId;
    try {
      embeddedIframe.contentWindow.postMessage({ autoBrowser: true, id: id, cmd: cmd, data: data || {} }, "*");
    } catch (e) {
      return Promise.resolve(null);
    }
    if (!waitForReply) return Promise.resolve(null);
    return new Promise(function(resolve) {
      pendingResolvers[id] = resolve;
      var checkInterval = setInterval(function() {
        if (!isOpen() && pendingResolvers[id]) {
          clearInterval(checkInterval);
          delete pendingResolvers[id];
          resolve(null);
        }
      }, 1000);
      setTimeout(function() {
        clearInterval(checkInterval);
        if (pendingResolvers[id]) { delete pendingResolvers[id]; resolve(null); }
      }, customTimeout || 10000);
    });
  }

  return {
    get currentUrl() { return currentUrl; },
    get agentMode() { return agentMode; },
    get directMode() { return directMode; },
    get tabs() { return currentTabs; },
    initListener: initListener,
    isOpen: isOpen,
    open: open,
    setEmbeddedIframe: function(el) { embeddedIframe = el; },
    onShowBrowser: function(cb) { onShowBrowserCb = cb; },
    close: function() { embeddedIframe = null; isReady = false; currentUrl = ""; },
    setDirectMode: function(on) { _send("setDirectMode", { direct: !!on }); directMode = !!on; },
    navigate: function(url, tabId) { if (!isOpen()) { open(url); return Promise.resolve(null); } return _send("navigate", { url: url, tabId: tabId }, true, 25000); },
    waitForReady: function() {
      if (isReady && isOpen()) return Promise.resolve();
      return new Promise(function(resolve) {
        readyResolvers.push(resolve);
        setTimeout(function() { var idx = readyResolvers.indexOf(resolve); if (idx >= 0) { readyResolvers.splice(idx, 1); resolve(); } }, 10000);
      });
    },
    click: function(sel) { return _send("click", { selector: sel }, true, 10000); },
    type: function(sel, text) { return _send("type", { selector: sel, text: text }, true, 10000); },
    read: function() { return _send("read", {}, true, 10000); },
    scroll: function(dir) { return _send("scroll", { direction: dir }, true, 5000); },
    find: function(q) { return _send("find", { query: q }, true, 8000); },
    screenshot: function() { return _send("screenshot", {}, true, 8000); },
    newTab: function(url) { return _send("newTab", { url: url || "" }, true, 15000); },
    closeTab: function(tabId) { return _send("closeTab", { tabId: tabId }, true); },
    switchTab: function(tabId) { return _send("switchTab", { tabId: tabId }, true); },
    getTabs: function() { return _send("getTabs", {}, true); },
    logMsg: function(msg, type) { _send("logMsg", { msg: msg, type: type || "" }); },
    focus: function() { /* embedded - no-op */ },
  };
})();

function openBrowserPopup(url) { agentBrowser.open(url); }
function isBrowserOpen() { return agentBrowser.isOpen(); }

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
  const [sidebarTab, setSidebarTab] = useState("browser"); // "browser" | "memory" | "terminal"
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [searchBusy, setSearchBusy] = useState(false);
  const [browserUrl, setBrowserUrl] = useState("");
  const [usage, setUsage] = useState({ i: 0, o: 0 });
  const [apiKey, setApiKey] = useState("");
  const [groqApiKey, setGroqApiKey] = useState("");
  const [researchStatus, setResearchStatus] = useState("");
  const [agentBrowserUrl, setAgentBrowserUrl] = useState("");
  const [agentUserTookOver, setAgentUserTookOver] = useState(false);
  const [popupBlocked, setPopupBlocked] = useState(false);
  const [expression, setExpression] = useState("happy"); // "happy" | "serious" | "veryHappy"
  const [isBlinking, setIsBlinking] = useState(false);
  const blinkRef = useRef(null);
  const [terminalHistory, setTerminalHistory] = useState([{ type: "system", text: "Auto Terminal v1.0 — JavaScript execution environment\nType JavaScript code and press Enter to execute.\nUse clear() to clear the terminal.\n" }]);
  const [terminalInput, setTerminalInput] = useState("");
  const [terminalCmdHistory, setTerminalCmdHistory] = useState([]);
  const [terminalHistoryIdx, setTerminalHistoryIdx] = useState(-1);
  const [attachments, setAttachments] = useState([]); // [{name, type, content, size}]
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [showBrowser, setShowBrowser] = useState(false);
  const [browserBlobUrl, setBrowserBlobUrl] = useState(null);
  const browserIframeRef = useRef(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const attachInputRef = useRef(null);
  const abortRef = useRef(null);
  const terminalScrollRef = useRef(null);
  const terminalInputRef = useRef(null);
  const msgsRef = useRef([]);
  const memRef = useRef("");
  const busyRef = useRef(false);

  const promptForApiKey = useCallback((reason = "Enter your OpenRouter API key:") => {
    const enteredKey = window.prompt(reason);
    const normalizedKey = (enteredKey || "").trim();
    if (!normalizedKey) return "";
    setApiKey(normalizedKey);
    saveApiKey(normalizedKey);
    return normalizedKey;
  }, []);

  const promptForGroqKey = useCallback((reason = "Enter your Groq API key:") => {
    const enteredKey = window.prompt(reason);
    const normalizedKey = (enteredKey || "").trim();
    if (!normalizedKey) return "";
    setGroqApiKey(normalizedKey);
    saveGroqKey(normalizedKey);
    return normalizedKey;
  }, []);

  // ─── Terminal execution ───
  const executeTerminal = useCallback((code) => {
    if (!code.trim()) return;
    const entry = { type: "input", text: code };
    const newHistory = [...terminalHistory, entry];

    if (code.trim() === "clear()") {
      setTerminalHistory([{ type: "system", text: "Terminal cleared.\n" }]);
      return;
    }

    // Capture console output
    const logs = [];
    const origLog = console.log;
    const origWarn = console.warn;
    const origError = console.error;
    const origInfo = console.info;
    const stringify = (args) => args.map(a => {
      if (a === undefined) return "undefined";
      if (a === null) return "null";
      if (typeof a === "object") { try { return JSON.stringify(a, null, 2); } catch { return String(a); } }
      return String(a);
    }).join(" ");
    console.log = (...args) => { logs.push({ level: "log", text: stringify(args) }); origLog(...args); };
    console.warn = (...args) => { logs.push({ level: "warn", text: stringify(args) }); origWarn(...args); };
    console.error = (...args) => { logs.push({ level: "error", text: stringify(args) }); origError(...args); };
    console.info = (...args) => { logs.push({ level: "info", text: stringify(args) }); origInfo(...args); };

    let result;
    try {
      // eslint-disable-next-line no-eval
      result = { type: "output", text: String(eval(code)), logs };
    } catch (e) {
      result = { type: "error", text: String(e), logs };
    }

    console.log = origLog;
    console.warn = origWarn;
    console.error = origError;
    console.info = origInfo;

    setTerminalHistory([...newHistory, result]);
    setTerminalCmdHistory(prev => [...prev, code]);
    setTerminalHistoryIdx(-1);
    setTimeout(() => { terminalScrollRef.current?.scrollIntoView({ behavior: "smooth" }); }, 50);
  }, [terminalHistory]);

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
    (async () => {
      const envKey = readEnvApiKey();
      if (envKey) { setApiKey(envKey); return; }
      const storedKey = await loadApiKey();
      if (storedKey) setApiKey(storedKey);
    })();
    // Init agent browser event listeners
    agentBrowser.initListener(
      (url) => setAgentBrowserUrl(url),
      () => setAgentUserTookOver(true),
      () => setPopupBlocked(true)
    );
    // Wire up embedded browser show callback
    agentBrowser.onShowBrowser(function(blobUrl) {
      setBrowserBlobUrl(blobUrl);
      setShowBrowser(true);
    });
  }, [promptForGroqKey]);

  // Sync embedded browser iframe ref with agentBrowser and scroll into view
  useEffect(() => {
    if (showBrowser && browserIframeRef.current) {
      agentBrowser.setEmbeddedIframe(browserIframeRef.current);
      setTimeout(() => scrollRef.current?.scrollIntoView({ behavior: "smooth" }), 100);
    }
  }, [showBrowser, browserBlobUrl]);

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

  const doSearch = useCallback(async (query) => {
    if (!query?.trim()) return [];
    setSearchBusy(true);
    setSearchResults([]);
    try {
      const results = await performSearch(query.trim());
      setSearchResults(results);
      return results;
    } catch (e) {
      console.error("Search error:", e);
      return [];
    } finally {
      setSearchBusy(false);
    }
  }, []);

  const handleSearchSubmit = useCallback((e) => {
    if (e) e.preventDefault();
    doSearch(searchQuery);
  }, [searchQuery, doSearch]);

  const handleBrowserGo = useCallback(() => {
    if (!browserUrl.trim()) return;
    let url = browserUrl.trim();
    if (!/^https?:\/\//i.test(url)) url = "https://" + url;
    openBrowserPopup(url);
  }, [browserUrl]);

  // ─── System prompt builder ───
  const buildSystem = useCallback(() => {
    const today = new Date().toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });
    let s = `You are Auto, a brutally honest, exceptionally loyal, warm AI assistant with internet research capabilities. You are curious, honest, loyal, trustworthy, helpful, and thorough. Use markdown formatting. Today is ${today}. Trust is your number 1 value.`;

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

    // Research / web search instructions
    s += `\n\n## Web Research Capability
You can search the internet and read web pages! You have a built-in research tool that searches the web without needing to open browser tabs or windows. When you need current information, facts, news, or want to research a topic:

1. **To search the web**: Include <web_search>your search query</web_search> in your response. This searches DuckDuckGo, Google, and Brave for comprehensive results.
2. **To read a webpage**: Include <read_url>https://example.com</read_url> in your response. This fetches and extracts text content from the page, with automatic fallbacks for dynamic sites (Google Cache, Wayback Machine).
3. **To open a page in the user's browser popup**: Include <open_browser>https://example.com</open_browser> in your response.

You should PROACTIVELY research when:
- The user asks about current events, news, or recent information
- You need to verify facts or find up-to-date data
- The user asks you to look something up or research a topic
- The topic involves recent developments, prices, statistics, etc.
- You want to provide accurate, up-to-date information on ANY topic
- Contains content that may be risky for the individual

When researching, search multiple queries if needed. Cite sources with URLs. You can chain multiple <web_search> and <read_url> tags in a single response to gather information from multiple sources at once.

## AI Browser Agent Capability
You can DIRECTLY CONTROL a visual browser window with TABS! The user will see the browser popup and can take over at any time.

⚠️ CRITICAL FORMAT RULE: Use ONLY the exact XML tags listed below. Do NOT use \`<tool_call>\`, \`<function=...>\`, \`<parameter=...>\`, JSON tool syntax, or any other wrapper format. Output the tags DIRECTLY in your response text:

**Navigation & Reading:**
1. **Navigate**: <browser_navigate>https://example.com</browser_navigate>
2. **Read page**: <browser_read/> — always use this after navigating to see the page!
3. **Click**: <browser_click>button text or CSS selector</browser_click>
4. **Type**: <browser_type>selector :: text to type</browser_type>
5. **Scroll**: <browser_scroll>down</browser_scroll> (up/down/top/bottom)
6. **Find elements**: <browser_find>search text</browser_find>

**Tab Management:**
7. **New tab**: <browser_new_tab>https://example.com</browser_new_tab> — open a URL in a new tab
8. **Close tab**: <browser_close_tab>tab_id</browser_close_tab> — close a specific tab
9. **Switch tab**: <browser_switch_tab>tab_id</browser_switch_tab> — switch to a specific tab

**Correct example** — researching in multiple tabs:
I'll open the docs in a new tab while keeping the current page.
<browser_new_tab>https://docs.example.com</browser_new_tab>
<browser_read/>

**Browser workflow**: Navigate → Read page → Click/type → Read again → Repeat as needed
- You can chain multiple browser actions in one response — they execute in sequence
- After browser actions you'll receive the results and can continue the task
- Use **tabs** to research multiple pages simultaneously or keep reference pages open
- The user can click "Take Over" in the browser popup to control it themselves anytime

Use the browser agent for: filling forms, searching websites, web apps, booking, shopping, research with multiple tabs, etc.

**Important**: The browser has multiple loading strategies:
- **Proxy mode** (default): AI has full page control (click, type, read). Assets (CSS, images, fonts) are automatically resolved. Best for most sites.
- **Jina Reader mode** (automatic fallback): For SPAs (x.com, reddit, etc.) and sites that block iframes, the browser uses Jina Reader API which renders pages via headless Chrome and returns clean, readable content. AI can read all content on these pages.
- **Direct mode**: Pages load with full JavaScript support. AI can still **read** page content. The user can toggle Direct mode manually.

The browser has an intelligent fallback chain: Proxy → Jina Reader (headless Chrome) → Wayback Machine archive. This means virtually ALL websites are accessible and readable, including SPAs, paywalled sites, and sites with strict CSP/X-Frame-Options.

## Terminal / Code Execution
You have a built-in JavaScript terminal! You can execute code directly in the browser environment.

To execute JavaScript code, include a <terminal_exec> tag in your response:
<terminal_exec>console.log("Hello world!"); 2 + 2</terminal_exec>

You can use this to:
- Perform calculations and data processing
- Test JavaScript code snippets
- Manipulate data structures (arrays, objects, JSON)
- Run utility functions (Date, Math, string operations, etc.)
- Create and test functions
- Access browser APIs (DOM, fetch, localStorage, etc.)

You can chain multiple <terminal_exec> blocks in one response. The execution results (return value + console output) will be returned to you so you can continue the task.

**Important**: Code runs in the browser context with full access to the page. Use this for calculations, data processing, and prototyping.

## Expressions
You have a visual avatar that shows your mood! Include an <expression> tag in EVERY response to set your expression:
- <expression>happy</expression> — use when greeting, helping, giving good news, being playful, or general conversation
- <expression>serious</expression> — use when thinking deeply, explaining complex topics, giving warnings, or discussing serious matters
- <expression>veryHappy</expression> — use when celebrating, super excited, receiving amazing news, completing a big task successfully, or when the user achieves something great

Always include exactly ONE <expression> tag per response. Place it at the very START of your response, before any other text. Default to happy if unsure.`;

    // ─── Skills System ───
    s += `\n\n## Agent Skills
You have access to specialized skills that enhance your capabilities. Skills are automatically detected based on the user's message, but you can also invoke them explicitly.

**Available Skills:**
${buildSkillsSummary()}
### How Skills Work
- Skills are **auto-detected** from the user's message and their instructions are injected into your context
- You can also explicitly invoke a skill by including \`<use_skill>skill-id</use_skill>\` in your response
- Skills provide specialized instructions, formulas, templates, and workflows for their domain
- Combine multiple skills when a task spans categories (e.g., data-analysis + data-visualization)
- When a skill is active, follow its specific instructions for output format and methodology

### Skill Usage Guidelines
- Use the **most specific skill** that matches the task
- For complex tasks, chain skills: research → analyze → visualize
- Skills enhance your existing capabilities — use them alongside web search, browser, and terminal
- Always verify computations with \`<terminal_exec>\` when a skill involves calculations`;

    return s;
  }, [mem]);

  // ─── Parse AI response (memory updates, search triggers, browser commands) ───
  const parseResponse = useCallback((text) => {
    // Safety: ensure we always work with a string
    if (!text || typeof text !== "string") return { text: String(text || ""), actions: { memoryUpdate: null, searches: [], readUrls: [], openUrls: [], browserActions: [], expression: null, terminalCommands: [] } };
    try {
    let cleaned = text;
    const actions = { memoryUpdate: null, searches: [], readUrls: [], openUrls: [], browserActions: [], expression: null, terminalCommands: [] };

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

    // Extract web search requests
    const searchMatches = cleaned.matchAll(/<web_search>([\s\S]*?)<\/web_search>/g);
    for (const m of searchMatches) actions.searches.push(m[1].trim());
    cleaned = cleaned.replace(/<web_search>[\s\S]*?<\/web_search>/g, "").trim();

    // Extract read URL requests
    const readMatches = cleaned.matchAll(/<read_url>([\s\S]*?)<\/read_url>/g);
    for (const m of readMatches) actions.readUrls.push(m[1].trim());
    cleaned = cleaned.replace(/<read_url>[\s\S]*?<\/read_url>/g, "").trim();

    // Extract open browser requests
    const openMatches = cleaned.matchAll(/<open_browser>([\s\S]*?)<\/open_browser>/g);
    for (const m of openMatches) actions.openUrls.push(m[1].trim());
    cleaned = cleaned.replace(/<open_browser>[\s\S]*?<\/open_browser>/g, "").trim();

    // ─── Browser Agent Actions ───
    // Navigate: <browser_navigate>https://example.com</browser_navigate>
    for (const m of cleaned.matchAll(/<browser_navigate>([\s\S]*?)<\/browser_navigate>/g))
      actions.browserActions.push({ type: "navigate", url: m[1].trim() });
    cleaned = cleaned.replace(/<browser_navigate>[\s\S]*?<\/browser_navigate>/g, "").trim();

    // Click: <browser_click>button text or CSS selector</browser_click>
    for (const m of cleaned.matchAll(/<browser_click>([\s\S]*?)<\/browser_click>/g))
      actions.browserActions.push({ type: "click", selector: m[1].trim() });
    cleaned = cleaned.replace(/<browser_click>[\s\S]*?<\/browser_click>/g, "").trim();

    // Type: <browser_type>selector :: text to type</browser_type>
    for (const m of cleaned.matchAll(/<browser_type>([\s\S]*?)<\/browser_type>/g)) {
      const parts = m[1].split(" :: ");
      if (parts.length >= 2) actions.browserActions.push({ type: "type", selector: parts[0].trim(), text: parts.slice(1).join(" :: ").trim() });
      else actions.browserActions.push({ type: "type", selector: "input,textarea", text: m[1].trim() });
    }
    cleaned = cleaned.replace(/<browser_type>[\s\S]*?<\/browser_type>/g, "").trim();

    // Read: <browser_read/> or <browser_read></browser_read>
    if (/<browser_read\s*\/?>/.test(cleaned) || /<browser_read>[\s\S]*?<\/browser_read>/.test(cleaned))
      actions.browserActions.push({ type: "read" });
    cleaned = cleaned.replace(/<browser_read\s*\/?>/g, "").replace(/<browser_read>[\s\S]*?<\/browser_read>/g, "").trim();

    // Scroll: <browser_scroll>down</browser_scroll>  (up/down/top/bottom)
    for (const m of cleaned.matchAll(/<browser_scroll>([\s\S]*?)<\/browser_scroll>/g))
      actions.browserActions.push({ type: "scroll", direction: m[1].trim() });
    cleaned = cleaned.replace(/<browser_scroll>[\s\S]*?<\/browser_scroll>/g, "").trim();

    // Find: <browser_find>search button</browser_find>
    for (const m of cleaned.matchAll(/<browser_find>([\s\S]*?)<\/browser_find>/g))
      actions.browserActions.push({ type: "find", query: m[1].trim() });
    cleaned = cleaned.replace(/<browser_find>[\s\S]*?<\/browser_find>/g, "").trim();

    // Tab management: <browser_new_tab>url</browser_new_tab>
    for (const m of cleaned.matchAll(/<browser_new_tab>([\s\S]*?)<\/browser_new_tab>/g))
      actions.browserActions.push({ type: "newTab", url: m[1].trim() });
    cleaned = cleaned.replace(/<browser_new_tab>[\s\S]*?<\/browser_new_tab>/g, "").trim();

    // Close tab: <browser_close_tab>tab_id</browser_close_tab>
    for (const m of cleaned.matchAll(/<browser_close_tab>([\s\S]*?)<\/browser_close_tab>/g))
      actions.browserActions.push({ type: "closeTab", tabId: parseInt(m[1].trim()) || 0 });
    cleaned = cleaned.replace(/<browser_close_tab>[\s\S]*?<\/browser_close_tab>/g, "").trim();

    // Switch tab: <browser_switch_tab>tab_id</browser_switch_tab>
    for (const m of cleaned.matchAll(/<browser_switch_tab>([\s\S]*?)<\/browser_switch_tab>/g))
      actions.browserActions.push({ type: "switchTab", tabId: parseInt(m[1].trim()) || 0 });
    cleaned = cleaned.replace(/<browser_switch_tab>[\s\S]*?<\/browser_switch_tab>/g, "").trim();

    // Terminal execution: <terminal_exec>code here</terminal_exec>
    for (const m of cleaned.matchAll(/<terminal_exec>([\s\S]*?)<\/terminal_exec>/g))
      actions.terminalCommands.push(m[1].trim());
    cleaned = cleaned.replace(/<terminal_exec>[\s\S]*?<\/terminal_exec>/g, "").trim();

    // ─── Handle <tool_call> format (some models output this instead of plain XML tags) ───
    // <tool_call><function=browser_navigate><parameter=url>URL</parameter></function></tool_call>
    for (const m of cleaned.matchAll(/<tool_call>[\s\S]*?<function=browser_navigate>[\s\S]*?<parameter=[^>]*>([\s\S]*?)<\/parameter>[\s\S]*?<\/function>[\s\S]*?<\/tool_call>/g))
      actions.browserActions.push({ type: "navigate", url: m[1].trim() });
    for (const m of cleaned.matchAll(/<tool_call>[\s\S]*?<function=browser_click>[\s\S]*?<parameter=[^>]*>([\s\S]*?)<\/parameter>[\s\S]*?<\/function>[\s\S]*?<\/tool_call>/g))
      actions.browserActions.push({ type: "click", selector: m[1].trim() });
    for (const m of cleaned.matchAll(/<tool_call>[\s\S]*?<function=browser_scroll>[\s\S]*?<parameter=[^>]*>([\s\S]*?)<\/parameter>[\s\S]*?<\/function>[\s\S]*?<\/tool_call>/g))
      actions.browserActions.push({ type: "scroll", direction: m[1].trim() });
    for (const m of cleaned.matchAll(/<tool_call>[\s\S]*?<function=web_search>[\s\S]*?<parameter=[^>]*>([\s\S]*?)<\/parameter>[\s\S]*?<\/function>[\s\S]*?<\/tool_call>/g))
      actions.searches.push(m[1].trim());
    if (/<tool_call>[\s\S]*?<function=(?:browser_read|web_read)/.test(cleaned))
      actions.browserActions.push({ type: "read" });
    // Strip all remaining <tool_call> blocks from display text
    cleaned = cleaned.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "").trim();

    // Handle <web_read/> as alias for <browser_read/>
    if (/<web_read[\s\/]/.test(cleaned) || cleaned.includes("<web_read>"))
      actions.browserActions.push({ type: "read" });
    cleaned = cleaned.replace(/<web_read\s*\/?>/g, "").replace(/<web_read>[\s\S]*?<\/web_read>/g, "").trim();

    // Strip any stray <function=...> tags that weren't inside a <tool_call>
    cleaned = cleaned.replace(/<function=[^>]*>[\s\S]*?<\/function>/g, "").trim();

    return { text: cleaned, actions };
    } catch (err) {
      console.warn("parseResponse error:", err);
      return { text: String(text), actions: { memoryUpdate: null, searches: [], readUrls: [], openUrls: [], browserActions: [], expression: null, terminalCommands: [] } };
    }
  }, []);

  // ─── Call AI API ───
  const callAI = useCallback(async (apiMsgs, key, groqKey) => {
    const buildBody = (model) => ({ model, messages: apiMsgs });
    let data = null;
    let usedModel = DEFAULT_MODEL;
    let lastErr = null;
    const delay = ms => new Promise(r => setTimeout(r, ms));

    // Try Groq first (default) with retry on 429 rate limits
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

    // Fall back to OpenRouter if Groq failed or unavailable
    if (!data && key) {
      for (const model of MODEL_FALLBACKS) {
        const MAX_RETRIES = 3;
        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
          let res;
          try {
            res = await fetch(API, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                Authorization: `Bearer ${key}`,
                "HTTP-Referer": window.location.origin,
                "X-Title": "Auto",
              },
              body: JSON.stringify(buildBody(model)),
              signal: abortRef.current?.signal,
            });
          } catch (e) {
            if (e.name === "AbortError") throw e;
            lastErr = e;
            break;
          }

          if (res.ok) {
            data = await res.json();
            usedModel = model;
            break;
          }

          const rawBody = await res.text();
          const msg = parseErrorMessage(rawBody, res.status);
          lastErr = new Error(msg);

          // Retry on 429 (rate limit) with exponential backoff
          if (res.status === 429 && attempt < MAX_RETRIES - 1) {
            await delay(1500 * (attempt + 1));
            continue;
          }

          // Non-retryable error — break inner retry loop, outer loop tries next model
          break;
        }
        if (data) break;
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
    setErr(null); setBusy(true); busyRef.current = true; setResearchStatus("");

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
      let key = (apiKey || readEnvApiKey() || (await loadApiKey()) || "").trim();
      if (!groqKey && !key) {
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
        // Detect relevant skills from the latest user message and inject context
        const latestUserMsg = [...currentMsgs].reverse().find(m => m.role === "user");
        const detectedSkills = latestUserMsg ? detectRelevantSkills(typeof latestUserMsg.content === "string" ? latestUserMsg.content : "") : [];
        let systemContent = buildSystem();
        if (detectedSkills.length > 0) {
          systemContent += "\n\n## Active Skills for This Query\nThe following skills have been auto-detected as relevant. Apply their methodology:\n";
          for (const skill of detectedSkills.slice(0, 4)) {
            systemContent += `\n**[${skill.name}]** (${skill.id}): ${skill.description}\n`;
          }
        }
        const apiMsgs = [
          { role: "system", content: systemContent },
          ...currentMsgs.map(m => ({ role: m.role, content: typeof m.content === "string" ? m.content.slice(0, 12000) : m.content })),
        ];

        if (researchRound > 0) {
          setResearchStatus(`Researching... (round ${researchRound})`);
          // Pace API calls to avoid 429 rate limits
          await new Promise(r => setTimeout(r, 800));
        }

        const { data, usedModel } = await callAI(apiMsgs, key, groqKey);
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

        // Handle open browser commands
        for (const url of actions.openUrls) {
          openBrowserPopup(url);
        }

        // Check if AI requested research (searches or page reads)
        if (actions.searches.length > 0 || actions.readUrls.length > 0) {
          researchRound++;
          let researchContext = "";

          // Execute searches (wrapped in try/catch for resilience)
          for (const query of actions.searches) {
            try {
              setResearchStatus(`Searching: "${query}"...`);
              setSearchQuery(query);
              const results = await doSearch(query);
              if (results.length > 0) {
                researchContext += `\n\n<search_results query="${query}">\n`;
                results.forEach((r, idx) => {
                  researchContext += `${idx + 1}. [${r.title}](${r.url})\n   ${r.snippet}\n`;
                });
                researchContext += `</search_results>`;
              } else {
                researchContext += `\n\n<search_results query="${query}">No results found.</search_results>`;
              }
            } catch (searchErr) {
              console.warn("Search failed:", searchErr);
              researchContext += `\n\n<search_results query="${query}">Search failed: ${searchErr.message || "unknown error"}</search_results>`;
            }
          }

          // Fetch pages (wrapped in try/catch for resilience)
          for (const url of actions.readUrls) {
            try {
              setResearchStatus(`Reading: ${url.slice(0, 50)}...`);
              const pageText = await fetchPageText(url);
              if (pageText) {
                researchContext += `\n\n<page_content url="${url}">\n${pageText}\n</page_content>`;
              } else {
                researchContext += `\n\n<page_content url="${url}">Could not fetch page content.</page_content>`;
              }
            } catch (readErr) {
              console.warn("Page read failed:", readErr);
              researchContext += `\n\n<page_content url="${url}">Failed to read page: ${readErr.message || "unknown error"}</page_content>`;
            }
          }

          // Feed research results back as a system-like user message
          currentMsgs = [...currentMsgs, {
            role: "user",
            content: `[SYSTEM: Research results from your web search/page read requests]${researchContext}\n\nNow please provide a comprehensive answer using these research results. Cite sources with URLs. If you need more information, you can search again.`
          }];
          setMsgs([...currentMsgs]);
          saveChat(currentMsgs); // Save after each research round

          continue; // Loop back for AI to process research results
        }

        // ─── Browser Agent Actions ───
        if (actions.browserActions.length > 0) {
          researchRound++;
          let browserContext = "";
          // Deduplicate: keep only the first 'read' action to avoid redundant page reads
          let seenRead = false;
          const dedupedActions = actions.browserActions.filter(a => {
            if (a.type === "read") { if (seenRead) return false; seenRead = true; }
            return true;
          });

          // Ensure popup is open and ready — auto-reopen if closed
          const ensureBrowserOpen = async () => {
            if (!agentBrowser.isOpen()) {
              agentBrowser.open();
              await agentBrowser.waitForReady();
            }
          };
          await ensureBrowserOpen();

          for (const action of dedupedActions) {
           try {
            // Re-check browser is still open before each action
            if (!agentBrowser.isOpen()) { await ensureBrowserOpen(); }
            if (action.type === "navigate") {
              setResearchStatus(`Browser: navigating to ${action.url.slice(0, 40)}...`);
              const navResult = await agentBrowser.navigate(action.url);
              // Give page time to render after navigation
              await new Promise(r => setTimeout(r, 800));
              browserContext += `\n\n<browser_result action="navigate">Navigated to ${action.url}. Current URL: ${agentBrowser.currentUrl || action.url}${navResult && !navResult.success ? " (Error: " + navResult.error + ")" : ""}</browser_result>`;
            } else if (action.type === "click") {
              setResearchStatus(`Browser: clicking "${action.selector}"...`);
              const res = await agentBrowser.click(action.selector);
              if (res?.success) {
                browserContext += `\n\n<browser_result action="click">Clicked "${action.selector}" — element: ${res.element || ""}, text: "${res.text || ""}"</browser_result>`;
                await new Promise(r => setTimeout(r, 400));
              } else {
                browserContext += `\n\n<browser_result action="click" error="true">Could not click "${action.selector}": ${res?.error || "not found"}</browser_result>`;
              }
            } else if (action.type === "type") {
              setResearchStatus(`Browser: typing into "${action.selector}"...`);
              const res = await agentBrowser.type(action.selector, action.text);
              if (res?.success) {
                browserContext += `\n\n<browser_result action="type">Typed "${action.text}" into "${action.selector}"</browser_result>`;
              } else {
                browserContext += `\n\n<browser_result action="type" error="true">Could not type into "${action.selector}": ${res?.error || "not found"}</browser_result>`;
              }
            } else if (action.type === "read") {
              setResearchStatus(`Browser: reading page...`);
              const res = await agentBrowser.read();
              const hasContent = res && res.text && res.text.trim().length > 50;
              if (hasContent) {
                const linksStr = (res.links || []).slice(0, 10).map(l => `  - ${l.text}: ${l.href}`).join("\n");
                const inputsStr = (res.inputs || []).slice(0, 10).map(inp => `  - ${inp.tag}[${inp.placeholder || inp.name || inp.type || ""}]${inp.text ? " \"" + inp.text + "\"" : ""}`).join("\n");
                browserContext += `\n\n<browser_page title="${res.title || ""}" url="${agentBrowser.currentUrl}">\n${res.text || "(no text)"}\n\nLinks on page:\n${linksStr || "  (none)"}\n\nForm inputs:\n${inputsStr || "  (none)"}\n</browser_page>`;
              } else {
                // Fallback: use Jina Reader API to read cross-origin / blocked pages
                const readUrl = agentBrowser.currentUrl;
                if (readUrl && readUrl.startsWith("http")) {
                  setResearchStatus(`Browser: reading via Jina Reader (${readUrl.slice(0, 40)})...`);
                  try {
                    const jinaText = await fetchWithJinaReader(readUrl, 20000);
                    if (jinaText && jinaText.length > 50) {
                      browserContext += `\n\n<browser_page title="(via Jina Reader)" url="${readUrl}">\n${jinaText.slice(0, 12000)}\n</browser_page>`;
                    } else {
                      browserContext += `\n\n<browser_result action="read" error="true">Could not read page — both iframe and Jina Reader returned no content</browser_result>`;
                    }
                  } catch {
                    browserContext += `\n\n<browser_result action="read" error="true">Could not read page (cross-origin blocked, Jina Reader also failed)</browser_result>`;
                  }
                } else {
                  browserContext += `\n\n<browser_result action="read" error="true">Could not read page (popup may be closed or page still loading)</browser_result>`;
                }
              }
            } else if (action.type === "scroll") {
              setResearchStatus(`Browser: scrolling ${action.direction}...`);
              await agentBrowser.scroll(action.direction);
              await new Promise(r => setTimeout(r, 100));
              browserContext += `\n\n<browser_result action="scroll">Scrolled ${action.direction}</browser_result>`;
            } else if (action.type === "find") {
              setResearchStatus(`Browser: finding "${action.query}"...`);
              const res = await agentBrowser.find(action.query);
              if (res?.matches?.length > 0) {
                const matchStr = res.matches.map(m => `  - ${m.tag}[id="${m.id}"] "${m.text}"${m.href ? " href=" + m.href : ""}${m.name ? " name=" + m.name : ""}`).join("\n");
                browserContext += `\n\n<browser_result action="find">Found ${res.matches.length} element(s) matching "${action.query}":\n${matchStr}</browser_result>`;
              } else {
                browserContext += `\n\n<browser_result action="find">No elements found matching "${action.query}"</browser_result>`;
              }
            } else if (action.type === "newTab") {
              setResearchStatus(`Browser: opening new tab${action.url ? ": " + action.url.slice(0, 35) + "..." : ""}...`);
              const res = await agentBrowser.newTab(action.url);
              if (action.url) {
                await new Promise(r => setTimeout(r, 1000));
              }
              browserContext += `\n\n<browser_result action="newTab">Opened new tab${action.url ? " with " + action.url : ""}${res?.tabId ? " (tab ID: " + res.tabId + ")" : ""}</browser_result>`;
            } else if (action.type === "closeTab") {
              setResearchStatus(`Browser: closing tab ${action.tabId}...`);
              await agentBrowser.closeTab(action.tabId);
              browserContext += `\n\n<browser_result action="closeTab">Closed tab ${action.tabId}</browser_result>`;
            } else if (action.type === "switchTab") {
              setResearchStatus(`Browser: switching to tab ${action.tabId}...`);
              await agentBrowser.switchTab(action.tabId);
              await new Promise(r => setTimeout(r, 200));
              browserContext += `\n\n<browser_result action="switchTab">Switched to tab ${action.tabId}. URL: ${agentBrowser.currentUrl || "(unknown)"}</browser_result>`;
            }
           } catch (browserErr) {
              console.warn("Browser action failed:", browserErr);
              browserContext += `\n\n<browser_result action="${action.type}" error="true">Action failed: ${browserErr.message || "unknown error"}. The browser may have closed or become unresponsive.</browser_result>`;
           }
          }

          // Include tab info in the browser results
          let tabInfo = "";
          try {
            const tabsResult = await agentBrowser.getTabs();
            if (tabsResult?.tabs?.length > 0) {
              tabInfo = "\nOpen tabs:\n" + tabsResult.tabs.map(t => `  - Tab ${t.id}: ${t.title || t.url || "New Tab"}${t.active ? " (active)" : ""}`).join("\n");
            }
          } catch {}

          currentMsgs = [...currentMsgs, {
            role: "user",
            content: `[SYSTEM: Browser agent results]\nCurrent browser URL: ${agentBrowser.currentUrl || "(unknown)"}${tabInfo}${browserContext}\n\nContinue your task. You can take more browser actions (including tab management) or provide your answer to the user.`
          }];
          setMsgs([...currentMsgs]);
          saveChat(currentMsgs); // Save after each browser round
          continue;
        }

        // ─── Terminal Command Execution ───
        if (actions.terminalCommands.length > 0) {
          researchRound++;
          let termContext = "";
          for (const code of actions.terminalCommands) {
           try {
            setResearchStatus(`Terminal: executing code...`);
            // Capture console output
            const logs = [];
            const origLog = console.log, origWarn = console.warn, origError = console.error;
            const stringify = (args) => args.map(a => {
              if (a === undefined) return "undefined";
              if (a === null) return "null";
              if (typeof a === "object") { try { return JSON.stringify(a, null, 2); } catch { return String(a); } }
              return String(a);
            }).join(" ");
            console.log = (...args) => { logs.push(stringify(args)); origLog(...args); };
            console.warn = (...args) => { logs.push("[warn] " + stringify(args)); origWarn(...args); };
            console.error = (...args) => { logs.push("[error] " + stringify(args)); origError(...args); };

            let result;
            try {
              // eslint-disable-next-line no-eval
              const evalResult = eval(code);
              const resultStr = evalResult !== undefined ? String(evalResult) : "(no return value)";
              const consoleOutput = logs.length > 0 ? "\nConsole output:\n" + logs.join("\n") : "";
              result = `<terminal_result>\n$ ${code}\n=> ${resultStr}${consoleOutput}\n</terminal_result>`;
              // Add to terminal UI
              setTerminalHistory(prev => [...prev, { type: "input", text: code }, { type: "output", text: resultStr, logs: logs.map(l => ({ level: "log", text: l })) }]);
            } catch (e) {
              const consoleOutput = logs.length > 0 ? "\nConsole output:\n" + logs.join("\n") : "";
              result = `<terminal_result error="true">\n$ ${code}\nError: ${String(e)}${consoleOutput}\n</terminal_result>`;
              setTerminalHistory(prev => [...prev, { type: "input", text: code }, { type: "error", text: String(e), logs: logs.map(l => ({ level: "log", text: l })) }]);
            }

            console.log = origLog;
            console.warn = origWarn;
            console.error = origError;
            termContext += "\n\n" + result;
           } catch (termErr) {
            console.warn("Terminal execution wrapper failed:", termErr);
            termContext += `\n\n<terminal_result error="true">Terminal execution failed: ${termErr.message || "unknown error"}</terminal_result>`;
           }
          }

          currentMsgs = [...currentMsgs, {
            role: "user",
            content: `[SYSTEM: Terminal execution results]${termContext}\n\nContinue your task. You can execute more code or provide your answer.`
          }];
          setMsgs([...currentMsgs]);
          saveChat(currentMsgs); // Save after each terminal round
          continue;
        }

        // No more research needed
        if (usedModel !== DEFAULT_MODEL) {
          setErr(`Primary model unavailable; used ${usedModel}.`);
        }
        break;
      }
    } catch (e) {
      if (e.name !== "AbortError") setErr(e.message);
      // Save whatever we have even on error
      try { if (currentMsgs && currentMsgs.length > 0) saveChat(currentMsgs); } catch {}
    } finally {
      setBusy(false);
      busyRef.current = false;
      setResearchStatus("");
      abortRef.current = null;
    }
  }, [input, msgs, busy, buildSystem, parseResponse, callAI, apiKey, groqApiKey, promptForApiKey, doSearch, attachments]);

  // ─── Expression image resolver — blink overrides all other states ───
  const getExprImg = useCallback((speakingOverride = false) => {
    if (isBlinking) return "./Expressions/Blink.png";
    if (speakingOverride || busy) return "./Expressions/HappySpeak.png";
    if (expression === "serious") return "./Expressions/Serious.png";
    if (expression === "veryHappy") return "./Expressions/VeryHappy.png";
    return "./Expressions/Happy.png";
  }, [isBlinking, busy, expression]);

  const clearChat = () => { setMsgs([]); saveChat([]); clearVal(LEGACY_CHAT_STORAGE_KEY); setSearchResults([]); setErr(null); };
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
              <span style={{ fontWeight: 700, fontSize: "13px", letterSpacing: "-0.2px" }}>Internet Browser</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} style={{ background: "none", border: "none", color: "var(--dm)", cursor: "pointer", fontSize: "16px" }}>×</button>
          </div>

          {/* Sidebar Tabs */}
          <div style={{ display: "flex", borderBottom: "1px solid var(--bd)" }}>
            <button
              onClick={() => setSidebarTab("browser")}
              style={{ flex: 1, padding: "8px", background: sidebarTab === "browser" ? "rgba(136,187,204,0.08)" : "transparent", border: "none", borderBottom: sidebarTab === "browser" ? "2px solid var(--ac2)" : "2px solid transparent", color: sidebarTab === "browser" ? "var(--ac2)" : "var(--dm)", cursor: "pointer", fontSize: "11px", fontFamily: "var(--m)", fontWeight: 600 }}
            >Search & Browse</button>
            <button
              onClick={() => setSidebarTab("memory")}
              style={{ flex: 1, padding: "8px", background: sidebarTab === "memory" ? "rgba(124,224,138,0.08)" : "transparent", border: "none", borderBottom: sidebarTab === "memory" ? "2px solid var(--ac)" : "2px solid transparent", color: sidebarTab === "memory" ? "var(--ac)" : "var(--dm)", cursor: "pointer", fontSize: "11px", fontFamily: "var(--m)", fontWeight: 600 }}
            >Memory</button>
            <button
              onClick={() => setSidebarTab("terminal")}
              style={{ flex: 1, padding: "8px", background: sidebarTab === "terminal" ? "rgba(200,160,255,0.08)" : "transparent", border: "none", borderBottom: sidebarTab === "terminal" ? "2px solid #c8a0ff" : "2px solid transparent", color: sidebarTab === "terminal" ? "#c8a0ff" : "var(--dm)", cursor: "pointer", fontSize: "11px", fontFamily: "var(--m)", fontWeight: 600 }}
            >Terminal</button>
          </div>

          {/* ─── Browser / Search Tab ─── */}
          {sidebarTab === "browser" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
              {/* Agent Browser Status */}
              <div style={{ padding: "7px 10px", borderBottom: "1px solid var(--bd)", background: (showBrowser || isBrowserOpen()) ? "rgba(124,224,138,0.04)" : "transparent" }}>
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "5px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
                    <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: (showBrowser || isBrowserOpen()) ? (agentUserTookOver ? "#cc7777" : "#7ce08a") : "#444", display: "inline-block", flexShrink: 0 }}></span>
                    <span style={{ fontSize: "10px", fontFamily: "var(--m)", fontWeight: 700, color: (showBrowser || isBrowserOpen()) ? (agentUserTookOver ? "#cc7777" : "#7ce08a") : "#555" }}>
                      {(showBrowser || isBrowserOpen()) ? (agentUserTookOver ? "USER CONTROL" : "AI AGENT ACTIVE") : "BROWSER CLOSED"}
                    </span>
                  </div>
                  <button
                    onClick={() => {
                      setPopupBlocked(false);
                      if (showBrowser || isBrowserOpen()) {
                        // Scroll browser into view
                        scrollRef.current?.scrollIntoView({ behavior: "smooth" });
                      } else {
                        agentBrowser.open();
                        setAgentUserTookOver(false);
                      }
                    }}
                    style={{ ...btn(popupBlocked ? "#cc7777" : (showBrowser || isBrowserOpen()) ? "#7ce08a" : "#88bbcc"), fontSize: "9px", padding: "2px 7px" }}
                  >{(showBrowser || isBrowserOpen()) ? "Focus" : "Open Browser"}</button>
                </div>
                {popupBlocked && (
                  <div style={{ fontSize: "9px", fontFamily: "var(--m)", color: "#cc7777", background: "rgba(204,119,119,0.08)", border: "1px solid rgba(204,119,119,0.2)", borderRadius: "4px", padding: "3px 6px", marginTop: "4px" }}>
                    ⚠ Browser failed to load. Click <strong>Open Browser</strong> above to retry.
                  </div>
                )}
                {agentBrowserUrl && !popupBlocked && (showBrowser || isBrowserOpen()) && (
                  <div style={{ fontSize: "9px", fontFamily: "var(--m)", color: "#445", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }} title={agentBrowserUrl}>
                    {agentBrowserUrl.slice(0, 50)}{agentBrowserUrl.length > 50 ? "…" : ""}
                  </div>
                )}
              </div>
              {/* URL Bar */}
              <div style={{ padding: "8px 10px", borderBottom: "1px solid var(--bd)" }}>
                <div style={{ display: "flex", gap: "4px", marginBottom: "6px" }}>
                  <input
                    value={browserUrl}
                    onChange={e => setBrowserUrl(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") handleBrowserGo(); }}
                    placeholder="Enter URL to open in agent browser..."
                    style={{ flex: 1, padding: "6px 8px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--bd)", borderRadius: "5px", color: "var(--tx)", fontSize: "11px", fontFamily: "var(--m)", outline: "none" }}
                  />
                  <button onClick={handleBrowserGo} style={btn("#88bbcc")} title="Open in agent browser">Go</button>
                </div>
                <form onSubmit={handleSearchSubmit} style={{ display: "flex", gap: "4px" }}>
                  <input
                    value={searchQuery}
                    onChange={e => setSearchQuery(e.target.value)}
                    placeholder="Search the web..."
                    style={{ flex: 1, padding: "6px 8px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--bd)", borderRadius: "5px", color: "var(--tx)", fontSize: "11px", fontFamily: "var(--m)", outline: "none" }}
                  />
                  <button type="submit" disabled={searchBusy} style={{ ...btn("#7ce08a"), opacity: searchBusy ? 0.5 : 1 }}>
                    {searchBusy ? "..." : "Search"}
                  </button>
                </form>
              </div>

              {/* Search Results */}
              <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px" }}>
                {searchBusy && (
                  <div style={{ color: "var(--dm)", fontSize: "11px", padding: "8px 0", fontFamily: "var(--m)" }}>
                    Searching...
                  </div>
                )}
                {!searchBusy && searchResults.length === 0 && (
                  <div style={{ color: "var(--dm)", fontSize: "11px", padding: "12px 0", textAlign: "center", lineHeight: 1.8 }}>
                    Search the web or enter a URL above.<br/>
                    AI can also search autonomously during chat.
                  </div>
                )}
                {searchResults.map((r, i) => (
                  <div key={i} style={{ marginBottom: "10px", padding: "8px", background: "rgba(255,255,255,0.02)", borderRadius: "6px", border: "1px solid var(--bd)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "6px" }}>
                      <a
                        href={r.url}
                        target="_blank"
                        rel="noopener"
                        onClick={(e) => { e.preventDefault(); openBrowserPopup(r.url); }}
                        style={{ color: "var(--ac2)", fontSize: "11.5px", fontWeight: 600, textDecoration: "none", cursor: "pointer", flex: 1, lineHeight: 1.4 }}
                        title={r.url}
                      >
                        {r.title}
                      </a>
                      <button
                        onClick={() => openBrowserPopup(r.url)}
                        style={{ ...btn("#88bbcc"), padding: "2px 6px", fontSize: "9px", flexShrink: 0 }}
                        title="Open in popup"
                      >Open</button>
                    </div>
                    {r.snippet && (
                      <div style={{ fontSize: "10.5px", color: "var(--dm)", marginTop: "4px", lineHeight: 1.5 }}>
                        {r.snippet.slice(0, 150)}{r.snippet.length > 150 ? "..." : ""}
                      </div>
                    )}
                    <div style={{ fontSize: "9px", color: "#335", marginTop: "3px", fontFamily: "var(--m)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {r.url.slice(0, 60)}{r.url.length > 60 ? "..." : ""}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ─── Memory Tab ─── */}
          {sidebarTab === "memory" && (
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

          {/* ─── Terminal Tab ─── */}
          {sidebarTab === "terminal" && (
            <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: "#0a0a10" }}>
              <div style={{ flex: 1, overflowY: "auto", padding: "8px 10px", fontFamily: "var(--m)", fontSize: "11px", lineHeight: 1.6 }}>
                {terminalHistory.map((entry, i) => (
                  <div key={i} style={{ marginBottom: "2px" }}>
                    {entry.type === "system" && (
                      <div style={{ color: "#c8a0ff", whiteSpace: "pre-wrap", opacity: 0.7 }}>{entry.text}</div>
                    )}
                    {entry.type === "input" && (
                      <div style={{ color: "#7ce08a" }}>
                        <span style={{ color: "#c8a0ff", marginRight: "6px" }}>{">"}</span>
                        <span style={{ whiteSpace: "pre-wrap" }}>{entry.text}</span>
                      </div>
                    )}
                    {entry.type === "output" && (
                      <div>
                        {entry.logs && entry.logs.map((log, li) => (
                          <div key={li} style={{ color: log.level === "error" ? "#cc7777" : log.level === "warn" ? "#ccaa55" : "#88bbcc", whiteSpace: "pre-wrap", paddingLeft: "12px" }}>
                            {log.text}
                          </div>
                        ))}
                        <div style={{ color: "#ccc", whiteSpace: "pre-wrap", paddingLeft: "12px" }}>{entry.text !== "undefined" ? entry.text : ""}</div>
                      </div>
                    )}
                    {entry.type === "error" && (
                      <div>
                        {entry.logs && entry.logs.map((log, li) => (
                          <div key={li} style={{ color: log.level === "error" ? "#cc7777" : "#88bbcc", whiteSpace: "pre-wrap", paddingLeft: "12px" }}>
                            {log.text}
                          </div>
                        ))}
                        <div style={{ color: "#cc7777", whiteSpace: "pre-wrap", paddingLeft: "12px" }}>{entry.text}</div>
                      </div>
                    )}
                  </div>
                ))}
                <div ref={terminalScrollRef} />
              </div>
              <div style={{ padding: "6px 8px", borderTop: "1px solid var(--bd)", display: "flex", alignItems: "center", gap: "4px", background: "rgba(0,0,0,0.3)" }}>
                <span style={{ color: "#c8a0ff", fontFamily: "var(--m)", fontSize: "12px", flexShrink: 0 }}>{">"}</span>
                <input
                  ref={terminalInputRef}
                  value={terminalInput}
                  onChange={e => setTerminalInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      executeTerminal(terminalInput);
                      setTerminalInput("");
                    } else if (e.key === "ArrowUp") {
                      e.preventDefault();
                      if (terminalCmdHistory.length > 0) {
                        const newIdx = terminalHistoryIdx < 0 ? terminalCmdHistory.length - 1 : Math.max(0, terminalHistoryIdx - 1);
                        setTerminalHistoryIdx(newIdx);
                        setTerminalInput(terminalCmdHistory[newIdx] || "");
                      }
                    } else if (e.key === "ArrowDown") {
                      e.preventDefault();
                      if (terminalHistoryIdx >= 0) {
                        const newIdx = terminalHistoryIdx + 1;
                        if (newIdx >= terminalCmdHistory.length) {
                          setTerminalHistoryIdx(-1);
                          setTerminalInput("");
                        } else {
                          setTerminalHistoryIdx(newIdx);
                          setTerminalInput(terminalCmdHistory[newIdx] || "");
                        }
                      }
                    }
                  }}
                  placeholder="Enter JavaScript..."
                  style={{ flex: 1, padding: "5px 6px", background: "rgba(255,255,255,0.03)", border: "1px solid var(--bd)", borderRadius: "4px", color: "#7ce08a", fontSize: "11px", fontFamily: "var(--m)", outline: "none" }}
                />
              </div>
              <div style={{ padding: "4px 8px", borderTop: "1px solid var(--bd)", display: "flex", gap: "4px" }}>
                <button onClick={() => setTerminalHistory([{ type: "system", text: "Terminal cleared.\n" }])} style={btn("#c8a0ff")}>Clear</button>
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
            <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>Groq (qwen3-32b) · OpenRouter fallback</span>
          </div>
          <div style={{ display: "flex", gap: "4px", alignItems: "center", flexWrap: "wrap" }}>
            <button
              onClick={() => promptForGroqKey("Set or update your Groq API key:")}
              style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", color: groqApiKey ? "var(--ac2)" : "var(--dg)", borderColor: groqApiKey ? "rgba(136,187,204,0.2)" : "rgba(204,119,119,0.2)" }}
              title={groqApiKey ? "Groq API key set (default)" : "Groq API key missing (default)"}
            >
              {groqApiKey ? "GROQ ✓" : "GROQ !"}
            </button>
            <button
              onClick={() => promptForApiKey("Set or update your OpenRouter API key:")}
              style={{ ...hdr(), fontSize: "10px", fontFamily: "var(--m)", color: apiKey ? "var(--ac)" : "var(--dg)", borderColor: apiKey ? "rgba(124,224,138,0.2)" : "rgba(204,119,119,0.2)" }}
              title={apiKey ? "OpenRouter API key set (fallback)" : "OpenRouter API key missing (fallback)"}
            >
              {apiKey ? "OpenRouter ✓" : "OpenRouter !"}
            </button>
            <span style={{ fontSize: "9px", color: "var(--dm)", fontFamily: "var(--m)", padding: "2px 6px", background: "rgba(255,255,255,0.02)", borderRadius: "3px" }}>↑{ft(usage.i)} ↓{ft(usage.o)}</span>
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              style={{ ...hdr(), background: sidebarOpen ? "rgba(136,187,204,0.08)" : undefined, color: sidebarOpen ? "var(--ac2)" : undefined, borderColor: sidebarOpen ? "rgba(136,187,204,0.15)" : undefined, display: "flex", alignItems: "center", gap: "4px" }}
              title="Internet Browser & Memory Panel"
            >
              <span style={{ fontSize: "13px" }}>🧠</span>
              <span style={{ fontSize: "10px", fontFamily: "var(--m)" }}>Browser</span>
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
                  AI agent with persistent memory, web search, and a visual browser it can control.<br/>
                  Ask it to browse, click, fill forms — or open the sidebar to control the browser yourself.
                </div>
              </div>
            )}
            <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
              {msgs.map((m, i) => {
                // Hide system research messages from display
                if (m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM: Research results")) {
                  return (
                    <div key={i} style={{ padding: "6px 10px", background: "rgba(136,187,204,0.05)", border: "1px solid rgba(136,187,204,0.1)", borderRadius: "8px", fontSize: "11px", color: "var(--ac2)", fontFamily: "var(--m)" }}>
                      Research data received — AI processing results...
                    </div>
                  );
                }
                if (m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM: Browser agent results]")) {
                  return (
                    <div key={i} style={{ padding: "6px 10px", background: "rgba(124,224,138,0.05)", border: "1px solid rgba(124,224,138,0.12)", borderRadius: "8px", fontSize: "11px", color: "var(--ac)", fontFamily: "var(--m)", display: "flex", alignItems: "center", gap: "6px" }}>
                      <span style={{ fontSize: "9px" }}>●</span> Browser action results received — AI continuing task...
                    </div>
                  );
                }
                if (m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM: Terminal execution results]")) {
                  return (
                    <div key={i} style={{ padding: "6px 10px", background: "rgba(200,160,255,0.05)", border: "1px solid rgba(200,160,255,0.12)", borderRadius: "8px", fontSize: "11px", color: "#c8a0ff", fontFamily: "var(--m)", display: "flex", alignItems: "center", gap: "6px" }}>
                      <span style={{ fontSize: "9px" }}>●</span> Terminal execution results received — AI continuing task...
                    </div>
                  );
                }
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
                  {researchStatus && <span style={{ color: "var(--ac2)", fontFamily: "var(--m)", fontSize: "10px" }}>{researchStatus}</span>}
                </div>
              )}
              {err && <div style={{ color: "#f88", fontSize: "12px", padding: "6px 2px" }}>{err}</div>}

              {/* ═══ EMBEDDED BROWSER ═══ */}
              {showBrowser && browserBlobUrl && (
                <div style={{ marginTop: "10px", borderRadius: "10px", border: "1px solid var(--bd)", overflow: "hidden", background: "#08080f", animation: "fadeIn .25s ease" }}>
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "6px 10px", background: "rgba(124,224,138,0.04)", borderBottom: "1px solid var(--bd)" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                      <span style={{ width: "7px", height: "7px", borderRadius: "50%", background: agentUserTookOver ? "#cc7777" : "#7ce08a", display: "inline-block" }}></span>
                      <span style={{ fontSize: "10px", fontFamily: "var(--m)", fontWeight: 700, color: agentUserTookOver ? "#cc7777" : "#7ce08a" }}>
                        {agentUserTookOver ? "USER CONTROL" : "AI AGENT BROWSER"}
                      </span>
                      {agentBrowserUrl && (
                        <span style={{ fontSize: "9px", fontFamily: "var(--m)", color: "#445", marginLeft: "6px" }} title={agentBrowserUrl}>
                          {agentBrowserUrl.length > 60 ? agentBrowserUrl.slice(0, 60) + "…" : agentBrowserUrl}
                        </span>
                      )}
                    </div>
                    <button
                      onClick={() => { setShowBrowser(false); agentBrowser.close(); }}
                      style={{ background: "none", border: "none", color: "var(--dm)", cursor: "pointer", fontSize: "16px", lineHeight: 1, padding: "0 4px" }}
                      title="Close browser"
                    >×</button>
                  </div>
                  <iframe
                    ref={browserIframeRef}
                    src={browserBlobUrl}
                    onLoad={() => {
                      if (browserIframeRef.current) {
                        agentBrowser.setEmbeddedIframe(browserIframeRef.current);
                      }
                    }}
                    style={{ width: "100%", height: "500px", border: "none", display: "block", background: "#0d0d14" }}
                    sandbox="allow-scripts allow-forms allow-popups allow-same-origin allow-popups-to-escape-sandbox allow-top-navigation-by-user-activation"
                  />
                </div>
              )}

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
                placeholder={attachments.length > 0 ? "Add a message about your files... (optional)" : "Type a message... (Auto can search the web for you!)"}
                style={{ flex: 1, minHeight: "44px", maxHeight: "180px", resize: "vertical", borderRadius: "8px", border: "1px solid var(--bd)", background: "rgba(255,255,255,0.02)", color: "var(--tx)", padding: "10px 12px", fontFamily: "var(--f)", fontSize: "13px", outline: "none" }}
              />
            </div>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "7px" }}>
              <span style={{ fontSize: "10px", color: "var(--dm)", fontFamily: "var(--m)" }}>
                {msgs.filter(m => !(m.role === "user" && typeof m.content === "string" && m.content.startsWith("[SYSTEM:"))).length} msgs
                {attachments.length > 0 && <span style={{ color: "var(--ac)", marginLeft: "8px" }}>{attachments.length} file{attachments.length > 1 ? "s" : ""} attached</span>}
                {(showBrowser || isBrowserOpen()) && <span style={{ color: agentUserTookOver ? "var(--dg)" : "var(--ac)", marginLeft: "8px" }}>
                  {agentUserTookOver ? "browser: user control" : "browser: AI agent active"}
                </span>}
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
