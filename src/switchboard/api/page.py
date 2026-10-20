"""The `/activity` viewer: one self-contained HTML page, no build step, no external scripts.

It reads `/v1/activity` and `/v1/activity/histogram` and draws the frequency chart as inline
SVG. All event data is inserted with `textContent`, never as HTML.
"""

ACTIVITY_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Switchboard Activity</title>
<style>
:root {
  --bg: #0b0d10; --panel: #12151a; --line: #222831; --text: #d6dae0; --muted: #7d8590;
  --info: #8b98a9; --warn: #e3b341; --error: #f85149; --accent: #6cb6ff;
  --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
header { padding: 16px 20px 8px; display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
h1 { font-size: 15px; font-weight: 600; margin: 0; letter-spacing: .02em; }
#status { color: var(--muted); font-size: 12px; font-family: var(--mono); }
form { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 20px 12px; }
input, select, button { background: var(--panel); color: var(--text); border: 1px solid var(--line);
  border-radius: 6px; padding: 6px 9px; font: 13px var(--mono); }
input { min-width: 0; flex: 1 1 160px; }
button { cursor: pointer; }
button:hover { border-color: var(--accent); }
label.live { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 12px; }
section { margin: 0 20px 16px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 8px; }
#chart { padding: 12px 12px 4px; }
#chart svg { width: 100%; height: 140px; display: block; }
#legend { display: flex; flex-wrap: wrap; gap: 12px; padding: 4px 12px 10px;
  font: 11px var(--mono); color: var(--muted); }
#legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px;
  margin-right: 5px; vertical-align: -1px; }
.table { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 7px 12px; border-bottom: 1px solid var(--line);
  vertical-align: top; }
th { color: var(--muted); font-weight: 500; font-size: 12px; }
tr.row { cursor: pointer; }
tr.row:hover td { background: #161a20; }
td.time, td.conv, td.leg { font: 12px var(--mono); color: var(--muted); white-space: nowrap; }
td.type { font: 12px var(--mono); white-space: nowrap; }
td.conv a { color: var(--accent); text-decoration: none; }
.sev { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
.sev.info { background: var(--info); }
.sev.warn { background: var(--warn); }
.sev.error { background: var(--error); }
tr.warn td.summary { color: var(--warn); } tr.error td.summary { color: var(--error); }
pre { margin: 0; padding: 10px 12px; font: 12px var(--mono); color: var(--muted);
  white-space: pre-wrap; word-break: break-word; }
#more { margin: 0 20px 24px; }
#error { color: var(--error); padding: 0 20px 8px; font: 12px var(--mono); }
@media (max-width: 640px) { td.leg, th.leg { display: none; } }
</style>
</head>
<body>
<header><h1>Switchboard activity</h1><span id="status"></span></header>
<form id="filters">
  <input name="type" placeholder="type or prefix, e.g. policy." autocomplete="off">
  <input name="conversation_id" placeholder="conversation id" autocomplete="off">
  <input name="q" placeholder="search payload" autocomplete="off">
  <select name="bucket" title="histogram bucket">
    <option value="auto">auto buckets</option><option value="1">1 s</option>
    <option value="10">10 s</option><option value="60">1 min</option>
    <option value="300">5 min</option>
    <option value="3600">1 h</option>
  </select>
  <button type="submit">Apply</button>
  <label class="live"><input type="checkbox" id="live"> live</label>
</form>
<div id="error"></div>
<section><div id="chart"></div><div id="legend"></div></section>
<section class="table">
  <table>
    <thead><tr>
      <th>time</th><th>type</th><th>conversation</th><th class="leg">leg</th><th>summary</th>
    </tr></thead>
    <tbody id="rows"></tbody>
  </table>
</section>
<button id="more" hidden>Load more</button>
<script>
"use strict";
const COLORS = { conversation: "#8b98a9", leg: "#768390", turn: "#6cb6ff", brain: "#b083f0",
  tool: "#57ab5a", policy: "#e3b341", speech: "#39c5cf", procedure: "#d2a8ff", child: "#f69d50",
  knowledge: "#96d0ff", handoff: "#f85149" };
const form = document.getElementById("filters");
const rows = document.getElementById("rows");
const more = document.getElementById("more");
const errorBox = document.getElementById("error");
const statusBox = document.getElementById("status");
let nextCursor = null;
let timer = null;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function filterParams() {
  const data = new FormData(form);
  const params = new URLSearchParams();
  for (const key of ["type", "conversation_id", "q"]) {
    const value = String(data.get(key) || "").trim();
    if (!value) continue;
    if (key === "type") value.split(/[ ,]+/).filter(Boolean).forEach(t => params.append("type", t));
    else params.set(key, value);
  }
  return params;
}

async function getJSON(url) {
  const res = await fetch(url, { headers: { accept: "application/json" } });
  const body = await res.json();
  if (!res.ok) {
    const detail = Array.isArray(body.detail)
      ? body.detail.map(d => (d.loc ? d.loc[d.loc.length - 1] + ": " : "") + d.msg).join("; ")
      : body.detail;
    const err = new Error(detail || res.statusText);
    err.status = res.status;
    throw err;
  }
  return body;
}

function formatTime(item) {
  if (item.wall_time) return item.wall_time.replace("T", " ").replace("+00:00", "Z").slice(0, 23);
  return "t+" + item.at.toFixed(3) + "s";
}

function renderItems(items) {
  for (const item of items) {
    const tr = el("tr", "row " + item.severity);
    const time = el("td", "time", formatTime(item));
    const type = el("td", "type");
    type.append(el("span", "sev " + item.severity), document.createTextNode(item.type));
    const conv = el("td", "conv");
    const link = el("a", null, item.conversation_id);
    link.href = "#";
    link.title = "filter to this conversation";
    link.addEventListener("click", ev => {
      ev.preventDefault(); ev.stopPropagation();
      form.elements.conversation_id.value = item.conversation_id;
      reload();
    });
    conv.append(link);
    const leg = el("td", "leg", item.leg_id || "");
    tr.append(time, type, conv, leg, el("td", "summary", item.summary));
    const detail = el("tr");
    detail.hidden = true;
    const cell = el("td");
    cell.colSpan = 5;
    cell.append(el("pre", null, JSON.stringify(item.payload, null, 2)));
    detail.append(cell);
    tr.addEventListener("click", () => { detail.hidden = !detail.hidden; });
    rows.append(tr, detail);
  }
}

async function loadPage(reset) {
  const params = filterParams();
  params.set("limit", "100");
  if (!reset && nextCursor) params.set("cursor", nextCursor);
  const body = await getJSON("/v1/activity?" + params.toString());
  if (reset) rows.replaceChildren();
  renderItems(body.items);
  nextCursor = body.next_cursor;
  more.hidden = !nextCursor;
  statusBox.textContent = rows.querySelectorAll("tr.row").length + " events shown";
}

async function loadHistogram() {
  const choice = form.elements.bucket.value;
  let bucket = choice === "auto" ? 1 : Number(choice);
  for (let attempt = 0; attempt < 6; attempt++) {
    const params = filterParams();
    params.set("bucket_s", String(bucket));
    try {
      const body = await getJSON("/v1/activity/histogram?" + params.toString());
      if (choice === "auto" && body.buckets.length > 120) { bucket *= 10; continue; }
      drawChart(body.buckets, body.bucket_s);
      return;
    } catch (err) {
      if (choice === "auto" && err.status === 422) { bucket *= 10; continue; }
      throw err;
    }
  }
}

function drawChart(buckets, bucketS) {
  const chart = document.getElementById("chart");
  const legend = document.getElementById("legend");
  const NS = "http://www.w3.org/2000/svg";
  const W = 1000, H = 140, PAD = 18;
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", "0 0 " + W + " " + H);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "event frequency");
  const max = Math.max(1, ...buckets.map(b => b.total));
  const width = buckets.length ? W / buckets.length : W;
  const seen = new Set();
  buckets.forEach((b, i) => {
    let y = H - PAD;
    for (const [type, count] of Object.entries(b.counts)) {
      const family = type.split(".")[0];
      seen.add(family);
      const h = (count / max) * (H - PAD - 6);
      const rect = document.createElementNS(NS, "rect");
      rect.setAttribute("x", String(i * width + 0.5));
      rect.setAttribute("y", String(y - h));
      rect.setAttribute("width", String(Math.max(1, width - 1)));
      rect.setAttribute("height", String(h));
      rect.setAttribute("fill", COLORS[family] || "#768390");
      const title = document.createElementNS(NS, "title");
      title.textContent = type + ": " + count + " (bucket " + b.start + "s, " + bucketS + "s wide)";
      rect.append(title);
      svg.append(rect);
      y -= h;
    }
  });
  const axis = document.createElementNS(NS, "line");
  axis.setAttribute("x1", "0"); axis.setAttribute("x2", String(W));
  axis.setAttribute("y1", String(H - PAD)); axis.setAttribute("y2", String(H - PAD));
  axis.setAttribute("stroke", "#222831");
  svg.append(axis);
  const label = document.createElementNS(NS, "text");
  label.setAttribute("x", "4"); label.setAttribute("y", "12");
  label.setAttribute("fill", "#7d8590"); label.setAttribute("font-size", "11");
  label.setAttribute("font-family", "ui-monospace, monospace");
  label.textContent = "max " + max + " / " + bucketS + "s";
  svg.append(label);
  chart.replaceChildren(svg);
  legend.replaceChildren();
  for (const family of [...seen].sort()) {
    const span = el("span");
    const swatch = el("i");
    swatch.style.background = COLORS[family] || "#768390";
    span.append(swatch, document.createTextNode(family));
    legend.append(span);
  }
  if (!buckets.length) legend.append(el("span", null, "no events in range"));
}

async function reload() {
  errorBox.textContent = "";
  nextCursor = null;
  try {
    await Promise.all([loadPage(true), loadHistogram()]);
  } catch (err) {
    errorBox.textContent = String(err.message || err);
  }
}

form.addEventListener("submit", ev => { ev.preventDefault(); reload(); });
more.addEventListener("click", () => {
  loadPage(false).catch(err => { errorBox.textContent = err.message; });
});
document.getElementById("live").addEventListener("change", ev => {
  if (timer) { clearInterval(timer); timer = null; }
  if (ev.target.checked) timer = setInterval(reload, 5000);
});
reload();
</script>
</body>
</html>
"""
