"use strict";

// ---------- helpers ----------
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const num = (x, d = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));
const fmtInt = (x) => Number(x).toLocaleString();
const img = (id, size) => `/img/${encodeURIComponent(id)}?size=${size}`;
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const duration = (s) =>
  s < 60 ? `${s.toFixed(1)} s` : s < 3600 ? `${(s / 60).toFixed(1)} min` : s < 86400 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`;

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (t.hidden = true), 2600);
}

// ---------- state ----------
const state = {
  data: null,
  filter: "multi",
  sort: "time",
  query: "",
  visible: [], // groups currently shown, in order (compare view walks this list)
};
const groupById = new Map();

/** Members of a group: the current pick first, then by automatic rank. */
function ordered(group) {
  const pick = group.members.find((m) => m.id === group.pick_id);
  return [pick, ...group.members.filter((m) => m.id !== group.pick_id)];
}

// ---------- load ----------
async function load() {
  const res = await fetch("/api/run");
  if (!res.ok) throw new Error(`server said ${res.status}`);
  const data = await res.json();
  state.data = data;
  for (const g of data.groups) groupById.set(g.id, g);
  renderStats();
  renderGroups();
  renderRejected();
}

function renderStats() {
  const { stats, model, groups, rejected } = state.data;
  const fewer = Math.round(stats.reduction_pct); // same figure as the CLI summary
  $("#stats").innerHTML =
    `<span><b>${fmtInt(stats.input)}</b> photos</span>` +
    `<span><b>${fmtInt(stats.rejected_sharpness)}</b> blurry</span>` +
    `<span><b>${fmtInt(stats.clusters)}</b> picks</span>` +
    `<span><b>${fewer}%</b> fewer to review</span>`;
  $("#count-groups").textContent = fmtInt(groups.length);
  $("#count-rejected").textContent = fmtInt(rejected.length);
  if (model && model.trained === false) {
    const b = $("#model-banner");
    b.hidden = false;
    b.textContent = "Aesthetic model is an untrained placeholder: aesthetic scores are not meaningful yet, so picks mostly follow sharpness. Grouping and the blur filter are unaffected.";
  }
  updateCounts();
}

function updateCounts() {
  const gs = state.data.groups;
  $("#n-all").textContent = gs.length;
  $("#n-multi").textContent = gs.filter((g) => g.members.length > 1).length;
  $("#n-single").textContent = gs.filter((g) => g.members.length === 1).length;
  $("#n-changed").textContent = gs.filter((g) => g.pick_id !== g.best_id).length;
}

// ---------- groups view ----------
function matches(group) {
  const n = group.members.length;
  if (state.filter === "multi" && n < 2) return false;
  if (state.filter === "single" && n !== 1) return false;
  if (state.filter === "changed" && group.pick_id === group.best_id) return false;
  if (state.query) return group.members.some((m) => m.name.toLowerCase().includes(state.query));
  return true;
}

function sortGroups(groups) {
  const byTime = (a, b) => (a.start ?? Infinity) - (b.start ?? Infinity) || a.id - b.id;
  if (state.sort === "size") return groups.sort((a, b) => b.members.length - a.members.length || byTime(a, b));
  return groups.sort(byTime);
}

function cardHtml(group, m) {
  const isPick = m.id === group.pick_id;
  const hit = state.query && m.name.toLowerCase().includes(state.query);
  const label = isPick ? "PICK" : `#${m.rank}`;
  return `<button class="card${isPick ? " pick" : ""}" data-gid="${group.id}" data-id="${esc(m.id)}"
      ${hit ? 'style="outline:2px solid var(--focus)"' : ""}
      title="${esc(m.name)} · score ${num(m.final)} · click to compare with the pick">
    <div class="img"><img loading="lazy" decoding="async" src="${img(m.id, "thumb")}" alt="${esc(m.name)}"></div>
    <span class="badge">${label}</span>${m.raw ? '<span class="raw">RAW</span>' : ""}
    <div class="bar"><i style="width:${Math.round((m.final ?? 0) * 100)}%"></i></div>
    <div class="cap"><span class="nm">${esc(m.name)}</span><span class="sc">${num(m.final)}</span></div>
  </button>`;
}

function groupHtml(g) {
  const n = g.members.length;
  const pick = g.members.find((m) => m.id === g.pick_id);
  const when = pick.clock ? pick.clock.slice(5, 16).replace(" ", " · ") : "no capture time";
  const span = n > 1 && g.span ? ` · over ${duration(g.span)}` : "";
  const changed = g.pick_id !== g.best_id ? '<span class="changed">pick changed by you</span>' : "";
  return `<section class="group" id="g${g.id}">
    <div class="group-head">
      <h3>Group ${g.id} · ${n} photo${n > 1 ? "s" : ""}</h3>
      <span class="meta">${esc(when)}${span}</span>
      <span class="meta">pick: ${esc(pick.name)}</span>${changed}
    </div>
    <div class="photos">${ordered(g).map((m) => cardHtml(g, m)).join("")}</div>
  </section>`;
}

function renderGroups() {
  state.visible = sortGroups(state.data.groups.filter(matches));
  $("#groups").innerHTML = state.visible.map(groupHtml).join("");
  $("#groups-empty").hidden = state.visible.length > 0;
}

function rerenderGroup(g) {
  const el = $(`#g${g.id}`);
  if (el) el.outerHTML = groupHtml(g);
  updateCounts();
}

// ---------- rejected view ----------
function renderRejected() {
  $("#rejected").innerHTML = state.data.rejected
    .slice()
    .sort((a, b) => (a.sharpness ?? 0) - (b.sharpness ?? 0))
    .map((r) => `<button class="card" data-rid="${esc(r.id)}" title="${esc(r.name)}">
        <div class="img"><img loading="lazy" decoding="async" src="${img(r.id, "thumb")}" alt="${esc(r.name)}"></div>
        <span class="badge">${num(r.sharpness, 1)}</span>
        <div class="cap"><span class="nm">${esc(r.name)}</span><span class="sc">${esc(r.reason ?? "")}</span></div>
      </button>`)
    .join("");
}

// ---------- synced zoom / pan viewer ----------
/**
 * Panes share one view: scale (1 = fit) and a centre in normalised image coordinates,
 * so the same spot of every photo stays under the cursor while zooming and panning.
 * Each pane shows a 2048 px preview and swaps to the original file once zoomed past it.
 */
class SyncViewer {
  constructor(paneEls, onChange) {
    this.panes = paneEls.map((el) => ({ el, vp: $(".viewport", el), img: $("img", el), spin: $(".spinner", el),
      id: null, prevW: 0, prevH: 0, fullW: 0, fullH: 0, fullLoaded: false, fullLoading: false }));
    this.view = { scale: 1, cx: 0.5, cy: 0.5 };
    this.onChange = onChange || (() => {});
    for (const p of this.panes) this.bind(p);
    new ResizeObserver(() => this.layout()).observe(this.panes[0].vp);
  }

  setImage(i, id) {
    const p = this.panes[i];
    if (p.id === id) return;
    Object.assign(p, { id, prevW: 0, prevH: 0, fullW: 0, fullH: 0, fullLoaded: false, fullLoading: false });
    p.spin.hidden = false;
    p.img.style.visibility = "hidden";
    p.img.onload = () => {
      if (p.id !== id) return;
      if (!p.fullLoaded) { p.prevW = p.img.naturalWidth; p.prevH = p.img.naturalHeight; }
      p.spin.hidden = true;
      p.img.style.visibility = "visible";
      this.layout();
    };
    p.img.src = img(id, "preview");
  }

  fit(p) {
    const W = p.vp.clientWidth, H = p.vp.clientHeight;
    const a = p.prevW && p.prevH ? p.prevW / p.prevH : 1.5;
    const fw = Math.min(W, H * a);
    return { W, H, fw, fh: fw / a };
  }

  /** scale at which the reference pane shows its original at 100% */
  scale100() {
    const p = this.panes[0];
    const { fw } = this.fit(p);
    const w = p.fullW || p.prevW * 3; // guess until the original is loaded
    return w && fw ? w / fw : 4;
  }

  layout() {
    for (const p of this.panes) {
      if (!p.id || !p.vp.clientWidth) continue;
      const { W, H, fw, fh } = this.fit(p);
      const dw = fw * this.view.scale, dh = fh * this.view.scale;
      const cx = dw <= W ? 0.5 : clamp(this.view.cx, W / 2 / dw, 1 - W / 2 / dw);
      const cy = dh <= H ? 0.5 : clamp(this.view.cy, H / 2 / dh, 1 - H / 2 / dh);
      Object.assign(p.img.style, { width: `${dw}px`, height: `${dh}px`, left: `${W / 2 - cx * dw}px`, top: `${H / 2 - cy * dh}px` });
      if (p.prevW && dw > p.prevW * 1.05) this.loadFull(p);
      p.vp.classList.toggle("pixels", p.fullW > 0 && dw > p.fullW * 1.5);
    }
    this.onChange(this.label());
  }

  loadFull(p) {
    if (p.fullLoaded || p.fullLoading) return;
    p.fullLoading = true;
    p.spin.hidden = false;
    const id = p.id, pre = new Image();
    pre.onload = () => {
      if (p.id !== id) return;
      Object.assign(p, { fullW: pre.naturalWidth, fullH: pre.naturalHeight, fullLoaded: true });
      p.img.src = pre.src;
      p.spin.hidden = true;
      this.layout();
    };
    pre.onerror = () => { p.spin.hidden = true; p.fullLoading = false; };
    pre.src = img(id, "full");
  }

  label() {
    if (this.view.scale <= 1.001) return "fit";
    const p = this.panes[0];
    if (!p.fullW) return `${this.view.scale.toFixed(1)}×`;
    return `${Math.round((this.fit(p).fw * this.view.scale / p.fullW) * 100)}%`;
  }

  zoomAt(p, factor, mx, my) {
    const { W, H, fw, fh } = this.fit(p);
    const max = Math.max(this.scale100() * 4, 2);
    const s0 = this.view.scale, s1 = clamp(s0 * factor, 1, max);
    if (s1 === s0) return;
    // keep the image point under the cursor fixed
    const dw0 = fw * s0, dh0 = fh * s0;
    const left0 = W / 2 - this.view.cx * dw0, top0 = H / 2 - this.view.cy * dh0;
    const px = (mx - left0) / dw0, py = (my - top0) / dh0;
    const dw1 = fw * s1, dh1 = fh * s1;
    this.view = { scale: s1, cx: (W / 2 - (mx - px * dw1)) / dw1, cy: (H / 2 - (my - py * dh1)) / dh1 };
    this.clampView(p);
    this.layout();
  }

  zoomBy(factor) {
    const p = this.panes[0];
    this.zoomAt(p, factor, p.vp.clientWidth / 2, p.vp.clientHeight / 2);
  }

  clampView(p) {
    const { W, H, fw, fh } = this.fit(p);
    const dw = fw * this.view.scale, dh = fh * this.view.scale;
    this.view.cx = dw <= W ? 0.5 : clamp(this.view.cx, W / 2 / dw, 1 - W / 2 / dw);
    this.view.cy = dh <= H ? 0.5 : clamp(this.view.cy, H / 2 / dh, 1 - H / 2 / dh);
  }

  reset() { this.view = { scale: 1, cx: 0.5, cy: 0.5 }; this.layout(); }

  oneToOne() {
    const p = this.panes[0];
    const apply = () => { this.view.scale = this.scale100(); this.clampView(p); this.layout(); };
    if (p.fullLoaded) return apply();
    this.loadFull(p);
    const wait = setInterval(() => { if (p.fullLoaded) { clearInterval(wait); apply(); } }, 50);
    setTimeout(() => clearInterval(wait), 15000);
  }

  bind(p) {
    p.vp.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = p.vp.getBoundingClientRect();
      this.zoomAt(p, Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
    }, { passive: false });
    p.vp.addEventListener("dblclick", (e) => {
      if (this.view.scale > 1.001) return this.reset();
      const r = p.vp.getBoundingClientRect();
      this.zoomAt(p, this.scale100() / this.view.scale, e.clientX - r.left, e.clientY - r.top);
    });
    let drag = null;
    p.vp.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY };
      p.vp.setPointerCapture(e.pointerId);
      p.vp.classList.add("dragging");
    });
    p.vp.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const { fw, fh } = this.fit(p);
      this.view.cx -= (e.clientX - drag.x) / (fw * this.view.scale);
      this.view.cy -= (e.clientY - drag.y) / (fh * this.view.scale);
      drag = { x: e.clientX, y: e.clientY };
      this.clampView(p);
      this.layout();
    });
    const end = () => { drag = null; p.vp.classList.remove("dragging"); };
    p.vp.addEventListener("pointerup", end);
    p.vp.addEventListener("pointercancel", end);
  }
}

// ---------- compare view ----------
const cmp = { group: null, otherId: null, viewer: null };

function metricRows(m, ref, group) {
  const start = group.start;
  const diff = (a, b, d = 2) => {
    if (!ref || a === null || a === undefined || b === null || b === undefined || m.id === ref.id) return "";
    const v = a - b;
    if (Math.abs(v) < Math.pow(10, -d) / 2) return ' <span class="muted">=</span>';
    return ` <span class="${v > 0 ? "better" : "worse"}">${v > 0 ? "▲" : "▼"} ${Math.abs(v).toFixed(d)}</span>`;
  };
  const offset = m.time !== null && start !== null ? ` (+${(m.time - start).toFixed(1)} s)` : "";
  return [
    ["Final score", `${num(m.final)}${diff(m.final, ref?.final)}`],
    ["Rank in group", `#${m.rank}${m.id === group.best_id ? " (automatic pick)" : ""}`],
    ["Sharpness in group (0–1)", `${num(m.sharp_norm)}${diff(m.sharp_norm, ref?.sharp_norm)}`],
    ["Sharpness (raw)", `${num(m.sharpness, 1)}${diff(m.sharpness, ref?.sharpness, 1)}`],
    ["Aesthetic (1–10)", `${num(m.aes_mean)} ± ${num(m.aes_std)}${diff(m.aes_mean, ref?.aes_mean)}`],
    ["Captured", `${esc(m.clock ?? "–")}${offset}`],
    ["ISO · shutter · focal", `${m.iso ?? "–"} · ${esc(m.shutter ?? "–")} · ${esc(m.focal ?? "–")}`],
    ["RAW", esc(m.raw ?? "–")],
  ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

function openCompare(gid, id) {
  const g = groupById.get(gid);
  cmp.group = g;
  if (id === g.pick_id) id = ordered(g)[1]?.id ?? id; // clicked the pick: compare with the runner-up
  cmp.otherId = id;
  $("#compare").hidden = false;
  document.body.style.overflow = "hidden";
  if (!cmp.viewer) cmp.viewer = new SyncViewer($$("#panes .pane"), (lbl) => ($("#zoom-label").textContent = lbl));
  cmp.viewer.view = { scale: 1, cx: 0.5, cy: 0.5 };
  renderCompare();
}

function renderCompare() {
  const g = cmp.group;
  const pick = g.members.find((m) => m.id === g.pick_id);
  const other = g.members.find((m) => m.id === cmp.otherId) ?? pick;
  const single = g.members.length === 1;
  const list = ordered(g);

  $("#cmp-title").textContent = `Group ${g.id} · ${g.members.length} photo${single ? "" : "s"}`;
  $("#cmp-sub").textContent = single ? "Only photo in this group" : `comparing ${list.indexOf(other)} of ${list.length - 1} with the pick`;
  $("#panes").classList.toggle("single", single);
  $('#panes .pane[data-side="other"]').hidden = single;

  const [pp, op] = $$("#panes .pane");
  $(".pane-name", pp).textContent = pick.name;
  $(".pane-name", op).textContent = other.name;
  $(".metrics", pp).innerHTML = metricRows(pick, null, g);
  $(".metrics", op).innerHTML = metricRows(other, pick, g);
  cmp.viewer.setImage(0, pick.id);
  if (!single) cmp.viewer.setImage(1, other.id);
  cmp.viewer.layout();

  const btn = $("#make-pick");
  btn.disabled = single || other.id === g.pick_id;
  btn.textContent = other.id === g.best_id && g.pick_id !== g.best_id ? "Restore automatic pick" : "Make this the pick";

  $("#filmstrip").innerHTML = list.map((m) => `<button class="film${m.id === g.pick_id ? " pick" : ""}${m.id === other.id && !single ? " current" : ""}"
      role="option" aria-selected="${m.id === other.id}" data-id="${esc(m.id)}" title="${esc(m.name)}">
      <img src="${img(m.id, "thumb")}" alt=""><span>${m.id === g.pick_id ? "PICK" : "#" + m.rank} · ${num(m.final)}</span></button>`).join("");
  $("#filmstrip .current")?.scrollIntoView({ block: "nearest", inline: "nearest" });

  // warm the cache for the neighbours
  const i = list.indexOf(other);
  for (const m of [list[i + 1], list[i + 2]]) if (m) new Image().src = img(m.id, "preview");
}

function stepPhoto(delta) {
  const list = ordered(cmp.group).slice(1); // everything except the pick
  if (!list.length) return;
  const i = Math.max(0, list.findIndex((m) => m.id === cmp.otherId));
  cmp.otherId = list[(i + delta + list.length) % list.length].id;
  renderCompare();
}

function stepGroup(delta) {
  const vis = state.visible;
  const i = vis.indexOf(cmp.group);
  const next = vis[i + delta];
  if (!next) return toast(delta > 0 ? "Last group" : "First group");
  openCompare(next.id, ordered(next)[1]?.id ?? next.pick_id);
}

async function makePick() {
  const g = cmp.group, id = cmp.otherId;
  if (id === g.pick_id) return;
  const res = await fetch("/api/pick", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ group: g.id, id }) });
  if (!res.ok) return toast(`Could not save: ${(await res.json()).error ?? res.status}`);
  const previous = g.pick_id;
  g.pick_id = id;
  cmp.otherId = previous; // keep the old pick on the right for comparison
  rerenderGroup(g);
  renderCompare();
  const name = g.members.find((m) => m.id === id).name;
  toast(id === g.best_id ? `Group ${g.id}: back to the automatic pick (${name})` : `${name} is now the pick for group ${g.id} (saved)`);
}

function closeCompare() {
  $("#compare").hidden = true;
  document.body.style.overflow = "";
  $(`#g${cmp.group.id}`)?.scrollIntoView({ block: "nearest" });
}

// ---------- lightbox (rejected) ----------
const lb = { viewer: null };
function openLightbox(rid) {
  const r = state.data.rejected.find((x) => x.id === rid);
  $("#lightbox").hidden = false;
  document.body.style.overflow = "hidden";
  $("#lb-title").textContent = r.name;
  $("#lb-sub").textContent = `sharpness ${num(r.sharpness, 1)} · ${r.reason ?? ""}`;
  if (!lb.viewer) lb.viewer = new SyncViewer($$("#lightbox .pane"));
  lb.viewer.view = { scale: 1, cx: 0.5, cy: 0.5 };
  lb.viewer.setImage(0, r.id);
  lb.viewer.layout();
}
function closeLightbox() { $("#lightbox").hidden = true; document.body.style.overflow = ""; }

// ---------- events ----------
function bindEvents() {
  $$(".tab").forEach((t) => t.addEventListener("click", () => {
    $$(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
    $("#view-groups").hidden = t.dataset.view !== "groups";
    $("#view-rejected").hidden = t.dataset.view !== "rejected";
  }));
  $$(".segmented button").forEach((b) => b.addEventListener("click", () => {
    $$(".segmented button").forEach((x) => x.classList.toggle("active", x === b));
    state.filter = b.dataset.filter;
    renderGroups();
  }));
  $("#sort").addEventListener("change", (e) => { state.sort = e.target.value; renderGroups(); });
  let typing;
  $("#search").addEventListener("input", (e) => {
    clearTimeout(typing);
    typing = setTimeout(() => {
      state.query = e.target.value.trim().toLowerCase();
      if (state.query && state.filter !== "all") $('.segmented button[data-filter="all"]').click();
      else renderGroups();
    }, 150);
  });
  $("#zoom").addEventListener("input", (e) => document.documentElement.style.setProperty("--thumb", `${e.target.value}px`));

  $("#groups").addEventListener("click", (e) => {
    const c = e.target.closest(".card");
    if (c) openCompare(Number(c.dataset.gid), c.dataset.id);
  });
  $("#rejected").addEventListener("click", (e) => {
    const c = e.target.closest(".card");
    if (c) openLightbox(c.dataset.rid);
  });
  $("#filmstrip").addEventListener("click", (e) => {
    const f = e.target.closest(".film");
    if (!f || f.dataset.id === cmp.group.pick_id) return;
    cmp.otherId = f.dataset.id;
    renderCompare();
  });

  $("#compare").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    const v = cmp.viewer;
    ({ close: closeCompare, "prev-group": () => stepGroup(-1), "next-group": () => stepGroup(1),
       fit: () => v.reset(), one: () => v.oneToOne(), "zoom-in": () => v.zoomBy(1.5), "zoom-out": () => v.zoomBy(1 / 1.5),
       pick: makePick })[act]?.();
  });
  $("#lightbox").addEventListener("click", (e) => {
    const act = e.target.closest("[data-act]")?.dataset.act;
    ({ close: closeLightbox, fit: () => lb.viewer.reset(), one: () => lb.viewer.oneToOne() })[act]?.();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select")) return;
    if (!$("#lightbox").hidden) {
      if (e.key === "Escape") closeLightbox();
      else if (e.key === "z" || e.key === "Z") lb.viewer.oneToOne();
      else if (e.key === "f" || e.key === "F") lb.viewer.reset();
      return;
    }
    if ($("#compare").hidden) return;
    const v = cmp.viewer;
    const keys = {
      Escape: closeCompare, ArrowRight: () => stepPhoto(1), ArrowLeft: () => stepPhoto(-1),
      ArrowDown: () => stepGroup(1), ArrowUp: () => stepGroup(-1),
      z: () => v.oneToOne(), Z: () => v.oneToOne(), f: () => v.reset(), F: () => v.reset(),
      p: makePick, P: makePick, "+": () => v.zoomBy(1.5), "=": () => v.zoomBy(1.5), "-": () => v.zoomBy(1 / 1.5),
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });
}

bindEvents();
load().catch((err) => {
  $("#groups").innerHTML = `<p class="empty">Could not load the run: ${esc(err.message)}. Is the server still running?</p>`;
});
