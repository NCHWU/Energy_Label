/* LLM Energy Label – frontend logic */

const API = "";

let currentDomain = null;
let domainsMeta = {};

async function fetchJSON(url) {
  const res = await fetch(url);
  return res.json();
}

/* ── Domain switching ──────────────────────────────────── */

async function initDomainPills() {
  // Pre-fetch domain metadata for insight panel
  const domains = await fetchJSON(`${API}/api/domains`);
  domains.forEach(d => { domainsMeta[d.id] = d; });

  document.querySelectorAll(".domain-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".domain-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      const domain = pill.dataset.domain;
      showInsight(domain);
      loadLeaderboard(domain);
    });
  });

  // Auto-select first domain
  const first = document.querySelector(".domain-pill");
  if (first) {
    first.classList.add("active");
    showInsight(first.dataset.domain);
    loadLeaderboard(first.dataset.domain);
  }
}

/* ── Insight panel ─────────────────────────────────────── */

function showInsight(domain) {
  const meta = domainsMeta[domain];
  const panel = document.getElementById("domain-insight");

  if (!meta || !meta.accuracy_metric) {
    panel.style.display = "none";
    return;
  }

  document.getElementById("insight-metric").textContent = meta.accuracy_metric;
  document.getElementById("insight-tasks").textContent =
    meta.tasks_count ? `${meta.tasks_count} tasks` : "\u2014";
  document.getElementById("insight-detail").textContent =
    meta.accuracy_detail || "\u2014";
  document.getElementById("insight-why").textContent =
    meta.why_this_metric || "\u2014";

  panel.style.display = "block";
  // Re-trigger animation
  panel.style.animation = "none";
  panel.offsetHeight;
  panel.style.animation = "";
}

/* ── Leaderboard rendering ─────────────────────────────── */

async function loadLeaderboard(domain) {
  currentDomain = domain;
  const board = document.getElementById("leaderboard-body");
  const card = document.getElementById("leaderboard-card");
  const domainTitle = document.getElementById("domain-title");

  const data = await fetchJSON(`${API}/api/leaderboard/${domain}`);

  domainTitle.textContent = domain.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  if (!data.length) {
    board.innerHTML = `
      <tr><td colspan="7" class="empty-state">
        <p>No benchmark results yet for this domain.</p>
      </td></tr>`;
    card.style.display = "block";
    return;
  }

  // Find max EPCA for bar scaling (exclude null)
  const epcaValues = data.map(r => r.epca_j).filter(v => v != null);
  const maxEpca = Math.max(...epcaValues, 1);

  board.innerHTML = data.map((row, i) => {
    const rank = i + 1;
    const rankClass = rank <= 3 ? `rank-${rank}` : "";
    const epca = row.epca_j != null ? row.epca_j.toFixed(2) : "\u221e";
    const passRate = (row.pass_rate * 100).toFixed(1);
    const energy = row.total_energy_j != null ? row.total_energy_j.toFixed(1) : "\u2014";
    const runtime = row.avg_runtime_ms != null ? row.avg_runtime_ms.toFixed(1) : "\u2014";
    const barPct = row.epca_j != null ? Math.min((row.epca_j / maxEpca) * 100, 100) : 100;
    const barColor = labelColor(row.label);

    return `<tr>
      <td class="rank ${rankClass}">${rank}</td>
      <td class="model-name">${escHtml(row.model)}</td>
      <td><span class="label-badge label-${row.label}">${row.label}</span></td>
      <td class="metric">${passRate}<span class="metric-unit">%</span></td>
      <td class="bar-cell">
        <span class="metric">${epca}<span class="metric-unit">J</span></span>
        <div class="bar-bg"><div class="bar-fill" style="width:${barPct}%;background:${barColor}"></div></div>
      </td>
      <td class="metric">${energy}<span class="metric-unit">J</span></td>
      <td class="metric">${runtime}<span class="metric-unit">ms</span></td>
    </tr>`;
  }).join("");

  card.style.display = "block";
  // Re-trigger animation
  card.style.animation = "none";
  card.offsetHeight; // force reflow
  card.style.animation = "";
}

/* ── Helpers ───────────────────────────────────────────── */

function labelColor(label) {
  const map = {
    A: "#34c759", B: "#6dd400", C: "#a8d600",
    D: "#ffd60a", E: "#ff9f0a", F: "#ff6723", G: "#ff3b30"
  };
  return map[label] || "#ccc";
}

function escHtml(str) {
  const d = document.createElement("div");
  d.textContent = str;
  return d.innerHTML;
}

/* ── Init ──────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", initDomainPills);
