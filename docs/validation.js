"use strict";

(async function historicalValidation() {
  const get = (id) => document.getElementById(id);
  const el = (tag, text) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; return n; };
  const svgNode = (tag, attrs = {}, text) => { const n = document.createElementNS("http://www.w3.org/2000/svg", tag); Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v)); if (text !== undefined) n.textContent = text; return n; };
  const pct = (v) => `${(v * 100).toFixed(1)}%`;
  const filters = ["league", "from", "to", "outcome"];
  const outputs = ["validation-calibration", "validation-timeline", "validation-summary", "validation-bins", "validation-periods"];
  // Keep the enhancement usable when the browser still has older page markup.
  const share = get("validation-share") || el("a", "分享当前验证范围 ↗");
  share.id = "validation-share"; share.href = "#historical-validation"; share.hidden = true;
  if (!share.isConnected) get("validation-status").before(share);
  filters.forEach((id) => { get(`validation-${id}`).disabled = true; });
  function clearView() {
    outputs.forEach((id) => get(id).replaceChildren());
    get("validation-export").disabled = true;
    get("validation-export").onclick = null;
    get("validation-share").hidden = true;
  }
  function unavailable(error) {
    clearView();
    filters.forEach((id) => { get(`validation-${id}`).disabled = true; });
    get("validation-status").textContent = `历史验证数据未就绪（${error.message}）。请刷新页面重试。当前预测与发布检查仍独立显示。`;
  }
  function chart(title, desc) {
    const s = svgNode("svg", {xmlns: "http://www.w3.org/2000/svg", viewBox: "0 0 600 430", role: "img", "aria-label": title});
    s.append(svgNode("title", {}, title), svgNode("desc", {}, desc)); return s;
  }
  function table(id, headers, rows) {
    const t = el("table"), h = el("thead"), tr = el("tr"), b = el("tbody");
    headers.forEach((text) => { const th = el("th", text); th.scope = "col"; tr.append(th); }); h.append(tr);
    rows.forEach((values) => { const r = el("tr"); values.forEach((v) => r.append(el("td", v))); b.append(r); }); t.append(h, b); get(id).replaceChildren(t);
  }
  function options(id, values, label) {
    get(id).replaceChildren(); values.forEach((v) => { const o = el("option", label(v)); o.value = v; get(id).append(o); }); get(id).disabled = !values.length;
  }
  let catalog;
  function shareUrl(league) {
    const defaults = [catalog.leagues[0].id, league.periods[0].month, league.periods.at(-1).month, "top"];
    const url = new URL(location.href);
    filters.forEach((id, i) => {
      const key = `validation_${id}`, value = get(`validation-${id}`).value;
      if (value === defaults[i]) url.searchParams.delete(key); else url.searchParams.set(key, value);
    });
    history.replaceState(null, "", url);
    url.hash = "historical-validation";
    return url.href;
  }
  function render() {
    const league = catalog.leagues.find((r) => r.id === get("validation-league").value);
    const from = get("validation-from").value, to = get("validation-to").value, kind = get("validation-outcome").value;
    const periods = league.periods.filter((r) => r.month >= from && r.month <= to);
    clearView();
    const url = shareUrl(league);
    if (!periods.length) { get("validation-status").textContent = "当前时间范围没有样本。起始月份不能晚于结束月份。"; return; }
    const count = periods.reduce((n, p) => n + p.count, 0), bins = Array.from({length: 10}, () => ({count: 0, probability_sum: 0, hits: 0}));
    periods.forEach((p) => p.calibration[kind].forEach((b, i) => { Object.keys(bins[i]).forEach((key) => bins[i][key] += b[key]); }));
    const sum = (key) => periods.reduce((n, p) => n + p[key], 0);
    const label = get("validation-outcome").selectedOptions[0].textContent;
    get("validation-status").textContent = `${league.name} · ${from} 至 ${to} · ${count} 场 · ${league.model} · 训练截止 ${league.training_end} · 留出 ${league.test_start} 至 ${league.test_end}`;
    [["所选比赛", `${count} 场`], ["模型 log loss · 越低越好", (sum("log_loss_sum") / count).toFixed(3)], ["模型三分类 Brier · 越低越好", (sum("brier_sum") / count).toFixed(3)], ["训练频率基线 Brier", (sum("baseline_brier_sum") / count).toFixed(3)], ["模型命中率", pct(sum("correct") / count)], ["训练多数类命中率", pct(sum("baseline_correct") / count)]].forEach(([title, value]) => { const d = el("div"); d.append(el("span", title), el("strong", value)); get("validation-summary").append(d); });
    const c = chart(`${label}校准`, "横轴为平均预测概率，纵轴为实际比例。点为非空的 10% 分箱，无样本分箱不画点。虚线是完全校准参考线，没有置信区间。"), x = (v) => 70 + v * 460, y = (v) => 355 - v * 290;
    [0, .25, .5, .75, 1].forEach((v) => {
      c.append(svgNode("line", {x1: x(0), x2: x(1), y1: y(v), y2: y(v), stroke: "#e2e8ee"}), svgNode("text", {x: 54, y: y(v) + 4, "text-anchor": "end", fill: "#647382", "font-size": 13}, `${v * 100}%`), svgNode("text", {x: x(v), y: 380, "text-anchor": "middle", fill: "#647382", "font-size": 13}, `${v * 100}%`));
    });
    c.append(svgNode("path", {d: `M${x(0)} ${y(0)}L${x(1)} ${y(1)}`, stroke: "#203344", "stroke-dasharray": "5 5", fill: "none"}), svgNode("text", {x: 70, y: 28, fill: "#203344", "font-size": 15}, kind === "top" ? "实际命中率" : "实际发生比例"), svgNode("text", {x: 300, y: 415, "text-anchor": "middle", fill: "#203344", "font-size": 14}, "平均预测概率"));
    bins.forEach((b) => { if (!b.count) return; const cx = x(b.probability_sum / b.count), cy = y(b.hits / b.count); c.append(svgNode("circle", {cx, cy, r: 5, fill: "#2468a0", stroke: "white", "stroke-width": 1}), svgNode("text", {x: Math.max(90, Math.min(515, cx)), y: Math.max(47, cy - 11), "text-anchor": "middle", fill: "#647382", "font-size": 12}, `n=${b.count}`)); });
    get("validation-calibration").append(c);
    const t = chart("按月命中率与多数类基线", "按真实月份间隔绘图。实线为模型命中率，虚线为训练多数类基线，无比赛月份断线。纵轴固定 0–100%，每月样本量见数值表。"), width = 460;
    const monthNumber = (m) => Number(m.slice(0, 4)) * 12 + Number(m.slice(5, 7)) - 1;
    const first = monthNumber(periods[0].month), last = monthNumber(periods.at(-1).month);
    const tx = (m) => last === first ? 300 : 70 + (monthNumber(m) - first) / (last - first) * width;
    [0, .25, .5, .75, 1].forEach((v) => t.append(svgNode("line", {x1: 70, x2: 530, y1: y(v), y2: y(v), stroke: "#e2e8ee"}), svgNode("text", {x: 54, y: y(v) + 4, "text-anchor": "end", fill: "#647382", "font-size": 13}, `${v * 100}%`)));
    for (const [key, color, dashed] of [["correct", "#2468a0", false], ["baseline_correct", "#647382", true]]) {
      let d = ""; periods.forEach((p, i) => { d += `${i && monthNumber(p.month) - monthNumber(periods[i - 1].month) === 1 ? "L" : "M"}${tx(p.month)} ${y(p[key] / p.count)} `; t.append(svgNode(dashed ? "rect" : "circle", dashed ? {x: tx(p.month) - 3, y: y(p[key] / p.count) - 3, width: 6, height: 6, fill: "white", stroke: color} : {cx: tx(p.month), cy: y(p[key] / p.count), r: 3.5, fill: color})); });
      t.append(svgNode("path", {d, fill: "none", stroke: color, "stroke-width": 2, ...(dashed ? {"stroke-dasharray": "5 4"} : {})}));
    }
    const ticks = [periods[0], ...(periods.length > 6 ? [periods[Math.floor(periods.length / 2)]] : []), ...(periods.length > 1 ? [periods.at(-1)] : [])];
    ticks.forEach((p) => t.append(svgNode("text", {x: tx(p.month), y: 382, "text-anchor": "middle", fill: "#647382", "font-size": 13}, p.month)));
    t.append(svgNode("text", {x: 70, y: 28, fill: "#2468a0", "font-size": 14}, "● 模型，实线"), svgNode("text", {x: 255, y: 28, fill: "#647382", "font-size": 14}, "□ 多数类基线，虚线")); get("validation-timeline").append(t);
    table("validation-bins", ["概率分箱", "样本数", "平均预测概率", kind === "top" ? "实际命中率" : "实际发生比例"], bins.map((b, i) => [`${i * 10}–${(i + 1) * 10}%${i === 9 ? "（含100%）" : "（不含上界）"}`, b.count, b.count ? pct(b.probability_sum / b.count) : "无样本", b.count ? pct(b.hits / b.count) : "无样本"]));
    table("validation-periods", ["月份", "比赛数", "模型命中率", "基线命中率", "log loss", "Brier"], periods.map((p) => [p.month, p.count, pct(p.correct / p.count), pct(p.baseline_correct / p.count), (p.log_loss_sum / p.count).toFixed(3), (p.brier_sum / p.count).toFixed(3)]));
    get("validation-export").disabled = false;
    get("validation-share").href = url;
    get("validation-share").hidden = false;
    get("validation-export").onclick = () => {
      const lines = [["league", "month", "matches", "accuracy", "baseline_accuracy", "log_loss", "brier"], ...periods.map((p) => [league.id, p.month, p.count, p.correct / p.count, p.baseline_correct / p.count, p.log_loss_sum / p.count, p.brier_sum / p.count])];
      const url = URL.createObjectURL(new Blob([lines.map((r) => r.join(",")).join("\n")], {type: "text/csv;charset=utf-8"}));
      const link = el("a"); link.href = url; link.download = `touchline-${league.id}-${from}-${to}.csv`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    };
  }
  function restore(id, value, fallback) {
    const select = get(`validation-${id}`);
    select.value = [...select.options].some((option) => option.value === value) ? value : fallback;
  }
  function leagueChanged(parameters = new URLSearchParams()) {
    const periods = catalog.leagues.find((r) => r.id === get("validation-league").value).periods;
    ["validation-from", "validation-to"].forEach((id) => options(id, periods.map((p) => p.month), String));
    restore("from", parameters.get("validation_from"), periods[0].month);
    restore("to", parameters.get("validation_to"), periods.at(-1).month);
    render();
  }
  try {
    const response = await fetch("./data/research/validation.json", {cache: "no-cache"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    catalog = await response.json();
    if (!catalog.leagues?.length) throw new Error("没有历史留出样本");
    get("validation-scope").textContent = catalog.scope;
    get("validation-limits").textContent = `${catalog.limits} 命中率基线取训练集中最多的赛果；Brier 基线使用训练赛果频率，三分类 Brier 为三类误差平方之和，范围 0–2。校准图虚线代表预测概率与实际比例相同；月度图虚线为多数类命中率基线。`;
    options("validation-league", catalog.leagues.map((r) => r.id), (id) => catalog.leagues.find((r) => r.id === id).name);
    const parameters = new URLSearchParams(location.search);
    restore("league", parameters.get("validation_league"), catalog.leagues[0].id);
    restore("outcome", parameters.get("validation_outcome"), "top");
    filters.forEach((id) => { get(`validation-${id}`).disabled = false; });
    get("validation-league").addEventListener("change", () => { try { leagueChanged(); } catch (error) { unavailable(error); } });
    ["validation-from", "validation-to", "validation-outcome"].forEach((id) => get(id).addEventListener("change", () => { try { render(); } catch (error) { unavailable(error); } }));
    leagueChanged(parameters);
  } catch (error) { unavailable(error); }
})();
