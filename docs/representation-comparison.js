"use strict";

const representationLossMetrics = {
  completion_log_loss: ["完成概率 log loss", "log loss 差值", 6],
  completion_brier: ["完成概率 Brier", "Brier 差值", 6],
  forward_displacement_mae_pitch_pct: ["向前位移 MAE", "球场长度百分点", 4],
};

function representationComparisonRows(report) {
  const raw = report.test_comparisons, test = report.data?.splits?.test;
  if (!raw || raw.method !== "paired_match_case_bootstrap" || !test ||
      raw.test_sequences !== test.sequences_used || raw.test_matches !== test.matches_used ||
      !Number.isInteger(raw.test_matches) || raw.test_matches < 2 || raw.test_sequences < raw.test_matches ||
      !Number.isInteger(raw.replicates) || raw.replicates < 200 || !Array.isArray(raw.comparisons)) throw new Error("缺少配对重采样结果或测试样本不一致");
  const methods = new Map(report.summaries.map((row) => [row.method, row.runs])), seen = new Set();
  if (!methods.has(raw.reference)) throw new Error("参照模型缺失");
  const rows = raw.comparisons.map((row) => {
    if (row.method === raw.reference || !methods.has(row.method) || methods.get(row.method) !== row.runs || seen.has(row.method)) throw new Error("比较方法重复或不一致");
    seen.add(row.method);
    const clean = {method: row.method, runs: row.runs};
    for (const metric of Object.keys(representationLossMetrics)) {
      const value = row[metric];
      if (!value || ![value.difference, value.lower, value.upper].every(Number.isFinite) || value.lower > value.upper) throw new Error("配对差值或范围无效");
      clean[metric] = {difference: value.difference, lower: value.lower, upper: value.upper};
    }
    return clean;
  });
  if (seen.size !== methods.size - 1) throw new Error("比较方法不完整");
  return rows;
}

async function representationComparisonView() {
  const get = (id) => document.getElementById(`representation-paired-${id}`);
  if (!get("status")) return;
  let report, rows;
  function clear() {
    for (const id of ["chart", "table", "scope", "limits"]) get(id).replaceChildren();
    get("metric").disabled = true; get("export").disabled = true; get("share").hidden = true;
  }
  function unavailable(error) {
    clear(); get("retry").hidden = false;
    get("status").textContent = `配对比较未就绪（${error.message}）。可重试读取数据。`;
  }
  function render() {
    const metric = get("metric").value, [label, unit, precision] = representationLossMetrics[metric];
    const raw = report.test_comparisons, name = (method) => methodNames[method] || method;
    const low = Math.min(0, ...rows.map((r) => r[metric].lower)), high = Math.max(0, ...rows.map((r) => r[metric].upper));
    const padding = Math.max((high - low) * .12, .0001), min = low - padding, max = high + padding;
    const height = 170 + rows.length * 38, x = (v) => 260 + (v - min) / (max - min) * 230;
    const svg = explorationSvg(`${label} · 方法减去${name(raw.reference)}`, "点为差值；线为逐项95%重采样范围，正值表示误差更大。", height);
    for (let i = 0; i <= 4; i++) {
      const value = min + (max - min) * i / 4;
      svg.append(svgElement("line", {x1: x(value), x2: x(value), y1: 90, y2: height - 65, stroke: "#e2e8ee"}));
      svg.append(svgElement("text", {x: x(value), y: height - 43, "text-anchor": "middle", "font-size": 11, fill: "#647382"}, number(value, precision === 6 ? 3 : 2)));
    }
    svg.append(svgElement("line", {x1: x(0), x2: x(0), y1: 90, y2: height - 65, stroke: "#647382", "stroke-dasharray": "4 4"}),
      svgElement("text", {x: x(0), y: 88, "text-anchor": "middle", "font-size": 11, fill: "#647382"}, "零差值"),
      svgElement("text", {x: 510, y: 88, "font-size": 12, fill: "#647382"}, "差值 [2.5%,97.5%]"));
    rows.forEach((row, index) => {
      const value = row[metric], y = 110 + index * 38;
      svg.append(svgElement("text", {x: 30, y: y + 4, "font-size": 13, fill: "#203344"}, name(row.method)),
        svgElement("line", {x1: x(value.lower), x2: x(value.upper), y1: y, y2: y, stroke: "#2468a0", "stroke-width": 2}),
        svgElement("circle", {cx: x(value.difference), cy: y, r: 4, fill: "#2468a0"}),
        svgElement("text", {x: 510, y: y + 4, "font-size": 12, fill: "#203344"}, `${number(value.difference, precision)} [${number(value.lower, precision)}, ${number(value.upper, precision)}]`));
    });
    svg.append(svgElement("text", {x: 375, y: height - 18, "text-anchor": "middle", "font-size": 12, fill: "#647382"}, unit));
    get("chart").replaceChildren(svg);
    get("scope").textContent = raw.scope;
    get("limits").replaceChildren(...raw.limits.map((limit) => node("p", limit, "context-note")));
    get("status").textContent = `${raw.test_matches} 场测试比赛 · ${raw.test_sequences.toLocaleString()} 个抽样传球 · ${raw.replicates.toLocaleString()} 次整场配对重采样 · ${rows.length} 项探索比较。模型与种子固定；范围不是训练种子标准差。`;
    const shown = (v) => Number(v.toFixed(precision));
    explorationTable("representation-paired-table", ["方法", "固定评估次数", "误差差值", "2.5%下界", "97.5%上界"],
      rows.map((row) => [name(row.method), row.runs, shown(row[metric].difference), shown(row[metric].lower), shown(row[metric].upper)]),
      {"图表": "表示学习 · 测试比赛配对比较", "筛选": label, "单位": unit,
       "来源与范围": `${get("status").textContent} ${raw.scope} ${raw.limits.join(" ")}`,
       "参照模型": name(raw.reference), "运行": report.run_id, "测试样本指纹": raw.cohort_sha256,
       "重采样": `SciPy ${raw.scipy_version} · ${raw.replicates}次 · seed ${raw.seed} · 配对整场百分位范围`,
       "来源链接": `${raw.method_url} / ${raw.engine_url}`, "显示精度": `${precision}位小数独立四舍五入；固定种子先平均损失，不平均概率。`});
    const url = new URL(location.href); url.searchParams.set("representation_metric", metric);
    if (typeof window === "undefined" || !window.TouchlineWorkspace || window.TouchlineWorkspace.active("representations")) history.replaceState(null, "", url); url.hash = "representation-paired"; get("share").href = url.href;
    get("share").hidden = false; get("export").disabled = false;
  }
  async function load() {
    clear(); get("retry").hidden = true; get("status").textContent = "正在读取固定模型的配对测试结果。";
    try {
      report = await explorationJson("representation"); rows = representationComparisonRows(report);
      const metric = new URLSearchParams(location.search).get("representation_metric");
      get("metric").value = Object.hasOwn(representationLossMetrics, metric) ? metric : "completion_log_loss";
      render(); get("metric").disabled = false;
      if (location.hash === "#representation-paired") $("representation-paired").open = true;
    } catch (error) { unavailable(error); }
  }
  get("metric").addEventListener("change", () => { try { render(); } catch (error) { unavailable(error); } });
  get("retry").addEventListener("click", load);
  get("export").addEventListener("click", () => saveExplorationSvg("representation-paired-chart", `touchline-representation-${get("metric").value}`, [
    `Wyscout ${report.data.season} · ${report.data.target_league} · 测试 ${report.data.splits.test.date_start}–${report.data.splits.test.date_end} · run ${report.run_id}`,
    get("status").textContent, get("scope").textContent, ...report.test_comparisons.limits,
    `方法与实现：${report.test_comparisons.method_url} / ${report.test_comparisons.engine_url}`]));
  await load();
}

if (typeof module !== "undefined") module.exports = {representationComparisonRows};
if (typeof document !== "undefined") representationComparisonView();
