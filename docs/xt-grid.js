"use strict";

function xtGridRows(report) {
  const sampled = report.resampling;
  if (report.schema !== "SoccerAnalytics.XTNumerics.v1" || !sampled || sampled.method !== "whole_match_case_bootstrap" ||
      sampled.numerical_failures !== 0 || !Number.isInteger(sampled.replicates) || sampled.replicates < 200 ||
      !Number.isInteger(sampled.clusters) || sampled.clusters < 2 || sampled.clusters !== report.counts.matches ||
      JSON.stringify(sampled.quantiles) !== "[0.025,0.5,0.975]") throw new Error("缺少完整的整场重采样结果");
  const matrices = [report.grid, sampled.lower_grid, sampled.median_grid, sampled.upper_grid,
    sampled.unobserved_replicates, sampled.zero_value_replicates,
    report.support.shots, report.support.goals, ...Object.values(sampled.match_support)];
  if (matrices.some((m) => !Array.isArray(m) || m.length !== 8 || m.some((row) => !Array.isArray(row) || row.length !== 12))) throw new Error("网格尺寸无效");
  const rows = [];
  for (let y = 0; y < 8; y++) for (let x = 0; x < 12; x++) {
    const row = {x: x * 10, y: y * 10, value: report.grid[y][x], lower: sampled.lower_grid[y][x],
      median: sampled.median_grid[y][x], upper: sampled.upper_grid[y][x], shots: report.support.shots[y][x],
      goals: report.support.goals[y][x], observedMatches: sampled.match_support.observations[y][x],
      shotMatches: sampled.match_support.shots[y][x], goalMatches: sampled.match_support.goals[y][x],
      unseen: sampled.unobserved_replicates[y][x], zero: sampled.zero_value_replicates[y][x]};
    if ([row.value, row.lower, row.median, row.upper].some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
        !(row.lower <= row.median && row.median <= row.upper) ||
        [row.shots, row.goals, row.observedMatches, row.shotMatches, row.goalMatches, row.unseen, row.zero].some((v) => !Number.isInteger(v) || v < 0) ||
        row.goals > row.shots || row.shotMatches > row.shots || row.goalMatches > row.goals ||
        row.goalMatches > row.shotMatches || row.shotMatches > row.observedMatches || row.observedMatches > sampled.clusters ||
        row.unseen > row.zero || row.zero > sampled.replicates) throw new Error("网格数值、范围或样本计数无效");
    row.width = row.upper - row.lower; rows.push(row);
  }
  if (rows.reduce((n, r) => n + r.shots, 0) !== report.counts.shots || rows.reduce((n, r) => n + r.goals, 0) !== report.counts.goals) throw new Error("网格样本总数不一致");
  return rows;
}

async function xtGridView() {
  const get = (id) => document.getElementById(`xt-grid-${id}`);
  if (!get("status")) return;
  const labels = {width: "95% 重采样范围宽度", value: "拟合 xT", shots: "射门样本数", goals: "进球样本数"};
  let report, rows;
  function clear() {
    for (const id of ["chart", "table", "scope", "limits"]) get(id).replaceChildren();
    get("view").disabled = true; get("export").disabled = true; get("share").hidden = true;
  }
  function unavailable(error) {
    clear(); report = null; rows = null; get("retry").hidden = false;
    get("status").textContent = `网格检查未就绪（${error.message}）。请更新研究数据后重试。`;
  }
  const format = (value, metric) => ["shots", "goals"].includes(metric) ? String(value) : value.toFixed(4);
  function color(value, maximum) {
    const t = maximum ? value / maximum : 0, rgb = [236, 245, 252].map((v, i) => Math.round(v + t * ([36, 104, 160][i] - v)));
    const linear = rgb.map((v) => v / 255 <= .04045 ? v / 255 / 12.92 : ((v / 255 + .055) / 1.055) ** 2.4);
    const luminance = .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2];
    return {fill: `rgb(${rgb.join(",")})`, text: luminance > .179 ? "#000" : "#fff"};
  }
  function render() {
    const metric = get("view").value, maximum = Math.max(...rows.map((r) => r[metric]));
    const svg = explorationSvg(labels[metric], "8×12 网格 · 当前指标单独设色 · x 向右、y 向下", 650);
    svg.classList.add("xt-grid");
    const left = 65, top = 100 + Number(svg.dataset.headerOffset), step = 52;
    const defs = svgElement("defs", {}), pattern = svgElement("pattern", {id: "xt-grid-unobserved", width: 8, height: 8, patternUnits: "userSpaceOnUse"});
    pattern.append(svgElement("rect", {width: 8, height: 8, fill: "#f2f4f6"}), svgElement("path", {d: "M0 8L8 0", stroke: "#647382"})); defs.append(pattern); svg.append(defs);
    for (const row of rows) {
      const x = left + row.x / 10 * step, y = top + row.y / 10 * step, colors = color(row[metric], maximum);
      svg.append(svgElement("rect", {x, y, width: step, height: step, fill: row.observedMatches ? colors.fill : "url(#xt-grid-unobserved)", stroke: "white", "stroke-width": 1}),
        svgElement("text", {x: x + step / 2, y: y + step / 2 + 4, "text-anchor": "middle", "font-size": 13, fill: row.observedMatches ? colors.text : "#203344"}, row.observedMatches ? format(row[metric], metric) : "无观测"));
    }
    for (let x = 0; x <= 12; x += 2) svg.append(svgElement("text", {x: left + x * step, y: top + 440, "text-anchor": "middle", "font-size": 13, fill: "#647382"}, String(x * 10)));
    for (let y = 0; y <= 8; y += 2) svg.append(svgElement("text", {x: left - 12, y: top + y * step + 4, "text-anchor": "end", "font-size": 13, fill: "#647382"}, String(y * 10)));
    svg.append(svgElement("text", {x: left, y: top - 14, "font-size": 13, fill: "#647382"}, "y"),
      svgElement("text", {x: left + 12 * step + 18, y: top + 440, "font-size": 13, fill: "#647382"}, "x"));
    for (let i = 0; i < 12; i++) svg.append(svgElement("rect", {x: left + i * step, y: top + 478, width: step, height: 14, fill: color(maximum * i / 11, maximum).fill}));
    svg.append(svgElement("text", {x: left, y: top + 515, "font-size": 13, fill: "#203344"}, `0 → ${format(maximum, metric)} · ${labels[metric]} · 斜线为原样本无观测格`));
    get("chart").replaceChildren(svg);
    get("status").textContent = `${report.source.provider} ${report.source.league} ${report.source.season} · ${report.counts.matches} 场训练比赛 · ${report.resampling.replicates} 次整场重采样 · ${labels[metric]}。颜色上限随指标改变，不能跨指标比较深浅。`;
    get("scope").textContent = `${report.window.start} 至 ${report.window.end}。${report.resampling.scope} 运行 ${report.run_id}。`;
    get("limits").replaceChildren(...report.resampling.limits.map((text) => node("li", text)));
    const decimal = (v) => Number(v.toFixed(4));
    explorationTable("xt-grid-table", ["x 范围", "y 范围", "拟合 xT", "2.5%分位", "中位数", "97.5%分位", "范围宽度", "射门数", "进球数", "观测场数", "无观测抽样数", "零值抽样数"],
      rows.map((r) => [`${r.x}–${r.x + 10}`, `${r.y}–${r.y + 10}`, ...[r.value, r.lower, r.median, r.upper, r.width].map(decimal), r.shots, r.goals, r.observedMatches, r.unseen, r.zero]), {
        "来源与范围": `${report.scope} ${report.window.start}–${report.window.end}。Wyscout 百分比坐标缩放为120×80单位，非米制；x向右、y向下；每格左闭右开，最右与最下格包含边界。xT、分位数与宽度各自四舍五入至四位小数。${report.resampling.scope} ${report.resampling.limits.join(" ")} 运行 ${report.run_id}。`,
        "来源链接": `${report.source.url} / https://arxiv.org/abs/2604.21087v4 / ${report.resampling.method_url}`,
      });
    const url = new URL(location.href);
    if (metric === "width") url.searchParams.delete("xt_metric"); else url.searchParams.set("xt_metric", metric);
    if (typeof window === "undefined" || !window.TouchlineWorkspace || window.TouchlineWorkspace.active("xt-grid")) history.replaceState(null, "", url); url.hash = "xt-grid"; get("share").href = url.href;
    get("share").hidden = false; get("export").disabled = false;
  }
  async function load() {
    clear(); get("retry").hidden = true; get("status").textContent = "正在读取网格与整场重采样结果。";
    try {
      report = await explorationJson("xt-numerics"); rows = xtGridRows(report);
      const metric = new URLSearchParams(location.search).get("xt_metric");
      get("view").value = Object.hasOwn(labels, metric) ? metric : "width";
      render(); get("view").disabled = false;
    } catch (error) { unavailable(error); }
  }
  get("view").addEventListener("change", () => { try { render(); } catch (error) { unavailable(error); } });
  get("retry").addEventListener("click", load);
  get("export").addEventListener("click", () => saveExplorationSvg("xt-grid-chart", `touchline-xt-${get("view").value}`, [get("status").textContent,
    get("scope").textContent, "Wyscout 百分比坐标缩放为120×80单位，非米制；原训练样本口径含点球及直接任意球。", ...report.resampling.limits]));
  await load();
}

if (typeof module !== "undefined") module.exports = {xtGridRows};
if (typeof document !== "undefined") xtGridView();
