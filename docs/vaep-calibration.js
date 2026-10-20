"use strict";

async function vaepCalibration() {
  const prefix = "vaep-calibration-", get = (id) => $(prefix + id);
  if (!get("status")) return;
  const controls = [["league", "England"], ["model", "context"], ["head", "score"], ["bins", "10"], ["range", "0.1"]];
  const outputs = ["summary", "chart", "bin-table", "pav-table", "score-table", "crop", "scope", "limits"];
  const percent = (v) => `${(v * 100).toFixed(3)}%`;
  const decimal = (v) => Number(v.toFixed(8));
  const label = (id) => get(id).selectedOptions[0].textContent;
  let report;

  function unavailable(error) {
    outputs.forEach((id) => get(id).replaceChildren());
    controls.forEach(([id]) => { get(id).disabled = true; });
    get("share").hidden = true; get("retry").hidden = false;
    get("status").textContent = `校准诊断未就绪（${error.message}）。可重试读取数据。`;
  }

  function draw(group, bins, maximum) {
    const svg = explorationSvg(`${label("league")} · ${label("model")} · ${label("head")}概率校准`, "蓝线 PAV 同样本拟合 · 橙色方点等宽分箱 · 灰色虚线相等参考", 660);
    const x = (v) => 90 + v / maximum * 560, y = (v) => 430 - v / maximum * 325;
    const clip = svgElement("clipPath", {id: "vaep-calibration-clip"});
    clip.append(svgElement("rect", {x: 90, y: 105, width: 560, height: 325}));
    const defs = svgElement("defs", {}); defs.append(clip); svg.append(defs);
    for (const v of [0, .25, .5, .75, 1]) {
      const value = v * maximum, text = `${Number((value * 100).toFixed(2))}%`;
      svg.append(svgElement("line", {x1: 90, x2: 650, y1: y(value), y2: y(value), stroke: "#e2e8ee"}),
        svgElement("text", {x: 78, y: y(value) + 4, "text-anchor": "end", fill: "#647382", "font-size": 13}, text),
        svgElement("text", {x: x(value), y: 452, "text-anchor": "middle", fill: "#647382", "font-size": 13}, text));
    }
    svg.append(svgElement("text", {x: 90, y: 87, fill: "#203344", "font-size": 14}, "实际发生比例"),
      svgElement("text", {x: 370, y: 480, "text-anchor": "middle", fill: "#203344", "font-size": 14}, "平均预测概率"));
    const marks = svgElement("g", {"clip-path": "url(#vaep-calibration-clip)"});
    marks.append(svgElement("path", {d: `M90 430L650 105`, stroke: "#647382", "stroke-dasharray": "5 5", fill: "none"}));
    let path = "";
    group.pav.forEach((b, i) => {
      path += `${i ? "L" : "M"}${x(b.lower)} ${y(b.observed_rate)}L${x(b.upper)} ${y(b.observed_rate)}`;
    });
    marks.append(svgElement("path", {d: path, stroke: "#2468a0", "stroke-width": 2.5, fill: "none"}));
    if (group.pav.length === 1) marks.append(svgElement("circle", {cx: x(group.pav[0].mean_probability), cy: y(group.pav[0].observed_rate), r: 5, fill: "#2468a0"}));
    bins.filter((b) => b.count).forEach((b) => marks.append(svgElement("rect", {x: x(b.mean_probability) - 4, y: y(b.observed_rate) - 4, width: 8, height: 8, fill: "#b66b2b", stroke: "white", "stroke-width": 1})));
    svg.append(marks, svgElement("text", {x: 90, y: 518, fill: "#203344", "font-size": 14}, "预测分布 · 固定 0–100% 横轴 · 柱高为样本占比"));
    for (const [cy, text] of [[530, "100%"], [575, "0%"]]) svg.append(svgElement("line", {x1: 90, x2: 650, y1: cy, y2: cy, stroke: "#e2e8ee"}),
      svgElement("text", {x: 78, y: cy + 4, "text-anchor": "end", fill: "#647382", "font-size": 12}, text));
    bins.forEach((b) => svg.append(svgElement("rect", {x: 90 + b.lower * 560, y: 575 - b.count / group.states * 45,
      width: (b.upper - b.lower) * 560 - 1, height: b.count / group.states * 45, fill: "#8b9baa"})));
    for (const v of [0, .25, .5, .75, 1]) svg.append(svgElement("text", {x: 90 + v * 560, y: 597, "text-anchor": "middle", fill: "#647382", "font-size": 13}, `${v * 100}%`));
    svg.append(svgElement("text", {x: 90, y: 635, fill: "#647382", "font-size": 13}, `${group.states.toLocaleString("zh-CN")} 个状态 · 当前进球与不完整窗口已排除 · 无置信带`));
    get("chart").replaceChildren(svg);
  }

  function render() {
    const group = report.groups.find((g) => g.league === get("league").value && g.method === get("model").value && g.head === get("head").value);
    if (!group) throw new Error("所选范围缺少校准记录");
    const block = group.binning[get("bins").value], maximum = Number(get("range").value);
    get("status").textContent = `${label("league")} · ${label("model")} · ${label("head")} · ${group.states.toLocaleString("zh-CN")} 个状态，其中 ${group.positives.toLocaleString("zh-CN")} 个窗口含${label("head")}`;
    const values = [["实际发生比例", percent(group.observed_rate)], ["平均预测概率", percent(group.predicted_rate)],
      ["原模型二分类 Brier", decimal(group.brier)], [`${label("bins")} ECE · 描述性`, percent(block.ece)]];
    get("summary").replaceChildren(...values.map(([title, value]) => { const item = node("div"); item.append(node("span", title), node("strong", String(value))); return item; }));
    const outside = block.bins.filter((b) => b.count && (b.mean_probability > maximum || b.observed_rate > maximum));
    get("crop").textContent = maximum < 1 ? `校准图两轴均为 0–10%。${outside.length} 个非空分箱点在图外，涉及 ${outside.reduce((n, b) => n + b.count, 0).toLocaleString("zh-CN")} 个状态。PAV 也按此范围裁切。下方表格与预测分布保留全部状态。` : "校准图两轴均为 0–100%。表格与预测分布保留全部状态。";
    draw(group, block.bins, maximum);
    explorationTable(prefix + "bin-table", ["概率下界", "概率上界", "状态数", "正标签数", "平均预测概率", "实际发生比例"], block.bins.map((b) => [percent(b.lower), percent(b.upper), b.count, b.positives, b.count ? percent(b.mean_probability) : "无样本", b.count ? percent(b.observed_rate) : "无样本"]));
    explorationTable(prefix + "pav-table", ["原概率最小值", "原概率最大值", "状态数", "正标签数", "PAV 拟合概率"], group.pav.map((b) => [percent(b.lower), percent(b.upper), b.count, b.positives, percent(b.observed_rate)]));
    explorationTable(prefix + "score-table", ["指标", "数值", "含义"], [
      ["原模型 Brier", decimal(group.brier), "原始预测的二分类均方误差"],
      ["PAV Brier · 同样本拟合", decimal(group.pav_brier), "用于诊断，不是校准后的留出表现"],
      ["MCB", decimal(group.mcb), "原模型 Brier 减同样本 PAV Brier"],
      ["DSC", decimal(group.dsc), "UNC 减同样本 PAV Brier"], ["UNC", decimal(group.unc), "样本事件频率 ×（1 − 样本事件频率）"],
      ...["9", "10", "11"].map((size) => [`${size} 箱 ECE`, percent(group.binning[size].ece), "分箱加权绝对校准差，随分箱变化"])]);
    const url = new URL(location.href);
    controls.forEach(([id, fallback]) => {
      const key = `vaep_${id}`;
      if (get(id).value === fallback) url.searchParams.delete(key); else url.searchParams.set(key, get(id).value);
    });
    if (typeof window === "undefined" || !window.TouchlineWorkspace || window.TouchlineWorkspace.active("vaep-calibration")) history.replaceState(null, "", url); url.hash = "vaep-calibration";
    get("share").href = url.href; get("share").hidden = false;
  }

  async function load() {
    get("retry").hidden = true; get("status").textContent = "正在读取校准诊断。";
    try {
      report = await explorationJson("vaep-calibration");
      if (report.schema !== "SoccerAnalytics.VAEPCalibration.v1" || report.groups?.length !== 12 || !Array.isArray(report.limits)) throw new Error("诊断摘要不完整");
      for (const g of report.groups) {
        if (!Number.isInteger(g.states) || g.states <= 0 || !g.pav?.length || !["9", "10", "11"].every((size) => g.binning?.[size]?.bins?.length === Number(size))) throw new Error("诊断分组不完整");
        if (![g.observed_rate, g.predicted_rate, g.brier, g.pav_brier, g.mcb, g.dsc, g.unc].every(Number.isFinite)) throw new Error("诊断数值无效");
      }
      get("scope").textContent = `${report.scope} ${report.method} 原实验运行 ${report.source_run_id}。诊断运行 ${report.run_id}。`;
      get("limits").replaceChildren(...report.limits.map((text) => node("li", text)));
      const parameters = new URLSearchParams(location.search);
      controls.forEach(([id, fallback]) => {
        const value = parameters.get(`vaep_${id}`);
        get(id).value = [...get(id).options].some((o) => o.value === value) ? value : fallback;
        get(id).disabled = false;
      });
      render();
    } catch (error) { unavailable(error); }
  }
  controls.forEach(([id]) => get(id).addEventListener("change", () => { try { render(); } catch (error) { unavailable(error); } }));
  get("retry").addEventListener("click", load);
  await load();
}

vaepCalibration();
