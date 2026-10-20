"use strict";

async function styleRetrieval() {
  const status = $("style-status");
  if (!status) return;
  if ($("style-retry")) { $("style-retry").hidden = true; $("style-retry").onclick = styleRetrieval; }
  const leagues = {England: "英超", France: "法甲", Germany: "德甲", Italy: "意甲", Spain: "西甲"};
  const methods = {frequency: "动作频率", heatmaps: "未压缩空间热图", nmf_start_only: "NMF · 仅起点", nmf_start_end: "NMF · 传球起终点"};
  function unavailable(error) {
    if ($("style-retry")) $("style-retry").hidden = false;
    $("style-table").replaceChildren(); $("style-share").hidden = true;
    $("style-seed-table").replaceChildren(); $("style-seed-status").textContent = "随机初始化诊断不可用。";
    for (const id of ["style-league", "style-pool"]) $(id).disabled = true;
    status.textContent = `球风检索实验未就绪（${error.message}），可重试读取公开摘要。`;
  }
  try {
    const report = await explorationJson("spatial-style");
    if (report.schema !== "SoccerAnalytics.SpatialStyle.v1" || !report.summary?.length) throw new Error("实验摘要不完整");
    $("style-scope").textContent = report.scope;
    $("style-method").textContent = report.method;
    $("style-run").textContent = `早段 ${report.windows.early.start}–${report.windows.early.end}，后段 ${report.windows.late.start}–${report.windows.late.end}。实验运行 ${report.run_id}。`;
    $("style-data-source").replaceChildren(safeLink("Wyscout 公开数据来源 ↗", report.source.source_url));
    $("style-limits").replaceChildren(...report.limits.map((text) => node("li", text)));
    const diagnostic = report.seed_stability;
    if (diagnostic) {
      if (diagnostic.initialization !== "random" || diagnostic.metric !== "cosine" || !Array.isArray(diagnostic.pairs)) throw new Error("初始化诊断不完整");
      const kinds = {shots: "射门", crosses: "传中", take_ons: "持球进攻对抗", passes: "传球"};
      const rows = diagnostic.pairs.map((row) => {
        if (!methods[row.method] || !kinds[row.kind] || !Array.isArray(row.matched_cosines) ||
            row.matched_cosines.some((v) => !Number.isFinite(v) || v < 0 || v > 1) ||
            [row.mean_cosine, row.min_cosine].some((v) => row.matched_cosines.length ? !Number.isFinite(v) || v < 0 || v > 1 : v !== null) ||
            !Number.isInteger(row.unassessed_pairs) || row.unassessed_pairs < 0) throw new Error("初始化诊断数值无效");
        return [methods[row.method], kinds[row.kind], `${row.seed_a} / ${row.seed_b}`,
          row.matched_cosines.length, row.mean_cosine === null ? "无法评估" : number(row.mean_cosine, 4),
          row.min_cosine === null ? "无法评估" : number(row.min_cosine, 4), row.unassessed_pairs];
      });
      $("style-seed-status").textContent = rows.length ? `${report.parameters.seeds.join("、")} 共 ${report.parameters.seeds.length} 个随机种子 · ${rows.length} 组分量匹配 · random 初始化 · 与上方筛选无关` : "不足两个种子，无法比较初始化稳定性。";
      explorationTable("style-seed-table", ["方法", "动作", "种子对", "有效匹配", "平均余弦", "最小余弦", "未评估分量"], rows, {
        "图表": "早段空间基底的随机初始化敏感性",
        "筛选": `五大联赛共享早段基底 · 全部种子对 ${report.parameters.seeds.join(" / ")} · 不按联赛或候选池筛选`,
        "来源与范围": `${report.source.provider} · 早段 ${report.windows.early.start}–${report.windows.early.end} · 运行 ${report.run_id} · ${$("style-seed-scope").textContent}`,
      });
    } else {
      $("style-seed-status").textContent = "当前摘要没有随机初始化诊断，请重新生成球风实验。";
      $("style-seed-table").replaceChildren();
    }
    resetOptions("style-league", ["all", ...Object.keys(leagues)], (key) => leagues[key] || "五大联赛总体");
    const parameters = new URLSearchParams(location.search);
    for (const [key, id] of [["style_league", "style-league"], ["style_pool", "style-pool"]]) {
      const value = parameters.get(key);
      if ([...$(id).options].some((option) => option.value === value)) $(id).value = value;
    }
    function render() {
      const rows = report.summary.filter((r) => r.scope === $("style-league").value && r.pool === $("style-pool").value);
      $("style-table").replaceChildren();
      if (rows.length !== 4) throw new Error("所选范围缺少对照方法");
      const first = rows[0];
      status.textContent = `${first.queries} 名球员 · 每次检索 ${first.candidate_min}–${first.candidate_max} 名候选 · 随机 Top-1 ${number(first.random_top1 * 100, 2)}% · NMF 拟合警告 ${report.convergence.warnings} 条`;
      explorationTable("style-table", ["方法", "维度", "Top-1", "Top-5", "MRR", "Top-1 种子范围"], rows.map((r) => [methods[r.method], r.features,
        `${number(r.top1.mean * 100, 1)}%`, `${number(r.top5.mean * 100, 1)}%`, number(r.mrr.mean, 3),
        r.runs === 1 ? "确定性基线" : `${number(r.top1.min * 100, 1)}–${number(r.top1.max * 100, 1)}%（${r.runs} 种子）`]));
      const url = new URL(location.href);
      for (const [key, id, fallback] of [["style_league", "style-league", "all"], ["style_pool", "style-pool", "same_league"]]) {
        if ($(id).value === fallback) url.searchParams.delete(key); else url.searchParams.set(key, $(id).value);
      }
      if (typeof window === "undefined" || !window.TouchlineWorkspace || window.TouchlineWorkspace.active("style-retrieval")) history.replaceState(null, "", url);
      url.hash = "style-retrieval"; $("style-share").href = url.href; $("style-share").hidden = false;
    }
    for (const id of ["style-league", "style-pool"]) {
      $(id).disabled = false; $(id).onchange = () => { try { render(); } catch (error) { unavailable(error); } };
    }
    render();
  } catch (error) {
    unavailable(error);
  }
}

styleRetrieval();
