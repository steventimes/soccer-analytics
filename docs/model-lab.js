"use strict";

const modelLabNames = {poisson_constant:"固定进球率", poisson:"Poisson 攻防", poisson_decay:"时间衰减 Poisson", dixon_coles:"Dixon–Coles", bivariate_poisson:"双变量 Poisson", identity:"原始概率", temperature:"温度缩放", dirichlet_odir:"Dirichlet ODIR", draw_constant:"平局常数", draw_logistic:"平局 Logistic", draw_beta:"平局 Beta", elo_ordered_logit:"Elo 有序 Logistic", elo_multinomial:"Elo 多分类", logistic_l2:"L2 Logistic", xgboost:"XGBoost", lightgbm:"LightGBM", lightgbm_goss:"LightGBM GOSS", catboost:"CatBoost"};
function modelLabRows(group, stage, metric) {
  if (stage === "development" && metric !== "log_loss") return [];
  return group.candidates.map((candidate) => ({...candidate, value:stage === "development" ? candidate.selection_log_loss : candidate.holdout[metric]}));
}
if (typeof module !== "undefined") module.exports = {modelLabRows};
if (typeof document !== "undefined") (async function modelLab() {
  if (!$("lab-status")) return;
  let report;
  function render() {
    const group = report.groups.find((entry) => entry.id === $("lab-group").value);
    const stage = $("lab-stage").value, metric = $("lab-metric").value, development = stage === "development";
    const rows = modelLabRows(group, stage, metric);
    const label = (row) => `${modelLabNames[row.model] || row.model}${row.scope ? " · " + ({default:"默认",model:"模型搜索",joint:"联合搜索"}[row.scope] || row.scope) : ""}`;
    $("lab-chart").replaceChildren(); $("lab-cohort").replaceChildren(); $("lab-table").replaceChildren(); $("lab-notes").replaceChildren();
    $("lab-status").textContent = `${group.label} · ${development ? "开发期时间折选择分数" : "最终时间留出评价"} · ${group.candidates.length} 个候选。`;
    const cohort = node("span"); cohort.append(node("strong", `${group.holdout_rows} 场`), node("span", ` 最终留出 · ${group.holdout_start.slice(0,10)} — ${group.holdout_end.slice(0,10)}`));
    $("lab-cohort").append(cohort, node("span", `来源：${group.sources.map((source) => `${source.provider} ${source.season}/${Number(source.season)+1}`).join(" / ")}`));
    const url = new URL(location.href); url.searchParams.set("experiment",group.id); url.searchParams.set("metric",metric); url.searchParams.set("stage",stage);
    if (!window.TouchlineWorkspace || window.TouchlineWorkspace.active("model-lab")) history.replaceState(null,"",url);
    if (!rows.length) { $("lab-chart").append(node("p", "公开摘要仅提供开发期 log loss。请选择 Log loss，或切回最终时间留出。", "context-note")); return; }
    const ceiling = metric === "accuracy" ? 1 : Math.max(...rows.map((row) => row.value), .01) * 1.08;
    $("lab-chart").setAttribute("aria-label", `${development ? "开发期" : "留出集"} ${metric}，所有柱形共用从零开始的坐标，最大值 ${ceiling.toFixed(3)}`);
    for (const row of rows) {
      const selected = row.id === group.selected, item = node("article", undefined, `lab-result${selected ? " is-selected" : ""}`), heading = node("div", undefined, "lab-result-heading");
      heading.append(node("strong", label(row)), node("span", number(row.value,4)));
      const track = node("div", undefined, "paired-track"), bar = node("i"); bar.style.width = `${row.value/ceiling*100}%`; track.setAttribute("aria-hidden","true"); track.append(bar);
      item.append(heading,track,node("small", selected ? "开发期选定方案" : "同组对照")); $("lab-chart").append(item);
    }
    $("lab-chart").append(node("p", `共用坐标：0–${number(ceiling,3)} · ${metric === "accuracy" ? "比例，越高越好" : "越低越好"}`, "context-note"));
    explorationTable("lab-table", ["候选", development ? "开发期 log loss" : `留出 ${metric}`, "开发期选择"], rows.map((row) => [label(row),number(row.value,4),row.id===group.selected?"是":"否"]), {"样本":`${group.competition} ${group.seasons.join(",")} · ${group.holdout_rows} 场最终留出 · ${group.holdout_start} 至 ${group.holdout_end}`,"选择规则":group.selected_on,"来源身份":group.sources.map((source)=>`${source.provider} ${source.sha256}`).join(" / ")});
    $("lab-notes").append(node("p","这些是历史离线实验。候选在开发期选择，映射冻结后进行时间留出评价；各组输入和预热规则分别解释。"),node("p",`选择规则：${group.selected_on}。公开表格保留四位小数；各组样本与配方分别解释。`));

  }
  async function load() {
    $("lab-retry").hidden=true; for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=true;
    try {
      report=await explorationJson("model-lab");
      if (report.schema!=="Touchline.ModelLab.v1" || !report.groups?.length || report.groups.some((group)=>!group.candidates?.length || group.candidates.some((row)=>![row.selection_log_loss,...Object.values(row.holdout)].every((value)=>typeof value==="number"&&Number.isFinite(value)&&value>=0)))) throw new Error("实验摘要格式不完整");
      resetOptions("lab-group",report.groups.map((group)=>group.id),(id)=>report.groups.find((group)=>group.id===id).label);
      const query=new URLSearchParams(location.search);
      for(const [id,key] of [["lab-group","experiment"],["lab-metric","metric"],["lab-stage","stage"]]) if([...$(id).options].some((option)=>option.value===query.get(key))) $(id).value=query.get(key);
      for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=false;
      render();
    } catch(error) { report=null; for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=true; $("lab-status").textContent=`实验汇总暂不可用（${error.message}）。`; $("lab-retry").hidden=false; $("lab-chart").replaceChildren(); $("lab-table").replaceChildren(); $("lab-cohort").replaceChildren(); }
  }
  for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).addEventListener("change",()=>{if(report)render();});
  $("lab-retry").addEventListener("click",load); await load();
})();
