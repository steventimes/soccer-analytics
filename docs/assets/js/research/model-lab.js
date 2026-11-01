"use strict";

const modelLabNames = {poisson_constant:"固定进球率", poisson:"Poisson 攻防", poisson_decay:"时间衰减 Poisson", dixon_coles:"Dixon–Coles", bivariate_poisson:"双变量 Poisson", identity:"原始概率", temperature:"温度缩放", dirichlet_odir:"Dirichlet ODIR", draw_constant:"平局常数", draw_logistic:"平局 Logistic", draw_beta:"平局 Beta", elo_ordered_logit:"Elo 有序 Logistic", elo_multinomial:"Elo 多分类", logistic_l2:"L2 Logistic", xgboost:"XGBoost", lightgbm:"LightGBM", lightgbm_goss:"LightGBM GOSS", catboost:"CatBoost"};
function modelLabRows(group, stage, metric) {
  if (stage === "development" && metric !== "log_loss") return [];
  return group.candidates.map((candidate) => ({...candidate, value:stage === "development" ? candidate.selection_log_loss : candidate.holdout[metric]}));
}
function modelLabTrace(group) {
  const trace = group.provenance;
  if (!trace) return {schema:"Touchline.AnalysisProvenance.v1",status:"legacy_missing",timing_policy:"not_recorded",result:null,run_id:null,code:null,environment:null,configuration:null,configuration_sha256:null};
  if (trace.schema!=="Touchline.AnalysisProvenance.v1" || !["legacy_missing","recorded"].includes(trace.status) || !["not_recorded","kickoff_order","observed_result"].includes(trace.timing_policy)) throw new Error("分析追溯格式不完整");
  if ((trace.result && (!/^[a-f0-9]{64}$/.test(trace.result.sha256) || typeof trace.result.path!=="string")) || (trace.status==="recorded" && (!/^[a-f0-9]{64}$/.test(trace.run_id) || !/^[a-f0-9]{64}$/.test(trace.code?.application_sha256) || !/^[a-f0-9]{64}$/.test(trace.configuration_sha256)))) throw new Error("分析追溯身份不完整");
  if(trace.status==="legacy_missing" && (trace.run_id||trace.code||trace.configuration_sha256||trace.timing_policy!=="not_recorded"))throw new Error("旧结果包含冲突的追溯声明");
  return trace;
}
function searchComparisonRows(comparison) {
  const names={random:"随机",bayesian:"GP",optuna:"TPE",genetic:"遗传",beam1:"Beam width 1",beam2:"Beam width 2"};
  if (!comparison || comparison.timing_policy!=="observed_result" || !Array.isArray(comparison.runs) || !comparison.runs.length || !Number.isInteger(comparison.outer_rows) || comparison.outer_rows<=0) throw new Error("搜索对照格式不完整");
  return comparison.runs.map((run)=>{
    const metrics=[run.selection_log_loss,run.outer_log_loss,run.elapsed_seconds];
    const counts=[run.objective_requests,run.inner_fit_attempts,run.outer_fit_attempts,run.failed_trials];
    if (!names[run.arm] || !Number.isInteger(run.search_seed) || metrics.some(value=>typeof value!=="number"||!Number.isFinite(value)||value<0) || counts.some(value=>!Number.isInteger(value)||value<0) || !Array.isArray(run.curve) || run.curve.some(point=>!Number.isInteger(point.fit_attempts)||point.fit_attempts<0||typeof point.elapsed_seconds!=="number"||!Number.isFinite(point.elapsed_seconds)||point.elapsed_seconds<0||point.best_log_loss!==null&&(typeof point.best_log_loss!=="number"||!Number.isFinite(point.best_log_loss)||point.best_log_loss<0))) throw new Error("搜索成本或成绩不完整");
    const diversity=run.parent_diversity||[];
    if (!Array.isArray(diversity) || diversity.some(step=>!Number.isInteger(step.parent_count)||step.parent_count<1||typeof step.parent_distance_unit_l2!=="number"||!Number.isFinite(step.parent_distance_unit_l2)||step.parent_distance_unit_l2<0)) throw new Error("父配置多样性不完整");
    return {...run,label:names[run.arm],parentDistance:diversity.length?Math.max(...diversity.map(step=>step.parent_distance_unit_l2)):null};
  });
}
function readinessUtc(value) {
  if (value===undefined || value===null || value==="") return "未记录";
  if (typeof value!=="string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/.test(value)) throw new Error("搜索准备时刻格式不完整");
  const instant=new Date(value);
  if (!Number.isFinite(instant.getTime())) throw new Error("搜索准备时刻格式不完整");
  // Preserve the recorded fractional precision while converting the offset to UTC.
  return instant.toISOString().replace("T"," ").replace(/\.\d{3}Z$/,`${value.match(/\.\d+/)?.[0]||""} UTC`);
}
function searchReadinessRows(summary) {
  const counts = [summary?.unique_training_rows,summary?.unique_validation_rows,summary?.followup?.observed_rows,summary?.followup?.pending_rows];
  if (!summary || summary.timing_state!=="admitted" || !["insufficient","not_blocked"].includes(summary.state) || summary.model_fit_attempts!==0 || counts.some(value=>!Number.isInteger(value)||value<0) || !Array.isArray(summary.folds) || !summary.folds.length || !Array.isArray(summary.reason_codes) || !Array.isArray(summary.warning_codes) || !["unavailable","available_unregistered"].includes(summary.followup.state) || summary.followup.registered!==false) throw new Error("搜索准备摘要格式不完整");
  return summary.folds.map((fold,index)=>{
    if ([fold.training_rows,fold.validation_rows,fold.training_rows_without_history].some(value=>!Number.isInteger(value)||value<0) || !Array.isArray(fold.constant_features) || !Array.isArray(fold.varying_features)) throw new Error("搜索准备窗口格式不完整");
    return [index+1,readinessUtc(fold.fit_as_of),fold.training_rows,fold.validation_rows,fold.training_rows_without_history,fold.constant_features.length,fold.varying_features.length];
  });
}
function renderSearchComparisons(report, group, searchReport, searchError, retry) {
  let host=$("lab-search-comparisons");
  if (!host) {host=node("div");host.id="lab-search-comparisons";$("lab-table").after(host);}
  host.replaceChildren();
  if (searchError) {
    const notice=node("div",undefined,"signal signal-warning"),body=node("div"),button=node("button","重试搜索准备摘要");
    button.type="button"; button.addEventListener("click",retry);
    body.append(node("strong","搜索准备摘要暂不可用"),node("p","时间准入与效益准备状态未知；现有历史模型与已完成比较仍单独展示。"),button);
    notice.append(node("span",undefined,"signal-mark"),body); host.append(notice);
  }
  const reasons={NO_OBSERVED_HISTORY_IN_TRAINING_ROWS:"所有折的训练行均无已观察历史，无法支持本轮特征搜索采用。",INDEPENDENT_CONFIRMATION_NOT_REGISTERED:"后续独立确认尚未注册，不能据此形成采用结论。",NO_OBSERVED_FOLLOWUP_BEFORE_FINAL_HOLDOUT:"最终留出之前尚无已观察的后续队列。",CONSTANT_BASELINE_TRAINING_FEATURES:"基准配方存在整折常数训练特征；不能据此推断所有算法无效。"};
  for (const [index,{summary,rows,asOf,sources}] of (searchReport?.readiness||[]).filter(item=>!group||item.summary.competition===group.competition).entries()) {
    const section=node("section"),notice=node("div",undefined,"signal signal-warning"),body=node("div"),details=node("details"),table=node("div");
    section.id=`lab-readiness-${index}`; table.id=`lab-readiness-table-${index}`; table.className="table-scroll";
    section.append(node("h3",`${summary.competition} · 搜索准备诊断`));
    body.append(node("strong",summary.state==="insufficient"?"效益准备不足 · 未执行搜索":"尚未发现结构性阻断 · 有效性未证明"),node("p","时间准入：通过。时间准入通过不等于可以采纳搜索结果。"));
    notice.append(node("span",undefined,"signal-mark"),body); section.append(notice);
    const scope=node("div",undefined,"readiness-time-scope");
    scope.append(node("p",`诊断观察时点（UTC）：${asOf}`));
    for (const source of sources) scope.append(node("p",source.label));
    if (!sources.length) scope.append(node("p","来源赛季与获取时刻：未记录"));
    scope.append(node("p","本节为当前无拟合准备盘点，与上方历史模型组的赛季及最终留出分别解释。来源获取时刻不代表比赛完赛或结果最早可用时刻；每折训练截点不是最终留出。","context-note"));
    section.append(scope);
    section.append(node("p",`去重训练 ${summary.unique_training_rows} 场 · 去重验证 ${summary.unique_validation_rows} 场 · 后续已观察 ${summary.followup.observed_rows} 场 · 后续待观察 ${summary.followup.pending_rows} 场`));
    if (summary.development_rows!==undefined && summary.outer_rows!==undefined) section.append(node("p",`开发标签 ${summary.development_rows} 场 · 开发期外层 ${summary.outer_rows} 场`));
    if (summary.provenance?.code?.application_sha256) {
      const identity=node("details",undefined,"table-inspector"),list=node("dl");
      identity.append(node("summary","查看准备摘要来源与身份"));
      const values={"诊断状态":summary.execution_status,"协议身份":summary.protocol_sha256,"代码身份":summary.provenance.code.application_sha256,"运行身份":summary.provenance.run_id,"来源":sources.map(source=>source.identity).join(" / ")||"未记录"};
      list.append(...Object.entries(values).flatMap(([name,value])=>[node("dt",name),node("dd",value)]));identity.append(list);section.append(identity);
    }
    for (const code of [...new Set([...summary.reason_codes,...summary.warning_codes])]) section.append(node("p",reasons[code]||"未识别的准备原因，请核对来源摘要。","context-note"));
    section.append(node("p","诊断未拟合模型；重叠训练窗口不是独立数据。后续队列仅为盘点，未注册为确认实验。","context-note"));
    details.append(node("summary","查看逐折训练、验证与常数特征"),table); section.append(details); host.append(section);
    explorationTable(table.id,["窗口","训练截点（UTC）","实际训练（场）","验证（场）","无历史训练（场）","常数特征（列）","变化特征（列）"],rows,{"诊断观察时点（UTC）":asOf,"来源赛季与获取时刻（UTC）":sources.map(source=>source.label).join(" / ")||"未记录","范围":"无拟合基准配方诊断，不是已完成优化结果。来源获取时刻不代表比赛完赛或结果最早可用时刻；每折训练截点不是最终留出。"});
    table.setAttribute("aria-label",`${summary.competition} 搜索准备逐折数值表，可滚动查看`);table.querySelector("caption").textContent=`${summary.competition} 搜索准备逐折计数`;
  }
  const comparisons=new Map([...(report?.search_comparisons||[]),...(searchReport?.comparisons||[])].map(item=>[`${item.comparison.competition}:${item.comparison.protocol_sha256}`,item]));
  for (const [index,{comparison,rows}] of [...comparisons.values()].filter(item=>!group||item.comparison.competition===group.competition).entries()) {
    const section=node("section"),table=node("div"),details=node("details"),curve=node("div");
    table.id=`lab-search-table-${index}`; curve.id=`lab-search-curve-${index}`;
    table.className="table-scroll"; curve.className="table-scroll";
    section.append(node("h3","共同起点搜索对照"),node("p",`${comparison.competition} · ${comparison.development_rows} 场开发 · ${comparison.outer_rows} 场开发期外层 · ${comparison.evaluation_start} 至 ${comparison.evaluation_end}`),node("p","以下是独立的开发期外层检验，并非上方历史组的最终留出。各臂共享四个初始配置；拟合数和秒为实际记录，未使用完整试验缓存。","context-note"),table);
    details.append(node("summary","查看逐次评价与实际拟合成本"),curve); section.append(details); host.append(section);
    explorationTable(table.id,["方法","搜索种子","开发 log loss","开发外层 log loss","目标请求","内层拟合","外层拟合","失败","耗时（秒）","最大父配置距离（单位空间 L2）"],rows.map(row=>[row.label,row.search_seed,number(row.selection_log_loss,4),number(row.outer_log_loss,4),row.objective_requests,row.inner_fit_attempts,row.outer_fit_attempts,row.failed_trials,number(row.elapsed_seconds,3),number(row.parentDistance,3)]),{"协议身份":comparison.protocol_sha256,"最终留出起点":comparison.final_holdout_start,"采用结论":"尚未确定"});
    explorationTable(curve.id,["方法","搜索种子","目标请求累计","实际拟合累计","累计秒","最佳开发 log loss"],rows.flatMap(row=>row.curve.map(point=>[row.label,row.search_seed,point.objective_requests,point.fit_attempts,number(point.elapsed_seconds,3),point.best_log_loss===null?"尚无成功完整评价":number(point.best_log_loss,4)])),{"曲线范围":"仅完整开发评价，失败仍计真实成本；不以开发最小值证明未来优势。"});
    for (const [container,label] of [[table,`${comparison.competition} 共同起点方法与外层成绩`],[curve,`${comparison.competition} 逐次评价与实际拟合成本`]]) {
      container.setAttribute("aria-label",`${label}数值表，可滚动查看`);
      container.querySelector("caption").textContent=label;
    }
  }
}
if (typeof module !== "undefined") module.exports = {modelLabRows,modelLabTrace,searchComparisonRows};
if (typeof document !== "undefined") window.TouchlineWorkspace.loadWhenActive("model-lab", async function modelLab() {
  if (!$("lab-status")) return;
  let report, searchReport=null, searchError=false;
  function render() {
    const group = report.groups.find((entry) => entry.id === $("lab-group").value);
    const stage = $("lab-stage").value, metric = $("lab-metric").value, development = stage === "development";
    const rows = modelLabRows(group, stage, metric);
    const trace=group.provenance, timing={not_recorded:"原结果未记录",kickoff_order:"开球顺序",observed_result:"结果实际可用时间"}[trace.timing_policy];
    const metricLabel = $("lab-metric").selectedOptions[0].textContent;
    const label = (row) => `${modelLabNames[row.model] || row.model}${row.scope ? " · " + ({default:"默认",model:"模型搜索",joint:"联合搜索"}[row.scope] || row.scope) : ""}`;
    $("lab-chart").replaceChildren(); $("lab-cohort").replaceChildren(); $("lab-table").replaceChildren(); $("lab-notes").replaceChildren(); $("lab-trace").replaceChildren();
    $("lab-status").textContent = `${group.label} · ${development ? "开发期时间折选择分数" : "最终时间留出评价"} · ${group.candidates.length} 个候选。`;
    const cohort = node("span"); cohort.append(node("strong", `${group.holdout_rows} 场`), node("span", ` 最终留出 · ${group.holdout_start.slice(0,10)} — ${group.holdout_end.slice(0,10)}`));
    $("lab-cohort").append(cohort, node("span", `来源：${group.sources.map((source) => `${source.provider} ${source.season}/${Number(source.season)+1}`).join(" / ")}`));
    const url = new URL(location.href); url.searchParams.set("experiment",group.id); url.searchParams.set("metric",metric); url.searchParams.set("stage",stage);
    if (!window.TouchlineWorkspace || window.TouchlineWorkspace.active("model-lab")) history.replaceState(null,"",url);
    renderSearchComparisons(report,group,searchReport,searchError,load);
    if (!rows.length) { $("lab-chart").append(node("p", "公开摘要仅提供开发期 log loss。请选择 Log loss，或切回最终时间留出。", "context-note")); return; }
    const ceiling = metric === "accuracy" ? 1 : Math.max(...rows.map((row) => row.value), .01) * 1.08;
    $("lab-chart").setAttribute("aria-label", `${development ? "开发期" : "留出集"} ${metricLabel}，所有柱形共用从零开始的坐标，最大值 ${ceiling.toFixed(3)}`);
    for (const row of rows) {
      const selected = row.id === group.selected, item = node("article", undefined, `lab-result${selected ? " is-selected" : ""}`), heading = node("div", undefined, "lab-result-heading");
      heading.append(node("strong", label(row)), node("span", number(row.value,4)));
      const track = node("div", undefined, "paired-track"), bar = node("i"); bar.style.width = `${row.value/ceiling*100}%`; track.setAttribute("aria-hidden","true"); track.append(bar);
      item.append(heading,track,node("small", selected ? "开发期选定方案" : "同组对照")); $("lab-chart").append(item);
    }
    $("lab-chart").append(node("p", `共用坐标：0–${number(ceiling,3)} · ${metric === "accuracy" ? "比例，越高越好" : "越低越好"}`, "context-note"));
    explorationTable("lab-table", ["候选", development ? "开发期 log loss" : `留出 ${metricLabel}`, "开发期选择"], rows.map((row) => [label(row),number(row.value,4),row.id===group.selected?"是":"否"]), {"样本":`${group.competition} ${group.seasons.join(",")} · ${group.holdout_rows} 场最终留出 · ${group.holdout_start} 至 ${group.holdout_end}`,"选择规则":group.selected_on,"来源身份":group.sources.map((source)=>`${source.provider} ${source.sha256}`).join(" / "),"时点路径":timing,"运行身份":trace.run_id||"原结果未记录","结果文件":trace.result?`${trace.result.path} · ${trace.result.sha256}`:"未提供","配置身份":trace.configuration_sha256||"原结果未记录"});
    $("lab-notes").append(node("p","这些是历史离线实验。候选在开发期选择，映射冻结后进行时间留出评价；各组输入和预热规则分别解释。"),node("p",`选择规则：${group.selected_on}。公开表格保留四位小数；各组样本与配方分别解释。`));
    const details=node("details",undefined,"table-inspector"), metadata=node("dl"), download=node("button","下载追溯 JSON");
    details.id="lab-provenance"; details.append(node("summary","查看运行与结果身份"));
    const values={"时点路径":timing,"结果文件":trace.result?.path||"未提供","结果 SHA-256":trace.result?.sha256||"未提供","运行身份":trace.run_id||"原结果未记录","应用代码 SHA-256":trace.code?.application_sha256||"原结果未记录","Git 提交":trace.code?.git_head||"原结果未记录","完整配置 SHA-256":trace.configuration_sha256||"原结果未记录"};
    metadata.append(...Object.entries(values).flatMap(([name,value])=>[node("dt",name),node("dd",value)])); details.append(metadata);
    if(group.resultUrl){const link=node("a","打开固定版本结果文件");link.href=group.resultUrl;details.append(link);}
    if(trace.status==="legacy_missing")details.append(node("p","这些旧结果未保存运行收据；代码、配置及实际时点路径保持未知，不用当前实现补写历史。", "context-note"));
    else details.append(node("pre",JSON.stringify({environment:trace.environment,configuration:trace.configuration},null,2)));
    download.type="button"; download.id="lab-trace-export"; download.addEventListener("click",()=>{
      const body={schema:"Touchline.AnalysisTraceExport.v1",group:group.id,sources:group.sources,selection:group.selected_on,provenance:trace};
      const url=URL.createObjectURL(new Blob([JSON.stringify(body,null,2)+"\n"],{type:"application/json"})),link=node("a");link.href=url;link.download=`touchline-${group.id}-trace.json`;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
    });details.append(download);$("lab-trace").append(details);


  }
  async function load() {
    $("lab-retry").hidden=true; for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=true;
    const [modelResult,searchResult]=await Promise.allSettled([explorationJson("model-lab"),explorationJson("search-comparison")]);
    searchReport=null; searchError=false;
    try {
      if (searchResult.status!=="fulfilled") throw searchResult.reason;
      const value=searchResult.value;
      if (value.schema!=="Touchline.SearchComparison.v1" || !Array.isArray(value.comparisons) || (value.readiness!==undefined&&!Array.isArray(value.readiness))) throw new Error("搜索摘要格式不完整");
      searchReport={
        comparisons:value.comparisons.map(comparison=>({comparison,rows:searchComparisonRows(comparison)})),
        readiness:(value.readiness||[]).map(summary=>{
          const rows=searchReadinessRows(summary),asOf=readinessUtc(summary.provenance?.configuration?.as_of);
          if (summary.sources!==undefined && summary.sources!==null && !Array.isArray(summary.sources)) throw new Error("搜索准备来源格式不完整");
          const sources=(summary.sources||[]).map(source=>{
            if (!source || typeof source!=="object" || Array.isArray(source) || [source.competition,source.provider,source.sha256].some(value=>value!==undefined&&value!==null&&typeof value!=="string") || (source.season!==undefined&&source.season!==null&&typeof source.season!=="string"&&!Number.isInteger(source.season))) throw new Error("搜索准备来源格式不完整");
            return {label:`${source.competition??"联赛未记录"} · 来源赛季 ${source.season??"未记录"} · ${source.provider??"来源未记录"} · 获取时刻（UTC）：${readinessUtc(source.fetched_at)}`,identity:`${source.provider??"未记录"} ${source.season??"未记录"} · ${source.sha256??"未记录"}`};
          });
          return {summary,rows,asOf,sources};
        })
      };
    } catch(error) { searchError=true; }
    try {
      if(modelResult.status!=="fulfilled")throw modelResult.reason;
      report=modelResult.value;
      if (report.schema!=="Touchline.ModelLab.v1" || !report.groups?.length || report.groups.some((group)=>!group.candidates?.length || group.candidates.some((row)=>![row.selection_log_loss,...Object.values(row.holdout)].every((value)=>typeof value==="number"&&Number.isFinite(value)&&value>=0)))) throw new Error("实验摘要格式不完整");
      if (report.search_comparisons!==undefined && !Array.isArray(report.search_comparisons)) throw new Error("搜索对照格式不完整");
      report={...report,groups:report.groups.map(group=>{
        const provenance=modelLabTrace(group);
        let resultUrl=null;
        if(provenance.result?.repository_url){
          const url=new URL(provenance.result.repository_url);
          if(url.origin!=="https://github.com"||!url.pathname.startsWith("/steventimes/soccer-analytics/blob/"))throw new Error("结果链接格式不完整");
          resultUrl=url.href;
        }
        return {...group,provenance,resultUrl};
      }),search_comparisons:(report.search_comparisons||[]).map(comparison=>({comparison,rows:searchComparisonRows(comparison)}))};
      resetOptions("lab-group",report.groups.map((group)=>group.id),(id)=>report.groups.find((group)=>group.id===id).label);
      const query=new URLSearchParams(location.search);
      for(const [id,key] of [["lab-group","experiment"],["lab-metric","metric"],["lab-stage","stage"]]) if([...$(id).options].some((option)=>option.value===query.get(key))) $(id).value=query.get(key);
      for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=false;
      render();
    } catch(error) { report=null; $("lab-search-comparisons")?.replaceChildren(); for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).disabled=true; $("lab-status").textContent=`实验汇总暂不可用（${error.message}）。`; $("lab-retry").hidden=false; $("lab-chart").replaceChildren(); $("lab-table").replaceChildren(); $("lab-cohort").replaceChildren(); $("lab-notes").replaceChildren(); $("lab-trace").replaceChildren(); renderSearchComparisons(null,null,searchReport,searchError,load); }
  }
  for(const id of ["lab-group","lab-metric","lab-stage"]) $(id).addEventListener("change",()=>{if(report)render();});
  $("lab-retry").addEventListener("click",load); await load();
});
