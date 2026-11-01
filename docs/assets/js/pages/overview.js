"use strict";
(async function () {
  const get = (id) => document.getElementById(id);
  const text = (tag, value) => { const node = document.createElement(tag); node.textContent = value; return node; };
  const read = async (name) => (await window.TouchlineData.read(`./data/${name}.json`)).payload;
  let matches = [], visibleLimit = 6;
  function outcome(match) { return match.score[0] === match.score[1] ? "draw" : match.score[0] > match.score[1] ? "first" : "second"; }
  function renderMatches() {
    const query = get("overview-team-query").value.trim().toLocaleLowerCase();
    const result = get("overview-result-filter").value;
    const selected = matches.filter((row) => row.teams.join(" ").toLocaleLowerCase().includes(query) && (result === "all" || outcome(row) === result));
    get("overview-matches").replaceChildren(...selected.slice(0,visibleLimit).map((match) => {
      const link = document.createElement("a"), url = new URL("./matches.html", location.href);
      url.searchParams.set("match", match.id); url.hash = "match-explorer"; link.href = url.href;
      link.append(text("small", match.date), text("strong", match.teams.join(" — ")), text("b", match.score.join(" : ")), text("span", "→"));
      return link;
    }));
    get("overview-match-filter-status").textContent = selected.length ? `当前 ${selected.length} 场 · 显示 ${Math.min(selected.length,visibleLimit)} 场` : matches.length ? "没有匹配比赛。调整球队名称或赛果筛选后重试。" : "比赛目录暂无比赛。";
    get("overview-more").hidden = selected.length <= visibleLimit;
    get("overview-outcome-sample").textContent = `${selected.length} 场`;
    get("overview-outcomes").replaceChildren(...[["first","列前胜"],["draw","平局"],["second","列后胜"]].map(([key,label]) => {
      const count = selected.filter((match) => outcome(match) === key).length;
      const row = text("div", ""); row.className = "outcome-row";
      const track = text("div", ""); track.className = "outcome-track"; track.setAttribute("aria-hidden","true");
      const bar = text("span", ""); bar.style.width = `${selected.length ? count/selected.length*100 : 0}%`; track.append(bar);
      row.append(text("span",label), track, text("b", `${count} 场`)); return row;
    }));
  }
  for (const id of ["overview-team-query","overview-result-filter"]) get(id).addEventListener(id.endsWith("query") ? "input" : "change", () => { visibleLimit=6; renderMatches(); });
  get("overview-more").addEventListener("click", () => { visibleLimit+=6; renderMatches(); });
  const tasks = [
    {name:"research/matches", render(data) {
      if (data.schema !== "Touchline.MatchExplorer.v1" || !Array.isArray(data.matches) || !data.matches.every((row) => row && typeof row.id === "string" && typeof row.date === "string" && Array.isArray(row.teams) && row.teams.length===2 && row.teams.every((value) => typeof value === "string") && Array.isArray(row.score) && row.score.length===2 && row.score.every((value) => Number.isInteger(value) && value>=0))) throw new Error("比赛目录格式不完整");
      matches=data.matches; get("overview-match-count").textContent=matches.length;
      get("overview-team-query").disabled=false; get("overview-result-filter").disabled=false;
      get("overview-team-count").textContent=new Set(matches.flatMap((row) => row.teams)).size;
      renderMatches();
    }},
    {name:"research/fifa-offers", render(data) { if (!Array.isArray(data.matches)) throw new Error("报告目录格式不完整"); get("overview-report-count").textContent=data.matches.length; }},
    {name:"research/figures", render(data) { if (!Array.isArray(data.figures)) throw new Error("图表目录格式不完整"); get("overview-figure-count").textContent=data.figures.length; }},
    {name:"research/coverage", render(data) {
      if (typeof data.observed_at !== "string" || !Array.isArray(data.coverage) || !data.coverage.every((row) => row && typeof row.provider === "string")) throw new Error("覆盖范围格式不完整");
      get("overview-source-count").textContent=new Set(data.coverage.map((row) => row.provider)).size;
      get("overview-observed-date").textContent=data.observed_at.slice(0,10);
      get("overview-coverage-count").textContent=`${data.coverage.length} 条范围记录`;
      get("overview-snapshot").textContent="按来源核对赛事与赛季；范围记录与来源间可能重叠，不累加为独立比赛数。";
    }},
  ];
  async function load() {
    get("overview-retry").disabled=true;
    const results=await Promise.allSettled(tasks.filter((task) => !task.ready).map(async (task) => { task.render(await read(task.name)); task.ready=true; }));
    const failed=results.filter((result) => result.status === "rejected").length;
    get("overview-status").textContent=failed ? `${failed} 项公开摘要暂不可用，可重试；其余入口仍可使用。` : "2024 男足欧洲杯事件样本 · 历史目录，不是实时赛程。";
    get("overview-retry").hidden=!failed; get("overview-retry").disabled=false;
  }
  get("overview-retry").addEventListener("click",load);
  await Promise.allSettled([load(), (async () => {
    try {
      const data=await read("manifest");
      if (typeof data.generated_at !== "string" || !Number.isInteger(data.prediction_count)) throw new Error("Invalid manifest");
      get("overview-prediction-status").textContent=`预测快照 ${data.generated_at.slice(0,10)} · ${data.prediction_count} 场预测。发布状态以预测页的完整校验为准。`;
    } catch { get("overview-prediction-status").textContent="预测快照暂不可用，可进入预测页核对发布状态。"; }
  })()]);
})();
