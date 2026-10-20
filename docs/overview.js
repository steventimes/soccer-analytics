"use strict";
(async function () {
  const get = (id) => document.getElementById(id);
  const text = (tag, value) => { const element = document.createElement(tag); element.textContent = value; return element; };
  const read = async (name) => { const response = await fetch(`./data/${name}.json`, {cache:"no-cache"}); if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); };
  const tasks = [
    {name:"research/matches", render(data) {
      if (data.schema !== "Touchline.MatchExplorer.v1" || !Array.isArray(data.matches) || !data.matches.length) throw new Error("比赛目录格式不完整");
      get("overview-match-count").textContent = data.matches.length;
      get("overview-matches").replaceChildren(...data.matches.slice(0, 5).map((match) => {
        const link = document.createElement("a"), url = new URL("./matches.html", location.href);
        url.searchParams.set("match", match.id); url.hash = "match-explorer"; link.href = url.href;
        link.append(text("small", match.date), text("strong", match.teams.join(" — ")), text("b", match.score.join(" : ")), text("span", "→"));
        return link;
      }));
    }},
    {name:"research/fifa-offers", render(data) { if (!Array.isArray(data.matches)) throw new Error("报告目录格式不完整"); get("overview-report-count").textContent = data.matches.length; }},
    {name:"research/figures", render(data) { if (!Array.isArray(data.figures)) throw new Error("图表目录格式不完整"); get("overview-figure-count").textContent = data.figures.length; }},
    {name:"research/coverage", render(data) { if (typeof data.observed_at !== "string") throw new Error("覆盖日期缺失"); get("overview-snapshot").textContent = `研究覆盖快照 ${data.observed_at.slice(0,10)}。按来源核对赛事与赛季，不累加重叠比赛。`; }},
  ];
  async function load() {
    get("overview-retry").disabled = true;
    const results = await Promise.allSettled(tasks.filter((task) => !task.ready).map(async (task) => { task.render(await read(task.name)); task.ready = true; }));
    const failed = results.filter((result) => result.status === "rejected").length;
    get("overview-status").textContent = failed ? `${failed} 项公开摘要暂不可用，可重试；其余入口仍可使用。` : "2024 男足欧洲杯 · 全部 51 场事件样本。目录不是实时赛程。";
    get("overview-retry").hidden = !failed; get("overview-retry").disabled = false;
  }
  get("overview-retry").addEventListener("click", load);
  await Promise.allSettled([load(), (async () => {
    try {
      const data = await read("manifest");
      if (typeof data.generated_at !== "string" || !Number.isInteger(data.prediction_count)) throw new Error("Invalid manifest");
      get("overview-prediction-status").textContent = `预测快照 ${data.generated_at.slice(0,10)} · ${data.prediction_count} 场预测。发布状态以预测页的完整校验为准。`;
    } catch { get("overview-prediction-status").textContent = "预测快照暂不可用，可进入预测页核对发布状态。"; }
  })()]);
})();
