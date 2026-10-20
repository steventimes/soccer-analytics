"use strict";

(function footballMarketReference() {
  const get = (id) => document.getElementById(id);
  if (!get("market-reference")) return;
  const leagues = {PL: "英超", PD: "西甲", BL1: "德甲", SA: "意甲", FL1: "法甲"};
  const books = {bet365: "Bet365", average: "市场平均", pinnacle: "Pinnacle"};
  const fields = ["league", "season", "book"];
  const el = (tag, text) => { const node = document.createElement(tag); if (text !== undefined) node.textContent = text; return node; };
  const percent = (n) => `${(n * 100).toFixed(1)}%`;
  let data;
  function render() {
    const league = get("market-league").value, season = Number(get("market-season").value), book = get("market-book").value;
    const row = data.results.find((r) => r.competition === league && r.season === season && r.book === book);
    get("market-table").replaceChildren();
    if (!row) { get("market-status").textContent = "该联赛和赛季没有已核查的报价。"; return; }
    const missing = row.excluded.incomplete_pre_close + row.excluded.incomplete_closing + row.excluded.incomplete_both;
    get("market-status").textContent = `${leagues[league]} · ${season}/${String(season + 1).slice(-2)} · ${books[book]} · 归档比赛 ${row.test_start} 至 ${row.test_end}。共 ${row.population} 场，配对 ${row.paired_matches} 场，缺少完整三项报价 ${missing} 场，供应商滞后警告排除 ${row.excluded.provider_stale_warning} 场。`;
    get("market-training").textContent = `频率基线使用 ${row.training_rows} 场较早比赛，截止 ${row.training_end}。表中三个参考使用相同的 ${row.paired_matches} 场比赛。不同报价来源的可用比赛可能不同。`;
    if (row.paired_matches) {
      const table = el("table"), head = el("thead"), body = el("tbody"), header = el("tr");
      table.append(el("caption", "相同比赛上的回顾性参考，log loss 和 Brier 越低越好"));
      ["参考", "比赛数", "Log loss ↓", "Brier ↓", "命中率", "平均超额倒数和"].forEach((label) => { const th = el("th", label); th.scope = "col"; header.append(th); });
      head.append(header);
      for (const [key, label] of [["prior", "历史赛果频率"], ["pre_close", "收盘前报价"], ["closing", "收盘报价"]]) {
        const score = row.metrics[key], tr = el("tr"), th = el("th", label); th.scope = "row"; tr.append(th);
        [row.paired_matches, score.log_loss.toFixed(3), score.brier.toFixed(3), percent(score.accuracy), score.mean_overround === null ? "不适用" : percent(score.mean_overround)].forEach((value) => tr.append(el("td", value)));
        body.append(tr);
      }
      table.append(head, body); get("market-table").append(table);
    } else {
      get("market-table").append(el("p", "本组合没有通过质量检查的配对报价，请选择其他赛季或报价来源。"));
    }
    const url = new URL(location.href);
    fields.forEach((key) => url.searchParams.set(`market_${key}`, get(`market-${key}`).value));
    history.replaceState(null, "", url);
  }
  async function load() {
    fields.forEach((key) => get(`market-${key}`).disabled = true);
    get("market-retry").hidden = true;
    get("market-status").textContent = "正在读取足球市场参考。";
    try {
      const response = await fetch("./data/research/market-reference.json", {cache: "no-cache"});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const catalog = await response.json();
      if (catalog.schema !== "SoccerAnalytics.MarketReference.v1" || !catalog.results?.length) throw new Error("没有已核查的数据");
      data = catalog;
      const params = new URL(location.href).searchParams;
      for (const [field, values, label] of [["league", Object.keys(leagues), (v) => leagues[v]], ["season", [...new Set(data.results.map((r) => String(r.season)))].sort(), (v) => `${v}/${String(Number(v) + 1).slice(-2)}`], ["book", Object.keys(books), (v) => books[v]]]) {
        const select = get(`market-${field}`); select.replaceChildren();
        values.forEach((value) => { const option = el("option", label(value)); option.value = value; select.append(option); });
        const saved = params.get(`market_${field}`); if (values.includes(saved)) select.value = saved;
        select.disabled = false;
      }
      get("market-method").textContent = data.method;
      get("market-limits").textContent = data.limits;
      get("market-source").textContent = `归档获取于 ${data.source.snapshot_retrieved_at.slice(0, 10)}。Pinnacle 从 ${data.pinnacle_warning_from} 起的比赛报价按官方滞后警告排除。超额倒数和为主、平、客三项赔率倒数之和减 1。`;
      render();
    } catch (error) {
      get("market-table").replaceChildren(); get("market-training").textContent = "";
      get("market-status").textContent = `足球市场参考未就绪（${error.message}），可重试读取。`;
      get("market-retry").hidden = false;
    }
  }
  fields.forEach((key) => get(`market-${key}`).addEventListener("change", render));
  get("market-retry").addEventListener("click", load);
  load();
})();
