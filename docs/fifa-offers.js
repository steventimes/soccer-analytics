"use strict";

async function fifaOffers() {
  if (!$("fifa-status")) return;
  const categories = [["in_front", "前方"], ["in_between", "线间"], ["out_to_in", "外向内"],
    ["in_to_out", "内向外"], ["in_behind", "身后"], ["no_movement", "无移动"]];
  const seriesNames = {fcwc2025: "2025 世俱杯", fwc2026: "2026 世界杯"};
  let report, selected;
  const percentage = (n, d) => d ? `${number(n / d * 100, 1)}%` : "无接应记录";
  function render() {
    if (!report?.matches?.length || !$("fifa-match").value) return;
    selected = report.matches.find((m) => m.id === $("fifa-match").value);
    if (!selected) return;
    const shares = $("fifa-mode").value === "shares", [a, b] = selected.teams;
    const scope = `${selected.date} · ${a.team} ${selected.goals.join(":")} ${b.team}`;
    $("fifa-status").textContent = `${seriesNames[selected.series]} · ${scope} · 全场报告，已核对球员记录与全队总数`;
    $("fifa-summary").replaceChildren(); $("fifa-sources").replaceChildren();
    for (const [i, team] of selected.teams.entries()) {
      const card = node("div");
      card.append(node("strong", `${i === 0 ? "A ●" : "B ◇"} ${team.team}`),
        node("p", `${team.total_offers} 次接应 · 其中 ${team.offers_received} 次接球（${percentage(team.offers_received, team.total_offers)}） · ${team.players} 条球员记录`));
      $("fifa-summary").append(card);
      const source = node("a", `${team.team}：球员表第 ${team.player_table_page} 页 ↗`);
      source.href = `${selected.source_url}#page=${team.player_table_page}`;
      source.target = "_blank"; source.rel = "noopener noreferrer";
      const overview = node("a", `全队总数第 ${team.overview_page} 页 ↗`);
      overview.href = `${selected.source_url}#page=${team.overview_page}`;
      overview.target = "_blank"; overview.rel = "noopener noreferrer";
      const line = node("p"); line.append(source, node("span", " · "), overview); $("fifa-sources").append(line);
    }
    const svg = explorationSvg(scope, `FIFA 官方报告 · ${shares ? "以各队接应总数为分母，坐标 0–100%" : "六类接应次数，两队共用从零开始的坐标"}`, 700);
    const offset = Number(svg.dataset.headerOffset), left = 145, width = 500;
    const max = shares ? 100 : Math.max(10, Math.ceil(Math.max(...selected.teams.flatMap((t) => categories.map(([key]) => t[key]))) / 10) * 10);
    const value = (team, key) => shares ? (team.total_offers ? team[key] / team.total_offers * 100 : 0) : team[key];
    const colors = ["#2468a0", "#647382"];
    const defs = svgElement("defs", {}), pattern = svgElement("pattern", {id: "fifa-offers-pattern", width: 7, height: 7, patternUnits: "userSpaceOnUse"});
    pattern.append(svgElement("rect", {width: 7, height: 7, fill: colors[1]}), svgElement("path", {d: "M-1 1L1 -1M0 7L7 0M6 8L8 6", stroke: "white", "stroke-opacity": .55})); defs.append(pattern); svg.append(defs);
    [a, b].forEach((team, i) => {
      wrapExplorationText(`${i ? "B ◇" : "A ●"} ${team.team}`, 42).forEach((line, n) => svg.append(svgElement("text", {x: i ? 395 : 30, y: 89 + offset + n * 19, fill: colors[i], "font-size": 14}, line)));
    });
    [0, max / 2, max].forEach((tick) => {
      const x = left + tick / max * width;
      svg.append(svgElement("line", {x1: x, x2: x, y1: 136 + offset, y2: 605 + offset, stroke: "#e2e8ee"}),
        svgElement("text", {x, y: 630 + offset, "text-anchor": "middle", fill: "#647382", "font-size": 13}, `${tick}${shares ? "%" : ""}`));
    });
    categories.forEach(([key, label], index) => {
      const y = 149 + offset + index * 77;
      svg.append(svgElement("text", {x: 126, y: y + 24, "text-anchor": "end", fill: "#203344", "font-size": 15}, label));
      selected.teams.forEach((team, i) => {
        const v = value(team, key), w = v / max * width, yy = y + i * 25;
        const bar = svgElement("rect", {x: left, y: yy, width: w, height: 17, fill: i ? "url(#fifa-offers-pattern)" : colors[0]});
        bar.append(svgElement("title", {}, `${team.team} · ${label}：${team[key]} 次，${percentage(team[key], team.total_offers)}`));
        svg.append(bar, svgElement("text", {x: left + w + 7, y: yy + 13, fill: "#203344", "font-size": 13}, shares ? percentage(team[key], team.total_offers) : team[key]));
      });
    });
    svg.append(svgElement("text", {x: 145, y: 664 + offset, fill: "#647382", "font-size": 13}, shares ? "各队分类占比加总为 100%；整场接应总数见摘要。" : "全场次数，未校准控球、出场时间与比赛状态。"));
    svg.classList.add("export-only");
    $("fifa-chart").replaceChildren(pairedMetricView(categories.map(([key,title]) => ({title, ceiling:max, unit:shares ? "%" : "次", values:selected.teams.map((team) => ({value:value(team,key), label:shares ? percentage(team[key],team.total_offers) : `${team[key]} 次`}))}))), svg);
    const fields = [["total_offers", "接应总数"], ...categories, ["offers_received", "其中接球"]];
    explorationTable("fifa-table", ["项目", `${a.team} 次数`, "占本队接应总数", `${b.team} 次数`, "占本队接应总数"],
      fields.map(([key, label]) => [label, a[key], percentage(a[key], a.total_offers), b[key], percentage(b[key], b.total_offers)]));
    $("fifa-export").disabled = false;
    const url = new URL(location.href); url.searchParams.set("fifa", selected.id); url.searchParams.set("offers_view", $("fifa-mode").value); if (!window.TouchlineWorkspace || window.TouchlineWorkspace.active("fifa-offers")) history.replaceState(null, "", url);
  }
  function seriesChanged(matchId) {
    const matches = report.matches.filter((m) => m.series === $("fifa-series").value);
    resetOptions("fifa-match", matches.map((m) => m.id), (id) => {
      const m = matches.find((r) => r.id === id);
      return `${m.date} · ${m.teams.map((t) => t.team).join(" — ")} · ${m.goals.join(":")}`;
    });
    $("fifa-match").value = matches.some((m) => m.id === matchId) ? matchId : matches.at(-1).id;
    render();
  }
  async function load() {
    $("fifa-retry").hidden = true; $("fifa-export").disabled = true;
    $("fifa-mode").disabled = true;
    $("fifa-status").textContent = "正在读取接应记录…";
    try {
      report = await explorationJson("fifa-offers");
      if (report.schema !== "SoccerAnalytics.FifaOffers.v1" || !report.matches?.length) throw new Error("没有通过核对的报告");
      const query = new URLSearchParams(location.search), requested = report.matches.find((m) => m.id === query.get("fifa"));
      resetOptions("fifa-series", report.coverage.map((r) => r.series), (id) => seriesNames[id]);
      if (requested) $("fifa-series").value = requested.series;
      if (["counts", "shares"].includes(query.get("offers_view"))) $("fifa-mode").value = query.get("offers_view");
      seriesChanged(requested?.id);
      $("fifa-mode").disabled = false;

    } catch (error) {
      ["fifa-chart", "fifa-summary", "fifa-table", "fifa-sources"].forEach((id) => $(id).replaceChildren());
      ["fifa-series", "fifa-match"].forEach((id) => { $(id).disabled = true; });
      $("fifa-status").textContent = `接应记录未能读取（${error.message}）。可重试或查看其他研究图表。`;
      $("fifa-retry").hidden = false;
    }
  }
  $("fifa-series").addEventListener("change", () => seriesChanged());
  ["fifa-match", "fifa-mode"].forEach((id) => $(id).addEventListener("change", render));
  $("fifa-retry").addEventListener("click", load);
  $("fifa-export").addEventListener("click", () => saveExplorationSvg("fifa-chart", `${selected.id}-offers-${$("fifa-mode").value}`, [
    $("fifa-status").textContent, ...[...$("fifa-summary").children].map((n) => n.textContent),
    "FIFA 整场报告，可能包含加时；不是每 90 分钟数据，不能据此判断能力或跑位质量。", selected.source_url,
    ...selected.teams.map((t) => `${t.team}：球员表第 ${t.player_table_page} 页；全队总数第 ${t.overview_page} 页。`),
  ]));
  await load();
}
fifaOffers();
