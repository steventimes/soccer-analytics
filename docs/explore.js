"use strict";

// Uses the research page's existing DOM and SVG helpers.
async function explorationJson(name) {
  const response = await fetch(`./data/research/${name}.json`, {cache: "no-cache"});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}
function explorationTable(target, headings, rows, contextOverrides = {}) {
  const table = node("table"), head = node("thead"), tr = node("tr"), body = node("tbody");
  headings.forEach((h) => { const th = node("th", h); th.scope = "col"; tr.append(th); });
  head.append(tr);
  rows.forEach((values) => { const row = node("tr"); values.forEach((v) => row.append(node("td", v))); body.append(row); });
  table.append(head, body);
  const button = node("button", "下载当前数值 CSV", "table-csv-export");
  button.type = "button"; button.disabled = !rows.length;
  function context() {
    const section = $(target).closest("section");
    return {
      "图表": section.querySelector("h2")?.textContent || target,
      "筛选": [...section.querySelectorAll("select:not(:disabled)")].filter((select) => !select.closest("label")?.hidden).map((select) => select.selectedOptions[0]?.textContent || "").join(" / "),
      "来源与范围": [...section.querySelectorAll("[role=status],p.context-note:not(.table-csv-note)")].filter((entry) => !entry.hidden).map((entry) => entry.textContent.trim()).filter(Boolean).join(" "),
      "来源链接": [...new Set([...section.querySelectorAll("a[href]")].filter((link) => !link.id.endsWith("-share") && !link.getAttribute("href").startsWith("#")).map((link) => link.href).filter((url) => /^https?:\/\//u.test(url)))].join(" / "),
      ...contextOverrides,
    };
  }
  function download(content, extension, type) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const link = node("a"); link.href = url; link.download = `touchline-${target}.${extension}`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  button.addEventListener("click", () => download(displayedTableCsv(headings, rows, context()), "csv", "text/csv;charset=utf-8"));
  const inspector = node("details", undefined, "table-inspector"), metadata = node("dl"), jsonButton = node("button", "下载当前视图 JSON");
  jsonButton.type = "button"; jsonButton.disabled = !rows.length;
  jsonButton.addEventListener("click", () => download(JSON.stringify(displayedTableSnapshot(headings, rows, context()), null, 2) + "\n", "json", "application/json;charset=utf-8"));
  inspector.append(node("summary", "检查数据与来源"),
    node("p", `${rows.length} 行 · ${headings.length} 列。保存表格显示值与单位，保留四舍五入，不包含原始事件或更高精度数值。`), metadata, jsonButton);
  inspector.addEventListener("toggle", () => {
    if (inspector.open) metadata.replaceChildren(...Object.entries(context()).flatMap(([label, value]) => [node("dt", label), node("dd", value || "未提供")]));
  });
  $(target).replaceChildren(button, node("p", "导出当前表格的显示值，保留当前筛选与来源说明。", "context-note table-csv-note"), inspector, table);
}
function explorationSvg(title, subtitle, height = 600) {
  const lines = wrapExplorationText(title, 68), offset = (lines.length - 1) * 24;
  const svg = svgElement("svg", {xmlns: "http://www.w3.org/2000/svg", viewBox: `0 0 760 ${height + offset}`, role: "img", "aria-label": `${title}。${subtitle}`, "font-family": "system-ui, sans-serif", "data-header-offset": offset, "data-export-height": height + offset});
  svg.append(svgElement("title", {}, title), svgElement("desc", {}, subtitle), svgElement("rect", {width: 760, height: height + offset, fill: "white"}));
  lines.forEach((line, i) => svg.append(svgElement("text", {x: 30, y: 29 + i * 24, fill: "#203344", "font-size": 18, "font-weight": 600}, line)));
  svg.append(svgElement("text", {x: 30, y: 55 + offset, fill: "#647382", "font-size": 13}, subtitle));
  if (typeof window !== "undefined" && window.TouchlineWorkspace) svg.setAttribute("viewBox", `0 ${70 + offset} 760 ${height - 70}`);
  return svg;
}
function wrapExplorationText(text, capacity = 90) {
  const lines = []; let line = "", width = 0;
  for (const char of text) {
    const size = char.codePointAt(0) > 255 ? 2 : 1;
    if (width + size > capacity) { lines.push(line); line = ""; width = 0; }
    line += char; width += size;
  }
  if (line) lines.push(line);
  return lines;
}
function saveExplorationSvg(container, filename, notes = []) {
  const original = $(container).querySelector("svg");
  if (!original) return;
  const svg = original.cloneNode(true), height = Number(original.dataset.exportHeight) || svg.viewBox.baseVal.height;
  svg.classList.remove("export-only");
  const lines = notes.flatMap((line) => wrapExplorationText(line));
  svg.setAttribute("viewBox", `0 0 760 ${height + 30 + lines.length * 22}`);
  svg.querySelector("rect").setAttribute("height", height + 30 + lines.length * 22);
  lines.forEach((line, i) => svg.append(svgElement("text", {x: 30, y: height + 15 + i * 22, fill: "#647382", "font-size": 13}, line)));
  const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg)], {type: "image/svg+xml;charset=utf-8"}));
  const link = node("a"); link.href = url; link.download = `${filename}.svg`; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function pairedMetricView(metrics) {
  const list = node("div", undefined, "paired-metrics");
  for (const {title, ceiling, unit, values} of metrics) {
    const item = node("div", undefined, "paired-metric"), heading = node("div", undefined, "paired-metric-heading");
    heading.append(node("strong", title), node("span", `0–${number(ceiling, 1)} ${unit}`)); item.append(heading);
    values.forEach(({value, label}, index) => {
      const row = node("div", undefined, `paired-value${index ? " is-b" : ""}`), track = node("span", undefined, "paired-track"), bar = node("i");
      const valid = typeof value === "number" && Number.isFinite(value) && value >= 0;
      bar.style.width = `${valid ? Math.min(100, value / ceiling * 100) : 0}%`; track.setAttribute("aria-hidden", "true"); track.append(bar);
      row.append(node("span", index ? "B ◇" : "A ●"), track, node("span", label)); item.append(row);
    });
    list.append(item);
  }
  return list;
}
function resetOptions(id, values, label) {
  $(id).replaceChildren(); options(id, values, label); $(id).disabled = !values.length;
}
function matchPitch(svg) {
  const offset = Number(svg.dataset.headerOffset);
  const scale = 5.6, x = (v) => 40 + v * scale, y = (v) => 82 + offset + v * scale;
  const line = {fill: "none", stroke: "#98a5b1", "stroke-width": 1.2};
  svg.append(svgElement("rect", {x: x(0), y: y(0), width: 672, height: 448, ...line}),
    svgElement("line", {x1: x(60), x2: x(60), y1: y(0), y2: y(80), ...line}),
    svgElement("circle", {cx: x(60), cy: y(40), r: 10 * scale, ...line}));
  [0, 102].forEach((start) => svg.append(svgElement("rect", {x: x(start), y: y(18), width: 18 * scale, height: 44 * scale, ...line})));
  [0, 114].forEach((start) => svg.append(svgElement("rect", {x: x(start), y: y(30), width: 6 * scale, height: 20 * scale, ...line})));
  svg.append(svgElement("text", {x: 40, y: 555 + offset, fill: "#647382", "font-size": 13}, "StatsBomb 120×80 单位 · 进攻方向 →"));
  return {x, y};
}
function drawPassing(svg, graph, names, view, focus, minimum, selectPlayer) {
  const byId = new Map(graph.players.map((p) => [p.id, p]));
  const label = (id) => names[id] || `球员 ${id}`;
  const links = graph.edges.filter((e) => e.count >= minimum && (!focus || e.source === focus || e.target === focus));
  const shown = links.reduce((n, e) => n + e.count, 0);
  const filter = `${focus ? label(focus) + " 的传出与接入" : "全队联系"} · 每条至少 ${minimum} 次`;
  $("match-filter-summary").textContent = `${filter} · 保留 ${links.length} 条有向联系，${shown} / ${graph.total} 次已识别传球；另有 ${graph.unlinked} 次完成传球缺少队员标识。`;
  if (!links.length) $("match-filter-summary").textContent += "没有符合条件的联系，请选择全队联系或降低次数门槛。";
  const offset = Number(svg.dataset.headerOffset);
  let hidden = 0;
  if (view === "matrix") {
    svg.classList.add("passing-matrix");
    const step = Math.min(34, 540 / Math.max(1, graph.players.length)), left = 100, top = 122 + offset;
    const max = Math.max(1, ...graph.edges.map((e) => e.count));
    const counts = new Map(links.map((e) => [`${e.source}:${e.target}`, e.count]));
    const recorded = new Map(graph.edges.map((e) => [`${e.source}:${e.target}`, e.count]));
    svg.append(svgElement("text", {x: left, y: top - 44, fill: "#203344", "font-size": 14}, "接球队员 →（列）"),
      svgElement("text", {x: 28, y: top - 14, fill: "#203344", "font-size": 13}, "传球队员 ↓"));
    graph.players.forEach((p, row) => {
      const color = p.id === focus ? "#2468a0" : "#203344";
      svg.append(svgElement("text", {x: left - 14, y: top + (row + .5) * step + 4, "text-anchor": "end", fill: color, "font-size": 13}, p.number),
        svgElement("text", {x: left + (row + .5) * step, y: top - 12, "text-anchor": "middle", fill: color, "font-size": 13}, p.number));
      graph.players.forEach((q, col) => {
        const key = `${p.id}:${q.id}`, count = counts.get(key) || 0;
        const omitted = !counts.has(key) && recorded.has(key);
        const fill = count ? `rgb(${Math.round(238 - 202 * count / max)},${Math.round(245 - 141 * count / max)},${Math.round(250 - 90 * count / max)})` : "#f3f6f9";
        const cell = svgElement("rect", {x: left + col * step, y: top + row * step, width: step, height: step, fill, stroke: "white", "stroke-width": 1});
        cell.append(svgElement("title", {}, `${label(p.id)} → ${label(q.id)}：${omitted ? "已筛除，原记录 " + recorded.get(key) : count} 次`));
        svg.append(cell, svgElement("text", {x: left + (col + .5) * step, y: top + (row + .5) * step + 4, "text-anchor": "middle", fill: count / max > .87 ? "white" : "black", "font-size": 12}, omitted ? "·" : count));
      });
    });
    svg.append(svgElement("text", {x: 40, y: 694 + offset, fill: "#647382", "font-size": 13}, `深色为次数更多，色阶 0–${max} 次（所选时段全队）· “·” 表示已筛除`));
  } else {
    const {x, y} = matchPitch(svg);
    const defs = svgElement("defs", {}), marker = svgElement("marker", {id: "network-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto"});
    marker.append(svgElement("path", {d: "M0 0L10 5L0 10z", fill: "#647382"})); defs.append(marker); svg.append(defs);
    links.forEach((e) => {
      const a = byId.get(e.source), b = byId.get(e.target);
      if (!a.position || !b.position || a.id === b.id) { hidden += e.count; return; }
      let ax = x(a.position[0]), ay = y(a.position[1]), bx = x(b.position[0]), by = y(b.position[1]);
      const dx = bx - ax, dy = by - ay, length = Math.hypot(dx, dy);
      if (length < 28) { hidden += e.count; return; }
      ax += dx / length * 14; ay += dy / length * 14; bx -= dx / length * 15; by -= dy / length * 15;
      const path = svgElement("path", {d: `M${ax} ${ay} Q${(ax + bx) / 2 - dy / length * 9} ${(ay + by) / 2 + dx / length * 9} ${bx} ${by}`, fill: "none", stroke: "#647382", "stroke-opacity": .65, "stroke-width": Math.min(5, 1 + Math.sqrt(e.count) / 2), "marker-end": "url(#network-arrow)"});
      path.append(svgElement("title", {}, `${label(a.id)} → ${label(b.id)}：${e.count} 次`)); svg.append(path);
    });
    const involved = new Set(links.flatMap((e) => [e.source, e.target]));
    graph.players.forEach((p) => {
      if (!p.position) return;
      const cx = x(p.position[0]), cy = y(p.position[1]);
      svg.append(svgElement("circle", {cx, cy, r: 13, fill: involved.has(p.id) ? "#2468a0" : "#647382", "fill-opacity": involved.has(p.id) ? 1 : .3, stroke: p.id === focus ? "#203344" : "white", "stroke-width": p.id === focus ? 3 : 1.5}),
        svgElement("text", {x: cx, y: cy + 4, "text-anchor": "middle", fill: "white", "font-size": 12}, p.number));
    });
    svg.append(svgElement("text", {x: 40, y: 580 + offset, fill: "#647382", "font-size": 13}, `箭头为传球方向，线宽随次数增加 · ${hidden} 次联系请切换矩阵或查看数值表`));
  }
  const list = node("ol", undefined, "network-roster");
  graph.players.forEach((p) => {
    const item = node("li"), button = node("button", label(p.id));
    button.type = "button"; button.setAttribute("aria-pressed", String(focus === p.id));
    button.addEventListener("click", () => selectPlayer(focus === p.id ? "" : p.id));
    item.append(button, node("span", `传出 ${p.outgoing} · 接入 ${p.incoming}${p.position ? "" : " · 无起点坐标"}`)); list.append(item);
  });
  $("match-roster").append(node("h4", "球员编号 · 点击聚焦"), list);
  explorationTable("match-table", ["传球队员", "接球队员", "完成传球次数", "占该球员已识别传出"], links.map((e) => [label(e.source), label(e.target), e.count, `${number(e.count / byId.get(e.source).outgoing * 100, 1)}%`]));
  if (!links.length) $("match-table").append(node("p", "没有符合条件的联系。请选择全队联系或降低次数门槛。"));
  svg.append(svgElement("text", {x: 40, y: (view === "matrix" ? 722 : 608) + offset, fill: "#647382", "font-size": 13}, filter));
}
async function matchExplorer() {
  if (!$("match-select")) return;
  let catalog, data, selected, request = 0;
  const cache = new Map();
  const controls = ["match-team", "match-view", "match-period", "match-start", "match-end", "match-player", "match-minimum", "match-phase", "match-action", "match-motif", "match-window"];
  resetOptions("match-start", [0, 15, 30, 45, 60, 75, 90, 105], String);
  resetOptions("match-end", [15, 30, 45, 60, 75, 90, 105, 120], String);
  $("match-end").value = "90";
  function updateUrl() {
    if (typeof window !== "undefined" && window.TouchlineWorkspace && !window.TouchlineWorkspace.active("match-explorer")) return;
    const url = new URL(location.href), view = $("match-view").value, period = $("match-period").value;
    url.searchParams.set("match", selected.id); url.searchParams.set("team", $("match-team").value);
    url.searchParams.set("view", view); url.searchParams.set("period", period);
    for (const [key, id] of [["start", "match-start"], ["end", "match-end"]]) {
      if (period === "custom") url.searchParams.set(key, $(id).value); else url.searchParams.delete(key);
    }
    for (const [key, id] of [["minimum", "match-minimum"], ["player", "match-player"]]) {
      if ((view === "network" || view === "matrix") && !$(id).disabled && $(id).value) url.searchParams.set(key, $(id).value);
      else url.searchParams.delete(key);
    }
    for (const [key, id] of [["phase", "match-phase"], ["action", "match-action"]]) {
      if ((view === "sequence" || view === "motifs" && id === "match-action") && !$(id).disabled && $(id).value) url.searchParams.set(key, $(id).value);
      else url.searchParams.delete(key);
    }
    for (const [key, id] of [["motif", "match-motif"], ["window", "match-window"]]) {
      if (view === "motifs" && !$(id).disabled && $(id).value) url.searchParams.set(key, $(id).value);
      else url.searchParams.delete(key);
    }
    history.replaceState(null, "", url);
  }
  function clearMotifWindow() {
    resetOptions("match-window", [], String);
    if ($("match-view").value === "motifs") {
      resetOptions("match-action", [], String);
      $("match-action-label").hidden = true; $("match-step-navigation").hidden = true;
    }
  }
  function render() {
    if (!data) return;
    if ($("match-heading")) $("match-heading").textContent = `${selected.teams[0]} ${selected.score.join(" : ")} ${selected.teams[1]}`;
    const period = $("match-period").value, custom = period === "custom";
    if ($("match-options-summary")) $("match-options-summary").textContent = $("match-period").selectedOptions[0].textContent;
    $("match-start").disabled = !custom; $("match-end").disabled = !custom;
    const max = Math.max(...data.segments.map((r) => r.segment), 5) * 15 + 15;
    const range = {all: [0, max], first: [0, 45], second: [45, 90], extra1: [90, 105], extra2: [105, 120], custom: [Number($("match-start").value), Number($("match-end").value)]}[period];
    const team = $("match-team").value, view = $("match-view").value;
    const sequence = view === "sequence", motifs = view === "motifs", passing = view === "network" || view === "matrix";
    ["match-player-label", "match-minimum-label", "match-filter-summary"].forEach((id) => { $(id).hidden = !passing; });
    ["match-phase-label", "match-action-label", "match-step-navigation", "match-sequence-reading"].forEach((id) => { $(id).hidden = !sequence; });
    ["match-motif-label", "match-window-label", "match-motif-reading"].forEach((id) => { $(id).hidden = !motifs; });
    $("match-overview-reading").hidden = sequence || motifs;
    $("match-phase").disabled = true; $("match-action").disabled = true;
    $("match-previous-action").disabled = true; $("match-next-action").disabled = true;
    $("match-filter-summary").textContent = "";
    $("match-chart").replaceChildren(); $("match-roster").replaceChildren(); $("match-table").replaceChildren();
    $("match-export").disabled = true;
    if (range[0] >= range[1]) {
      clearMotifWindow();
      $("match-status").textContent = "结束分钟必须晚于起始分钟。请调整自选时段。";
      $("match-summary").replaceChildren(); updateUrl(); return;
    }
    const rows = data.segments.filter((r) => r.team === team && r.segment * 15 >= range[0] && r.segment * 15 < range[1]);
    const sum = (key) => rows.reduce((total, r) => total + r[key], 0);
    const scope = `${selected.date} · ${team} · ${range[0]}–${range[1]} 分钟（含对应半场补时）`;
    const subtitle = "StatsBomb 开放数据 · " + (motifs ? "连续三传球 · 可重叠窗口次数，不是战术效果" : sequence ? "控球段内动作顺序 · 逐步查看，不填补动作间隔" : view === "shots" ? "10×10 分区汇总，圆心不是精确位置" : view === "matrix" ? "行传给列 · 次数，不是传球概率 · 0 为未记录" : "有向完成传球，节点为传球起点均值");
    $("match-status").textContent = `${selected.competition} · ${scope} · ${rows.length ? "所选时段已更新" : "所选时段没有记录"}`;
    $("match-summary").replaceChildren(node("h3", team), node("p", `${sum("shots")} 次射门 · ${number(sum("xg"), 2)} xG · ${sum("goals")} 个射门进球`), node("p", `${sum("completed_passes")} / ${sum("passes")} 次传球完成`));
    if (!rows.length) { clearMotifWindow(); $("match-chart").append(node("p", "这场比赛在所选时段没有记录，请选择其他时段。")); updateUrl(); return; }
    let svg = explorationSvg(scope, subtitle, view === "matrix" ? 750 : 636);
    if (motifs) {
      if (!renderPassingMotifs(svg, data.passing_motifs, team, range, rows.map((r) => r.segment))) { updateUrl(); return; }
    } else if (sequence) {
      const phases = shotSequencePhases(data.shot_sequences?.phases || [], team, range);
      const previous = $("match-phase").value;
      resetOptions("match-phase", phases.map((p) => p.id), (id) => {
        const p = phases.find((p) => p.id === id);
        return `${sequencePeriods[p.period]} ${sequenceClock(p.actions[0])}–${sequenceClock(p.actions.at(-1))} · ${p.shots} 次射门 · ${p.goals} 个进球 · ${number(p.xg, 2)} xG`;
      });
      if (phases.some((p) => p.id === previous)) $("match-phase").value = previous;
      if (!phases.length) {
        resetOptions("match-action", [], String);
        $("match-summary").replaceChildren();
        $("match-status").textContent += data.shot_sequences ? " · 本队在该时段没有可显示的射门前控球段" : " · 当前快照没有控球段摘要，请重新生成探索数据";
        $("match-chart").append(node("p", "请选择其他球队或时段，或切换射门分区查看汇总。")); updateUrl(); return;
      }
      const phase = phases.find((p) => p.id === $("match-phase").value);
      const token = `${selected.id}:${team}:${phase.id}`, oldAction = $("match-action").value;
      const samePhase = $("match-action").dataset.phase === token;
      resetOptions("match-action", phase.actions.map((_, i) => String(i)), (i) => `${Number(i) + 1}. ${sequenceClock(phase.actions[i])} · ${sequenceKinds[phase.actions[i].kind]} · ${phase.actions[i].player}`);
      $("match-action").dataset.phase = token;
      $("match-action").value = samePhase && [...$("match-action").options].some((option) => option.value === oldAction) ? oldAction : String(phase.actions.length - 1);
      const step = Number($("match-action").value);
      $("match-previous-action").disabled = step === 0;
      $("match-next-action").disabled = step === phase.actions.length - 1;
      $("match-status").textContent += ` · ${phases.length} 段含射门的控球过程 · 当前 ${sequenceClock(phase.actions[0])}–${sequenceClock(phase.actions.at(-1))} · 未能归段的射门 ${data.shot_sequences.unassigned_shots[team]} 次（整场）`;
      svg = explorationSvg(`${scope} · ${sequencePeriods[phase.period]} ${sequenceClock(phase.actions[0])}–${sequenceClock(phase.actions.at(-1))}`, subtitle, 636);
      svg.querySelector("title").textContent += ` · 控球段 ${phase.id} · 动作 ${step + 1}`;
      drawShotSequence(svg, phase, step);
    } else if (view === "shots") {
      const {x, y} = matchPitch(svg);
      const cells = new Map();
      rows.flatMap((r) => r.cells).forEach((c) => {
        const a = cells.get(c.cell) || {cell: c.cell, shots: 0, goals: 0, xg: 0};
        a.shots += c.shots; a.goals += c.goals; a.xg += c.xg; cells.set(c.cell, a);
      });
      cells.forEach((c) => {
        const cx = x((c.cell % 12) * 10 + 5), cy = y(Math.floor(c.cell / 12) * 10 + 5);
        svg.append(svgElement("circle", {cx, cy, r: 7 * Math.sqrt(c.shots), fill: "#2468a0", "fill-opacity": .16, stroke: "#2468a0", "stroke-width": 1.4}));
        svg.append(svgElement("text", {x: cx, y: cy + 4, "text-anchor": "middle", fill: "#203344", "font-size": 13}, c.shots));
      });
      svg.append(svgElement("text", {x: 40, y: 580 + Number(svg.dataset.headerOffset), fill: "#647382", "font-size": 13}, `圆内为射门次数，圆面积与次数成正比 · 缺坐标 ${sum("unlocated_shots")} 次`));
      explorationTable("match-table", ["射门分区 x / y", "射门", "进球", "xG"], [...cells.values()].sort((a, b) => a.cell - b.cell).map((c) => [`${c.cell % 12 * 10}–${c.cell % 12 * 10 + 10} / ${Math.floor(c.cell / 12) * 10}–${Math.floor(c.cell / 12) * 10 + 10}`, c.shots, c.goals, number(c.xg, 3)]));
    } else {
      const graph = passingGraph(rows), previous = $("match-player").value;
      resetOptions("match-player", ["", ...graph.players.map((p) => p.id)], (id) => id ? `${graph.players.find((p) => p.id === id).number}. ${data.players[id] || id}` : "全队联系");
      if (graph.players.some((p) => p.id === previous)) $("match-player").value = previous;
      drawPassing(svg, graph, data.players, view, $("match-player").value, Number($("match-minimum").value), (id) => {
        $("match-player").value = id; render();
        $("match-player").focus();
      });
    }
    $("match-chart").append(svg); $("match-export").disabled = false;
    updateUrl();
  }
  function restoreMatchState(query) {
    if (data?.teams.includes(query.get("team"))) { $("match-team").value = query.get("team"); render(); }
    if ([...$("match-player").options].some((o) => o.value === query.get("player"))) { $("match-player").value = query.get("player"); render(); }
    for (const [id, key] of [["match-phase", "phase"], ["match-window", "window"], ["match-action", "action"]]) {
      if ([...$(id).options].some((o) => o.value === query.get(key))) { $(id).value = query.get(key); render(); }
    }
  }
  async function select(restoreQuery = null) {
    const current = ++request;
    if ($("match-retry")) $("match-retry").hidden = true;
    selected = catalog.matches.find((m) => m.id === $("match-select").value);
    data = null; $("match-filter-summary").textContent = ""; $("match-player").disabled = true; $("match-chart").replaceChildren(); $("match-table").replaceChildren(); $("match-summary").replaceChildren(); $("match-roster").replaceChildren(); $("match-export").disabled = true;
    for (const id of ["match-phase", "match-action", "match-window"]) resetOptions(id, [], String);
    $("match-motif").disabled = true;
    $("match-previous-action").disabled = true; $("match-next-action").disabled = true;
    resetOptions("match-team", selected.teams, String);
    $("match-status").textContent = "正在读取所选比赛…";
    try {
      const result = cache.get(selected.id) || await explorationJson(`matches/${selected.id}`);
      if (current !== request) return;
      if (result.id !== selected.id || !Array.isArray(result.segments)) throw new Error("比赛摘要不匹配");
      data = result; render();
      if (restoreQuery instanceof URLSearchParams) restoreMatchState(restoreQuery);
      cache.set(selected.id, data);
      if ($("match-catalog")) for (const button of $("match-catalog").querySelectorAll("button")) button.setAttribute("aria-pressed", String(button.dataset.match === selected.id));
    } catch (error) { if (current === request) { cache.delete(selected.id); data = null; $("match-chart").replaceChildren(); $("match-table").replaceChildren(); $("match-export").disabled = true; $("match-status").textContent = `比赛未能加载（${error.message}）。请重试或选择其他比赛。`; if ($("match-retry")) $("match-retry").hidden = false; } }
  }
  if ($("match-retry")) $("match-retry").onclick = () => catalog ? select(new URLSearchParams(location.search)) : matchExplorer();
  try {
    catalog = await explorationJson("matches");
    if (!catalog.matches?.length) throw new Error("没有可探索的比赛");
    resetOptions("match-select", catalog.matches.map((m) => m.id), (id) => { const m = catalog.matches.find((v) => v.id === id); return `${m.date} · ${m.teams.join(" — ")} · ${m.score.join(":")}`; });
    if ($("match-search")) {
      function catalogView() {
        const query = $("match-search").value.trim().toLocaleLowerCase();
        const matches = catalog.matches.filter((match) => `${match.teams.join(" ")} ${match.date}`.toLocaleLowerCase().includes(query));
        $("match-catalog-count").textContent = `${catalog.matches.length} 场公开样本`;
        $("match-search-status").textContent = matches.length ? `找到 ${matches.length} 场` : "没有匹配比赛，请换一个球队名或日期。";
        $("match-catalog").replaceChildren(...matches.map((match) => {
          const button = node("button"); button.type = "button"; button.dataset.match = match.id;
          button.setAttribute("aria-pressed", String(match.id === $("match-select").value));
          button.append(node("small", match.date), node("span", match.score.join(" : ")), node("strong", match.teams.join(" — ")));
          button.addEventListener("click", async () => { $("match-select").value = match.id; await select(); $("match-select").focus(); });
          return button;
        }));
      }
      $("match-search").addEventListener("input", catalogView); catalogView();
    }
    const query = new URLSearchParams(location.search);
    if (catalog.matches.some((m) => m.id === query.get("match"))) $("match-select").value = query.get("match");
    for (const [id, key] of [["match-view", "view"], ["match-period", "period"], ["match-start", "start"], ["match-end", "end"], ["match-minimum", "minimum"], ["match-motif", "motif"]]) if ([...$(id).options].some((o) => o.value === query.get(key))) $(id).value = query.get(key);
    await select(query);
    $("match-select").addEventListener("change", select);
    controls.forEach((id) => $(id).addEventListener("change", render));
    for (const [id, direction] of [["match-previous-action", -1], ["match-next-action", 1]]) $(id).addEventListener("click", () => {
      const next = Number($("match-action").value) + direction;
      if (next >= 0 && next < $("match-action").options.length) { $("match-action").value = String(next); render(); }
    });
    $("match-export").addEventListener("click", () => saveExplorationSvg("match-chart", `${selected.id}-${$("match-team").value}-${$("match-view").value}`, [
      `${selected.id} · ${selected.teams.join(" — ")} · ${selected.score.join(":")}`,
      $("match-status").textContent, $("match-summary").textContent, $("match-filter-summary").textContent,
      ["sequence", "motifs"].includes($("match-view").value) ? "StatsBomb 事件原始 120×80 起终点，均向右进攻，不补画动作间隔。" : catalog.scope,
      $("match-view").value === "motifs" ? $("match-motif-reading").textContent : $("match-view").value === "sequence" ? $("match-sequence-reading").textContent : $("match-view").value === "shots" ? "圆心为分区中心，不是精确射门点。" : "球员编号不是球衣号码；0 为本时段未记录，· 为已筛除。矩阵按行传给列，节点为传球起点均值。",
      ...[...$("match-roster").querySelectorAll("li")].map((e, i) => `${i + 1}. ${e.textContent}`),
    ]));
  } catch (error) { $("match-status").textContent = `比赛目录未就绪（${error.message}）。`; if ($("match-retry")) $("match-retry").hidden = false; }
}

const comparisonMetrics = {
  teams: [["points_per_match", "场均积分", "分/场", 3], ["goals_for_per_match", "场均进球", "球/场"], ["goals_against_per_match", "场均失球", "球/场"], ["win_rate", "胜率", "%", 1], ["clean_sheet_rate", "零封率", "%", 1]],
  players: [["passes_per90", "传球", "次/90分钟"], ["progressive_passes_per90", "推进传球", "次/90分钟"], ["shots_per90", "射门", "次/90分钟"], ["goals_per90", "进球", "球/90分钟"], ["defensive_duels_per90", "防守对抗", "次/90分钟"], ["interceptions_per90", "拦截", "次/90分钟"], ["pass_completion", "传球成功率", "%", 1], ["duel_win_rate", "对抗成功率", "%", 1]],
};
async function comparisonExplorer() {
  if (!$("compare-kind")) return;
  try {
    const catalog = await explorationJson("comparisons");
    if ($("compare-retry")) $("compare-retry").hidden = true;
    const query = new URLSearchParams(location.search);
    if (["teams", "players"].includes(query.get("kind"))) $("compare-kind").value = query.get("kind");
    const leagueLabels = {PL: "英超", PD: "西甲", BL1: "德甲", SA: "意甲", FL1: "法甲", England: "英超", Spain: "西甲", Germany: "德甲", Italy: "意甲", France: "法甲"};
    const positions = {Goalkeeper: "门将", Defender: "后卫", Midfielder: "中场", Forward: "前锋"};
    $("compare-position").replaceChildren(node("option", "全部位置")); $("compare-position").firstChild.value = "";
    options("compare-position", Object.keys(positions), (v) => positions[v]);
    function filter() {
      const kind = $("compare-kind").value;
      return catalog[kind].filter((r) => r.league === $("compare-league").value && r.season === $("compare-season").value && (kind === "teams" || ((!$("compare-position").value || r.position === $("compare-position").value) && r.minutes >= Number($("compare-minutes").value))));
    }
    function draw() {
      const rows = filter(), kind = $("compare-kind").value;
      const url = new URL(location.href);
      for (const [key,id] of [["kind","compare-kind"],["league","compare-league"],["season","compare-season"],["position","compare-position"],["minutes","compare-minutes"],["a","compare-a"],["b","compare-b"],["search","compare-search"]]) if ($(id)) url.searchParams.set(key, $(id).value);
      if (typeof window === "undefined" || !window.TouchlineWorkspace || window.TouchlineWorkspace.active("comparisons")) history.replaceState(null, "", url);
      const a = rows.find((r) => r.id === $("compare-a").value), b = rows.find((r) => r.id === $("compare-b").value);
      $("compare-chart").replaceChildren(); $("compare-samples").replaceChildren(); $("compare-table").replaceChildren(); $("compare-export").disabled = true;
      if ($("compare-options-summary")) $("compare-options-summary").textContent = `${leagueLabels[$("compare-league").value] || $("compare-league").value} · ${$("compare-season").selectedOptions[0]?.textContent || "无赛季"}`;
      $("compare-status").textContent = `${kind === "teams" ? catalog.team_source : catalog.player_source} · 当前筛选 ${rows.length} 个对象`;
      if (!a || !b || a.id === b.id) { $("compare-chart").append(node("p", "请选择两个不同对象；没有足够对象时，可降低分钟门槛或更换筛选。")); return; }
      [a, b].forEach((r, i) => { const card = node("div"); card.append(node("strong", `${i === 0 ? "A ●" : "B ◇"} ${r.name}`), node("p", `${r.matches} 场${kind === "players" ? ` · ${number(r.minutes, 0)} 分钟 · ${positions[r.position]}` : ` · ${r.date_start} 至 ${r.date_end}`}`)); $("compare-samples").append(card); });
      const metrics = comparisonMetrics[kind], svg = explorationSvg(`${a.name} 与 ${b.name}`, `${leagueLabels[a.league]} · ${a.season} · A 实心圆 / B 空心菱形 · 每项坐标从零开始`, 120 + metrics.length * 65);
      const offset = Number(svg.dataset.headerOffset);
      const values = [], displayMetrics = [];
      metrics.forEach(([key, title, unit, fixedMax], index) => {
        const av = a.metrics[key], bv = b.metrics[key], y = 104 + index * 65 + offset;
        const valid = (v) => typeof v === "number" && Number.isFinite(v) && v >= 0;
        const ceiling = fixedMax || Math.max(...rows.map((r) => r.metrics[key]).filter(valid), .01) * 1.05;
        const x = (v) => 235 + v / ceiling * 330;
        const format = (v) => valid(v) ? `${number(unit === "%" ? v * 100 : v, key === "goals_per90" ? 2 : 1)} ${unit}` : "暂无";
        svg.append(svgElement("text", {x: 30, y: y + 4, fill: "#203344", "font-size": 14}, title));
        svg.append(svgElement("line", {x1: x(0), x2: x(ceiling), y1: y, y2: y, stroke: "#e2e8ee", "stroke-width": 2}));
        [0, ceiling].forEach((v) => svg.append(svgElement("text", {x: x(v), y: y + 23, fill: "#647382", "font-size": 12, "text-anchor": "middle"}, number(unit === "%" ? v * 100 : v, 1))));
        if (valid(av)) svg.append(svgElement("circle", {cx: x(av), cy: y, r: 5, fill: "#2468a0"}));
        if (valid(bv)) svg.append(svgElement("path", {d: `M${x(bv)} ${y - 6}l6 6-6 6-6-6z`, fill: "white", stroke: "#c5722c", "stroke-width": 2}));
        svg.append(svgElement("text", {x: 595, y: y - 6, fill: "#2468a0", "font-size": 13}, `A ${format(av)}`), svgElement("text", {x: 595, y: y + 14, fill: "#a34f15", "font-size": 13}, `B ${format(bv)}`));
        displayMetrics.push({title, ceiling: ceiling * (unit === "%" ? 100 : 1), unit, values: [{value: valid(av) ? av * (unit === "%" ? 100 : 1) : null, label: format(av)}, {value: valid(bv) ? bv * (unit === "%" ? 100 : 1) : null, label: format(bv)}]});
        values.push([title, format(av), format(bv), valid(av) && valid(bv) ? `${number((av - bv) * (unit === "%" ? 100 : 1), 2)} ${unit === "%" ? "个百分点" : unit}` : "暂无"]);
      });
      svg.classList.add("export-only"); $("compare-chart").append(pairedMetricView(displayMetrics), svg); $("compare-export").disabled = false;
      explorationTable("compare-table", ["指标", a.name, b.name, "A − B"], values);
    }
    function subjects() {
      const rows = filter().sort((a, b) => a.name.localeCompare(b.name));
      const oldA = $("compare-a").value, oldB = $("compare-b").value;
      for (const id of ["compare-a", "compare-b"]) resetOptions(id, rows.map((r) => r.id), (v) => rows.find((r) => r.id === v).name);
      if (rows.some((r) => r.id === oldA)) $("compare-a").value = oldA;
      $("compare-b").value = rows.some((r) => r.id === oldB && r.id !== $("compare-a").value) ? oldB : (rows.find((r) => r.id !== $("compare-a").value)?.id || "");
      searchSubjects(); draw();
    }
    function searchSubjects() {
      if (!$("compare-search-results")) return;
      const query = $("compare-search").value.trim().toLocaleLowerCase(), matches = filter().filter((row) => row.name.toLocaleLowerCase().includes(query));
      $("compare-search-results").replaceChildren();
      if (!query) return;
      $("compare-search-results").append(node("p", `找到 ${matches.length} 个对象；选择加入 A 或 B。`, "context-note"));
      for (const row of matches.slice(0,30)) {
        const item = node("div", undefined, "subject-search-result"); item.append(node("span", row.name));
        for (const key of ["a","b"]) { const button = node("button", `设为 ${key.toUpperCase()}`); button.type="button"; button.setAttribute("aria-label", `${row.name} 设为 ${key.toUpperCase()}`); button.addEventListener("click",()=>{ $("compare-"+key).value=row.id; draw(); }); item.append(button); }
        $("compare-search-results").append(item);
      }
      if (matches.length > 30) $("compare-search-results").append(node("p","这里只显示前 30 项，请增加关键词。", "context-note"));
    }
    function seasons() {
      const rows = catalog[$("compare-kind").value].filter((r) => r.league === $("compare-league").value);
      resetOptions("compare-season", [...new Set(rows.map((r) => r.season))].sort().reverse(), (s) => s.length === 4 ? `${s}/${Number(s) + 1}` : s);
      subjects();
    }
    function kind() {
      const players = $("compare-kind").value === "players";
      $("compare-position-label").hidden = !players; $("compare-minutes-label").hidden = !players;
      resetOptions("compare-league", [...new Set(catalog[players ? "players" : "teams"].map((r) => r.league))].sort(), (v) => leagueLabels[v] || v); seasons();
    }
    $("compare-kind").addEventListener("change", kind); $("compare-league").addEventListener("change", seasons);
    ["compare-season", "compare-position", "compare-minutes"].forEach((id) => $(id).addEventListener("change", subjects));
    ["compare-a", "compare-b"].forEach((id) => $(id).addEventListener("change", draw));
    $("compare-export").addEventListener("click", () => saveExplorationSvg("compare-chart", "touchline-comparison", [
      $("compare-status").textContent, ...[...$("compare-samples").children].map((e) => e.textContent),
      "比例坐标 0–100%，场均积分 0–3；其他坐标上限取当前筛选样本最大值并留白。未校准对手强度和球队风格。",
    ]));
    if ($("compare-search")) { $("compare-search").value = query.get("search") || ""; $("compare-search").addEventListener("input", searchSubjects); }
    kind();
    function restore(id,key,change) { if ([...$(id).options].some((option) => option.value === query.get(key))) { $(id).value = query.get(key); change(); } }
    restore("compare-league","league",seasons); restore("compare-season","season",subjects);
    restore("compare-position","position",subjects); restore("compare-minutes","minutes",subjects);
    restore("compare-a","a",draw); restore("compare-b","b",draw);
    if ($("compare-retry")) $("compare-retry").onclick = comparisonExplorer;
  } catch (error) { if ($("compare-retry")) { $("compare-retry").hidden = false; $("compare-retry").onclick = comparisonExplorer; } $("compare-status").textContent = `比较数据未就绪（${error.message}）。请重新生成探索数据。`; }
}
matchExplorer();
comparisonExplorer();
