"use strict";

const passingMotifTypes = ["ABAB", "ABAC", "ABCA", "ABCB", "ABCD"];
function passingMotifWindows(summary, team, range, pattern = "") {
  if (summary?.schema !== "Touchline.PassingMotifs.v1") return [];
  return summary.chains.filter((c) => c.team === team).flatMap((chain) => chain.motifs.flatMap((motif, i) => {
    const segment = chain.passes[i + 2].segment;
    return segment * 15 >= range[0] && segment * 15 < range[1] && (!pattern || pattern === motif)
      ? [{id: `${chain.id}:${i}`, period: chain.period, pattern: motif, segment, passes: chain.passes.slice(i, i + 3)}] : [];
  }));
}
function motifClock(pass) { return `${sequenceClock(pass)}.${String(pass.clock_ms % 1000).padStart(3, "0")}`; }

function drawMotifWindow(svg, window, players, selected) {
  const {x, y} = matchPitch(svg);
  const defs = svgElement("defs", {}), marker = svgElement("marker", {id: "motif-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse"});
  marker.append(svgElement("path", {d: "M0 0L10 5L0 10z", fill: "context-stroke"})); defs.append(marker); svg.append(defs);
  const order = [0, 1, 2].filter((i) => i !== selected).concat(selected);
  for (const i of order) {
    const p = window.passes[i], active = i === selected;
    if (!p.start) continue;
    const g = svgElement("g", {opacity: active ? 1 : .35});
    g.append(svgElement("title", {}, `${i + 1}. ${motifClock(p)} ${players[p.player]} → ${players[p.recipient]}`));
    if (p.end) g.append(svgElement("line", {x1: x(p.start[0]), y1: y(p.start[1]), x2: x(p.end[0]), y2: y(p.end[1]), stroke: active ? "#2468a0" : "#647382", "stroke-width": active ? 3 : 1.4, "marker-end": "url(#motif-arrow)"}));
    g.append(svgElement("circle", {cx: x(p.start[0]), cy: y(p.start[1]), r: active ? 12 : 4, fill: active ? "#2468a0" : "#647382"}));
    if (active) g.append(svgElement("text", {x: x(p.start[0]), y: y(p.start[1]) + 4, "text-anchor": "middle", fill: "white", "font-size": 12}, i + 1));
    svg.append(g);
  }
  const missing = window.passes.filter((p) => !p.start || !p.end).length;
  svg.append(svgElement("text", {x: 40, y: 580 + Number(svg.dataset.headerOffset), fill: "#647382", "font-size": 13}, `${window.pattern} · 第 ${selected + 1} / 3 次传球 · ${missing} 次缺起点或终点 · 不连接动作间的空白`));
  const location = (p) => p ? p.map((v) => number(v, 1)).join(" / ") : "缺失";
  explorationTable("match-table", ["步骤", "发起时间", "传球队员", "接球队员", "距上次发起", "起点 x / y", "终点 x / y"], window.passes.map((p, i) => [i + 1, motifClock(p), players[p.player], players[p.recipient], i ? `${number((p.clock_ms - window.passes[i - 1].clock_ms) / 1000, 3)} 秒` : "—", location(p.start), location(p.end)]));
  $("match-table").querySelectorAll("tbody tr")[selected].setAttribute("aria-current", "step");
}

function drawMotifCounts(svg, all, pattern, segments) {
  const types = pattern ? [pattern] : passingMotifTypes, columns = [...new Set(segments)].sort((a, b) => a - b);
  const count = (type, segment) => all.filter((w) => w.pattern === type && w.segment === segment).length;
  const maximum = Math.max(1, ...passingMotifTypes.flatMap((type) => columns.map((s) => count(type, s))));
  const width = Math.min(92, 580 / columns.length), offset = Number(svg.dataset.headerOffset);
  columns.forEach((s, i) => svg.append(svgElement("text", {x: 120 + (i + .5) * width, y: 128 + offset, "text-anchor": "middle", fill: "#647382", "font-size": 13}, `${s * 15}–${s * 15 + 15}`)));
  types.forEach((type, row) => {
    const cy = 150 + row * 56 + offset;
    svg.append(svgElement("text", {x: 102, y: cy + 29, "text-anchor": "end", fill: "#203344", "font-size": 14}, type));
    columns.forEach((segment, col) => {
      const n = count(type, segment);
      svg.append(svgElement("rect", {x: 120 + col * width, y: cy, width: width - 3, height: 49, fill: "#2468a0", "fill-opacity": n / maximum * .3, stroke: "#d9e2e9"}),
        svgElement("text", {x: 120 + (col + .5) * width - 1.5, y: cy + 29, "text-anchor": "middle", fill: "#203344", "font-size": 14}, n));
    });
  });
  const bottom = 150 + types.length * 56 + offset;
  svg.dataset.exportHeight = String(bottom + 126);
  const cropped = typeof window !== "undefined" && window.TouchlineWorkspace;
  svg.setAttribute("viewBox", cropped ? `0 ${70 + offset} 760 ${bottom + 56 - offset}` : `0 0 760 ${bottom + 126}`);
  svg.append(svgElement("text", {x: 120, y: bottom + 34, fill: "#203344", "font-size": 14}, `每格为窗口次数 · 同队所选时段共用 0–${maximum} 色阶`),
    svgElement("text", {x: 120, y: bottom + 68, fill: "#647382", "font-size": 13}, "窗口可重叠，按第三次传球归档；次数未按控球时间归一化。"),
    svgElement("text", {x: 120, y: bottom + 96, fill: "#647382", "font-size": 13}, "在“回看窗口”选择一组三次传球，再逐步查看原始起终点。"));
  explorationTable("match-table", ["模式", "第三次传球所在时段", "窗口数", "该时段全部窗口", "占该时段全部窗口"], types.flatMap((type) => columns.map((segment) => {
    const n = count(type, segment), total = all.filter((w) => w.segment === segment).length;
    return [type, `${segment * 15}–${segment * 15 + 15} 分钟`, n, total, total ? `${number(n / total * 100, 2)}%` : "无窗口"];
  })));
}

function renderPassingMotifs(svg, summary, team, range, segments) {
  svg.classList.add("passing-motifs");
  const available = summary?.schema === "Touchline.PassingMotifs.v1";
  $("match-motif").disabled = !available;
  const all = passingMotifWindows(summary, team, range), pattern = $("match-motif").value;
  const windows = all.filter((w) => !pattern || w.pattern === pattern), old = $("match-window").value;
  resetOptions("match-window", ["", ...windows.map((w) => w.id)], (id) => {
    const w = windows.find((w) => w.id === id);
    return w ? `${w.pattern} · ${sequencePeriods[w.period]} ${motifClock(w.passes[0])}–${motifClock(w.passes[2])}` : "查看分时段计数";
  });
  if (windows.some((w) => w.id === old)) $("match-window").value = old;
  $("match-window").disabled = !windows.length;
  for (const id of ["match-action-label", "match-step-navigation"]) $(id).hidden = !$("match-window").value;
  if (!available) {
    $("match-status").textContent += " · 当前快照没有三传球模式，请重新生成探索数据";
    $("match-summary").replaceChildren(); $("match-chart").append(node("p", "可切换到传球网络，或重新生成三传球摘要。")); return false;
  }
  const audit = summary.audit[team];
  $("match-status").textContent += ` · 所选模式 ${windows.length} 个窗口，全部模式 ${all.length} 个窗口`;
  $("match-summary").replaceChildren(node("h3", `${team} · ${pattern || "全部五种模式"}`),
    node("p", `${windows.length} / ${all.length} 个三传球窗口，允许重叠。`),
    node("p", `整场 ${audit.completed_passes} 次完成传球，${audit.eligible_passes} 次有合规身份与时钟，${audit.chain_passes} 次进入至少三传球的连续链。`));
  const window = windows.find((w) => w.id === $("match-window").value);
  if (window) {
    const token = `motif:${window.id}`, previous = $("match-action").value, same = $("match-action").dataset.phase === token;
    resetOptions("match-action", ["0", "1", "2"], (i) => `${Number(i) + 1}. ${motifClock(window.passes[i])} · ${summary.players[window.passes[i].player]}`);
    $("match-action").dataset.phase = token; $("match-action").value = same && ["0", "1", "2"].includes(previous) ? previous : "0";
    const step = Number($("match-action").value);
    $("match-previous-action").disabled = step === 0; $("match-next-action").disabled = step === 2;
    $("match-summary").append(node("p", `${window.pattern} · ${sequencePeriods[window.period]} ${motifClock(window.passes[0])}–${motifClock(window.passes[2])}`));
    const action = window.passes[step], list = node("ul");
    $("match-summary").append(node("p", `${step + 1}. ${summary.players[action.player]} → ${summary.players[action.recipient]}`));
    [...new Set([window.passes[0].player, ...window.passes.map((p) => p.recipient)])].forEach((actor, i) => list.append(node("li", `${String.fromCharCode(65 + i)} · ${summary.players[actor]}`)));
    $("match-roster").append(node("h3", "本窗口内的球员代号"), list);
    svg.querySelector("title").textContent += ` · ${window.pattern} · ${motifClock(window.passes[0])}–${motifClock(window.passes[2])} · 第 ${step + 1} 次传球`;
    drawMotifWindow(svg, window, summary.players, step);
  } else {
    resetOptions("match-action", [], String);
    drawMotifCounts(svg, all, pattern, segments);
  }
  return true;
}

if (typeof module !== "undefined" && module.exports) module.exports = {passingMotifWindows};
