"use strict";

const sequenceKinds = {Pass: "传球", Carry: "带球", Dribble: "过人", Shot: "射门"};
const sequencePeriods = {1: "上半场", 2: "下半场", 3: "加时上半场", 4: "加时下半场"};
const sequenceOutcomes = {Complete: "完成", Incomplete: "未完成", Out: "出界", Goal: "进球", Saved: "扑救", Blocked: "封堵", OffT: "偏出", Wayward: "偏出", Post: "中框", Recorded: "已记录", Unknown: "未知", Won: "成功", Lost: "失败", "Saved Off Target": "扑救后偏出", "Saved to Post": "扑救后中框", "Pass Offside": "越位", "Injury Clearance": "伤停解围"};
function sequenceClock(action) { return `${action.minute}:${String(action.second).padStart(2, "0")}`; }
function shotSequencePhases(phases, team, range) {
  return phases.filter((phase) => phase.team === team && phase.actions.some((action) => {
    if (action.kind !== "Shot") return false;
    const [start, length] = {1: [0, 45], 2: [45, 45], 3: [90, 15], 4: [105, 15]}[phase.period];
    const minute = start + Math.min(length - 1, Math.max(0, action.minute - start));
    return minute >= range[0] && minute < range[1];
  }));
}
function drawShotSequence(svg, phase, selected) {
  const {x, y} = matchPitch(svg), action = phase.actions[selected];
  const defs = svgElement("defs", {}), marker = svgElement("marker", {id: "sequence-arrow", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 4, markerHeight: 4, orient: "auto-start-reverse"});
  marker.append(svgElement("path", {d: "M0 0L10 5L0 10z", fill: "context-stroke"})); defs.append(marker); svg.append(defs);
  const order = phase.actions.map((_, i) => i).filter((i) => i !== selected).concat(selected);
  order.forEach((i) => {
    const a = phase.actions[i];
    if (!a.start) return;
    const active = i === selected, color = active ? (a.kind === "Shot" ? "#a34f15" : "#2468a0") : "#647382";
    const group = svgElement("g", {opacity: active ? 1 : .25});
    group.append(svgElement("title", {}, `${i + 1}. ${sequenceClock(a)} ${a.player} · ${sequenceKinds[a.kind]} · ${sequenceOutcomes[a.outcome] || a.outcome}`));
    if (a.end) group.append(svgElement("line", {x1: x(a.start[0]), y1: y(a.start[1]), x2: x(a.end[0]), y2: y(a.end[1]), stroke: color, "stroke-width": active ? 3 : 1.4, "stroke-dasharray": a.kind === "Carry" ? "6 4" : "none", "marker-end": "url(#sequence-arrow)"}));
    group.append(svgElement("circle", {cx: x(a.start[0]), cy: y(a.start[1]), r: active ? 13 : 3, fill: color}));
    if (active) group.append(svgElement("text", {x: x(a.start[0]), y: y(a.start[1]) + 4, "text-anchor": "middle", fill: "white", "font-size": 12}, i + 1));
    svg.append(group);
  });
  const offset = Number(svg.dataset.headerOffset), missing = phase.actions.filter((a) => !a.start || !a.end).length;
  svg.append(svgElement("text", {x: 40, y: 580 + offset, fill: "#647382", "font-size": 13}, `突出动作 ${selected + 1} / ${phase.actions.length} · 虚线为带球 · ${missing} 步缺少起点或终点`));
  $("match-summary").replaceChildren(node("h3", `${phase.team} · ${phase.shots} 次射门`),
    node("p", `${phase.goals} 个进球 · ${number(phase.xg, 2)} xG`),
    node("p", `${selected + 1}. ${sequenceClock(action)} · ${action.player}`),
    node("p", `${sequenceKinds[action.kind]}${action.recipient ? " → " + action.recipient : ""} · ${sequenceOutcomes[action.outcome] || action.outcome}${action.xg === null ? "" : " · " + number(action.xg, 3) + " xG"}`),
    node("p", `同段另有 ${phase.opponent_events} 条对方事件。箭头仅表示单个已记录动作，不连接两个动作之间的空白。`));
  const location = (point) => point ? point.map((v) => number(v, 1)).join(" / ") : "缺失";
  explorationTable("match-table", ["步骤", "时间", "球员", "动作", "接球队员", "结果", "起点 x / y", "终点 x / y", "xG"], phase.actions.map((a, i) => [i + 1, sequenceClock(a), a.player, sequenceKinds[a.kind], a.recipient, sequenceOutcomes[a.outcome] || a.outcome, location(a.start), location(a.end), a.xg === null ? "—" : number(a.xg, 3)]));
  $("match-table").querySelectorAll("tbody tr")[selected].setAttribute("aria-current", "step");
}

if (typeof module !== "undefined" && module.exports) module.exports = {shotSequencePhases};
