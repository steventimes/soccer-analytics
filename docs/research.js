"use strict";

const competitionNames = {
  premier_league: "英超", la_liga: "西甲", bundesliga: "德甲", serie_a: "意甲", ligue_1: "法甲",
  champions_league: "欧冠", european_championship: "欧洲杯", world_cup: "世界杯", club_world_cup: "世俱杯",
};
const metricNames = {
  controlled_possession_pct: "受控控球占比", in_possession_long_ball: "长球阶段占比",
  out_of_possession_mid_block: "中位防守阶段占比", out_of_possession_counter_press: "反抢阶段占比",
};
const providerNames = {espn_recent: "ESPN", fifa_training_centre: "FIFA Training Centre", football_csv: "Football-Data CSV", statsbomb: "StatsBomb", understat: "Understat", wyscout: "Wyscout"};
const grainNames = {event: "逐事件", result: "赛果", result_and_basic_statistics: "赛果与基础统计", result_and_summary: "赛果与比赛摘要", schedule_with_xg: "赛程与 xG", tracking_derived_report: "追踪衍生官方报告"};
const coverageStatusNames = {complete: "已达到该赛事的预期比赛数", partial: "部分比赛", partial_or_selected_archive: "部分或精选档案", "source_records; completeness not inferred from row count": "按来源记录，数量不能证明赛事完整覆盖"};
const methodNames = {
  constant_train_prior: "训练集均值", linear_context: "线性上下文模型", separate_single_task: "两个独立单任务 GRU",
  hist_gradient_boosting_context: "梯度提升树上下文", mlp_context: "前馈 MLP 上下文",
  shared_multi_task: "共享多任务 GRU", current_event_only: "仅当前事件（移除历史）", source_zero_shot: "其他联赛直接迁移",
  supervised_transfer_frozen: "有标签预训练 · 冻结", supervised_transfer_finetune: "有标签预训练 · 微调",
  self_supervised_frozen: "自监督预训练 · 冻结", self_supervised_finetune: "自监督预训练 · 微调",
};
const $ = (id) => document.getElementById(id);
function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = String(text);
  if (className) element.className = className;
  return element;
}
function safeLink(title, rawUrl) {
  const link = node("a", title);
  try {
    const url = new URL(rawUrl);
    if (url.protocol !== "https:") throw new Error("Unsupported source protocol");
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  } catch { link.removeAttribute("href"); }
  return link;
}
function options(id, values, label = (value) => value) {
  values.forEach((value) => {
    const option = node("option", label(value));
    option.value = value;
    $(id).append(option);
  });
}
function number(value, decimals = 1) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(decimals) : "暂无";
}

function syncResearchQuery(panel, values) {
  if (typeof window !== "undefined" && window.TouchlineWorkspace && !window.TouchlineWorkspace.active(panel)) return;
  const url = new URL(location.href);
  Object.entries(values).forEach(([key, value]) => { if (value) url.searchParams.set(key, value); else url.searchParams.delete(key); });
  history.replaceState(null, "", url);
}
function restoreResearchControl(id, key) {
  const value = new URLSearchParams(location.search).get(key), control = $(id);
  if ([...control.options].some((option) => option.value === value)) control.value = value;
}
function coverageView(coverage) {
  const rows = coverage.coverage;
  let page = 0;
  let selected = null;
  const pageSize = 12;
  ["competition-filter", "provider-filter", "season-filter"].forEach((id) => {
    const control = $(id);
    while (control.options.length > 1) control.remove(1);
    control.value = "";
  });
  options("competition-filter", [...new Set(rows.map((r) => r.competition))].sort(), (v) => competitionNames[v] || v);
  options("provider-filter", [...new Set(rows.map((r) => r.provider))].sort(), (v) => providerNames[v] || v);
  options("season-filter", [...new Set(rows.map((r) => r.season))].sort((a, b) => b.localeCompare(a, undefined, {numeric: true})));
  for (const [id,key] of [["competition-filter","competition"],["provider-filter","provider"],["season-filter","season"]]) restoreResearchControl(id,key);
  function dataType(row) {
    const grain = grainNames[row.grain] || row.grain;
    return Number.isInteger(row.matched_summary_documents)
      ? `${grain} · 摘要 ${row.matched_summary_documents.toLocaleString()} / ${row.source_match_records.toLocaleString()}` : grain;
  }
  function detail(row) {
    const panel = $("coverage-detail");
    panel.replaceChildren(node("h3", `${competitionNames[row.competition] || row.competition} · ${row.season}`));
    const list = node("dl");
    [["来源", providerNames[row.provider] || row.provider], ["来源赛季标记", row.source_season], ["数据类型", grainNames[row.grain] || row.grain], ["该来源记录数", row.source_match_records.toLocaleString()],
      ["观察日期", `${row.date_start || "未提供"} 至 ${row.date_end || "未提供"}`], ["覆盖说明", coverageStatusNames[row.coverage_status] || row.coverage_status]]
      .forEach(([label, text]) => list.append(node("dt", label), node("dd", text)));
    if (Number.isInteger(row.matched_summary_documents)) {
      [["匹配摘要", `${row.matched_summary_documents.toLocaleString()} / ${row.source_match_records.toLocaleString()}`],
        ["缺失或未完整归档", row.missing_summary_documents.toLocaleString()],
        ["格式或赛果不符", row.invalid_summary_documents.toLocaleString()]]
        .forEach(([label, text]) => list.append(node("dt", label), node("dd", text)));
    }
    panel.append(list);
    if (Number.isInteger(row.matched_summary_documents)) panel.append(node("p", "摘要已核对比赛身份、主客队和终场比分。具体统计、阵容或事件字段仍可能缺失。", "context-note"));
  }
  function render() {
    const filtered = rows.filter((r) => [["competition-filter", "competition"], ["provider-filter", "provider"], ["season-filter", "season"]]
      .every(([control, field]) => !$(control).value || $(control).value === r[field]));
    syncResearchQuery("coverage", {competition:$("competition-filter").value,provider:$("provider-filter").value,season:$("season-filter").value});
    const pages = Math.max(1, Math.ceil(filtered.length / pageSize));
    page = Math.min(page, pages - 1);
    $("coverage-count").textContent = `${filtered.length} / ${rows.length} 条来源记录`;
    $("coverage-rows").replaceChildren();
    if (selected && !filtered.includes(selected)) selected = null;
    if (!selected) $("coverage-detail").replaceChildren(node("h3", "来源细节"), node("p", "选择一条来源记录，查看时间范围与覆盖说明。"));
    filtered.slice(page * pageSize, (page + 1) * pageSize).forEach((row) => {
      const tr = node("tr");
      if (row === selected) tr.classList.add("selected");
      const first = node("td");
      const button = node("button", competitionNames[row.competition] || row.competition, "row-select");
      button.type = "button";
      button.setAttribute("aria-label", `查看 ${competitionNames[row.competition] || row.competition} ${row.season} ${row.provider} 来源细节`);
      button.setAttribute("aria-pressed", String(row === selected));
      button.addEventListener("click", () => {
        selected = row;
        $("coverage-rows").querySelectorAll("tr").forEach((r) => r.classList.remove("selected"));
        $("coverage-rows").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
        tr.classList.add("selected");
        button.setAttribute("aria-pressed", "true");
        detail(row);
      });
      first.append(button, node("small", row.season));
      tr.append(first, node("td", providerNames[row.provider] || row.provider), node("td", row.source_match_records.toLocaleString()), node("td", dataType(row)));
      $("coverage-rows").append(tr);
    });
    if (!filtered.length) {
      const cell = node("td", "这个组合没有记录。重置筛选可查看其他来源。");
      cell.colSpan = 4;
      const tr = node("tr"); tr.append(cell); $("coverage-rows").append(tr);
    }
    $("page-status").textContent = `${page + 1} / ${pages} 页 · 每页最多 ${pageSize} 条`;
    $("previous-page").disabled = page === 0;
    $("next-page").disabled = page >= pages - 1;
  }
  ["competition-filter", "provider-filter", "season-filter"].forEach((id) => { $(id).onchange = () => { page = 0; render(); }; });
  $("reset-filters").onclick = () => {
    ["competition-filter", "provider-filter", "season-filter"].forEach((id) => { $(id).value = ""; });
    page = 0; selected = null; render();
  };
  $("previous-page").onclick = () => { page--; render(); };
  $("next-page").onclick = () => { page++; render(); };
  render();
}

function svgElement(tag, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, String(value)));
  if (text !== undefined) element.textContent = text;
  return element;
}
function isPercentage(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 100;
}
function metricPlot(team, metric, key) {
  const svg = svgElement("svg", {viewBox: "0 0 440 155", role: "img", "aria-label": `${team} ${metricNames[key]}：此前均值 ${number(metric.pre_final_mean)}%，决赛 ${number(metric.final)}%`});
  const x = (value) => 28 + value * 3.84;
  [0, 25, 50, 75, 100].forEach((tick) => {
    svg.append(svgElement("line", {x1: x(tick), x2: x(tick), y1: 42, y2: 109, stroke: "#e2e8ee"}));
    svg.append(svgElement("text", {x: x(tick), y: 137, "text-anchor": "middle", fill: "#647382", "font-size": 13}, `${tick}%`));
  });
  const prior = metric.pre_final_mean, final = metric.final;
  if ([prior, final].every(isPercentage)) {
    svg.append(svgElement("line", {x1: x(prior), x2: x(final), y1: 74, y2: 74, stroke: "#98a5b1", "stroke-width": 3}));
    svg.append(svgElement("circle", {cx: x(prior), cy: 74, r: 6, fill: "#fff", stroke: "#647382", "stroke-width": 2}));
    svg.append(svgElement("rect", {x: x(final) - 5, y: 69, width: 10, height: 10, fill: "#2468a0"}));
    [[prior, 33, `此前 ${number(prior)}%`, "#647382"], [final, 113, `决赛 ${number(final)}%`, "#2468a0"]].forEach(([value, y, text, color]) => {
      const labelX = Math.max(65, Math.min(375, x(value)));
      svg.append(svgElement("text", {x: labelX, y, "text-anchor": "middle", fill: color, "font-size": 15, "font-weight": 600}, text));
    });
  } else svg.append(svgElement("text", {x: 220, y: 75, "text-anchor": "middle", fill: "#647382", "font-size": 15}, "暂无可比较数据"));
  return svg;
}
function caseView(study) {
  const source = safeLink("FIFA 原始报告 ↗", study.match.source_url);
  $("case-source").replaceWith(source); source.id = "case-source";
  function render() {
    const key = $("metric-select").value;
    $("team-facets").replaceChildren();
    $("metric-detail").replaceChildren(node("strong", `${metricNames[key]} · 决赛减此前均值`));
    study.team_comparisons.forEach((team) => {
      const metric = team.metrics?.[key] || {};
      const facet = node("article", undefined, "team-facet");
      facet.append(node("h3", team.team), metricPlot(team.team, metric, key));
      $("team-facets").append(facet);
      const available = [metric.final, metric.pre_final_mean].every(isPercentage);
      const delta = available ? metric.final - metric.pre_final_mean : null;
      facet.append(node("p", available ? `决赛${delta >= 0 ? "增加" : "减少"} ${number(Math.abs(delta))} 个百分点` : "这项指标缺少可比较的数据。", "team-change"));
      $("metric-detail").append(node("p", available ? `${team.team}：${delta >= 0 ? "+" : ""}${number(delta)} 个百分点。此前 ${metric.pre_final_available_matches} 场，决赛 1 场。` : `${team.team}：暂无可比较数据。`));
    });
  }
  $("metric-select").onchange = render;
  render();
}

function representationView(report) {
  $("representation-date").textContent = `实验 ${report.observed_at.slice(0, 10)}`;
  const scope = report.data;
  $("representation-scope").textContent = `Wyscout ${scope.season} · 目标联赛 ${scope.target_league}；目标训练 ${scope.splits.train.sequences_used.toLocaleString()} 个传球序列，测试 ${scope.splits.test.sequences_used.toLocaleString()} 个。其他四个联赛的预训练样本截止 ${scope.splits.source.date_end}，验证从 ${scope.validation_start} 开始，测试从 ${scope.test_start} 开始。`;
  function render() {
    const metric = $("experiment-metric").value;
    const scores = report.summaries.map((r) => ({name: r.method, runs: r.runs, ...r.test[metric]})).filter((s) => Number.isFinite(s.mean) && Number.isFinite(s.seed_sd)).sort((a, b) => a.mean - b.mean);
    const low = Math.min(...scores.map((s) => s.mean - s.seed_sd));
    const high = Math.max(...scores.map((s) => s.mean + s.seed_sd));
    const padding = Math.max((high - low) * .16, high * .015, .001);
    const min = Math.max(0, low - padding), max = high + padding;
    const unit = metric === "completion_log_loss" ? "log loss" : "球场长度百分比";
    const precision = metric === "completion_log_loss" ? 5 : 3;
    $("experiment-chart").replaceChildren();
    $("experiment-chart-label").textContent = `测试集${metric === "completion_log_loss" ? "传球完成概率误差" : "向前位移平均绝对误差"}，越低越好。实心点与横线为 ${report.configuration.seeds.length} 个种子的均值与 ±1 标准差，空心菱形为拟合一次的固定基线。`;
    const height = 80 + scores.length * 44;
    const svg = svgElement("svg", {viewBox: `0 0 960 ${height}`, role: "img", "aria-label": `按均值排序的${unit}结果，点与横线表示均值和种子标准差。下方有完整数据表。`});
    const x = (value) => 272 + (value - min) / (max - min) * 440;
    for (let i = 0; i <= 4; i++) {
      const value = min + (max - min) * i / 4;
      svg.append(svgElement("line", {x1: x(value), x2: x(value), y1: 30, y2: height - 45, stroke: "#e2e8ee"}));
      svg.append(svgElement("text", {x: x(value), y: height - 17, "text-anchor": "middle", fill: "#647382", "font-size": 13}, number(value, metric === "completion_log_loss" ? 3 : 2)));
    }
    svg.append(svgElement("text", {x: 942, y: 18, "text-anchor": "end", fill: "#647382", "font-size": 13}, "均值 ± 种子标准差"));
    const baseline = scores.find((s) => s.name === "constant_train_prior");
    if (baseline) svg.append(svgElement("line", {x1: x(baseline.mean), x2: x(baseline.mean), y1: 30, y2: height - 45, stroke: "#98a5b1", "stroke-dasharray": "4 4"}));
    scores.forEach((score, index) => {
      const y = 49 + index * 44;
      const color = score.runs === 1 ? "#647382" : "#2468a0";
      svg.append(svgElement("text", {x: 0, y: y + 5, fill: "#203344", "font-size": 14}, methodNames[score.name] || score.name));
      svg.append(svgElement("line", {x1: x(score.mean - score.seed_sd), x2: x(score.mean + score.seed_sd), y1: y, y2: y, stroke: color, "stroke-width": 2.5, class: "seed-sd"}));
      [score.mean - score.seed_sd, score.mean + score.seed_sd].forEach((v) => svg.append(svgElement("line", {x1: x(v), x2: x(v), y1: y - 5, y2: y + 5, stroke: color, "stroke-width": 1.5})));
      if (score.runs === 1) svg.append(svgElement("path", {d: `M${x(score.mean)} ${y - 5}l5 5-5 5-5-5z`, fill: "#fff", stroke: color, "stroke-width": 1.8}));
      else svg.append(svgElement("circle", {cx: x(score.mean), cy: y, r: 4.5, fill: color}));
      svg.append(svgElement("text", {x: 942, y: y + 5, "text-anchor": "end", fill: "#203344", "font-size": 14}, `${number(score.mean, precision)}${score.runs === 1 ? " · 固定基线" : ` ± ${number(score.seed_sd, precision)}`}`));
    });
    svg.classList.add("export-only");
    const intervals = node("div", undefined, "interval-results");
    intervals.append(node("p", `共用坐标：${number(min,precision)}–${number(max,precision)} ${unit} · 范围未从零开始`, "context-note"));
    scores.forEach((score) => {
      const row = node("div", undefined, "interval-result"), heading = node("div", undefined, "interval-heading");
      heading.append(node("strong", methodNames[score.name] || score.name), node("span", `${number(score.mean,precision)}${score.runs===1 ? " · 固定基线" : ` ± ${number(score.seed_sd,precision)}`}`));
      const track = node("div", undefined, "interval-track"), range = node("i"), dot = node("b"); track.setAttribute("aria-hidden","true");
      range.style.left = `${(score.mean-score.seed_sd-min)/(max-min)*100}%`; range.style.width = `${2*score.seed_sd/(max-min)*100}%`;
      dot.style.left = `${(score.mean-min)/(max-min)*100}%`; if (score.runs===1) dot.className="fixed-baseline";
      track.append(range,dot); row.append(heading,track); intervals.append(row);
    });
    $("experiment-chart").append(intervals,svg);
    const details = node("details", undefined, "chart-data");
    details.append(node("summary", "查看实验数值"));
    const table = node("table");
    table.append(node("caption", `${unit}，均值与随机种子标准差`));
    const head = node("tr"); ["方法", "均值", "种子标准差"].forEach((label) => head.append(node("th", label))); table.append(head);
    scores.forEach((score) => { const row = node("tr"); [methodNames[score.name] || score.name, number(score.mean, 6), number(score.seed_sd, 6)].forEach((value) => row.append(node("td", value))); table.append(row); });
    details.append(table);
    $("experiment-chart").append(node("p", `横轴 ${number(min, 3)}–${number(max, 3)} ${unit}，虚线为训练集均值模型。点图使用局部刻度以便比较。`, "experiment-axis"), details);
  }
  $("experiment-metric").onchange = render;
  render();
}

function methodsView(catalog) {
  function render() {
    $("method-cards").replaceChildren();
    catalog.methods.filter((m) => !$("method-filter").value || m.task === $("method-filter").value).forEach((method) => {
      const card = node("article", undefined, "method-card");
      card.append(node("p", method.organization, "organization"), node("h3", method.name),
        node("p", method.method), node("p", method.application));
      const details = node("details");
      details.append(node("summary", "查看输入、公开参数与未公开项"));
      [["输入", method.features], ["已公开", method.disclosed_parameters], ["未确认", method.unknown]].forEach(([label, values]) => {
        details.append(node("strong", label));
        const list = node("ul"); values.forEach((value) => list.append(node("li", value))); details.append(list);
      });
      details.append(node("p", `${method.evidence_scope}${method.published_date ? " · " + method.published_date : ""}`));
      card.append(details);
      method.sources.forEach((source) => card.append(safeLink(source.title + " ↗", source.url)));
      $("method-cards").append(card);
    });
  }
  $("method-filter").onchange = render;
  render();
}

function galleryView(catalog) {
  const fields = ["id", "title", "topic", "filename", "format", "alt", "scope", "source"];
  const seen = new Set();
  const figures = catalog.figures.filter((figure) => {
    if (!figure || !fields.every((field) => typeof figure[field] === "string" && figure[field].trim()) ||
        !/^[a-z0-9][a-z0-9-]*\.(svg|png)$/.test(figure.filename) ||
        figure.format !== figure.filename.split(".").pop() || seen.has(figure.id)) return false;
    seen.add(figure.id);
    return true;
  });
  options("figure-topic", [...new Set(figures.map((figure) => figure.topic))]);
  const dialog = $("figure-dialog");
  const image = $("figure-dialog-image");
  const readerImage = $("figure-reader-image");
  let filtered = figures;
  const galleryQuery = new URLSearchParams(location.search);
  $("figure-search").value = galleryQuery.get("figure_search") || "";
  restoreResearchControl("figure-topic", "topic");
  let selected = figures.find((figure) => figure.id === galleryQuery.get("figure")) || figures[0];
  let opener;
  const imageUrl = (figure) => `./data/research/figures/${figure.filename}`;
  function zoom() {
    const mode = $("figure-zoom").value;
    image.style.width = mode === "fit" ? "100%" : `${(image.naturalWidth || 900) * (mode === "large" ? 1.5 : 1)}px`;
    image.style.maxWidth = mode === "fit" ? "100%" : "none";
  }
  image.addEventListener("load", () => { zoom(); $("figure-dialog-status").textContent = ""; });
  image.addEventListener("error", () => { $("figure-dialog-status").textContent = "图表未能加载。可关闭窗口后重试，或用上方链接打开原图。"; });
  readerImage.addEventListener("load", () => {
    $("figure-image-status").textContent = "";
    if (document.activeElement === $("figure-image-retry")) $("figure-enlarge").focus();
    $("figure-image-retry").hidden = true;
  });
  readerImage.addEventListener("error", () => {
    $("figure-image-status").textContent = "图表未能加载。请重试，或在新窗口打开原图。";
    $("figure-image-retry").hidden = false;
  });
  $("figure-image-retry").addEventListener("click", () => {
    $("figure-image-status").textContent = "正在重新加载图表…";
    readerImage.src = imageUrl(selected);
  });
  $("figure-zoom").addEventListener("change", zoom);
  $("close-figure").addEventListener("click", () => dialog.close());
  dialog.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const controls = Array.from(dialog.querySelectorAll('button:not(:disabled), select:not(:disabled), a[href], [tabindex="0"]')).filter(control => control.getClientRects().length);
    const first = controls[0], last = controls[controls.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault(); last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault(); first.focus();
    }
  });
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
    }
  });
  dialog.addEventListener("close", () => { document.body.classList.remove("figure-is-open"); opener?.focus(); });
  function open(figure, trigger) {
    opener = trigger;
    const url = imageUrl(figure);
    $("figure-dialog-title").textContent = figure.title;
    $("figure-dialog-topic").textContent = figure.topic;
    $("figure-dialog-scope").textContent = figure.scope;
    $("figure-dialog-source").textContent = `来源：${figure.source}`;
    $("figure-download").href = url;
    $("figure-download").textContent = `下载 ${figure.format.toUpperCase()} 原图`;
    $("figure-open").href = url;
    $("figure-zoom").value = "fit";
    image.alt = figure.alt;
    $("figure-dialog-status").textContent = "正在加载图表…";
    image.src = url;
    zoom();
    $("figure-viewport").scrollTo(0, 0);
    document.body.classList.add("figure-is-open");
    dialog.showModal();
    $("close-figure").focus();
  }
  function select(figure) {
    selected = figure;
    if (figure) syncResearchQuery("figures", {figure:figure.id,topic:$("figure-topic").value,figure_search:$("figure-search").value.trim()});
    const index = filtered.indexOf(figure);
    $("figure-reader").hidden = !figure;
    if (!figure) return;
    $("figure-reader-topic").textContent = figure.topic;
    $("figure-reader-title").textContent = figure.title;
    $("figure-reader-scope").textContent = figure.scope;
    $("figure-reader-source").textContent = `来源：${figure.source}`;
    $("figure-position").textContent = `${index + 1} / ${filtered.length}`;
    $("figure-select").value = figure.id;
    $("figure-previous").disabled = index <= 0;
    $("figure-next").disabled = index >= filtered.length - 1;
    $("figure-enlarge").setAttribute("aria-label", `放大查看：${figure.title}`);
    $("figure-reader-download").href = imageUrl(figure);
    $("figure-reader-download").download = figure.filename;
    $("figure-reader-download").textContent = `下载 ${figure.format.toUpperCase()}`;
    $("figure-reader-open").href = imageUrl(figure);
    readerImage.alt = figure.alt;
    $("figure-image-status").textContent = "正在加载图表…";
    $("figure-image-retry").hidden = true;
    readerImage.src = imageUrl(figure);
    $("figure-gallery").querySelectorAll("button").forEach((button) => {
      if (button.dataset.figureId === figure.id) button.setAttribute("aria-current", "true");
      else button.removeAttribute("aria-current");
    });
  }
  $("figure-enlarge").addEventListener("click", (event) => open(selected, event.currentTarget));
  $("figure-previous").addEventListener("click", () => select(filtered[filtered.indexOf(selected) - 1]));
  $("figure-next").addEventListener("click", () => select(filtered[filtered.indexOf(selected) + 1]));
  $("figure-select").addEventListener("change", (event) => select(filtered.find((figure) => figure.id === event.target.value)));
  function render() {
    const query = $("figure-search").value.trim().toLocaleLowerCase();
    const topic = $("figure-topic").value;
    filtered = figures.filter((figure) => (!topic || topic === figure.topic) && (!query || `${figure.title} ${figure.topic} ${figure.alt} ${figure.scope}`.toLocaleLowerCase().includes(query)));
    $("figure-count").textContent = `${filtered.length} / ${figures.length} 张图表`;
    $("figure-gallery").replaceChildren();
    $("figure-select").replaceChildren();
    $("figure-select").disabled = !filtered.length;
    $("figure-empty").hidden = !!filtered.length;
    filtered.forEach((figure) => {
      const choice = node("button", undefined, "figure-choice");
      choice.type = "button";
      choice.dataset.figureId = figure.id;
      choice.append(node("span", figure.topic), node("strong", figure.title));
      choice.addEventListener("click", () => select(figure));
      $("figure-gallery").append(choice);
      const option = node("option", figure.title);
      option.value = figure.id;
      $("figure-select").append(option);
    });
    if (!filtered.length) $("figure-empty").textContent = figures.length ? "没有匹配的图表。请更换关键词或主题。" : "当前快照没有可展示的图表。";
    select(filtered.includes(selected) ? selected : filtered[0]);
  }
  $("figure-search").addEventListener("input", render);
  $("figure-topic").addEventListener("change", render);
  render();
}

async function loadGallery() {
  if (!$("figure-count")) return;
  try {
    const response = await fetch("./data/research/figures.json", {cache: "no-cache"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const catalog = await response.json();
    if (!Array.isArray(catalog.figures)) throw new Error("目录格式不完整");
    galleryView(catalog);
    if ($("gallery-retry")) $("gallery-retry").hidden = true;
  } catch (error) {
    if ($("gallery-retry")) { $("gallery-retry").hidden = false; $("gallery-retry").onclick = loadGallery; }
    $("figure-count").textContent = "图表目录未就绪";
    $("figure-empty").hidden = false;
    $("figure-empty").textContent = `无法读取图表目录（${error.message}）。可使用重试按钮重新读取。`;
  }
}

function validResearchData(name, data) {
  if (!data || typeof data !== "object") return false;
  const dated = (value) => typeof value === "string" && /^\d{4}-\d{2}-\d{2}/.test(value);
  const count = (value) => Number.isSafeInteger(value) && value >= 0;
  if (name === "coverage") return dated(data.observed_at) && Array.isArray(data.coverage) && data.coverage.every((row) =>
    row && ["competition", "provider", "season", "grain"].every((key) => typeof row[key] === "string") && count(row.source_match_records) &&
    (row.matched_summary_documents === undefined ||
      ["matched", "missing", "invalid"].every((key) => count(row[`${key}_summary_documents`])) &&
      row.matched_summary_documents + row.missing_summary_documents + row.invalid_summary_documents === row.source_match_records));
  if (name === "case") return !!data.match && Array.isArray(data.team_comparisons) && data.team_comparisons.length === 2 &&
    data.team_comparisons.every((team) => team && typeof team.team === "string");
  if (name === "methods") return dated(data.checked_at) && Array.isArray(data.methods) && data.methods.every((method) =>
    method && ["features", "disclosed_parameters", "unknown", "sources"].every((key) => Array.isArray(method[key])) &&
    method.sources.every((source) => source && typeof source.title === "string"));
  if (name === "representation") return dated(data.observed_at) && !!data.data && !!data.configuration &&
    Array.isArray(data.configuration.seeds) && data.configuration.seeds.length > 0 &&
    ["train", "test"].every((key) => count(data.data.splits?.[key]?.sequences_used)) &&
    typeof data.data.splits?.source?.date_end === "string" &&
    Array.isArray(data.summaries) && data.summaries.length > 0 && data.summaries.every((row) => row &&
      ["completion_log_loss", "forward_displacement_mae_pitch_pct"].every((key) =>
        Number.isFinite(row.test?.[key]?.mean) && Number.isFinite(row.test?.[key]?.seed_sd)));
  return false;
}

async function loadResearchSection(section, fetchData = fetch) {
  if (section.state === "loading" || section.state === "ready") return;
  section.state = "loading";
  section.error = null;
  section.onState(section);
  let failure = "网络连接失败";
  try {
    const response = await fetchData(section.url, {cache: "no-cache"});
    failure = `服务器返回 HTTP ${response.status}`;
    if (!response.ok) throw new Error();
    failure = "返回的数据无法读取";
    const data = await response.json();
    failure = "返回的数据不完整";
    if (!section.validate(data)) throw new Error();
    failure = "图表暂时无法显示";
    section.render(data);
    section.data = data;
    section.state = "ready";
  } catch {
    section.error = failure;
    section.state = "error";
  }
  section.onState(section);
}

async function loadResearch() {
  const sections = [
    {name: "coverage", label: "数据覆盖", render: coverageView, controls: ["competition-filter", "provider-filter", "season-filter", "reset-filters"], content: ["coverage-content"]},
    {name: "case", label: "战术案例", render: caseView, controls: ["metric-select"], content: ["team-facets", "metric-detail"]},
    {name: "methods", label: "机构方法", render: methodsView, controls: ["method-filter"], content: ["method-cards"]},
    {name: "representation", label: "表示学习汇总", render: representationView, controls: ["experiment-metric"], content: ["experiment-chart", "representation-scope", "experiment-chart-label"]},
  ].filter((section) => $(`${section.name}-status`)).map((section) => ({...section, state: "idle", url: `./data/research/${section.name}.json`,
    validate: (data) => validResearchData(section.name, data), onState: update}));
  function update(section) {
    const status = $(`${section.name}-status`), retry = $(`${section.name}-retry`);
    if (section.state === "loading") section.restoreFocus = document.activeElement === retry;
    status.hidden = section.state === "ready";
    status.classList.toggle("error", section.state === "error");
    status.textContent = section.state === "error" ? `${section.label}暂时无法读取（${section.error}）。` : `正在读取${section.label}。`;
    retry.disabled = section.state === "loading";
    if (section.state === "error") retry.hidden = false;
    if (section.state === "ready") retry.hidden = true;
    section.controls.forEach((id) => { $(id).disabled = section.state !== "ready"; });
    section.content.forEach((id) => { $(id).hidden = section.state !== "ready"; });
    if (section.state === "ready" && section.restoreFocus) $(section.controls[0]).focus();
    const ready = sections.filter((item) => item.state === "ready").length;
    const errors = sections.filter((item) => item.state === "error");
    $("load-status").className = errors.length ? "error" : "";
    $("load-status").textContent = errors.length
      ? `${ready} / ${sections.length} 项研究摘要已加载。${errors.map((item) => item.label).join("、")}暂不可用，可在对应区域重试。`
      : `${ready} / ${sections.length} 项研究摘要已加载。${ready < sections.length ? "其余摘要正在读取。" : ""}`;
    const coverage = sections.find((item) => item.name === "coverage")?.data;
    const methods = sections.find((item) => item.name === "methods")?.data;
    $("snapshot-date").textContent = `${coverage ? `比赛数据快照 ${coverage.observed_at.slice(0, 10)}` : "比赛数据日期暂不可用"} · ${methods ? `方法核对 ${methods.checked_at}` : "方法核对日期暂不可用"}`;
  }
  sections.forEach((section) => {
    $(`${section.name}-retry`).addEventListener("click", () => loadResearchSection(section));
  });
  await Promise.all(sections.map((section) => loadResearchSection(section)));
}
async function loadLiterature() {
  const status = $("paper-status");
  if (!status) return;
  try {
    const response = await fetch("./data/research/literature.json", {cache: "no-cache"});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const catalog = await response.json();
    if (catalog.schema !== "SoccerAnalytics.FootballLiterature.v1" || !catalog.papers?.length) throw new Error("目录格式不完整");
    $("paper-scope").textContent = catalog.scope;
    const topics = {visualization: "图表与可视分析", "sports-analytics": "足球分析模型"};
    function card(paper, candidate = false) {
      const article = node("article", undefined, "method-card");
      article.append(node("p", `${paper.year} · ${paper.venue} · ${paper.publication_status === "preprint" ? "预印本" : "已发表"}`, "organization"), node("h3", paper.title), node("p", paper.application));
      const details = node("details"); details.append(node("summary", candidate ? "查看已核对范围" : "查看数据、验证与限制"));
      [["作者", paper.authors], ["数据", paper.data], ["验证", paper.validation], ["限制", paper.limitations], ["阅读范围", candidate ? "仅出版信息，未读正文" : paper.read_scope]].forEach(([label, text]) => details.append(node("p", `${label}：${text}`)));
      if (paper.evidence_pages.length) details.append(node("p", `所链接 PDF 的相关页码：${paper.evidence_pages.join("、")}。`));
      for (const check of paper.source_checks || []) {
        const evidence = node("p"); evidence.append(safeLink(check.label, check.url), node("span", ` ${check.note}`)); details.append(evidence);
      }
      article.append(details, safeLink(candidate ? "作者机构出版记录 ↗" : "作者或机构公开全文 ↗", paper.source_url), safeLink("出版记录 ↗", paper.publication_url));
      return article;
    }
    restoreResearchControl("paper-filter", "paper_topic");
    $("paper-search").value = new URLSearchParams(location.search).get("paper_search") || "";
    function render() {
      const topic = $("paper-filter").value;
      syncResearchQuery("literature", {paper_topic:topic,paper_search:$("paper-search").value.trim()});
      const query = $("paper-search")?.value.trim().toLocaleLowerCase() || "";
      const papers = catalog.papers.filter((paper) => (!topic || paper.topic === topic) && (!query ||
        [paper.title, paper.authors, paper.venue, paper.data, paper.validation, paper.limitations, paper.application].some((text) => text.toLocaleLowerCase().includes(query))))
        .sort((a, b) => a.priority - b.priority || b.year - a.year || a.id.localeCompare(b.id));
      $("paper-cards").replaceChildren(...papers.map((paper) => card(paper)));
      if (!papers.length) $("paper-cards").append(node("p", "没有匹配的论文，请更换关键词或研究方向。", "context-note"));
      status.textContent = `${topic ? topics[topic] : "全部方向"}${query ? ` · 搜索“${$("paper-search").value.trim()}”` : ""} ${papers.length} / ${catalog.papers.length} 篇 · 另有 ${catalog.candidates.length} 篇仅核对出版信息 · 核对日期 ${catalog.checked_at}。论文数字属于各自实验，不代表本项目的结果。`;
    }
    $("paper-candidates").replaceChildren(...catalog.candidates.map((paper) => card(paper, true)));
    $("paper-filter").disabled = false;
    $("paper-filter").addEventListener("change", render);
    if ($("paper-search")) { $("paper-search").disabled = false; $("paper-search").addEventListener("input", render); }
    render();
    if ($("paper-retry")) $("paper-retry").hidden = true;
  } catch (error) {
    if ($("paper-retry")) { $("paper-retry").hidden = false; $("paper-retry").onclick = loadLiterature; }
    $("paper-cards").replaceChildren();
    status.textContent = `论文目录未就绪（${error.message}），可使用重试按钮重新读取。`;
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {loadResearchSection, validResearchData};
} else {
  loadResearch();
  loadGallery();
  loadLiterature();
}
