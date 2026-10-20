const state = {
  predictions: [],
  predictionsAvailable: false,
  releaseTimer: null,
};

const RELEASE_MAX_AGE_MS = 24 * 60 * 60 * 1000;

async function loadJson(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path} returned ${response.status}`);
  }
  const raw = await response.text();
  const payload = JSON.parse(raw);
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    throw new Error(`${path} must contain a JSON object`);
  }
  const name = path.split("/").pop();
  const requiredRows = {
    "predictions.json": ["predictions"],
    "preset_questions.json": ["questions"],
    "scores.json": ["scores"],
    "release.json": ["blockers", "warnings"],
  }[name] || [];
  const rowFields = [...new Set([
    ...requiredRows, "errors", "refresh_errors", "alerts",
    "model_quality_by_competition", "coverage_by_competition",
  ])];
  for (const field of rowFields) {
    if (payload[field] === undefined && !requiredRows.includes(field)) continue;
    if (!Array.isArray(payload[field]) || payload[field].some((row) => !row || typeof row !== "object" || Array.isArray(row))) {
      throw new Error(`${path}: ${field} must be an array of objects`);
    }
  }
  return { payload, raw };
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function finiteNumber(value) {
  if (value === null || value === undefined || value === "" || typeof value === "boolean") {
    return null;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function percent(value, digits = 0) {
  const numeric = finiteNumber(value);
  return numeric === null ? "—" : `${(numeric * 100).toFixed(digits)}%`;
}

function formatDateTime(value) {
  if (!value) return "快照时间未知";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", { dateStyle: "medium", timeStyle: "short" });
}

function formatKickoff(value) {
  if (!value) return "开球时间待定";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString("zh-CN", {
    weekday: "short",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function setEmpty(container, title, message) {
  container.innerHTML = `
    <div class="empty-state">
      <strong>${escapeHtml(title)}</strong>
      <span>${escapeHtml(message)}</span>
    </div>
  `;
}

function outcomeLabel(value) {
  return {
    HOME_WIN: "主胜",
    DRAW: "平局",
    AWAY_WIN: "客胜",
  }[value] || value || "待定";
}

function renderCoverage(manifest, operations) {
  const coverageAvailable = Boolean(manifest || operations);
  const coverage = operations?.model_coverage || manifest?.model_coverage || {};
  const loaded = Math.max(0, finiteNumber(coverage.loaded) ?? 0);
  const requested = Math.max(0, finiteNumber(coverage.requested) ?? 0);
  const ratio = requested > 0 ? Math.min(1, loaded / requested) : 0;
  const track = document.getElementById("model-coverage-track");
  document.getElementById("coverage-fill").style.width = `${ratio * 100}%`;
  document.getElementById("coverage-ratio").textContent = requested ? `${loaded} / ${requested}` : "— / —";
  document.getElementById("coverage-copy").textContent = requested
    ? loaded === requested
      ? "所选赛事的模型均已就绪"
      : `${requested - loaded} 个赛事模型需要检查`
    : coverageAvailable ? "尚无赛事模型" : "覆盖数据未就绪";
  track.setAttribute("aria-valuenow", String(Math.round(ratio * 100)));

  const labels = operations?.coverage_by_competition || [];
  document.getElementById("coverage-labels").innerHTML = labels.length
    ? labels.map((item) => `<span><b>${escapeHtml(item.competition)}</b>${escapeHtml(item.prediction_count)} 场比赛</span>`).join("")
    : coverageAvailable
      ? "<span><b>暂无待赛比赛</b>运行训练与预测后查看赛事覆盖。</span>"
      : "<span><b>覆盖情况未知</b>请恢复覆盖摘要文件。</span>";
}

function recommendationsApproved(release) {
  return release?.release_decision === "approved"
    && release?.can_publish_recommendations === true
    && Array.isArray(release?.blockers)
    && release.blockers.length === 0;
}

function currentRelease(release, now = Date.now()) {
  if (!release) return null;
  const generated = Date.parse(release.generated_at);
  const expires = Date.parse(release.expires_at);
  const invalidTime = !Number.isFinite(generated) || generated > now
    || !Number.isFinite(expires) || expires <= generated
    || expires > generated + RELEASE_MAX_AGE_MS;
  const expired = Number.isFinite(generated)
    && (now >= generated + RELEASE_MAX_AGE_MS || (Number.isFinite(expires) && now >= expires));
  const inconsistent = release.release_decision === "approved" && !recommendationsApproved(release);
  if (!expired && !inconsistent && (!invalidTime || release.release_decision !== "approved")) return release;
  const blocker = expired
    ? { code: "RELEASE_EXPIRED", message: "本次发布许可已过期。请刷新快照后重新检查。" }
    : { code: "RELEASE_INVALID", message: "发布许可或有效期不完整。请刷新快照后重新检查。" };
  return {
    ...release,
    release_decision: "blocked",
    can_publish_recommendations: false,
    blockers: [...(release.blockers || []), blocker],
  };
}

function renderMetrics(predictions, operations, predictionsAvailable, release) {
  if (!predictionsAvailable) {
    ["metric-fixtures", "metric-picks", "metric-confidence", "metric-cold-start"].forEach((id) => {
      document.getElementById(id).textContent = "—";
    });
    return;
  }
  const confidences = predictions.map((item) => finiteNumber(item.confidence)).filter((value) => value !== null);
  const average = confidences.length
    ? confidences.reduce((sum, value) => sum + value, 0) / confidences.length
    : null;
  document.getElementById("metric-fixtures").textContent = String(predictions.length);
  const threshold = finiteNumber(release?.thresholds?.high_confidence) ?? 0.6;
  document.getElementById("metric-picks-threshold").textContent = `最高赛果概率至少 ${percent(threshold)}`;
  document.getElementById("metric-picks").textContent = recommendationsApproved(release)
    ? String(confidences.filter((value) => value >= threshold).length)
    : "0";
  document.getElementById("metric-confidence").textContent = percent(average, 1);
  document.getElementById("metric-cold-start").textContent = predictions.length
    ? percent(operations?.cold_start_ratio, 0)
    : "—";
}

function signalMarkup(item, fallbackLevel = "warning") {
  const level = item.level || fallbackLevel;
  const code = String(item.code || "").replace(/^CRITICAL_ALERT_/, "");
  const messages = {
    INSUFFICIENT_FIXTURE_COVERAGE: "已导出的比赛数未达到发布门槛。",
    OPERATIONS_NOT_HEALTHY: "运行摘要尚未通过检查，请先处理数据质量问题。",
    NO_PUBLISHABLE_PICKS: "达到概率门槛的候选数量不足。",
    LOW_AVERAGE_CONFIDENCE: "平均最高赛果概率未达到要求或尚不可用。",
    OPERATIONS_PREDICTION_AUDIT_MISMATCH: "运行摘要计数与去重后的预测集合不一致。",
    MODEL_COVERAGE_INCOMPLETE: "部分赛事模型尚未完成预测。",
    MODEL_EVIDENCE_INCOMPLETE: "部分模型缺少完整的质量验证依据。",
    NO_PREDICTIONS: "尚未导出待赛预测，请检查模型和赛程数据。",
    MODEL_COVERAGE_GAP: "赛事模型覆盖不完整。",
    PREDICTION_ERRORS: "部分赛事预测失败，具体原因见运行诊断。",
  };
  const translated = messages[code];
  return `
    <article class="signal signal-${escapeHtml(level)}">
      <span class="signal-mark" aria-hidden="true"></span>
      <div>
        <strong>${escapeHtml(item.code || "NOTICE")}</strong>
        <p>${escapeHtml(translated || item.message || "没有提供进一步说明。")}</p>
        ${translated && item.message ? `<details><summary>查看来源诊断原文</summary><p>${escapeHtml(item.message)}</p></details>` : ""}
      </div>
    </article>
  `;
}

function renderOperations(operations) {
  const container = document.getElementById("operations-summary");
  const alerts = document.getElementById("operations-alerts");
  const status = operations?.status || "unknown";
  const statusNode = document.getElementById("operations-status");
  statusNode.textContent = ({healthy: "正常", warning: "需检查", critical: "未通过", unknown: "未知"})[status] || "未知";
  statusNode.className = `status-token status-${status}`;
  if (!operations) {
    document.getElementById("operations-message-count").textContent = "未就绪";
    setEmpty(container, "运行摘要未就绪", "运行摘要文件未能加载。");
    setEmpty(alerts, "运行摘要文件未就绪", "模型质量尚未确认，本次快照未通过检查。");
    return;
  }
  const coverage = operations?.model_coverage || {};
  const hasFixtures = (finiteNumber(operations.prediction_count) ?? 0) > 0;
  const qualityRows = operations.model_quality_by_competition || [];
  const backtestsReady = qualityRows.filter((item) => finiteNumber(item.backtest_folds) !== null).length;
  const fallbackModels = (operations.models_recovered_from_previous || []).length;
  const staleModels = (operations.models_stale || []).length;
  const unstableModels = (operations.models_temporally_unstable || []).length;
  container.innerHTML = `
    <div><span>就绪模型</span><strong>${escapeHtml(coverage.loaded ?? "—")} / ${escapeHtml(coverage.requested ?? "—")}</strong></div>
    <div><span>历史不足占比</span><strong>${hasFixtures ? percent(operations.cold_start_ratio) : "—"}</strong></div>
    <div><span>平均不确定性</span><strong>${percent(operations?.average_uncertainty)}</strong></div>
    <div><span>已有回测</span><strong>${escapeHtml(backtestsReady)} / ${escapeHtml(coverage.loaded ?? "—")}</strong></div>
    <div><span>回退模型</span><strong>${escapeHtml(fallbackModels)}</strong></div>
    <div><span>过期模型</span><strong>${escapeHtml(staleModels)}</strong></div>
    <div><span>不稳定模型</span><strong>${escapeHtml(unstableModels)}</strong></div>
  `;
  const qualityByCompetition = new Map(
    qualityRows.map((item) => [item.competition, item]),
  );
  const qualityItems = (operations.models_below_baseline || []).map((competition) => {
    const item = qualityByCompetition.get(competition);
    if (!item) {
      return {
        level: "critical",
        code: `MODEL_BELOW_BASELINE / ${competition}`,
        message: "质量详情未就绪，请恢复验证依据后再发布。",
      };
    }
    return {
      level: "critical",
      code: `MODEL_BELOW_BASELINE / ${item.competition}`,
      message: `验证命中率 ${percent(item.validation_accuracy)}，基线 ${percent(item.baseline_accuracy)}，差值 ${percent(item.accuracy_lift, 1)}。`,
    };
  });
  const backtestItems = qualityRows
    .filter((item) => finiteNumber(item.backtest_folds) !== null)
    .map((item) => {
      const technology = [item.estimator, item.estimator_version]
        .filter((value) => typeof value === "string" && value.trim())
        .map((value) => escapeHtml(value.trim()))
        .join(" ");
      const explanationMethod = typeof item.explanation_method === "string" && item.explanation_method.trim()
        ? ` · ${escapeHtml(item.explanation_method.trim().replaceAll("_", " "))}`
        : "";
      const boostingRounds = finiteNumber(item.boosting_rounds) === null
        ? ""
        : ` · ${escapeHtml(Math.round(finiteNumber(item.boosting_rounds)))} 轮`;
      const eloBaseline = finiteNumber(item.backtest_elo_baseline_accuracy);
      const majorityBaseline = finiteNumber(item.backtest_majority_baseline_accuracy);
      const domainBaselines = eloBaseline === null || majorityBaseline === null
        ? ""
        : ` · Elo ${percent(eloBaseline)} / 多数类 ${percent(majorityBaseline)}`;
      const classLabels = ["客胜", "平局", "主胜"];
      const reliability = Array.isArray(item.backtest_reliability)
        ? item.backtest_reliability.map((entry) => {
          const classIndex = Number(entry.class_index);
          const label = classLabels[classIndex] || `类别 ${classIndex}`;
          return `${label} ECE ${percent(entry.calibration_error)} (${escapeHtml(entry.sample_count ?? "—")} 个样本)`;
        }).join(" · ")
        : "分类校准数据未就绪";
      return {
        level: "healthy",
        code: `BACKTEST / ${item.competition || "UNKNOWN"}`,
        message: `${technology ? `${technology}${explanationMethod}${boostingRounds} · ` : ""}${escapeHtml(item.backtest_folds)} 个扩展窗口 · 命中率 ${percent(item.backtest_accuracy)}，最强基线 ${percent(item.backtest_baseline_accuracy)}${domainBaselines} · RPS ${finiteNumber(item.backtest_ranked_probability_score) === null ? "—" : escapeHtml(finiteNumber(item.backtest_ranked_probability_score).toFixed(3))} · 校准误差 ${percent(item.backtest_calibration_error)} · ${reliability} · 平均最高概率 ${percent(item.backtest_sharpness)} · 最差窗口提升 ${percent(item.backtest_worst_fold_accuracy_lift, 1)} · 窗口标准差 ${percent(item.backtest_accuracy_lift_stddev, 1)}${finiteNumber(item.model_age_hours) === null ? "" : ` · 模型年龄 ${escapeHtml(Math.round(finiteNumber(item.model_age_hours)))}h`}.`,
      };
    });
  const items = [...qualityItems, ...backtestItems, ...(operations.alerts || [])];
  document.getElementById("operations-message-count").textContent = String(items.length);
  if (items.length) {
    alerts.innerHTML = items.map((item) => signalMarkup(item)).join("");
  } else {
    setEmpty(alerts, "没有模型质量提示", "本次快照的覆盖和模型检查已通过。");
  }
}

function renderRelease(release) {
  const decision = release?.release_decision || "unknown";
  const approved = recommendationsApproved(release);
  const expired = release?.blockers?.some((item) => item.code === "RELEASE_EXPIRED");
  const status = document.getElementById("release-status");
  status.textContent = approved ? "允许发布" : decision === "blocked" ? "暂停" : "未知";
  status.className = `status-token status-${approved ? "healthy" : decision === "blocked" ? "critical" : "unknown"}`;
  document.getElementById("release-command-title").textContent = approved
    ? "推荐已通过发布检查"
    : decision === "blocked" ? "暂停发布推荐" : "发布状态未就绪";
  document.getElementById("release-command-copy").textContent = approved
    ? "覆盖、模型质量和概率门槛均已通过。"
    : decision === "blocked"
      ? expired ? "本次发布许可已过期。请刷新快照后重新检查。" : "至少一项发布条件尚未满足，请先查看检查详情。"
      : "发布推荐前，请先读取最新的发布检查。";

  const audit = release?.audit || {};
  document.getElementById("release-command-facts").innerHTML = `
    <span><b>${escapeHtml(audit.prediction_count ?? 0)}</b> 场比赛</span>
    <span><b>${escapeHtml(audit.high_confidence_count ?? 0)}</b> 个候选</span>
    <span><b>${percent(audit.average_confidence)}</b> 最高赛果概率</span>
  `;
  document.getElementById("release-summary").innerHTML = `
    <div><span>发布结论</span><strong>${({approved: "通过", blocked: "未通过", unknown: "未知"})[decision] || "未知"}</strong></div>
    <div><span>可发布</span><strong>${approved ? "是" : "否"}</strong></div>
    <div><span>未通过项</span><strong>${escapeHtml((release?.blockers || []).length)}</strong></div>
    <div><span>提示项</span><strong>${escapeHtml((release?.warnings || []).length)}</strong></div>
  `;

  const blockers = document.getElementById("release-blockers");
  document.getElementById("release-message-count").textContent = release ? String((release.blockers || []).length) : "未就绪";
  if (release?.blockers?.length) {
    blockers.innerHTML = release.blockers.map((item) => signalMarkup(item, "critical")).join("");
  } else if (approved) {
    setEmpty(blockers, "发布检查通过", "发布时请保留使用说明。");
  } else {
    setEmpty(blockers, "发布检查文件未就绪", "请通过站点导出生成发布检查文件。");
  }

  const disclosure = release?.risk_disclosure;
  if (disclosure) {
    const copy = {
      "Model picks are probabilistic decision support, not guaranteed outcomes or financial advice.": "模型概率仅供分析参考，不保证比赛结果，也不构成财务建议。",
      "Do not publish picks when release_decision is blocked.": "发布检查未通过时，暂停发布推荐。",
      "Do not present confidence as certainty.": "最高赛果概率不等于结果确定性。",
      "Show bankroll and loss-limit guidance outside the model output.": "模型输出之外应保留资金和损失限额说明。",
      "Keep human review in the publishing workflow for paid or public recommendations.": "公开或付费推荐应保留人工复核。",
    };
    document.getElementById("risk-disclosure").innerHTML = `
      <p>${escapeHtml(copy[disclosure.summary] || disclosure.summary)}</p>
      <ul>${(disclosure.responsible_use || []).map((item) => `<li>${escapeHtml(copy[item] || item)}</li>`).join("")}</ul>
    `;
  }
}

function populateCompetitionFilter(predictions) {
  const select = document.getElementById("competition-filter");
  select.innerHTML = '<option value="all">全部赛事</option>';
  [...new Set(predictions.map((item) => item.competition).filter(Boolean))]
    .sort()
    .forEach((competition) => {
      const option = document.createElement("option");
      option.value = competition;
      option.textContent = competition;
      select.appendChild(option);
    });
}

function filteredPredictions() {
  const query = document.getElementById("prediction-search").value.trim().toLowerCase();
  const competition = document.getElementById("competition-filter").value;
  const confidence = Number(document.getElementById("confidence-filter").value);
  const sort = document.getElementById("prediction-sort").value;
  const filtered = state.predictions.filter((item) => {
    const teams = `${item.home_team || ""} ${item.away_team || ""}`.toLowerCase();
    return (!query || teams.includes(query))
      && (competition === "all" || item.competition === competition)
      && (finiteNumber(item.confidence) ?? 0) >= confidence;
  });
  return filtered.sort((left, right) => {
    if (sort === "confidence") return (finiteNumber(right.confidence) ?? -1) - (finiteNumber(left.confidence) ?? -1);
    if (sort === "uncertainty") return (finiteNumber(left.uncertainty) ?? 2) - (finiteNumber(right.uncertainty) ?? 2);
    return new Date(left.utc_date || 0) - new Date(right.utc_date || 0);
  });
}

function probabilityMarkup(probabilities = {}) {
  const rows = [
    ["主胜", probabilities.home_win ?? probabilities.Win, "home"],
    ["平局", probabilities.draw ?? probabilities.Draw, "draw"],
    ["客胜", probabilities.away_win ?? probabilities.Loss, "away"],
  ];
  if (rows.every(([, value]) => finiteNumber(value) === null)) return "";
  return `<div class="probability-table" role="group" aria-label="主胜、平局、客胜概率，各使用 0–100% 坐标">
    <div class="probability-axis" aria-hidden="true"><span>0</span><span>50</span><span>100%</span></div>
    ${rows.map(([label, value, outcome]) => {
    const raw = finiteNumber(value);
    const numeric = raw === null ? null : Math.max(0, Math.min(1, raw));
    return `
      <div class="probability-row probability-${outcome}" aria-label="${label}: ${percent(numeric, 1)}">
        <span>${label}</span>
        <i aria-hidden="true"><b style="--probability:${(numeric ?? 0) * 100}%"></b></i>
        <strong>${percent(numeric, 1)}</strong>
      </div>`;
  }).join("")}</div>`;
}

function evidenceMarkup(item) {
  const margin = finiteNumber(item.probability_margin);
  const explanation = item.explanation || {};
  const drivers = explanation.available === true && Array.isArray(explanation.top_drivers)
    ? explanation.top_drivers
    : [];
  const driverRows = drivers.map((driver) => {
    const contribution = finiteNumber(driver.contribution);
    const signed = contribution === null
      ? "—"
      : `${contribution >= 0 ? "+" : ""}${contribution.toFixed(2)}`;
    const label = String(driver.feature || "未知特征").replaceAll("_", " ");
    return `<li><span>${escapeHtml(label)}</span><b class="${contribution !== null && contribution < 0 ? "driver-negative" : ""}">${escapeHtml(signed)} logit</b></li>`;
  }).join("");
  const fullDrivers = Array.isArray(explanation.drivers) ? explanation.drivers : [];
  const bias = finiteNumber(explanation.bias_contribution);
  const omitted = finiteNumber(explanation.omitted_contribution);
  const rawMargin = finiteNumber(explanation.logit_margin);
  const signedMargin = value => value === null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(6)}`;
  const closureMarkup = fullDrivers.length && bias !== null && omitted !== null && rawMargin !== null
    ? `<p>以下数值比较最高类别与次高类别的模型原始分数，不是概率百分点或因果影响。</p>
       <dl class="explanation-closure">
         <dt>基线差</dt><dd>${escapeHtml(signedMargin(bias))}</dd>
         <dt>以上三项贡献合计</dt><dd>${escapeHtml(signedMargin(drivers.reduce((total, d) => total + (finiteNumber(d.contribution) ?? 0), 0)))}</dd>
         <dt>其余特征贡献合计</dt><dd>${escapeHtml(signedMargin(omitted))}</dd>
         <dt>原始分数差</dt><dd>${escapeHtml(signedMargin(rawMargin))}</dd>
       </dl>
       <details><summary>查看全部 ${fullDrivers.length} 项贡献</summary>
         <p>窄屏可左右滚动表格；键盘聚焦表格后可用方向键滚动。</p>
         <div class="explanation-table-scroll" tabindex="0" role="region" aria-label="全部特征贡献表">
           <table><caption>原始分数差的逐项分解，显示值四舍五入</caption>
             <thead><tr><th scope="col">特征</th><th scope="col">赛前取值</th><th scope="col">贡献</th></tr></thead>
             <tbody>${fullDrivers.map(d => `<tr><th scope="row">${escapeHtml(String(d.feature))}</th><td>${escapeHtml(finiteNumber(d.raw_value)?.toFixed(6) ?? "—")}</td><td>${escapeHtml(signedMargin(finiteNumber(d.contribution)))}</td></tr>`).join("")}</tbody>
           </table>
         </div>
       </details><p>TreeSHAP 使用当前模型的树路径统计作为解释参照。</p>`
    : (driverRows ? '<p>该记录未提供完整贡献分解。</p>' : "");
  const predictedAt = item.predicted_at ? formatDateTime(item.predicted_at) : "未就绪";
  const forecastId = String(item.forecast_id || "unavailable").slice(0, 8);
  const modelSha256 = String(item.model?.model_sha256 || "unavailable").slice(0, 8);
  const estimator = [item.model?.estimator, item.model?.estimator_version]
    .filter((value) => typeof value === "string" && value.trim())
    .map((value) => value.trim())
    .join(" ");
  return `
    <details class="prediction-evidence">
      <summary>查看预测依据</summary>
      <div class="evidence-facts">
        <span>领先第二赛果 <b>${percent(margin)}</b></span>
        <span>概率含义 <b>最高赛果概率</b></span>
        <span>预测标识 <b>${escapeHtml(forecastId)}</b></span>
        <span>模型标识 <b>${escapeHtml(modelSha256)}</b></span>
        ${estimator ? `<span>模型算法 <b>${escapeHtml(estimator)}</b></span>` : ""}
        <span>预测时间 <b>${escapeHtml(predictedAt)}</b></span>
      </div>
      ${driverRows
        ? `<p>${escapeHtml(explanation.predicted_vs_runner_up || "最高概率与第二赛果的比较")}</p><ol>${driverRows}</ol>`
        : '<p>该模型暂未提供特征贡献解释。</p>'}
      ${closureMarkup}
    </details>
  `;
}

function renderPredictions() {
  const predictions = filteredPredictions();
  const container = document.getElementById("predictions");
  if (!state.predictionsAvailable) {
    document.getElementById("fixture-result-count").textContent = "预测文件未就绪";
    setEmpty(container, "预测未就绪", "请恢复预测数据后查看比赛。");
    return;
  }
  document.getElementById("fixture-result-count").textContent = `显示 ${predictions.length} / ${state.predictions.length} 场比赛`;
  if (!predictions.length) {
    setEmpty(container, state.predictions.length ? "没有符合筛选的比赛" : "暂无待赛比赛",
      state.predictions.length ? "请调整或清除筛选。" : "请查看下方运行诊断，或在赛程更新后刷新快照。");
    return;
  }
  container.innerHTML = predictions.map((item) => {
    const confidence = Math.max(0, Math.min(1, finiteNumber(item.confidence) ?? 0));
    const coldStart = item.data_quality?.is_cold_start === true;
    const validation = finiteNumber(item.model?.validation_accuracy);
    const baseline = finiteNumber(item.model?.baseline_accuracy);
    const lift = validation !== null && baseline !== null ? validation - baseline : null;
    return `
      <article class="fixture-card">
        <div class="fixture-meta">
          <span>${escapeHtml(item.competition || "未标注赛事")}</span>
          <time datetime="${escapeHtml(item.utc_date || "")}">${escapeHtml(formatKickoff(item.utc_date))}</time>
        </div>
        <div class="matchup">
          <h3>${escapeHtml(item.home_team || "主队待定")} <span>对</span> ${escapeHtml(item.away_team || "客队待定")}</h3>
          <div class="confidence-value">
            <strong>${percent(confidence)}</strong><span>最高赛果概率</span>
          </div>
        </div>
        <div class="model-call">
          <span>最高概率赛果</span><strong>${escapeHtml(outcomeLabel(item.prediction))}</strong>
        </div>
        ${probabilityMarkup(item.probabilities || {})}
        ${evidenceMarkup(item)}
        <div class="quality-flags">
          <span class="${coldStart ? "flag-warning" : ""}">${coldStart ? "历史不足" : "已有历史"}</span>
          <span>不确定性 ${percent(item.uncertainty)}</span>
          <span>验证提升 ${lift === null ? "—" : `${lift >= 0 ? "+" : ""}${percent(lift, 1)}`}</span>
        </div>
      </article>
    `;
  }).join("");
}

function renderRecommended(preset, release) {
  const container = document.getElementById("recommended-bets");
  if (!preset) {
    setEmpty(container, "候选列表未就绪", "候选列表文件未能加载。");
    return;
  }
  if (!recommendationsApproved(release) || preset?.can_publish_recommendations !== true) {
    setEmpty(container, "暂停显示推荐", "发布检查未通过或尚未就绪。赛果概率仅供诊断与分析。");
    return;
  }
  const question = (preset?.questions || []).find((item) => item.id === "recommended_bets_today");
  const threshold = finiteNumber(release.thresholds?.high_confidence) ?? 0.6;
  const items = (Array.isArray(question?.items) ? question.items : []).filter((item) => {
    const confidence = finiteNumber(item?.confidence);
    return confidence !== null && confidence >= threshold && confidence <= 1;
  });
  if (!items.length) {
    setEmpty(container, "暂无推荐候选", "仅显示通过发布检查并达到概率门槛的比赛。");
    return;
  }
  container.innerHTML = items.map((item) => `
    <article class="compact-row">
      <div><strong>${escapeHtml(item.match)}</strong><span>${escapeHtml(item.competition || "赛事待定")}</span></div>
      <div class="compact-result"><b>${escapeHtml(outcomeLabel(item.prediction))}</b><span>${percent(item.confidence, 1)}</span></div>
    </article>
  `).join("");
}

function renderScores(scorePayload) {
  const container = document.getElementById("scores");
  if (!scorePayload) {
    setEmpty(container, "赛果数据未就绪", "赛果文件未能加载。");
    return;
  }
  const scores = scorePayload?.scores || [];
  if (!scores.length) {
    setEmpty(container, "暂无最新赛果", "数据来源更新后会显示最新赛果。");
    return;
  }
  container.innerHTML = scores.slice(0, 12).map((item) => `
    <article class="compact-row score-row">
      <div><strong>${escapeHtml(item.home_team || "主胜")}</strong><span>${escapeHtml(item.away_team || "客胜")}</span></div>
      <div class="scoreline"><b>${escapeHtml(item.home_score ?? "–")}</b><i>:</i><b>${escapeHtml(item.away_score ?? "–")}</b></div>
    </article>
  `).join("");
}

function renderErrors(predictionPayload, scorePayload, artifactFailures) {
  const container = document.getElementById("prediction-errors");
  const errors = predictionPayload?.errors || [];
  const refreshErrors = predictionPayload?.refresh_errors || [];
  const scoreErrors = scorePayload?.errors || [];
  const signals = [
    ...artifactFailures,
    ...errors.map((item) => ({
      level: "critical",
      code: `${item.competition || "UNKNOWN"} / ${item.code || "PREDICTION_FAILED"}`,
      message: item.message,
    })),
    ...refreshErrors.map((item) => ({
      level: "warning",
      code: `${item.competition || "UNKNOWN"} / REFRESH / ${item.code || "TRAINING_FAILED"}`,
      message: item.message,
    })),
    ...scoreErrors.map((item) => ({
      level: "warning",
      code: item.code || "SCORE_FETCH_FAILED",
      message: item.message,
    })),
  ];
  document.getElementById("diagnostic-message-count").textContent = String(signals.length);
  if (!signals.length) {
    setEmpty(container, "暂无运行问题", "所选赛事未报告预测或模型刷新错误。");
    return;
  }
  container.innerHTML = signals.map((item) => signalMarkup(item, "critical")).join("");
}

function settledValue(results, index) {
  return results[index].status === "fulfilled" ? results[index].value.payload : null;
}

function settledRaw(results, index) {
  return results[index].status === "fulfilled" ? results[index].value.raw : null;
}

function artifactFailures(paths, results) {
  return results.flatMap((result, index) => {
    if (result.status === "fulfilled") return [];
    const name = paths[index].toUpperCase();
    return [{
      level: "critical",
      code: `${name}_ARTIFACT_UNAVAILABLE`,
      message: `${paths[index]}.json 未能加载：${result.reason?.message || "未知错误"}`,
    }];
  });
}

async function sha256(raw) {
  if (!globalThis.crypto?.subtle || typeof TextEncoder === "undefined") return null;
  const bytes = new TextEncoder().encode(raw);
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
}

async function snapshotFailures(paths, values, rawValues) {
  const manifest = values[0];
  if (!manifest) return [];
  const expected = manifest.generated_at;
  if (typeof expected !== "string" || !Number.isFinite(Date.parse(expected)) || Date.parse(expected) > Date.now()) {
    return [{
      level: "critical",
      code: "MANIFEST_SNAPSHOT_MISMATCH",
      message: "快照时间无效或晚于当前时间，无法验证快照。",
    }];
  }
  const manifestKeys = {
    operations: "operations",
    release: "release_governance",
    predictions: "predictions",
    preset_questions: "preset_questions",
    scores: "scores",
  };
  const failures = [];
  for (let index = 1; index < values.length; index += 1) {
    const value = values[index];
    if (value === null) continue;
    const name = paths[index].toUpperCase();
    if (value.generated_at !== expected) {
      failures.push({
        level: "critical",
        code: `${name}_SNAPSHOT_MISMATCH`,
        message: `${paths[index]}.json 的快照时间为 ${value.generated_at || "未知"}，应为 ${expected}。`,
      });
      continue;
    }
    const metadata = manifest.artifacts?.[manifestKeys[paths[index]]];
    const raw = rawValues[index];
    const bytes = typeof raw === "string" && typeof TextEncoder !== "undefined"
      ? new TextEncoder().encode(raw)
      : null;
    const digest = typeof raw === "string" ? await sha256(raw) : null;
    if (
      !metadata
      || bytes === null
      || digest === null
      || metadata.size_bytes !== bytes.byteLength
      || metadata.sha256 !== digest
    ) {
      failures.push({
        level: "critical",
        code: `${name}_SNAPSHOT_MISMATCH`,
        message: `${paths[index]}.json 的摘要或文件大小与快照清单不符。`,
      });
    }
  }
  return failures;
}

async function init() {
  const paths = ["manifest", "operations", "release", "predictions", "preset_questions", "scores"];
  const results = await Promise.allSettled(paths.map((name) => loadJson(`./data/${name}.json`)));
  const values = paths.map((_, index) => settledValue(results, index));
  const rawValues = paths.map((_, index) => settledRaw(results, index));
  const manifest = values[0];
  const unavailable = artifactFailures(paths, results);
  const inconsistent = await snapshotFailures(paths, values, rawValues);
  const failures = [...unavailable, ...inconsistent];
  const snapshotUsable = Boolean(manifest) && failures.length === 0;
  const operations = snapshotUsable ? values[1] : null;
  const release = snapshotUsable ? values[2] : null;
  const predictionPayload = snapshotUsable ? values[3] : null;
  const preset = snapshotUsable ? values[4] : null;
  const scores = snapshotUsable ? values[5] : null;
  state.predictionsAvailable = Boolean(predictionPayload);
  state.predictions = predictionPayload?.predictions || [];

  const refreshPublishing = () => {
    const activeRelease = currentRelease(release);
    const stale = Date.now() >= Date.parse(manifest?.generated_at) + RELEASE_MAX_AGE_MS;
    document.getElementById("data-health").textContent = manifest
    ? snapshotUsable
      ? `快照 ${formatDateTime(manifest.generated_at)} · ${stale ? "已过期，请刷新" : `${manifest.prediction_count ?? 0} 场比赛`}`
      : `快照 ${formatDateTime(manifest.generated_at)} · 完整性检查未通过`
    : "快照清单未就绪，各区数据可能不一致";
    renderMetrics(state.predictions, operations, state.predictionsAvailable, activeRelease);
    renderRelease(activeRelease);
    renderRecommended(preset, activeRelease);
  };
  document.getElementById("footer-run-id").textContent = manifest?.run_id
    ? `运行 ${manifest.run_id}`
    : "旧版或不完整的快照";

  renderCoverage(snapshotUsable ? manifest : null, operations);
  renderOperations(operations);
  populateCompetitionFilter(state.predictions);
  renderPredictions();
  refreshPublishing();
  clearTimeout(state.releaseTimer);
  const expires = Date.parse(release?.expires_at);
  if (expires > Date.now() && expires <= Date.now() + RELEASE_MAX_AGE_MS) {
    state.releaseTimer = setTimeout(refreshPublishing, expires - Date.now() + 1);
  }
  renderScores(scores);
  renderErrors(predictionPayload, scores, failures);
  document.body.classList.add("is-ready");
}

["prediction-search", "competition-filter", "confidence-filter", "prediction-sort"].forEach((id) => {
  document.getElementById(id).addEventListener(id === "prediction-search" ? "input" : "change", renderPredictions);
});

init().catch((error) => {
  document.getElementById("data-health").textContent = `页面初始化失败： ${error.message}`;
  document.body.classList.add("is-ready");
});
