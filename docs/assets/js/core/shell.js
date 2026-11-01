"use strict";

// Application navigation and browser-local views. Analytical state stays in URLs.
(function () {
  const destinations = [
    ["index.html", "数据工作台", "工作空间", "公开摘要 比赛 来源"],
    ["studio.html", "分析中心", "工作空间", "查找 收藏 最近访问"],
    ["matches.html#match-explorer", "比赛事件分析", "比赛研究", "射门 传球 网络 矩阵 时段"],
    ["matches.html#fifa-offers", "FIFA 接应分析", "比赛研究", "接球 跑位 接应报告"],
    ["matches.html#case", "战术案例", "比赛研究", "球队 对比 案例"],
    ["entities.html#comparisons", "球队与球员比较", "对象研究", "每90分钟 差值 对比"],
    ["entities.html#style-retrieval", "球风与相似性", "对象研究", "风格 检索 表示"],
    ["models.html#model-lab", "赛果模型实验", "模型研究", "时间留出 评级 校准 比分"],
    ["models.html#representations", "事件表示学习", "模型研究", "GRU MLP 迁移 自监督"],
    ["models.html#vaep-calibration", "动作概率校准", "模型研究", "VAEP Brier 可靠性"],
    ["models.html#xt-grid", "xT 空间网格", "模型研究", "威胁 重采样 不确定性"],
    ["predictions.html", "预测与发布检查", "模型研究", "赛前 概率 时效 历史验证"],
    ["library.html#figures", "研究图表库", "数据与资料", "图集 导出 来源"],
    ["library.html#coverage", "数据覆盖", "数据与资料", "联赛 赛季 来源 缺失"],
    ["library.html#methods", "公开方法", "数据与资料", "机构 参数 模型"],
    ["library.html#literature", "论文与证据", "数据与资料", "文献 参考 已读"],
  ].map(([href, title, group, keywords]) => ({href, title, group, keywords}));
  const pages = new Set(destinations.map((item) => item.href.split("#")[0]));
  const key = "touchline.workspace.v1";
  const element = (tag, value, className) => {
    const node = document.createElement(tag);
    if (value !== undefined) node.textContent = value;
    if (className) node.className = className;
    return node;
  };
  function safeHref(value) {
    if (typeof value !== "string" || value.length > 3000) return null;
    try {
      const url = new URL(value, location.href);
      const directory = new URL("./", location.href);
      const filename = url.pathname.slice(directory.pathname.length);
      if (url.origin !== directory.origin || !url.pathname.startsWith(directory.pathname) || !pages.has(filename)) return null;
      return filename + url.search + url.hash;
    } catch { return null; }
  }
  function cleanViews(value, limit) {
    if (!Array.isArray(value)) return [];
    const seen = new Set();
    return value.filter((row) => {
      if (!row || typeof row.title !== "string" || !row.title.trim() || !safeHref(row.href) || seen.has(safeHref(row.href))) return false;
      seen.add(safeHref(row.href)); return true;
    }).slice(0, limit).map((row) => ({href:safeHref(row.href), title:row.title.slice(0, 80)}));
  }
  let damagedStorage = false;
  function load() {
    try {
      const raw = localStorage.getItem(key);
      let stored;
      try { stored = JSON.parse(raw || "{}"); }
      catch { damagedStorage = true; return {saved:[], recent:[], collapsed:false, compact:false}; }
      if (!stored || typeof stored !== "object" || Array.isArray(stored)) { damagedStorage = true; return {saved:[], recent:[], collapsed:false, compact:false}; }
      const saved = cleanViews(stored.saved ?? [], 40), recent = cleanViews(stored.recent ?? [], 8);
      if (("saved" in stored && (!Array.isArray(stored.saved) || saved.length !== stored.saved.length)) || ("recent" in stored && (!Array.isArray(stored.recent) || recent.length !== stored.recent.length))) { damagedStorage = true; return {saved:[], recent:[], collapsed:false, compact:false}; }
      damagedStorage = false;
      return {saved, recent, collapsed:stored?.collapsed === true, compact:stored?.compact === true};
    } catch { return {saved:[], recent:[], collapsed:false, compact:false}; }
  }
  let state = load();
  const currentHref = () => safeHref(location.href);
  function currentTitle() {
    const panel = document.querySelector('.workspace-tabs [aria-selected="true"]');
    const title = document.querySelector("h1").textContent.trim();
    return (panel ? `${title} · ${panel.textContent.trim()}` : title).slice(0, 80);
  }
  const announce = (message) => { document.getElementById("shell-status").textContent = message; };
  function persist(message) {
    if (damagedStorage) announce("本地保存记录无法读取，原记录未覆盖；本次会话的视图关闭页面后不会保留。");
    else try { localStorage.setItem(key, JSON.stringify(state)); if (message) announce(message); }
    catch { announce("浏览器未允许保存。本次会话仍可使用，关闭页面后不会保留。"); }
    renderHistory(); document.dispatchEvent(new CustomEvent("touchline:workspace"));
  }
  function visit() {
    const href = currentHref();
    if (!href) return;
    state.recent = [{href, title:currentTitle()}, ...state.recent.filter((row) => row.href !== href)].slice(0, 8);
    persist();
  }
  function renderHistory() {
    const container = document.getElementById("workspace-history-links");
    container.replaceChildren(...state.recent.slice(0, 6).map((row) => {
      const link = element("a", row.title); link.href = "./" + row.href;
      if (row.href === currentHref()) link.setAttribute("aria-current", "location");
      return link;
    }));
  }
  function applyPreferences() {
    document.body.classList.toggle("sidebar-collapsed", state.collapsed);
    document.documentElement.dataset.density = state.compact ? "compact" : "comfortable";
    const density = document.getElementById("density-toggle");
    density.setAttribute("aria-pressed", String(state.compact));
    density.textContent = state.compact ? "舒适视图" : "紧凑视图";
  }
  const sidebar = document.querySelector(".workspace-sidebar");
  const toggle = document.getElementById("sidebar-toggle");
  const narrow = matchMedia("(max-width: 760px)");
  function syncNavigation() {
    const open = document.body.classList.contains("navigation-open");
    sidebar.inert = narrow.matches && !open;
    toggle.setAttribute("aria-expanded", String(narrow.matches ? open : !state.collapsed));
    document.getElementById("navigation-shade").hidden = !narrow.matches || !open;
  }
  toggle.addEventListener("click", () => {
    if (narrow.matches) document.body.classList.toggle("navigation-open");
    else { state.collapsed = !state.collapsed; applyPreferences(); persist(); }
    syncNavigation();
  });
  function closeNavigation() { document.body.classList.remove("navigation-open"); syncNavigation(); toggle.focus(); }
  document.getElementById("navigation-shade").addEventListener("click", closeNavigation);
  sidebar.addEventListener("keydown", (event) => { if (event.key === "Escape" && narrow.matches) closeNavigation(); });
  sidebar.addEventListener("click", (event) => { if (event.target.closest("a") && narrow.matches) { document.body.classList.remove("navigation-open"); syncNavigation(); } });
  narrow.addEventListener("change", () => { document.body.classList.remove("navigation-open"); syncNavigation(); });
  document.getElementById("density-toggle").addEventListener("click", () => { state.compact = !state.compact; applyPreferences(); persist(); });

  const search = document.getElementById("navigation-search");
  const searchInput = document.getElementById("navigation-query");
  function renderSearch() {
    const terms = searchInput.value.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean);
    const matches = destinations.filter((row) => terms.every((term) => `${row.title} ${row.group} ${row.keywords}`.toLocaleLowerCase().includes(term)));
    document.getElementById("navigation-results").replaceChildren(...matches.map((row) => {
      const link = element("a", undefined, "navigation-result"); link.href = "./" + row.href;
      link.append(element("span", row.group), element("strong", row.title), element("small", row.keywords));
      link.addEventListener("click", () => search.close()); return link;
    }));
    document.getElementById("navigation-count").textContent = matches.length ? `${matches.length} 个分析入口` : "没有匹配入口。试试“传球”“模型”或“来源”。";
  }
  function openSearch() { searchInput.value = ""; renderSearch(); search.showModal(); searchInput.focus(); }
  document.getElementById("navigation-search-open").addEventListener("click", openSearch);
  searchInput.addEventListener("input", renderSearch);
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") { event.preventDefault(); if (!document.querySelector("dialog[open]")) openSearch(); }
    if (event.key === "Escape" && document.body.classList.contains("navigation-open")) closeNavigation();
  });
  for (const dialog of document.querySelectorAll(".shell-dialog")) {
    dialog.querySelector(".dialog-close").addEventListener("click", () => dialog.close());
    dialog.addEventListener("keydown", (event) => {
      // Search inputs consume Escape to clear text before the native dialog closes.
      if (event.key === "Escape") { event.preventDefault(); dialog.close(); }
    });
  }
  const saveDialog = document.getElementById("save-view-dialog");
  document.getElementById("save-view").addEventListener("click", () => {
    document.getElementById("save-view-message").textContent = "";
    document.getElementById("saved-view-name").value = state.saved.find((row) => row.href === currentHref())?.title || currentTitle();
    saveDialog.showModal(); document.getElementById("saved-view-name").select();
  });
  document.getElementById("save-view-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const title = document.getElementById("saved-view-name").value.trim(), href = currentHref();
    if (!title || !href) { document.getElementById("save-view-message").textContent = "请输入视图名称。"; return; }
    if (state.saved.some((row) => row.title === title && row.href !== href)) {
      document.getElementById("save-view-message").textContent = "已有同名视图，请使用其他名称。"; return;
    }
    if (state.saved.length >= 40 && !state.saved.some((row) => row.href === href)) {
      document.getElementById("save-view-message").textContent = "已保存 40 个视图。请先在分析中心移除不需要的收藏，再保存新视图。";
      return;
    }
    state.saved = [{title:title.slice(0,80), href}, ...state.saved.filter((row) => row.href !== href)].slice(0,40);
    persist("已保存当前视图，可在分析中心打开。"); saveDialog.close();
  });
  window.TouchlineShell = {
    destinations, safeHref,
    views:() => ({saved:state.saved.map((row) => ({...row})), recent:state.recent.map((row) => ({...row}))}),
    importSaved(backup) {
      if (damagedStorage) return "本地记录损坏，未导入或覆盖原记录。";
      if (!backup || backup.schema !== "Touchline.SavedViews.v1" || !Array.isArray(backup.saved)) return "备份格式无效，未导入。";
      const merged = [...state.saved];
      for (const row of backup.saved) {
        const href = safeHref(row?.href), title = typeof row?.title === "string" ? row.title.trim() : "";
        if (!href || !title || title.length > 80) return "备份包含无效视图，未导入。";
        const sameName = merged.find((item) => item.title === title), sameLink = merged.find((item) => item.href === href);
        if ((sameName && sameName.href !== href) || (sameLink && sameLink.title !== title)) return `视图“${title}”的名称或链接与现有视图冲突，未导入。`;
        if (!sameName && !sameLink) merged.push({title, href});
      }
      if (merged.length > 40) return "合并后超过 40 个视图，请先移除部分收藏。";
      const next = {...state, saved:merged};
      try { localStorage.setItem(key, JSON.stringify(next)); }
      catch { return "浏览器未允许保存，未导入。"; }
      const count = merged.length - state.saved.length;
      state = next; renderHistory(); document.dispatchEvent(new CustomEvent("touchline:workspace"));
      return `已导入 ${count} 个收藏。相同视图保留；最近访问未导入。`;
    },
    remove(href) { state.saved = state.saved.filter((row) => row.href !== href); persist("已移除收藏视图。"); },
    clearRecent() { state.recent = []; persist("已清空最近访问。"); },
    element,
  };
  window.addEventListener("storage", (event) => { if (event.key === key) { state = load(); applyPreferences(); syncNavigation(); renderHistory(); document.dispatchEvent(new CustomEvent("touchline:workspace")); } });
  document.body.classList.add("shell-ready"); applyPreferences(); syncNavigation();
  window.addEventListener("hashchange", visit);
  document.addEventListener("touchline:view", visit);
  // Filter controls update the URL via replaceState; capture the final URL when saved.
  visit();
})();
