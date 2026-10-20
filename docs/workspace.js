"use strict";

// Shared page composition. Scientific renderers keep their own typed data contracts.
(function () {
  if (typeof document === "undefined") return;
  if (document.body.dataset.page === "index" && ["historical-validation","fixtures-title","model-health","operations-title"].includes(location.hash.slice(1))) {
    const url = new URL(location.href); url.pathname = url.pathname.replace(/[^/]*$/, "predictions.html"); location.replace(url.href); return;
  }
  const tabs = [...document.querySelectorAll('.workspace-tabs [role="tab"]')];
  let knownSearch = location.search;
  const currentPanel = () => tabs.find((tab) => tab.getAttribute("aria-selected") === "true")?.getAttribute("aria-controls");
  function activate(id, push = false) {
    const target = document.getElementById(id), panel = target?.closest('[role="tabpanel"]');
    const selected = tabs.find((tab) => tab.getAttribute("aria-controls") === panel?.id) || tabs[0];
    if (!selected) return;
    for (const tab of tabs) {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active)); tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute("aria-controls")).hidden = !active;
    }
    if (id === "representation-paired" && target) target.open = true;
    if (push) {
      const url = new URL(location.href); url.hash = selected.getAttribute("aria-controls");
      history.pushState(null, "", url); knownSearch = location.search;
    }
    document.dispatchEvent(new CustomEvent("touchline:view", {detail: {panel: currentPanel()}}));
  }
  window.TouchlineWorkspace = {active: (id) => !tabs.length || currentPanel() === id};
  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => activate(tab.getAttribute("aria-controls"), true));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault(); tabs[next].focus(); activate(tabs[next].getAttribute("aria-controls"), true);
    });
  });
  activate(location.hash.slice(1));
  window.addEventListener("hashchange", () => activate(location.hash.slice(1)));
  window.addEventListener("popstate", () => {
    // A previous tab may carry a different provider-specific selection. Reload that
    // saved URL so each renderer restores validated options, never stale controls.
    if (location.search !== knownSearch) { location.reload(); return; }
    activate(location.hash.slice(1));
  });
  // Offer a native, focus-contained reader without duplicating interactive SVGs.
  const reader = document.createElement("dialog"); reader.className = "chart-reader";
  reader.setAttribute("aria-labelledby", "chart-reader-title");
  reader.innerHTML = '<header><h2 id="chart-reader-title">图表阅读</h2><button type="button" autofocus>关闭 ×</button></header><p class="context-note">图内操作与当前筛选保持一致。细节较密时可滚动查看，按 Escape 关闭。</p><div class="chart-reader-body"></div>';
  document.body.append(reader);
  let chart, anchor, opener;
  function closeReader() { reader.close(); }
  reader.querySelector("button").addEventListener("click", closeReader);
  reader.addEventListener("keydown", (event) => {
    if (event.key !== "Tab") return;
    const controls = [...reader.querySelectorAll('button:not(:disabled),a[href],select:not(:disabled),input:not(:disabled),[tabindex="0"]')].filter((element) => element.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
  });
  reader.addEventListener("close", () => {
    if (chart && anchor) anchor.replaceWith(chart);
    opener?.focus(); chart = null; anchor = null;
  });
  document.querySelectorAll('.interactive-chart, .experiment-chart').forEach((container) => {
    const button = document.createElement("button"); button.type = "button";
    button.className = "chart-open"; button.textContent = "放大图表 ↗"; button.setAttribute("aria-haspopup", "dialog");
    button.setAttribute("aria-label", `放大${container.closest("section")?.querySelector("h2")?.textContent || "当前图表"}`);
    container.before(button);
    button.addEventListener("click", () => {
      chart = container; opener = button; anchor = document.createComment("chart reader position");
      container.replaceWith(anchor); reader.querySelector(".chart-reader-body").replaceChildren(container);
      reader.querySelector("h2").textContent = button.getAttribute("aria-label"); reader.showModal();
    });
  });
})();
