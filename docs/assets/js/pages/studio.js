"use strict";
(function () {
  const shell = window.TouchlineShell, create = shell.element;
  const list = document.getElementById("studio-destinations");
  const search = document.getElementById("studio-query"), group = document.getElementById("studio-group");
  for (const value of new Set(shell.destinations.map((row) => row.group))) {
    const option = create("option", value); option.value = value; group.append(option);
  }
  function render() {
    const query = search.value.toLocaleLowerCase().trim();
    const rows = shell.destinations.filter((row) => (!group.value || row.group === group.value) && `${row.title} ${row.group} ${row.keywords}`.toLocaleLowerCase().includes(query));
    list.replaceChildren(...rows.map((row) => {
      const link = create("a", undefined, "studio-card"); link.href = "./" + row.href;
      link.append(create("small", row.group), create("h3", row.title), create("p", row.keywords), create("span", "打开分析 →")); return link;
    }));
    document.getElementById("studio-count").textContent = rows.length ? `${rows.length} 个可用入口` : "没有匹配入口，调整搜索或分类后重试。";
  }
  function views() {
    const {saved, recent} = shell.views();
    for (const [id, rows, empty] of [["studio-saved", saved, "还没有收藏。在任意分析页面调整筛选后，点顶部“保存视图”。"], ["studio-recent", recent, "还没有最近访问。打开一个分析入口即可开始。"]]) {
      const nodes = rows.map((row) => {
        const item = create("div", undefined, "saved-view-row"), link = create("a", row.title); link.href = "./"+row.href;
        item.append(link);
        if (id === "studio-saved") {
          const remove = create("button", "移除"); remove.type = "button"; remove.setAttribute("aria-label", `移除收藏：${row.title}`);
          remove.addEventListener("click", () => shell.remove(row.href)); item.append(remove);
        }
        return item;
      });
      document.getElementById(id).replaceChildren(...(nodes.length ? nodes : [create("p", empty, "empty-state")]));
    }
    document.getElementById("studio-saved-count").textContent = `${saved.length} 个视图`;
  }
  search.addEventListener("input", render); group.addEventListener("change", render);
  document.getElementById("clear-recent").addEventListener("click", () => shell.clearRecent());
  document.getElementById("export-workspace").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify({schema:"Touchline.SavedViews.v1", ...shell.views()},null,2)+"\n"], {type:"application/json"});
    const url = URL.createObjectURL(blob), link = create("a"); link.href=url; link.download="touchline-workspace.json"; link.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
  });
  document.getElementById("import-workspace").addEventListener("change", async (event) => {
    const input = event.currentTarget, file = input.files[0], status = document.getElementById("import-workspace-status");
    if (!file) return;
    let backup;
    try { backup = JSON.parse(await file.text()); }
    catch { status.textContent = "文件无法读取或不是有效 JSON，未导入。"; input.value = ""; return; }
    status.textContent = shell.importSaved(backup); input.value = "";
  });
  document.addEventListener("touchline:workspace", views); render(); views();
})();
