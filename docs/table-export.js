"use strict";

function displayedTableSnapshot(headings, rows, context = {}) {
  if (!Array.isArray(headings) || !Array.isArray(rows) || rows.some((row) => !Array.isArray(row) || row.length !== headings.length)) {
    throw new TypeError("Rows must match the displayed headings");
  }
  function value(cell) {
    if (cell === null || cell === undefined) return null;
    if (typeof cell === "number" && !Number.isFinite(cell)) throw new TypeError("Table numbers must be finite");
    if (!["number", "string", "boolean"].includes(typeof cell)) throw new TypeError("Unsupported table cell");
    return cell;
  }
  return {schema: "SoccerAnalytics.DisplayedTable.v1", precision: "displayed",
    row_count: rows.length, column_count: headings.length,
    columns: headings.map(value), rows: rows.map((row) => row.map(value)),
    context: Object.fromEntries(Object.entries(context).map(([key, cell]) => [key, value(cell)]))};
}

function displayedTableCsv(headings, rows, context = {}) {
  const snapshot = displayedTableSnapshot(headings, rows, context);
  function cell(value) {
    if (value === null || value === undefined) return '""';
    if (typeof value === "number" && !Number.isFinite(value)) throw new TypeError("CSV numbers must be finite");
    if (!["number", "string", "boolean"].includes(typeof value)) throw new TypeError("Unsupported CSV cell");
    let text = String(value);
    // Keep provider-supplied text inert when opened in a spreadsheet.
    if (typeof value === "string" && /^[\s\uFEFF]*[=+@-]/u.test(text)) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  }
  const metadata = Object.entries(snapshot.context);
  const values = [[...metadata.map(([key]) => key), ...snapshot.columns],
    ...snapshot.rows.map((row) => [...metadata.map(([, value]) => value), ...row])];
  return "\uFEFF" + values.map((row) => row.map(cell).join(",")).join("\r\n") + "\r\n";
}

if (typeof module !== "undefined") module.exports = {displayedTableCsv, displayedTableSnapshot};
