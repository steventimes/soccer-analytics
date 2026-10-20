"use strict";

// One directed graph for the pitch, matrix and table; IDs stay within a match.
function passingGraph(rows) {
  const players = new Map(), edges = new Map();
  const player = (id) => {
    if (!players.has(id)) players.set(id, {id, outgoing: 0, incoming: 0, location: [0, 0, 0]});
    return players.get(id);
  };
  let completed = 0;
  rows.forEach((row) => {
    completed += row.completed_passes;
    Object.entries(row.nodes).forEach(([id, values]) => {
      const p = player(id);
      values.forEach((v, i) => { p.location[i] += v; });
    });
    row.edges.forEach(([source, target, count]) => {
      player(source).outgoing += count;
      player(target).incoming += count;
      const key = `${source}:${target}`;
      if (!edges.has(key)) edges.set(key, {source, target, count: 0});
      edges.get(key).count += count;
    });
  });
  const roster = [...players.values()].sort((a, b) =>
    (b.incoming + b.outgoing) - (a.incoming + a.outgoing) || a.id.localeCompare(b.id));
  roster.forEach((p, i) => {
    p.number = i + 1;
    p.position = p.location[2] ? p.location.slice(0, 2).map((v) => v / p.location[2]) : null;
    delete p.location;
  });
  const links = [...edges.values()].sort((a, b) => b.count - a.count ||
    a.source.localeCompare(b.source) || a.target.localeCompare(b.target));
  const total = links.reduce((n, e) => n + e.count, 0);
  return {players: roster, edges: links, total, unlinked: completed - total};
}

if (typeof module !== "undefined" && module.exports) module.exports = {passingGraph};
