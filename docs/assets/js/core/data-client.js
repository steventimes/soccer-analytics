"use strict";

// One bounded transport for public JSON. Schema checks remain with each view.
function createDataClient({fetchData = (...args) => globalThis.fetch(...args), timeoutMs = 12000} = {}) {
  const pending = new Map(), remembered = new Map();
  function invalidate(path) { remembered.delete(path); }
  function cancel(path) {
    for (const [key, request] of pending) if (!path || request.path === path) {
      request.controller.abort(); pending.delete(key);
    }
    if (path) invalidate(path); else remembered.clear();
  }
  function read(path, options = {}) {
    if (options.signal?.aborted) return Promise.reject(new Error("读取已取消，可重试"));
    const cache = options.cache || "no-cache";
    const key = `${cache}:${path}`;
    if (options.remember && remembered.has(path)) return Promise.resolve(remembered.get(path));
    // A caller-owned signal owns its request; shared consumers use page cancellation.
    if (!options.signal && pending.has(key)) return pending.get(key).promise;
    const controller = new AbortController();
    let timer, abortListener;
    const deadline = new Promise((resolve, reject) => {
      abortListener = () => reject(new Error("读取已取消，可重试"));
      controller.signal.addEventListener("abort", abortListener, {once:true});
      timer = setTimeout(() => {
        reject(new Error("读取超时，请重试")); controller.abort();
      }, options.timeoutMs ?? timeoutMs);
    });
    const forwardAbort = () => controller.abort();
    if (options.signal) {
      options.signal.addEventListener("abort", forwardAbort, {once:true});
      if (options.signal.aborted) controller.abort();
    }
    const request = {path, controller}, requestKey = options.signal ? Symbol(path) : key;
    const transport = (async () => {
      const response = await fetchData(path, {cache, signal:controller.signal});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const raw = await response.text();
      return {payload:JSON.parse(raw), raw};
    })();
    request.promise = Promise.race([transport, deadline]).then((value) => {
      if (options.remember) remembered.set(path, value);
      return value;
    }).finally(() => {
      clearTimeout(timer); controller.signal.removeEventListener("abort", abortListener);
      options.signal?.removeEventListener("abort", forwardAbort);
      if (pending.get(requestKey) === request) pending.delete(requestKey);
    });
    pending.set(requestKey, request);
    return request.promise;
  }
  return {read, invalidate, cancel};
}

const TouchlineData = createDataClient();
if (typeof module !== "undefined") module.exports = {...TouchlineData, createDataClient};
if (typeof window !== "undefined") {
  window.TouchlineData = TouchlineData;
  window.addEventListener("pagehide", () => TouchlineData.cancel());
}
