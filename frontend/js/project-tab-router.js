(() => {
  "use strict";

  const tabToRoute = Object.freeze({
    "tab-test": "test",
    "tab-train": "train",
    "tab-results": "results",
    "tab-map": "map",
    "tab-postprocess": "post-process",
    "tab-logs": "logs",
  });
  const routeToTab = new Map(Object.entries(tabToRoute).map(([tab, route]) => [route, tab]));
  routeToTab.set("postprocess", "tab-postprocess");

  function current(fallback = "tab-test") {
    const route = new URL(window.location.href).searchParams.get("tab")?.trim().toLowerCase();
    return routeToTab.get(route) || fallback;
  }

  function navigate(tabId, { replace = false } = {}) {
    const route = tabToRoute[tabId];
    if (!route) return;
    const url = new URL(window.location.href);
    if (url.searchParams.get("tab") === route) return;
    url.searchParams.set("tab", route);
    window.history[replace ? "replaceState" : "pushState"]({ tab: route }, "", url);
  }

  function listen(handler) {
    window.addEventListener("popstate", () => handler(current()));
  }

  window.ProjectTabRouter = { current, navigate, listen };
})();
