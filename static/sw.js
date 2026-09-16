/* Balatro Strategist service worker — cache-first shell, fresh API. */
const CACHE = "bs-v201";
const PRECACHE = ["/", "/manifest.json", "/icon-192.png", "/icon-512.png"];

// Shell requests bypass the browser HTTP cache (revalidate with the server);
// otherwise a heuristically-fresh stale copy can be precached on a new release
// and every background refresh keeps reading the same stale copy.
const shellOpts = (pathname) => (pathname.startsWith("/img/") ? {} : { cache: "no-cache" });

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(PRECACHE.map((u) => new Request(u, { cache: "reload" }))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return; // network-only

  // /api/bootstrap: stale-while-revalidate so the shell opens instantly offline
  if (url.pathname === "/api/bootstrap") {
    e.respondWith(
      caches.open(CACHE).then(async (c) => {
        const cached = await c.match(e.request);
        const fresh = fetch(e.request)
          .then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })
          .catch(() => cached);
        return cached || fresh;
      })
    );
    return;
  }

  // every other API call is live-only (optimizer, discard, AI, runs)
  if (url.pathname.startsWith("/api/")) return;

  // shell + assets: cache-first, refresh in the background
  e.respondWith(
    caches.open(CACHE).then(async (c) => {
      const cached = await c.match(e.request);
      const fresh = fetch(e.request, shellOpts(url.pathname))
        .then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })
        .catch(() => cached);
      return cached || fresh;
    })
  );
});
