/* Balatro Strategist service worker — cache-first shell, fresh API. */
const CACHE = "bs-v140";
const PRECACHE = ["/", "/manifest.json", "/icon-192.png", "/icon-512.png"];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
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
      const fresh = fetch(e.request)
        .then((r) => { if (r.ok) c.put(e.request, r.clone()); return r; })
        .catch(() => cached);
      return cached || fresh;
    })
  );
});
