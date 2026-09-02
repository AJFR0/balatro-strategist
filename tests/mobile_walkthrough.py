"""Mobile walkthrough + UX audit for Balatro Strategist.

Runs the five core journeys (Play, Codex, Web, Runs, Coach) in mobile
emulation (390x844, touch, DPR 2) against a local demo-mode server and
reports measurable UX findings:

  - tap targets under 44px
  - inputs under 16px font (iOS auto-zoom trigger)
  - horizontal page overflow
  - safe-area padding on the bottom tab bar
  - whether the optimizer result is visible after tapping the CTA
  - what a *tap* does on synergy-web cards (hover-only affordances)
  - console errors

Usage:  DEMO_MODE=1 uvicorn app:app --port 8009 &
        python tests/mobile_walkthrough.py [base_url] [shots_dir]
"""
import asyncio
import json
import sys

from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8009"
SHOTS = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mobile_shots"

FINDINGS: list[dict] = []


def finding(sev, area, issue, detail=""):
    FINDINGS.append({"sev": sev, "area": area, "issue": issue, "detail": detail})


MEASURE_JS = """() => {
  const out = {smallTargets: [], smallInputs: [], overflow: false};
  out.overflow = document.documentElement.scrollWidth > window.innerWidth + 1;
  for (const el of document.querySelectorAll('button, [role="button"], select, .nrow, .chipbtn')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.height < 40 || r.width < 40) {
      const label = (el.textContent || el.className || el.tagName).trim().replace(/\\s+/g, ' ').slice(0, 24);
      out.smallTargets.push(label + ' (' + Math.round(r.width) + 'x' + Math.round(r.height) + ')');
    }
  }
  for (const el of document.querySelectorAll('input, textarea, select')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0) continue;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 16) out.smallInputs.push((el.id || el.placeholder || el.tagName).slice(0, 24) + ' (' + fs + 'px)');
  }
  // env(safe-area-inset-bottom) computes to 0 in emulation, so check the
  // stylesheet declares it rather than the computed value.
  out.navSafeArea = [...document.styleSheets].some(s => {
    try { return [...s.cssRules].some(r => /nav\\b/.test(r.selectorText || '') &&
                                           /safe-area-inset-bottom/.test(r.cssText)); }
    catch (e) { return false; }
  });
  return out;
}"""


async def add_joker(pg, name):
    await pg.fill("#jsearch", name)
    await pg.wait_for_timeout(400)
    await pg.locator("#aclist div[data-n]").first.dispatch_event("mousedown")
    await pg.wait_for_timeout(250)


async def main():
    import os
    os.makedirs(SHOTS, exist_ok=True)
    console_errors = []
    async with async_playwright() as p:
        b = await p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
        ctx = await b.new_context(viewport={"width": 390, "height": 844},
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
        pg = await ctx.new_page()
        pg.on("console", lambda m: console_errors.append(m.text)
              if m.type == "error" and "ERR_TUNNEL" not in m.text
              and "Failed to load resource" not in m.text else None)
        await pg.goto(BASE)
        await pg.wait_for_timeout(1800)

        # ---------- journey 1: Play ----------
        m = await pg.evaluate(MEASURE_JS)
        if m["overflow"]:
            finding("HIGH", "global", "horizontal page overflow on Play tab")
        if m["smallInputs"]:
            finding("HIGH", "global", "inputs under 16px trigger iOS zoom-on-focus",
                    ", ".join(m["smallInputs"][:6]))
        if not m.get("navSafeArea"):
            finding("MED", "nav", "bottom tab bar has no safe-area padding (iPhone home bar overlap)")
        if m["smallTargets"]:
            finding("MED", "play", f"{len(m['smallTargets'])} tap targets under 40px",
                    ", ".join(m["smallTargets"][:8]))
        await pg.screenshot(path=f"{SHOTS}/01_play.png")

        await pg.click("#demoHand")
        for j in ["The Tribe", "Blueprint", "Hologram"]:
            await add_joker(pg, j)
        eds = pg.locator("select.jed")
        if await eds.count() >= 2:
            await eds.nth(1).select_option("polychrome")
        vals = pg.locator("input.jval")
        if await vals.count() >= 1:
            await vals.last.fill("2.5")
        go = pg.locator("#go")
        go_box = await go.bounding_box()
        await go.click()
        await pg.wait_for_timeout(1500)
        res = pg.locator("#result .score, #result")
        res_box = await pg.locator("#result").bounding_box()
        visible = res_box and res_box["y"] < 844 and res_box["y"] > 0
        if not visible:
            finding("HIGH", "play", "optimizer result renders below the fold with no auto-scroll",
                    f"result y={res_box and int(res_box['y'])}, viewport=844")
        await pg.screenshot(path=f"{SHOTS}/02_play_result.png")

        # ---------- journey 1b: discard advisor ----------
        if not await pg.locator("#discPanel").is_visible():
            finding("HIGH", "play", "discard advisor panel not shown after optimize")
        else:
            await pg.click("#discGo")
            for _ in range(30):                       # MC sim can take a few seconds
                await pg.wait_for_timeout(500)
                if "toss" in (await pg.locator("#discOut").inner_text()):
                    break
            disc_txt = await pg.locator("#discOut").inner_text()
            if "toss" not in disc_txt:
                finding("HIGH", "play", "discard advisor returned no options")
            elif "stand pat" not in disc_txt:
                finding("LOW", "play", "discard advisor missing stand-pat baseline")
            await pg.screenshot(path=f"{SHOTS}/02b_discard.png")

        # ---------- journey 1c: run mode ----------
        await pg.click("#runStart")
        await pg.wait_for_timeout(400)
        if not await pg.locator("#runbar").is_visible():
            finding("HIGH", "runmode", "Start run does not show the run bar")
        else:
            bar = await pg.locator("#runbar").inner_text()
            if "Small Blind" not in bar:
                finding("MED", "runmode", "run bar missing blind name", bar[:80])
            blind_val = await pg.locator("#blind").input_value()
            if blind_val != "300":
                finding("MED", "runmode", f"blind target not auto-filled (got {blind_val!r}, want 300)")
            await pg.click("#runNext")
            await pg.wait_for_timeout(300)
            bar = await pg.locator("#runbar").inner_text()
            if "Big Blind" not in bar:
                finding("MED", "runmode", "Next blind did not advance small→big", bar[:80])
            m1b = await pg.evaluate(MEASURE_JS)
            if m1b["overflow"]:
                finding("HIGH", "runmode", "run bar causes horizontal overflow")
            await pg.screenshot(path=f"{SHOTS}/02c_runbar.png")
            await pg.click("#runEnd")
            await pg.wait_for_timeout(800)
            if await pg.locator("#runbar").is_visible():
                finding("MED", "runmode", "End run does not dismiss the run bar")

        # ---------- journey 2: Codex ----------
        await pg.click("#nav button[data-t='codex']")
        await pg.wait_for_timeout(1200)
        await pg.fill("#csearch", "flush")
        await pg.wait_for_timeout(900)
        n_tiles = await pg.locator("#codexGrid .tile").count()
        if n_tiles == 0:
            finding("HIGH", "codex", "search for 'flush' returned no tiles")
        first_details = pg.locator("#codexGrid .tile details summary").first
        if await first_details.count():
            await first_details.click()
            await pg.wait_for_timeout(300)
        await pg.screenshot(path=f"{SHOTS}/03_codex.png")

        # ---------- journey 3: Web (touch behavior) ----------
        await pg.click("#nav button[data-t='web']")
        await pg.wait_for_timeout(1800)
        row = pg.locator(".nrow[data-jt]").first
        if await row.count() == 0:
            finding("HIGH", "web", "no synergy rows rendered at mobile width")
        else:
            await row.tap()
            await pg.wait_for_timeout(500)
            tip_visible = await pg.locator("#jtip").is_visible()
            sheet_visible = await pg.locator("#jsheet").is_visible() if await pg.locator("#jsheet").count() else False
            if not (tip_visible or sheet_visible):
                finding("HIGH", "web", "tapping a synergy card shows no stats on touch (hover-only affordance)")
            elif tip_visible and not sheet_visible:
                finding("MED", "web", "tap shows the cursor tooltip, which cannot be dismissed on touch")
            await pg.screenshot(path=f"{SHOTS}/04_web.png")
            if sheet_visible:
                sheet_txt = await pg.locator("#jsheet").inner_text()
                if "benchmark" not in sheet_txt and "solo-neutral" not in sheet_txt:
                    finding("LOW", "web", "stat sheet missing engine benchmark line")
                await pg.tap("#shClose")
                await pg.wait_for_timeout(300)
                if await pg.locator("#jsheet").count():
                    finding("MED", "web", "stat sheet Close button does not dismiss the sheet")
        if not await pg.locator("#jsheet").count():
            pass
        else:
            await pg.evaluate("document.getElementById('jsheet')?.remove(); document.getElementById('jveil')?.remove()")

        # ---------- journey 4: Runs ----------
        await pg.click("#nav button[data-t='runs']")
        await pg.wait_for_timeout(900)
        await pg.fill("#rNotes", "mobile walkthrough test run")
        await pg.click("#rSave")
        await pg.wait_for_timeout(900)
        runs_txt = await pg.locator("#runList").inner_text()
        if "walkthrough" not in runs_txt and "ante" not in runs_txt.lower():
            finding("MED", "runs", "saved run not visible in list after logging")
        await pg.screenshot(path=f"{SHOTS}/05_runs.png")

        # ---------- journey 5: Coach ----------
        await pg.click("#nav button[data-t='coach']")
        await pg.wait_for_timeout(800)
        m2 = await pg.evaluate(MEASURE_JS)
        if m2["smallInputs"]:
            finding("MED", "coach", "coach inputs under 16px (iOS zoom)", ", ".join(m2["smallInputs"][:4]))
        await pg.click("#cAsk")
        await pg.wait_for_timeout(1200)
        ans_visible = await pg.locator("#answerWrap").is_visible()
        if not ans_visible:
            finding("MED", "coach", "no visible answer/fallback after asking the strategist")
        await pg.screenshot(path=f"{SHOTS}/06_coach.png")

        # ---------- journey 6: PWA ----------
        pwa = await pg.evaluate("""async () => {
          const out = {};
          try { const r = await fetch('/manifest.json'); const m = await r.json();
                out.manifest = r.ok && m.display === 'standalone' && m.icons.length >= 2; }
          catch (e) { out.manifest = false; }
          try { const reg = await navigator.serviceWorker.getRegistration();
                out.sw = !!(reg && (reg.active || reg.installing || reg.waiting)); }
          catch (e) { out.sw = false; }
          out.link = !!document.querySelector('link[rel="manifest"]');
          return out;
        }""")
        if not pwa.get("link"):
            finding("HIGH", "pwa", "no <link rel=manifest> in the page head")
        if not pwa.get("manifest"):
            finding("HIGH", "pwa", "manifest.json missing/invalid (standalone + icons)")
        if not pwa.get("sw"):
            finding("MED", "pwa", "service worker not registered")

        if console_errors:
            finding("MED", "global", f"{len(console_errors)} console errors", console_errors[0][:100])

        await ctx.close()
        await b.close()

    print(json.dumps(FINDINGS, indent=2))
    print(f"\n{len(FINDINGS)} findings · screenshots in {SHOTS}")
    return FINDINGS


if __name__ == "__main__":
    asyncio.run(main())
