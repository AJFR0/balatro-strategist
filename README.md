# 🃏 Balatro Strategist

An unofficial, fan-made **Balatro companion** that runs as a
[Databricks App](https://docs.databricks.com/en/dev-tools/databricks-apps/index.html) —
built entirely on **Databricks Free Edition**.

> Balatro is by LocalThunk. This project is not affiliated with or endorsed by
> LocalThunk or Playstack — it's a love letter with a SQL warehouse.
> Card data compiled from community sources (Polychrome dataset + Balatro wiki),
> planet-card values verified against the wiki.

## What it does

| Tab | What you get |
|-----|--------------|
| 🎯 **What do I play?** | Say or tap the 8 cards in your hand + your joker lineup (order matters). A deterministic scoring engine enumerates every legal play (all 218 subsets) and ranks them, with the full chips × mult math shown step by step. |
| 🎙 **Say the hand** (v2.1) | "ace king nine five two of hearts, gold king of spades" — the browser's own speech recogniser (Chrome, Safari, Edge; no keys, no cost) or any keyboard dictation tool (Wispr Flow, the iOS mic key) types into one box; `static/cardspeech.js` parses ranks-then-suit or suit-first phrasing, enhancements/editions/seals and common mis-hearings (*for*→4, *to*→2, *hard*→hearts), previews the cards, flags anything it didn't understand, and commits a clean recording when the mic stops. Undoable like every other edit. |
| 📖 **Joker-pedia** | All 150 jokers — searchable, filterable by rarity / category / synergy tags. Plus hands, planets, tarots, spectrals, vouchers. |
| 🕸️ **Synergy web** | Pick a joker, see which other jokers share its synergy tags, weighted by overlap. For deciding what to buy next. |
| 🧠 **AI strategist** | A foundation model (via Databricks Model Serving) reads the engine's output and your run context, then talks strategy. It is explicitly told not to do arithmetic — **AI narrates, math decides.** |

## The design principle

LLMs are lousy at Balatro math (order-dependent ×Mult chains break them) and
great at Balatro *talk* (builds, synergies, what to shop for). So the app splits
the job:

```
your hand + jokers ──► engine.py (deterministic, unit-tested) ──► exact scores
                                        │
                                        ▼  (scores injected as ground truth)
run context ─────────► foundation model on Databricks ─────────► strategy
```

## Scoring model honesty

- Every scoring-relevant joker that can be computed deterministically is in
  `engine.py`'s registry (flat, per-card, held-in-hand, hand-contains,
  retriggers, Blueprint/Brainstorm copying, rule-changers like Four Fingers /
  Shortcut / Smeared / Splash / Pareidolia).
- **Scaling jokers** (Hologram, Ride the Bus, …) take their *current value* as
  input — you read it off your screen, the engine takes it from there.
- **Probabilistic effects** (Lucky cards, Bloodstone, Misprint) score at
  expected value, or always-hit if you enable Optimist mode.
- Economy/shop/utility jokers don't change a played hand's score; the app knows
  them for synergy but excludes them from the math, and *tells you* it did.
- Known approximations are documented at the top of `engine.py`.

## Where the card data comes from (v1.7)

Every card table in `data/` is generated from a structured extract of
[balatrowiki.org](https://balatrowiki.org) (`data/wiki_dump.json`: infobox
parameters, Synergies / Anti-Synergies / Strategy sections, thumbnail paths)
by `tools/wiki_enrich.py`, which normalises the wikitext and merges in the
hand-written columns (archetype, playbook, test ideas). `tests/test_data.py`
fails if a CSV ever drifts from the extract. The first run of that audit
found real errors in the old hand-typed tables — see
[`docs/Data_Audit.md`](docs/Data_Audit.md) — none of which were in the engine.

Card art is © LocalThunk and is served through the app's `/img/` route from
the wiki's thumbnails (nothing is bundled); wiki text is CC BY-NC-SA 3.0 and
is credited in the Codex.

## Run it locally

```bash
pip install -r requirements.txt
DEMO_MODE=1 uvicorn app:app --port 8000
python tests/test_engine.py   # 26 tests, verified against known game math
```

`DEMO_MODE=1` runs with zero Databricks dependencies: the optimizer, codex,
synergy web and keyword search are fully local; the run log lands in a local
SQLite file and the coach answers from the deterministic playbook. Without the
flag, the app expects Databricks App credentials and lights up Lakebase
(Postgres + pgvector), semantic search, and the Llama coach.

## Deploy a review copy on AWS

The live review copy runs demo mode on **Lambda + CloudFront** at
[balatro.ajf.codes](https://balatro.ajf.codes): `lambda_handler.py` wraps the
FastAPI app with Mangum, dependencies ride the AWS SDK for pandas managed
layer, and each release's `balatro-lambda.zip` asset is the deployable
artifact (fetched into S3 by a tiny bootstrap function, then
`update-function-code`). `apprunner.yaml` is also included if you prefer App
Runner's build-from-repo flow.

## Deploy on Databricks Free Edition

1. Sign up at [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).
2. In the workspace: **New → App → Custom**, name it `balatro-strategist`.
3. Upload these files into the app's source folder (or point the app at a
   workspace folder containing them — keep the `data/` subfolder).
4. Deploy. That's it — Free Edition includes the compute.

Optional, for the Genie follow-up: run `setup_uc_tables.py` as a notebook to
load the CSVs into Unity Catalog, then create a Genie space over
`jokers`, `hands`, and friends and ask it things in English.

## Files

```
app.py               FastAPI backend (optimizer, codex, search, runs, coach)
static/index.html    hand-built SPA (5 tabs, mobile-first)
static/cardspeech.js spoken/typed hand → cards parser (engine-agnostic; node-testable)
engine.py            deterministic scoring engine + joker effect registry
db.py                Lakebase/pgvector/model-serving layer + demo-mode fallbacks
data/*.csv           150 jokers (wiki-verified + tagged), joker_notes (wiki
                     synergy prose), hands, planets, tarots, spectrals,
                     vouchers, decks, tags, blinds, enhancements, editions,
                     seals, stakes; wiki_dump.json is the raw extract
tools/wiki_enrich.py regenerates data/ from wiki_dump.json (tools/wikitext.py
                     is the wikitext → plain-text normaliser)
tests/test_engine.py 28 engine tests · tests/test_data.py data audit ·
                     tests/test_cardspeech.js 39 spoken-hand cases ·
                     tests/mobile_walkthrough.py tap-budget walkthrough
setup_uc_tables.py   optional: load data/ into Unity Catalog for Genie
app.yaml             Databricks Apps entry point
apprunner.yaml       AWS App Runner entry point (demo mode)
```
