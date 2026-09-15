# Data audit — v1.7.0 (September 2026)

**Question asked:** *do we feel confident that the data we pulled is accurate?*

**Short answer:** the scoring engine, yes — it was already correct. The card
*text* the app showed you (and fed to the coach, Genie and search), no — it
had real errors. v1.7 replaces every hand-typed card table with tables
generated from a structured extract of balatrowiki.org, and adds a test that
fails if they ever drift again.

## Method

1. Pulled the MediaWiki source of every card page on balatrowiki.org
   (151 joker pages, 22 tarot, 12 planet, 18 spectral, 32 voucher, 24 tag,
   15 deck, 30 blind, 8 enhancement, 5 edition, 4 seal, 8 stake pages) plus
   thumbnail paths — `data/wiki_dump.json`, 484 KB.
2. Wrote `tools/wikitext.py` to turn infobox wikitext (`{{Mult|+4}}`,
   `{{xmult|1.5}}`, `{{hl|orange|…}}`, `[[links]]`, `<br>`) into plain text.
3. Diffed the old `data/*.csv` against the normalised wiki text
   (`tools/wiki_enrich.py --check`), then regenerated the tables, keeping the
   hand-written columns (archetype, playbook, test ideas, synergy tags).
4. Cross-checked the engine registry (`engine.py`) against the wiki text for
   every joker whose effect is a plain `+N Mult / +N Chips / XN Mult`.

## What was wrong in the old tables

| Table | Finding | Fixed |
|---|---|---|
| jokers.csv | **33 effects were substantively wrong** — numbers, hand types or conditions. Worst: Clever Joker (+150 chips on Four of a Kind → really +80 on Two Pair), Mad Joker (+20 on Four of a Kind → +10 on Two Pair), Odd Todd (said *even* ranks +30 → *odd* ranks +31), Yorick, Hanging Chad (retriggers 2×, not 1×), Invisible Joker (2 rounds, not 3), Glass Joker (X0.75, not X0.5), Stuntman (+250, not +300), Gros Michel (1 in 6, not 1 in 4), Steel Joker (X0.2), Vampire (X0.1), Lucky Cat (X0.25), Campfire (X0.25), Banner (+30), Hiker (+5), Smiley Face (+5), Onyx Agate (+7), the four suit jokers (+3, not +4), Mail-In Rebate ($5), Golden Ticket ($4), Runner (+15). | ✅ all 150 effects now generated from the wiki |
| jokers.csv | **65 of 108 listed costs were wrong; 42 were missing.** Examples: Joker $2 (was $4), Blueprint $10 (was missing), Vagabond $8 (was $5). | ✅ all 150 costs from the wiki |
| jokers.csv | **6 rarities wrong**: Burnt Joker, Stuntman, Vagabond are Rare (were Uncommon); Reserved Parking is Common; Sixth Sense and Séance are Uncommon (were Rare). | ✅ |
| jokers.csv | image_url column pointed at a dead domain (`balatro.wiki`) for every row. | ✅ replaced by `image` (wiki file name → `/img/` proxy) |
| jokers.csv | spelling: `8-Ball`, `Riff-raff` → in-game `8 Ball`, `Riff-Raff` (also fixed in synergy_edges / benchmarks) | ✅ |
| tags.csv | Investment Tag said $15 (it is $25). "Skip" Tag was really the Speed Tag. Foil/Holo/Poly/Negative tags said "shop has a … joker" — they make the next base-edition shop joker free *and* give it the edition. | ✅ |
| decks.csv | 21 rows incl. a blank row and 5 junk rows (Braided, Foil, Holographic, Polychrome, Silver — not decks). Every cost was NaN. | ✅ 15 real decks with effect + unlock |
| vouchers.csv | junk "Name" header row; costs missing (all vouchers are $10); curly apostrophe in Director's Cut | ✅ |
| tarots/spectrals | wording drift only ("Spawns" vs "Creates"); costs were missing ($3 / $4); `Soul` → `The Soul`, `Déjà Vu` kept | ✅ |
| planets.csv, hands.csv | **no errors** — chips/mult per level match the wiki for all 12 hands | — |
| engine.py | **no errors** — 34 registry numbers cross-checked against the wiki text, 0 mismatches. The engine had the right values (e.g. Mad Joker +10 on Two Pair) even where the CSV text was wrong. | — |

So: anything the optimizer *scored* was right; anything the Codex/coach
*said* about a card could be wrong, and about one joker in four was. That is
exactly the kind of error an LLM coach amplifies — it will confidently
explain a +150-chip Four-of-a-Kind joker that does not exist.

## What v1.7 adds on top

- **Thumbnails** for every joker, tarot, planet, spectral, voucher, tag,
  deck, blind, enhancement, edition, seal and stake (328 images) — Codex
  tiles, lineup rows, quick-add chips, synergy list. Served through the
  app's own `/img/` route from the wiki (allow-listed file names only,
  cached 30 days, © LocalThunk credited in the Codex footer).
- **New Codex sets**: Tarot · Planet · Spectral, Vouchers (base/upgraded +
  prerequisite), Tags (minimum ante), Decks (effect + unlock), Blinds
  (effect, ante, score multiplier, reward), Editions · Seals · Enhancements ·
  Stakes.
- **Unlock conditions** for jokers, consumables, decks and tags — blurred
  by default (one global 🔒 toggle, or tap a single card), so the app keeps
  its "don't spoil the game" rule.
- **Joker metadata**: type (Chips / +Mult / xMult / Economy / Retrigger /
  Effect), activation timing (On Scored, On Held, Independent …), listed
  odds, and copyable / perishable / eternal compatibility.
- **Wiki synergy notes**: 2,233 paragraphs of Synergies / Anti-Synergies /
  Strategy prose from the wiki, per joker (`data/joker_notes.csv`), shown on
  demand in each Codex tile, folded into the BM25 / TF-IDF search documents
  ("jokers that punish discards" now hits on meaning), exposed to the coach
  and the MCP server as a `card_lookup` tool, and loaded into Unity Catalog
  for Genie.
- **Boss blind picker in run mode**: choose the boss and the target adjusts
  (The Wall ×4, Violet Vessel ×6, The Needle ×1), the effect is shown in the
  run bar, The Water zeroes discards / The Needle sets one hand, and the
  coach is told which boss you are facing.
- `tests/test_data.py`: fails the build if any table drifts from
  `wiki_dump.json`, if wikitext leaks into a CSV, if an image is not in the
  allow-list, or if an engine number disagrees with the wiki text.

## Limits, honestly

- The wiki is community-maintained. It tracks the current game version well
  (it already reflects the 1.0.1 suit-joker nerf the old CSV missed), but it
  is not LocalThunk's source code. When the wiki and the engine's tests
  disagree, the tests win and the disagreement gets investigated.
- 25 of the wiki's synergy sections were cut mid-sentence in the extract
  (my extractor's per-section cap); those tails are dropped rather than shown
  half-finished. Re-running the extract with a larger cap fixes it.
- `data/wiki_dump.json` is a snapshot (2026-09-15). Re-pull it when the game
  patches; `python tools/wiki_enrich.py` regenerates everything else.
- The hand-written columns (archetype, playbook, test ideas, synergy tags)
  are still opinions, not facts — they are labelled as such in the UI.

*Text from balatrowiki.org is CC BY-NC-SA 3.0. Card art © LocalThunk.*
