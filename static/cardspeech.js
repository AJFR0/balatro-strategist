/* cardspeech.js — turns a spoken or typed description of a Balatro hand into
   card objects the app understands: {rank, suit, enh, ed, seal}.

   Engine-agnostic: the text can come from the browser's speech recognizer,
   a keyboard dictation tool (Wispr Flow, the iOS mic key) or plain typing.

   Grammar (all of these work, and can be mixed):
     "ace king nine five two of hearts"          suit after a run of ranks
     "hearts: ace king nine, spades: five two"   suit before a run of ranks
     "ace of hearts, king of spades"             one suit per card
     "gold king of hearts, steel ace of spades"  enhancement before the rank
     "foil glass ten of diamonds"                edition + enhancement stack
     "red seal seven of clubs"                   "<colour> seal" before the rank
     "AH KH 9H 5H 2H"                            typed shorthand

   Homophones speech recognisers commonly emit are folded in: for→4, to/too→2,
   ate→8, tree→3, sex→6, hard/hart→hearts, spayed→spades, clover→clubs.

   parse(text) → {cards, unsuited, unknown, dangling}
     cards    complete cards, in the order they were said
     unsuited ranks that never got a suit (shown for the player to fix)
     unknown  tokens that mean nothing here (so a misheard word is visible)
     dangling modifiers said with no card after them ("gold" at the end)
*/
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.CardSpeech = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const RANK = {
    ace: "A", aces: "A", a: "A", "1": "A",
    king: "K", kings: "K", k: "K",
    queen: "Q", queens: "Q", q: "Q",
    jack: "J", jacks: "J", jax: "J", j: "J",
    ten: "10", tens: "10", "10": "10", t: "10",
    nine: "9", nines: "9", nein: "9", "9": "9",
    eight: "8", eights: "8", ate: "8", "8": "8",
    seven: "7", sevens: "7", "7": "7",
    six: "6", sixes: "6", sex: "6", "6": "6",
    five: "5", fives: "5", fife: "5", "5": "5",
    four: "4", fours: "4", for: "4", fore: "4", "4": "4",
    three: "3", threes: "3", tree: "3", free: "3", trey: "3", "3": "3",
    two: "2", twos: "2", to: "2", too: "2", deuce: "2", deuces: "2", "2": "2",
  };
  const SUIT = {
    heart: "H", hearts: "H", hart: "H", harts: "H", hard: "H", hards: "H", "♥": "H", "♡": "H",
    diamond: "D", diamonds: "D", dimond: "D", dimonds: "D", dime: "D", dimes: "D", "♦": "D", "♢": "D",
    club: "C", clubs: "C", clove: "C", cloves: "C", clover: "C", clovers: "C", "♣": "C", "♧": "C",
    spade: "S", spades: "S", spaid: "S", spaids: "S", spayed: "S", spaded: "S", "♠": "S", "♤": "S",
  };
  const ENH = {
    bonus: "bonus", mult: "mult", multi: "mult", wild: "wild", glass: "glass",
    steel: "steel", steal: "steel", stone: "stone", gold: "gold", golden: "gold", lucky: "lucky",
  };
  const ED = { foil: "foil", holo: "holo", holographic: "holo", hologram: "holo", polychrome: "polychrome", poly: "polychrome" };
  const SEAL_COLOR = { red: "red", gold: "gold", blue: "blue", purple: "purple" };
  // words that carry no meaning here (connectors, fillers, speech-recogniser noise)
  const FILLER = new Set(["of", "the", "an", "and", "then", "also", "plus", "with", "card", "cards",
    "hand", "my", "i", "have", "got", "is", "um", "uh", "ah", "as", "please", "add", "in", "on", "it", "its",
    "seal", "sealed", "edition", "enhanced", "enhancement", "playing", "okay", "ok", "so", "like",
    "comma", "period", "next", "new", "line"]);
  // the connector words that make the *next* suit apply to the ranks before it
  const AFTER = new Set(["of", "in"]);

  // raw tokens keep their case: "AH" is shorthand, a spoken "ah" is just a noise
  function rawTokens(text) {
    return String(text || "")
      .replace(/['’]s\b/gi, "")                 // heart's → heart
      .replace(/(\d)(?:st|nd|rd|th)\b/gi, "$1")  // 10th → 10 (some recognisers do this)
      .replace(/[^\p{L}\p{N}♥♡♦♢♣♧♠♤]+/gu, " ")  // punctuation → space, keep suit glyphs
      .trim()
      .split(/\s+/)
      .filter(Boolean)
      .flatMap(splitGlyph);
  }
  function tokenize(text) { return rawTokens(text).map(t => t.toLowerCase()); }
  // "a♥" / "10♠" typed with glyphs stuck to the rank
  function splitGlyph(tok) {
    const m = /^(.+?)([♥♡♦♢♣♧♠♤])$/u.exec(tok);
    return m ? [m[1], m[2]] : [tok];
  }
  const SHORT = /^(10|[2-9]|[ajqkt])([hdcs])$/;
  const SUIT_LETTER = { h: "H", d: "D", c: "C", s: "S" };

  function blankMods() { return { enh: "none", ed: "none", seal: "none" }; }
  function makeCard(rank, suit, mods) {
    return { rank, suit, enh: mods.enh, ed: mods.ed, seal: mods.seal };
  }

  function parse(text) {
    const raw = rawTokens(text);
    const toks = raw.map(t => t.toLowerCase());
    const out = { cards: [], unsuited: [], unknown: [], dangling: [] };
    let segment = [];            // ranks since the last suit word
    let tentative = null;        // suit named *before* these ranks, if any
    let mods = blankMods();      // modifiers waiting for the next rank
    let modsUsed = false;

    // Letter-rank shorthand (ah, as, kd…) collides with real words ("as", "ah"),
    // so it only counts when typed in caps or alongside other shorthand tokens.
    const shortCount = toks.filter(t => SHORT.test(t) && !FILLER.has(t)).length;
    const isShort = (i) => {
      const m = SHORT.exec(toks[i]); if (!m) return null;
      if (/\d/.test(m[1]) || /[A-Z]/.test(raw[i]) || shortCount >= 2) return m;
      return null;
    };

    const commitSegment = () => {
      for (const c of segment) (c.suit ? out.cards : out.unsuited).push(c);
      segment = [];
    };

    for (let i = 0; i < toks.length; i++) {
      const tok = toks[i], prev = toks[i - 1], next = toks[i + 1];

      // typed shorthand: AH, 10d, Ks (t♣ becomes "t" + glyph via splitGlyph)
      const sh = isShort(i);
      if (sh) {
        commitSegment(); tentative = null;
        out.cards.push(makeCard(RANK[sh[1]], SUIT_LETTER[sh[2]], mods));
        mods = blankMods(); modsUsed = false;
        continue;
      }

      // "<colour> seal" — must be checked before bare gold (enhancement)
      if (SEAL_COLOR[tok] && (next === "seal" || next === "sealed")) {
        mods.seal = SEAL_COLOR[tok]; modsUsed = true; i++; continue;
      }
      if (tok === "gold" || tok === "golden") { mods.enh = "gold"; modsUsed = true; continue; }
      if (SEAL_COLOR[tok]) { mods.seal = SEAL_COLOR[tok]; modsUsed = true; continue; } // bare "red seven" → red seal
      if (ENH[tok]) { mods.enh = ENH[tok]; modsUsed = true; continue; }
      if (ED[tok]) { mods.ed = ED[tok]; modsUsed = true; continue; }
      if (tok === "chrome" && prev === "poly") continue; // "poly chrome" already applied

      // "a" is only an ace when it clearly names a card ("a of hearts", "a hearts", "gold a")
      if (tok === "a" && !(AFTER.has(next) || SUIT[next] || modsUsed)) continue;

      if (RANK[tok]) {
        segment.push(makeCard(RANK[tok], tentative, mods));
        mods = blankMods(); modsUsed = false;
        continue;
      }

      if (SUIT[tok]) {
        const suit = SUIT[tok];
        // suit-after: "…of hearts", or ranks that have no suit yet followed by a bare suit word
        const after = AFTER.has(prev) || (segment.length && !tentative);
        if (after) {
          for (const c of segment) c.suit = suit;
          commitSegment(); tentative = null;
        } else {
          commitSegment(); tentative = suit;   // suit-first: applies to what follows
        }
        continue;
      }

      if (FILLER.has(tok)) continue;
      out.unknown.push(tok);
    }
    commitSegment();
    if (modsUsed) {
      out.dangling = [mods.enh, mods.ed, mods.seal !== "none" ? mods.seal + " seal" : "none"].filter(x => x !== "none");
    }
    return out;
  }

  /* one-line summary for screen readers / the preview strip */
  const RN = { A: "Ace", K: "King", Q: "Queen", J: "Jack" };
  const SN = { H: "Hearts", D: "Diamonds", S: "Spades", C: "Clubs" };
  function describe(c) {
    const mods = [c.enh, c.ed, c.seal !== "none" ? c.seal + " seal" : "none"].filter(x => x && x !== "none");
    return (mods.length ? mods.join(" ") + " " : "") + (RN[c.rank] || c.rank) + (c.suit ? " of " + SN[c.suit] : "");
  }

  return { parse, tokenize, describe };
});
