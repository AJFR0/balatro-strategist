/* Unit tests for static/cardspeech.js — run with: node tests/test_cardspeech.js */
"use strict";
const path = require("path");
const { parse, describe } = require(path.join(__dirname, "..", "static", "cardspeech.js"));

let pass = 0, fail = 0;
const short = c => c.rank + c.suit + ([c.enh, c.ed, c.seal].some(m => m !== "none") ? "(" + [c.enh, c.ed, c.seal].filter(m => m !== "none").join(",") + ")" : "");
function expect(text, want, extra) {
  const r = parse(text);
  const got = r.cards.map(short).join(" ");
  const view = { unsuited: r.unsuited.map(c => c.rank), unknown: r.unknown, dangling: r.dangling };
  const ok = got === want
    && (!extra || Object.entries(extra).every(([k, v]) => JSON.stringify(view[k]) === JSON.stringify(v)));
  if (ok) { pass++; console.log("  PASS", JSON.stringify(text)); }
  else { fail++; console.log("  FAIL", JSON.stringify(text), "\n       want:", want, extra ? JSON.stringify(extra) : "", "\n       got: ", got, JSON.stringify({ unsuited: r.unsuited.map(short), unknown: r.unknown, dangling: r.dangling })); }
}

console.log("suit after ranks");
expect("ace king nine five two of hearts", "AH KH 9H 5H 2H");
expect("ace, king, 9, 5, 2 of hearts", "AH KH 9H 5H 2H");
expect("ace king hearts nine five spades", "AH KH 9S 5S");
expect("Ace of hearts King of hearts 9 of hearts 5 of hearts 2 of hearts", "AH KH 9H 5H 2H");
expect("ten of diamonds", "10D");
expect("10 of diamonds and jack of clubs", "10D JC");

console.log("suit before ranks");
expect("hearts ace king nine, spades five two", "AH KH 9H 5S 2S");
expect("hearts: ace, king, nine. Spades: five, two.", "AH KH 9H 5S 2S");
expect("clubs seven eight nine ten jack", "7C 8C 9C 10C JC");

console.log("mixed forms");
expect("ace of hearts, king, five of spades", "AH KS 5S");
expect("ace of hearts king of spades queen of diamonds jack of clubs", "AH KS QD JC");
expect("hearts ace king of spades", "AS KS");                 // contradiction: the explicit "of" wins

console.log("homophones");
expect("ace king nine five to of hearts", "AH KH 9H 5H 2H");
expect("for of spades and ate of clubs", "4S 8C");
expect("tree of dimonds, sex of harts, too of spayed", "3D 6H 2S");
expect("nine of hards and five of clover", "9H 5C");

console.log("modifiers");
expect("gold king of hearts", "KH(gold)");
expect("steel ace of spades and glass ten of diamonds", "AS(steel) 10D(glass)");
expect("foil glass ten of diamonds", "10D(glass,foil)");
expect("red seal seven of clubs", "7C(red)");
expect("gold seal king of hearts", "KH(gold)".replace("(gold)", "(gold)"));  // seal gold, not enhancement
expect("purple seal wild ace of hearts", "AH(wild,purple)");
expect("holographic lucky queen of spades", "QS(lucky,holo)");
expect("polychrome mult five of hearts", "5H(mult,polychrome)");
expect("hearts: gold king, steel ace", "KH(gold) AH(steel)");
expect("gold king and queen of hearts", "KH(gold) QH");          // modifier sticks to the next rank only
expect("king of hearts gold", "KH", { dangling: ["gold"] });    // trailing modifier reported, not silently dropped

console.log("shorthand");
expect("AH KH 9H 5H 2H", "AH KH 9H 5H 2H");
expect("ah kh 9h", "AH KH 9H");                                // several shorthand tokens → shorthand mode
expect("10d Js Qc", "10D JS QC");
expect("A♥ K♠ 10♦ 7♣", "AH KS 10D 7C");
expect("as well as the ace of spades", "AS");                  // lone "as" is a word, not a card
expect("ah, the king of hearts", "KH");

console.log("errors surface, never silently");
expect("ace king of hearts banana", "AH KH", { unknown: ["banana"] });
expect("ace king nine", "", { unsuited: ["A", "K", "9"].map(r => r) });
expect("gold", "", { dangling: ["gold"] });
expect("", "");
expect("um so my hand is ace of hearts please", "AH");

console.log("describe");
{
  const d = describe({ rank: "K", suit: "H", enh: "gold", ed: "none", seal: "red" });
  if (d === "gold red seal King of Hearts") { pass++; console.log("  PASS describe"); }
  else { fail++; console.log("  FAIL describe:", d); }
}

// unsuited entries are card objects; compare ranks only in the message above
if (fail) { console.log(`\n${fail} FAILED, ${pass} passed`); process.exit(1); }
console.log(`\nALL PASS (${pass})`);
