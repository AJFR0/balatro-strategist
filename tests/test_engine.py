"""Unit tests for the Balatro scoring engine — verified against known game math."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import (parse_cards, score_hand, best_plays, detect_hand,
                    JokerState, Rules)


def S(played, held="", jokers=(), levels=None, rules=None, extra=None):
    return score_hand(parse_cards(played), parse_cards(held),
                      [JokerState(j) if isinstance(j, str) else j for j in jokers],
                      levels, rules, extra)


def test_pair_of_tens_level1():
    # Pair lvl1 = 10 chips x 2 mult; cards add 10+10 -> (10+20) x 2 = 60
    r = S("10H 10S")
    assert r.hand == "Pair"
    assert r.total == 60, r.steps


def test_high_card_ace():
    # High Card lvl1 = 5 x 1; ace adds 11 -> 16 x 1 = 16
    r = S("AH 3C")  # ace scores, 3 doesn't (only highest scores)
    assert r.hand == "High Card"
    assert r.total == 16, r.steps


def test_flush_level_2():
    # Flush lvl2 = (35+15) chips x (4+2) mult; cards 11+10+10+9+2 = 42 -> 92 x 6 = 552
    r = S("AH KH JH 9H 2H", levels={"Flush": 2})
    assert r.hand == "Flush"
    assert r.total == (35 + 15 + 42) * 6, r.steps


def test_plus_mult_joker():
    # Joker: +4 mult. Pair of tens -> 30 x (2+4) = 180
    r = S("10H 10S", jokers=["Joker"])
    assert r.total == 180, r.steps


def test_the_duo_xmult():
    # The Duo: x2 if contains a Pair -> 30 x (2*2) = 120
    r = S("10H 10S", jokers=["The Duo"])
    assert r.total == 120, r.steps


def test_blueprint_copies_right():
    # Blueprint left of The Duo => Duo applies twice: 30 x (2*2*2) = 240
    r = S("10H 10S", jokers=["Blueprint", "The Duo"])
    assert r.total == 240, r.steps


def test_blueprint_at_end_copies_nothing():
    r = S("10H 10S", jokers=["The Duo", "Blueprint"])
    assert r.total == 120, r.steps


def test_baron_kings_held():
    # Pair of 5s played (5+5 card chips): (10+10)=20 chips x 2 mult
    # 2 kings held: x1.5 x1.5 => 20 x (2*2.25) = 90
    r = S("5H 5S", held="KH KD 3C", jokers=["Baron"])
    assert r.total == 90, r.steps


def test_steel_card_held():
    r = S("10H 10S", held="KH(steel)")
    assert r.total == 30 * 3, r.steps  # 2 mult * 1.5 = 3


def test_glass_card_x2():
    # glass 10 pair: chips 30, mult 2 then x2 at the glass card
    r = S("10H(glass) 10S")
    assert r.total == 120, r.steps


def test_fibonacci():
    # Fibonacci: +8 mult per scored A/2/3/5/8. Pair of 8s -> (10+8+8) x (2+8+8) = 468
    r = S("8H 8S", jokers=["Fibonacci"])
    assert r.total == 26 * 18, r.steps


def test_sock_and_buskin_retrigger():
    # Pair of kings, Sock and Buskin retriggers faces: each K scores twice.
    # chips: 10 + 4*10 = 50, mult 2 -> 100
    r = S("KH KS", jokers=["Sock and Buskin"])
    assert r.total == 100, r.steps


def test_four_of_a_kind_detection():
    r = S("9H 9S 9C 9D 3H")
    assert r.hand == "Four of a Kind"
    # only the four 9s score: (60 + 36) x 7 = 672
    assert r.total == 672, r.steps


def test_full_house():
    r = S("9H 9S 9C KH KD")
    assert r.hand == "Full House"
    assert r.total == (40 + 27 + 20) * 4, r.steps


def test_straight_ace_low():
    r = S("AH 2S 3C 4D 5H")
    assert r.hand == "Straight"


def test_shortcut_straight():
    rules = Rules(shortcut=True)
    r = S("2H 4S 6C 8D 10H", rules=rules)
    assert r.hand == "Straight", r.hand


def test_four_fingers_flush():
    r = S("2H 5H 9H KH 3C", jokers=["Four Fingers"])
    assert r.hand == "Flush", r.hand


def test_stone_card_always_scores():
    # High card ace + stone: stone adds 50 chips
    r = S("AH 3C(stone)")
    assert r.total == (5 + 11 + 50) * 1, r.steps


def test_scaling_state_value():
    # Hologram at current x2.25
    r = S("10H 10S", jokers=[JokerState("Hologram", value=2.25)])
    assert r.total == int(30 * 2 * 2.25), r.steps


def test_polychrome_joker_edition():
    r = S("10H 10S", jokers=[JokerState("Joker", edition="polychrome")])
    # 30 x ((2+4) * 1.5) = 270
    assert r.total == 270, r.steps


def test_hand_contains_vs_played():
    # Full house contains a Pair (The Duo triggers on full house)
    r = S("9H 9S 9C KH KD", jokers=["The Duo"])
    assert any("contains Pair" in s for s in r.steps), r.steps


def test_best_plays_finds_flush_over_pair():
    cards = parse_cards("AH KH 9H 5H 2H AS 3C 7D")
    plays = best_plays(cards, [], top_n=3)
    assert plays[0]["result"].hand == "Flush", [p["result"].hand for p in plays]


def test_best_plays_respects_jokers():
    # With The Tribe (x2 flush) flush should dominate even harder;
    # with pair-focused build, pair of aces + holo etc. could win otherwise
    cards = parse_cards("AH KH 9H 5H 2H AS 3C 7D")
    plays = best_plays(cards, [JokerState("The Tribe")], top_n=1)
    assert plays[0]["result"].hand == "Flush"


def test_unknown_joker_reported():
    r = S("10H 10S", jokers=["Egg"])  # Egg is economy — not in scoring registry
    assert "Egg" in r.unknown_jokers


def test_blueprint_copies_scaling_value():
    # Blueprint left of Hologram(x2.5): both apply -> pair 30 x (2 * 2.5 * 2.5) = 375
    r = S("10H 10S", jokers=[JokerState("Blueprint"), JokerState("Hologram", value=2.5)])
    assert r.total == int(30 * 2 * 2.5 * 2.5), r.steps


if __name__ == "__main__":
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS {name}")
            except Exception:
                fails += 1
                print(f"  FAIL {name}")
                traceback.print_exc()
    print("ALL PASS" if fails == 0 else f"{fails} FAILURES")
    sys.exit(1 if fails else 0)