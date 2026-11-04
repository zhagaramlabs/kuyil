"""Kuyil's text frontend: Tamil script, English in either script, and numbers to one phone set.

    python frontend.py "நான் meeting-க்கு 10.30-க்கு வருவேன்"
    python frontend.py --coverage ../uraikal/data/d4_web.jsonl ../uraikal/data/bpb/eval.json

Tamil is read by rule: letters, plus context rules for the stops, ற்ற, ன்ற and word-final u.
English in Latin letters goes through CMUdict, mapped to how Tamil speakers say English.
Numbers are read as spoken Tamil by the agent's reader (uraikal.numbers). English written in Tamil script uses the Tamil rules.
Everything here is a first draft for a Tamil linguist to review
(plan: brainstorm/2026-11-02-tamil-tiny-tts-plan.md in the lab repo).
"""
import json, re, sys, unicodedata
from collections import Counter

import cmudict
from uraikal.numbers import UNITS, to_words

SPECIAL = ["<pad>", "<bos>", "<eos>", "<space>", ".", ",", "?", "!", ";", ":"]
VOWELS = ["a", "a:", "i", "i:", "u", "u:", "e", "e:", "o", "o:", "ai", "au", "ux"]
CONSONANTS = ["k", "g", "x", "ng", "c", "j", "s", "ny", "T", "D", "N", "t", "d", "dh", "n", "p", "b", "m",
              "y", "r", "l", "v", "zh", "L", "R", "tr", "dr", "sh", "h", "f", "z"]

INDEPENDENT = {"அ": "a", "ஆ": "a:", "இ": "i", "ஈ": "i:", "உ": "u", "ஊ": "u:", "எ": "e", "ஏ": "e:",
               "ஐ": "ai", "ஒ": "o", "ஓ": "o:", "ஔ": "au"}
SIGN = {"ா": "a:", "ி": "i", "ீ": "i:", "ு": "u", "ூ": "u:", "ெ": "e", "ே": "e:", "ை": "ai",
        "ொ": "o", "ோ": "o:", "ௌ": "au"}
LETTER = {"க": "k", "ங": "ng", "ச": "c", "ஞ": "ny", "ட": "T", "ண": "N", "த": "t", "ந": "n", "ப": "p",
          "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "zh", "ள": "L", "ற": "R", "ன": "n",
          "ஜ": "j", "ஷ": "sh", "ஶ": "sh", "ஸ": "s", "ஹ": "h"}
GEMINATES = [p + ":" for p in dict.fromkeys(LETTER.values()) if p != "R"]  # ற்ற is "tr"
SYMBOLS = SPECIAL + VOWELS + CONSONANTS + GEMINATES
PULLI, AYTHAM = "்", "ஃ"

STOP = {"k": ("g", "x"), "c": ("j", "s"), "T": ("D", "D"), "t": ("d", "dh"), "p": ("b", "b")}  # (after a nasal, between vowels)
NASAL = {"ng", "ny", "N", "n", "m"}
SONORANT = NASAL | {"y", "r", "l", "v", "zh", "L"}

ARPA = {"AA": ["a:"], "AE": ["e"], "AH": ["a"], "AO": ["a:"], "AW": ["au"], "AY": ["ai"], "EH": ["e"],
        "ER": ["a", "r"], "EY": ["e:"], "IH": ["i"], "IY": ["i:"], "OW": ["o:"], "OY": ["a:", "y"],
        "UH": ["u"], "UW": ["u:"], "B": ["b"], "CH": ["c"], "D": ["D"], "DH": ["d"], "F": ["f"],
        "G": ["g"], "HH": ["h"], "JH": ["j"], "K": ["k"], "L": ["l"], "M": ["m"], "N": ["n"],
        "NG": ["ng"], "P": ["p"], "R": ["r"], "S": ["s"], "SH": ["sh"], "T": ["T"], "TH": ["t"],
        "V": ["v"], "W": ["v"], "Y": ["y"], "Z": ["z"], "ZH": ["sh"]}  # Tamil-English: retroflex t/d, v=w

_cmu = None


def read_number(s):
    """Digits as spoken Tamil (the agent's reader, uraikal.numbers); phone numbers and leading zeros digit by digit."""
    s = s.replace(",", "")
    whole = s.partition(".")[0]
    if len(whole) >= 7 or (len(whole) > 1 and whole[0] == "0"):
        return " ".join(UNITS[int(d)] if d.isdigit() else "புள்ளி" for d in s)
    return to_words(float(s) if "." in s else int(s))


def tamil_word(w, stats=None):
    segs = []  # (phone, is_vowel)
    i = 0
    while i < len(w):
        ch = w[i]
        if ch in LETTER:
            segs.append((LETTER[ch], False))
            nxt = w[i + 1] if i + 1 < len(w) else ""
            if nxt in SIGN:
                segs.append((SIGN[nxt], True)); i += 1
            elif nxt == PULLI:
                i += 1
            else:
                segs.append(("a", True))
        elif ch in INDEPENDENT:
            segs.append((INDEPENDENT[ch], True))
        elif ch == AYTHAM:
            segs.append((AYTHAM, False))
        elif stats is not None:
            stats["unknown_chars"] += 1
        i += 1

    nvowels = sum(v for _, v in segs)
    out, skip = [], False
    for j, (p, is_vowel) in enumerate(segs):
        if skip:
            skip = False; continue
        prev = out[-1] if out else None
        nxt = segs[j + 1] if j + 1 < len(segs) else None
        if is_vowel:
            out.append("ux" if p == "u" and nxt is None and nvowels > 1 else p)
        elif p == AYTHAM:
            skip = bool(nxt) and nxt[0] in ("p", "j")  # ஃப = f, ஃஜ = z
            out.append({"p": "f", "j": "z"}[nxt[0]] if skip else "h")
        elif nxt == (p, False):
            out.append("tr" if p == "R" else p + ":"); skip = True
        elif p == "R" and prev == "n":
            out.append("dr")  # ன்ற
        elif p in STOP:
            voiced, between = STOP[p]
            if prev is None:
                out.append(p)
            elif prev in NASAL:
                native = p not in ("c", "T") or prev == {"c": "ny", "T": "N"}[p]  # ன்ச, ன்ட are loans
                out.append(voiced if native else ("s" if p == "c" else p))
            elif prev in SONORANT:
                out.append("s" if p == "c" else voiced)
            elif prev in VOWELS and nxt and nxt[1]:
                out.append(between)
            else:
                out.append(p)
        else:
            out.append(p)
    return out


def english_word(w, stats=None):
    global _cmu
    _cmu = _cmu or cmudict.dict()
    key = w.lower()
    if key in _cmu:
        arpa = _cmu[key][0]
    else:
        acronym = w.isupper() and len(w) <= 5
        if stats is not None:
            stats["acronyms" if acronym else "fallback_words"] += 1
        # shortcut: unknown words are spelled out letter by letter; add letter-to-sound rules when coverage needs it
        arpa = [a for ch in key if ch in _cmu for a in next(p for p in _cmu[ch] if any("1" in x for x in p))]
    return [p for a in arpa for p in ARPA[a.rstrip("012")]]


TOKEN = re.compile(r"(?P<num>\d[\d,]*(?:\.\d+)?)|(?P<ta>[஀-௿]+)|(?P<en>[A-Za-z]+(?:'[A-Za-z]+)?)"
                   r"|(?P<sp>\s+)|(?P<p>[.,?!;:।])|(?P<cur>₹)|(?P<pct>%)|(?P<other>.)")


def words(text, stats=None):
    """Text as a list of words (each a list of phones) and break tokens (<space> or punctuation)."""
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    text = text.translate({0x0BE6 + d: str(d) for d in range(10)})  # Tamil digits
    out, money = [], False

    def space():
        if out and isinstance(out[-1], list):
            out.append("<space>")

    for m in TOKEN.finditer(text):
        kind, s = m.lastgroup, m.group()
        if kind == "num":
            spoken = read_number(s).split() + (["ரூபாய்"] if money else [])
            money = False
            for k, w in enumerate(spoken):
                if k:
                    out.append("<space>")
                out.append(tamil_word(w, stats))
            if stats is not None:
                stats["numbers"] += 1
        elif kind == "ta":
            out.append(tamil_word(s, stats))
            if stats is not None:
                stats["tamil_words"] += 1
        elif kind == "en":
            out.append(english_word(s, stats))
            if stats is not None:
                stats["latin_words"] += 1
        elif kind == "sp":
            space()
        elif kind == "p":
            if out and out[-1] == "<space>":
                out.pop()
            out.append("." if s == "।" else s)
        elif kind == "cur":
            money = True
        elif kind == "pct":
            space(); out.append(tamil_word("சதவீதம்", stats))
        elif s != "-" and stats is not None:
            stats["unknown_chars"] += 1
    while out and out[-1] == "<space>":
        out.pop()
    return [w for w in out if w != []]


def phonemize(text, stats=None):
    """Text to phones from SYMBOLS, with <space> between words and punctuation kept."""
    return [p for w in words(text, stats) for p in (w if isinstance(w, list) else [w])]


def coverage(path):
    if path.endswith(".jsonl"):
        groups = {path: [json.loads(line)["text"] for line in open(path, encoding="utf-8")]}
    else:
        groups = json.load(open(path, encoding="utf-8"))
    for name, texts in groups.items():
        st = Counter()
        for t in texts:
            phonemize(t, st)
        words = st["tamil_words"] + st["latin_words"] + st["numbers"]
        print(json.dumps({"set": name, "texts": len(texts), **st,
                          "fallback_pct": round(100 * st["fallback_words"] / max(words, 1), 3)}, ensure_ascii=False))


if __name__ == "__main__":
    if sys.argv[1] == "--coverage":
        for p in sys.argv[2:]:
            coverage(p)
    else:
        print(" ".join(phonemize(" ".join(sys.argv[1:]))))
