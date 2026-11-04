"""python -m pytest test_frontend.py. Expected phones follow standard spoken Tamil; a linguist should check them."""
import pytest

from frontend import SYMBOLS, phonemize


@pytest.mark.parametrize("text, phones", [
    ("படம்", "p a D a m"),            # ட between vowels
    ("பட்டம்", "p a T: a m"),         # doubled ட: the only difference from படம்
    ("அது", "a dh ux"),               # த between vowels; word-final u reduced
    ("பூ", "p u:"),
    ("தங்கம்", "t a ng g a m"),       # voiced after a nasal
    ("பஞ்சு", "p a ny j ux"),
    ("அன்பு", "a n b ux"),
    ("மகன்", "m a x a n"),            # க between vowels
    ("பசி", "p a s i"),
    ("செய்து", "c e y d ux"),         # voiced after ய்
    ("வெற்றி", "v e tr i"),           # ற்ற
    ("ஒன்று", "o n dr ux"),           # ன்ற
    ("கற்க", "k a R k a"),
    ("பள்ளி", "p a L: i"),
    ("தமிழ்", "t a m i zh"),
    ("ஸ்ரீ", "s r i:"),
    ("ஃபோன்", "f o: n"),
    ("டாக்டர்", "T a: k T a r"),      # English in Tamil script uses the Tamil rules
    ("இன்டர்நெட்", "i n T a r n e T"),  # ன்ட is a loan cluster, so ட stays voiceless
    ("meeting-க்கு", "m i: T i ng k: u"),
    ("UPI", "y u: p i: ai"),          # unknown acronyms are spelled out
    ("₹500", "ai n u: R ux <space> r u: b a: y"),  # ஐநூறு ரூபாய்; ரூபாய் is said rūbāy
    ("அவன் வந்தான்.", "a v a n <space> v a n d a: n ."),
])
def test_phones(text, phones):
    assert " ".join(phonemize(text)) == phones


def test_numbers_use_the_agents_reader():
    assert phonemize("2026") == phonemize("ரெண்டாயிரத்தி இருபத்தி ஆறு")
    assert phonemize("3.5") == phonemize("மூணு புள்ளி அஞ்சு")
    assert phonemize("1,50,000") == phonemize("ஒரு லட்சத்தி அம்பதாயிரம்")
    assert phonemize("9876543210") == phonemize("ஒம்பது எட்டு ஏழு ஆறு அஞ்சு நாலு மூணு ரெண்டு ஒண்ணு பூஜ்யம்")


def test_every_phone_is_a_symbol():
    text = "அஆஇஈஉஊஎஏஐஒஓஔ கஙசஞடணதநபமயரலவழளறன ஜஷஸஹ க்ஷ meeting zoo thing 12345 ₹10 50%"
    assert set(phonemize(text)) <= set(SYMBOLS)
