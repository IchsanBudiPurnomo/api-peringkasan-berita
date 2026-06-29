"""
preprocessor.py
Modul preprocessing teks berita Indonesia.

Pipeline:
1. Noise removal (iklan, CTA, promosi media sosial, dll.)
2. Sentence segmentation
3. Text normalization
4. Stemming (hanya untuk TF-IDF, bukan output final)
5. Stopword removal (hanya untuk TF-IDF)
"""

import re
import unicodedata
from typing import List, Tuple

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------------------------
# Inisialisasi Sastrawi (sekali saja saat import)
# ---------------------------------------------------------------------------
_stemmer = StemmerFactory().create_stemmer()
_stop_word_remover = StopWordRemoverFactory().create_stop_word_remover()
_STOPWORDS: set = set(StopWordRemoverFactory().get_stop_words())

# ---------------------------------------------------------------------------
# Pola Noise Generik (tidak hardcode per situs)
# ---------------------------------------------------------------------------
_NOISE_LINE_PATTERNS: List[re.Pattern] = [re.compile(p, re.IGNORECASE) for p in [
    r"^(baca\s+juga|artikel\s+terkait|simak\s+juga|lihat\s+juga|tonton\s+juga|cek\s+juga)\s*[:\-–]?",
    r"^advertisement$",
    r"scroll\s+untuk\s+melanjutkan",
    r"^scroll\s+to\s+continue\b",
    r"iklan\s+di\s+bawah",
    r"(klik|tap)\s+(di\s+sini|untuk\s+(download|unduh|beli|berlangganan))",
    r"(follow|ikuti|gabung|join|subscribe|berlangganan)\s+.*(instagram|twitter|facebook|youtube|tiktok|telegram|whatsapp|channel)",
    r"(dapatkan|temukan)\s+berita\s+.*\s+(di|via)\s+(instagram|twitter|facebook|youtube|tiktok|telegram|whatsapp)",
    r"whatsapp\s+group",
    r"telegram\s+(channel|group)",
    r"download\s+(aplikasi|app)\s+",
    r"©\s*\d{4}",
    r"all\s+rights?\s+reserved",
    r"^editor\s*:\s*\w+",
    r"^reporter\s*:\s*\w+",
    r"^sumber\s*:",
    r"^tags?\s*:",
    r"^\d+\s+shares?$",
    r"^(share|bagikan|cetak|print)\s*$",
    r"(berlangganan|subscribe)\s+(premium|pro|plus)",
    r"baca\s+berita\s+tanpa\s+iklan",
    r"^(senin|selasa|rabu|kamis|jumat|sabtu|minggu),?\s+\d{1,2}\s+\w+\s+\d{4}$",
    r"^\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}$",
    r"\([cC]\)\s*(ap\s+photo|getty|reuters|afp|antara|epa|istimewa)",
    r"\(?foto\s*[:\-]",
]]

_INLINE_NOISE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*[A-Z][A-Z0-9-]+\.(?:com|co|id)(?:,\s*[A-Z][a-z]+)?\s*[-–—]\s*", re.IGNORECASE), ""),
    (re.compile(r"\bloading\s*\.\.\.\s*(?:A\s+){2,}A\b", re.IGNORECASE), " "),
    (re.compile(r"\bloading\s*\.\.\.", re.IGNORECASE), " "),
    (re.compile(r"\bscroll\s+ke\s+bawah\s+untuk\s+melanjutkan\s+membaca\s+iklan\b", re.IGNORECASE), " "),
    (re.compile(r"\bscroll\s+untuk\s+melanjutkan\s+membaca\s+iklan\b", re.IGNORECASE), " "),
    (
        re.compile(
            r"\b(?:baca|artikel|simak|lihat|tonton|cek)\s+juga\s*:\s*"
            r"(?=(?:Seperti|Menurut|Sementara|Selain|Adapun|Namun|Dengan|Karena|"
            r"Untuk|Sebagai|Di\s+tengah|Pemerintah)\b)",
            re.IGNORECASE,
        ),
        " ",
    ),
    (re.compile(r"@\w+"), ""),
    (re.compile(r"#\w+"), ""),
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"&[a-z]+;", re.IGNORECASE), " "),
    (
        re.compile(
            r"\b(?:baca|artikel|simak|lihat|tonton|cek)\s+juga\s*:\s*"
            r"[^.?!\n]{0,220}?"
            r"(?=\s+(?:Benar\s+saja|Kala\s+itu|Namun|Sementara\s+itu|Sebagai\s+catatan|"
            r"Sebagai\s+informasi|Adapun|Sebab|Grup\s+ini|Hasil\s+drawing|"
            r"Haryo\s+menjelaskan|Di\s+tengah)\b|[.?!]|$)",
            re.IGNORECASE,
        ),
        " ",
    ),
    (
        re.compile(
            r"\bKOMPAS\.com\s+berkomitmen\s+memberikan\s+fakta\s+jernih.*$",
            re.IGNORECASE,
        ),
        " ",
    ),
    (
        re.compile(
            r"\(\s*(?:(?:Foto\s+oleh|Foto\s*:|Photo\s+by|Dok\.|Dokumentasi)[^)]{0,240}|"
            r"[^)]{0,200}/(?:AFP|ANTARA|Reuters|AP Photo|Getty Images)[^)]*)\)",
            re.IGNORECASE,
        ),
        " ",
    ),
    (
        re.compile(
            r"\b(?:Foto\s+oleh|Foto\s*:|Photo\s+by|Dok\.|Dokumentasi)\s+[^.?!\n]{0,240}",
            re.IGNORECASE,
        ),
        " ",
    ),
    (re.compile(r"\(\s*\)"), ""),
    (re.compile(r"\(\s*$"), ""),
    (re.compile(r"\.\s+\."), "."),
    (re.compile(r"\s+\.\s+\."), "."),
    (re.compile(r"\s{2,}"), " "),
]

_MIN_SENTENCE_LEN: int = 30
_MIN_WORD_COUNT: int = 5
_LIST_ITEM_PATTERN = re.compile(r'^\d+\s*[\.\-\)]?\s*\w+', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Fungsi Publik
# ---------------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """Membersihkan teks dari noise secara line-by-line dan inline."""
    lines: List[str] = raw_text.splitlines()
    clean_lines: List[str] = []
    seen_lines = set()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if _is_noise_line(line):
            continue
        line = _clean_inline(line).strip()
        
        # Deduplikasi baris menggunakan representasi lowercase.
        # Beberapa situs multi-page mengulang paragraf akhir halaman sebelumnya
        # sebagai bagian dari baris panjang halaman berikutnya.
        line_lower = line.lower()
        if line_lower in seen_lines:
            continue
        if any(line_lower in seen for seen in seen_lines):
            continue
        seen_lines.add(line_lower)
        
        if len(line) < _MIN_SENTENCE_LEN:
            if not _LIST_ITEM_PATTERN.match(line):
                continue
        clean_lines.append(line)

    return "\n".join(clean_lines)


def segment_sentences(text: str) -> List[str]:
    """
    Memecah teks menjadi kalimat individual.
    Mempertahankan singkatan umum agar tidak terpotong salah.
    """
    abbreviations = [
        "Dr", "Prof", "Mr", "Mrs", "Ms", "Drs", "Ir", "H", "Hj",
        "dll", "dsb", "dkk", "hlm", "No", "Vol",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Ags", "Sep", "Okt", "Nov", "Des",
        "Rp", "km", "kg", "gr",
    ]
    protected = {}
    for abbr in abbreviations:
        placeholder = f"__ABBR_{abbr.upper()}__"
        pattern = re.compile(r"\b" + re.escape(abbr) + r"\.")
        text, count = pattern.subn(placeholder, text)
        if count:
            protected[placeholder] = f"{abbr}."

    # List bernomor sering muncul inline setelah titik dua:
    # "... berikut daftarnya: 1. Item pertama 2. Item kedua".
    # Jadikan marker list sebagai batas tanpa melindungi angka tanggal/skor
    # seperti "2023." atau "21-13.".
    text = re.sub(r"(?<!\n)\s+(?=\d{1,2}\.\s+[A-Z])", "\n", text)

    # Split di titik/seru/tanya yang diikuti spasi + huruf kapital
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\d\"'\u201c\u201d\(])", text)

    sentences: List[str] = []
    for chunk in raw_sentences:
        sub = [s.strip() for s in chunk.split("\n") if s.strip()]
        sentences.extend(sub)

    restored: List[str] = []
    seen_sentences = set()
    for sent in sentences:
        for placeholder, original in protected.items():
            sent = sent.replace(placeholder, original)
        sent = sent.strip()
        sentence_key = re.sub(r"\W+", "", sent.lower())
        if sentence_key in seen_sentences:
            continue
        if _is_valid_sentence(sent):
            seen_sentences.add(sentence_key)
            restored.append(sent)

    return restored


def preprocess_for_tfidf(sentence: str) -> str:
    """
    Preprocessing untuk TF-IDF: lowercase → hapus tanda baca →
    stopword removal → stemming.

    JANGAN gunakan output ini sebagai teks summary final.
    """
    text = sentence.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _stop_word_remover.remove(text)
    text = _stemmer.stem(text)
    return text


def get_stopwords() -> set:
    """Mengembalikan set stopword bahasa Indonesia dari Sastrawi."""
    return _STOPWORDS.copy()


# ---------------------------------------------------------------------------
# Helper Privat
# ---------------------------------------------------------------------------

def _is_noise_line(line: str) -> bool:
    for pattern in _NOISE_LINE_PATTERNS:
        if pattern.search(line):
            return True
    return False


def _clean_inline(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    for pattern, replacement in _INLINE_NOISE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _is_valid_sentence(sentence: str) -> bool:
    if _is_noise_line(sentence):
        return False
    if _LIST_ITEM_PATTERN.match(sentence):
        return True
    if len(sentence) < _MIN_SENTENCE_LEN:
        return False
    if len(sentence.split()) < _MIN_WORD_COUNT:
        return False
    alpha_ratio = sum(c.isalpha() for c in sentence) / max(len(sentence), 1)
    return alpha_ratio >= 0.4