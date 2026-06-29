"""
utils.py
Utilitas pendukung sistem peringkasan berita.

Berisi:
- reconstruct_summary      : merangkai kalimat terpilih menjadi teks ringkasan final
- count_summary_sentences  : menghitung kalimat aktual dari teks ringkasan final
- validate_url             : validasi format URL sederhana
- truncate_text            : memotong teks panjang untuk log/debug
"""

import re
from typing import List, Tuple


def reconstruct_summary(selected_sentences: List[Tuple[int, str]]) -> str:
    """
    Merangkai kalimat-kalimat terpilih menjadi teks ringkasan final.

    Kalimat sudah diurutkan berdasarkan posisi asli (logical order)
    sebelum masuk ke fungsi ini (dilakukan di mmr.select_sentences_mmr).

    Parameters
    ----------
    selected_sentences : List[Tuple[int, str]]
        Daftar (index_original, kalimat) dari MMR selector.

    Returns
    -------
    str
        Teks ringkasan final, setiap kalimat dipisahkan spasi.
    """
    if not selected_sentences:
        return ""

    sentences = [sent.strip() for _, sent in selected_sentences]

    # Pastikan setiap kalimat diakhiri tanda titik
    normalized: List[str] = []
    for sent in sentences:
        if sent and sent[-1] not in ".!?":
            sent = sent + "."
        normalized.append(sent)

    return " ".join(normalized)


def count_summary_sentences(text: str) -> int:
    """
    Menghitung jumlah kalimat aktual yang terlihat pada summary_text.

    Berbeda dari preprocessor.segment_sentences(), fungsi ini tidak dipakai
    untuk seleksi MMR. Tujuannya hanya untuk metadata response API, sehingga
    list marker seperti "1." tidak dihitung sebagai kalimat terpisah.
    """
    text = text.strip()
    if not text:
        return 0

    text = re.sub(r"\s+", " ", text)

    # Lindungi titik yang bukan akhir kalimat.
    protected_patterns = [
        r"\b(?:[A-Z]\.){2,}",  # K.H., U.S., dll.
        r"\b(?:Dr|Prof|Mr|Mrs|Ms|Drs|Ir|H|Hj|No|Vol|hlm|dll|dsb|dkk|Rp|km|kg|gr)\.",
        r"\b(?:Jan|Feb|Mar|Apr|Jun|Jul|Ags|Sep|Okt|Nov|Des)\.",
    ]

    protected = {}

    def _protect(match: re.Match) -> str:
        placeholder = f"__P{len(protected)}__"
        protected[placeholder] = match.group(0)
        return placeholder

    for pattern in protected_patterns:
        text = re.sub(pattern, _protect, text)

    # Kalau list inline muncul setelah titik dua/awal kalimat, anggap itemnya
    # sebagai entri baru tanpa menghitung titik pada marker list. Jangan
    # sentuh skor olahraga seperti "21-13. Hu Zhean..." karena itu memang
    # batas kalimat.
    text = re.sub(r"(^|[.!?:]\s+)(\d{1,2})\.(?=\s+[A-Z])", r"\1\2)", text)
    text = re.sub(r":\s+(?=\d{1,2}\)\s+[A-Z])", ". ", text)

    sentence_end = re.compile(
        r"[.!?]+[\"'\u201d\u2019)]*(?=\s+(?:[A-Z0-9\"'\u201c\u2018(]|__P\d+__)|$)"
    )
    count = len(sentence_end.findall(text))

    # Fallback untuk teks tanpa tanda baca akhir tetapi tetap berisi kata.
    if count == 0 and len(text.split()) >= 3:
        return 1

    return count


def validate_url(url: str) -> bool:
    """
    Memvalidasi apakah string merupakan URL HTTP/HTTPS yang valid.

    Parameters
    ----------
    url : str
        String URL yang akan divalidasi.

    Returns
    -------
    bool
        True jika valid, False jika tidak.
    """
    pattern = re.compile(
        r"^https?://"           # skema HTTP atau HTTPS
        r"(?:\S+(?::\S*)?@)?"   # optional auth
        r"(?:"
        r"(?!(?:10|127)(?:\.\d{1,3}){3})"  # bukan IP private
        r"(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})"
        r"(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})"
        r"(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])"
        r"(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}"
        r"(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))"
        r"|"
        r"(?:(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)"
        r"(?:\.(?:[a-z\u00a1-\uffff0-9]-*)*[a-z\u00a1-\uffff0-9]+)*"
        r"(?:\.(?:[a-z\u00a1-\uffff]{2,}))"
        r")"
        r"(?::\d{2,5})?"
        r"(?:/\S*)?$",
        re.IGNORECASE,
    )
    return bool(pattern.match(url))


def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    Memotong teks panjang untuk keperluan log/debug.

    Parameters
    ----------
    text : str
        Teks yang akan dipotong.
    max_chars : int
        Batas maksimum karakter.

    Returns
    -------
    str
        Teks yang sudah dipotong dengan elipsis jika diperlukan.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
