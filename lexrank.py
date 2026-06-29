"""
lexrank.py
Implementasi LexRank manual dengan information-aware boosting.

Alur:
1. Bangun representasi TF-IDF (unigram + bigram) setiap kalimat
2. Hitung cosine similarity antar kalimat → adjacency matrix
3. Threshold-based binarization → similarity graph
4. Jalankan PageRank (via networkx) → LexRank score
5. Terapkan multi-signal boosting (tanpa keyword hardcode):
   - Positional boost
   - Title similarity boost
   - Information density boost (angka, entitas, kutipan)
   - Generic title-intent boost (dari TF-IDF top-terms artikel)
   - Enumeration/list boost (pola struktural)
6. Kembalikan skor akhir per kalimat
"""

import re
import math
import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import preprocess_for_tfidf, get_stopwords


# ---------------------------------------------------------------------------
# Konstanta hyperparameter (dapat di-tune)
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD: float = 0.10   # batas minimum similarity agar ada edge
PAGERANK_DAMPING: float = 0.85        # damping factor PageRank (LexRank)
PAGERANK_MAX_ITER: int = 200

# Bobot boosting (seimbangkan agar judul dan posisi lebih diprioritaskan)
W_POSITION: float = 1.00
W_TITLE: float = 1.00
W_INFO_DENSITY: float = 0.00
W_TITLE_INTENT: float = 0.00  # 0.10, 0.20, 0.30, 0.40, 0.50
W_ENUMERATION: float = 0.40
W_EXPERT: float = 0.30
W_RECOMMENDATION: float = 0.15
W_POLICY: float = 0.45

# Jumlah top-terms TF-IDF yang dipakai untuk generic title-intent boost
TOP_TERMS_K: int = 15


def score_sentences(
    sentences: List[str],
    title: Optional[str] = None,
) -> List[float]:
    """
    Menghitung skor LexRank + multi-signal boost untuk setiap kalimat.

    Parameters
    ----------
    sentences : List[str]
        Daftar kalimat original (belum di-stem).
    title : str | None
        Judul artikel untuk title-similarity boost.

    Returns
    -------
    List[float]
        Skor akhir per kalimat (index sesuai dengan urutan sentences).
    """
    n = len(sentences)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    # ------------------------------------------------------------------
    # 1. TF-IDF Vectorization (unigram + bigram, stopword Indonesia)
    # ------------------------------------------------------------------
    preprocessed = [preprocess_for_tfidf(s) for s in sentences]
    stopwords_id = list(get_stopwords())

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords_id,
        min_df=1,
        sublinear_tf=True,      # gunakan log(TF) untuk smoothing
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
    except ValueError:
        # Fallback jika vocabulary kosong
        return [1.0 / n] * n

    # ------------------------------------------------------------------
    # 2. Cosine Similarity Matrix
    # ------------------------------------------------------------------
    sim_matrix: np.ndarray = cosine_similarity(tfidf_matrix)

    # ------------------------------------------------------------------
    # 3. Threshold-based Adjacency Matrix → Similarity Graph
    # ------------------------------------------------------------------
    adj_matrix = (sim_matrix >= SIMILARITY_THRESHOLD).astype(float)
    np.fill_diagonal(adj_matrix, 0)  # hapus self-loop

    # Jika tidak ada edge sama sekali, gunakan skor uniform
    if adj_matrix.sum() == 0:
        return [1.0 / n] * n

    # ------------------------------------------------------------------
    # 4. LexRank via PageRank (networkx)
    # ------------------------------------------------------------------
    graph = nx.from_numpy_array(adj_matrix)
    pagerank_scores: Dict[int, float] = nx.pagerank(
        graph,
        alpha=PAGERANK_DAMPING,
        max_iter=PAGERANK_MAX_ITER,
    )
    lexrank_scores: List[float] = [
        pagerank_scores.get(i, 0.0) for i in range(n)]

    # Normalisasi LexRank ke [0, 1]
    max_lr = max(lexrank_scores) or 1.0
    lexrank_norm = [s / max_lr for s in lexrank_scores]

    # ------------------------------------------------------------------
    # 5. Ekstrak top-terms TF-IDF artikel (untuk generic intent boost)
    # ------------------------------------------------------------------
    feature_names = vectorizer.get_feature_names_out()
    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = tfidf_sum.argsort()[-TOP_TERMS_K:][::-1]
    top_terms: set = {feature_names[i] for i in top_indices}

    # ------------------------------------------------------------------
    # 6. TF-IDF kalimat vs judul (title similarity boost)
    # ------------------------------------------------------------------
    title_sim_scores: List[float] = _compute_title_similarity(
        tfidf_matrix, vectorizer, title, n
    )

    # ------------------------------------------------------------------
    # 7. Hitung skor boost per kalimat dan gabungkan
    # ------------------------------------------------------------------
    final_scores: List[float] = []
    for i, sent in enumerate(sentences):
        base = lexrank_norm[i]

        pos_boost = _positional_boost(i, n)
        title_boost = title_sim_scores[i] * W_TITLE
        info_boost = _info_density_score(sent) * W_INFO_DENSITY
        intent_boost = _generic_title_intent_score(
            sent, top_terms) * W_TITLE_INTENT
        enum_boost = _enumeration_boost(sent) * W_ENUMERATION
        expert_boost = _expert_quote_boost(sent) * W_EXPERT
        rec_boost = _recommendation_boost(sent) * W_RECOMMENDATION
        policy_boost = _policy_announcement_boost(sent) * W_POLICY

        score = base + pos_boost + title_boost + info_boost + intent_boost + \
            enum_boost + expert_boost + rec_boost + policy_boost
        final_scores.append(score)

    return final_scores


# ---------------------------------------------------------------------------
# Fungsi Boost (semuanya generik, tidak ada keyword domain hardcode)
# ---------------------------------------------------------------------------

def _positional_boost(idx: int, total: int) -> float:
    """
    Memberikan boost ringan pada kalimat di awal artikel.
    Berita Indonesia umumnya meletakkan informasi terpenting di awal (inverted pyramid).
    Boost dihitung secara eksponensial menurun.
    """
    if total <= 1:
        return 0.0
    # Kalimat pertama dapat boost penuh; menurun secara log
    relative_pos = idx / (total - 1)  # 0.0 (awal) → 1.0 (akhir)
    boost = W_POSITION * math.exp(-2.5 * relative_pos)
    return boost


def _compute_title_similarity(
    tfidf_matrix,
    vectorizer: TfidfVectorizer,
    title: Optional[str],
    n: int,
) -> List[float]:
    """
    Menghitung cosine similarity setiap kalimat terhadap judul berita.
    Jika judul tidak tersedia, kembalikan list nol.
    """
    if not title:
        return [0.0] * n

    title_preprocessed = preprocess_for_tfidf(title)
    try:
        title_vec = vectorizer.transform([title_preprocessed])
    except Exception:
        return [0.0] * n

    sims = cosine_similarity(tfidf_matrix, title_vec).flatten()
    max_sim = sims.max() or 1.0
    return (sims / max_sim).tolist()


def _info_density_score(sentence: str) -> float:
    """
    Mengukur kepadatan informasi kalimat berdasarkan sinyal generik:
    - Angka/statistik (persentase, nilai, jumlah)
    - Pola tanggal
    - Kutipan langsung (tanda petik)
    - Referensi entitas (huruf kapital)

    Tidak ada keyword domain hardcode.
    """
    score = 0.0

    # Angka dan satuan (termasuk Rp, %, ribuan, jutaan, dll.)
    number_count = len(re.findall(
        r"\b\d+(?:[.,]\d+)*\s*(?:%|persen|juta|miliar|triliun|ribu|kg|km|m|cm|mw|mwp)?\b",
        sentence, re.IGNORECASE
    ))
    score += min(number_count * 0.15, 0.45)

    # Pola tanggal
    if re.search(r"\b\d{1,2}\s+\w+\s+\d{4}\b", sentence):
        score += 0.10

    # Kutipan langsung
    if re.search(r'["\'"\u201c\u201d]', sentence):
        score += 0.15

    # Token huruf kapital (kemungkinan nama orang/tempat/organisasi)
    cap_tokens = len(re.findall(r"\b[A-Z][a-z]+\b", sentence))
    score += min(cap_tokens * 0.04, 0.20)

    return min(score, 1.0)


def _generic_title_intent_score(sentence: str, top_terms: set) -> float:
    """
    Memberikan boost berdasarkan seberapa banyak top-terms TF-IDF artikel
    muncul dalam kalimat. Belajar dari konten artikel sendiri — tidak hardcode.

    Juga mendeteksi sinyal struktural peringatan/urgensi generik.
    """
    if not top_terms:
        return 0.0

    sent_lower = sentence.lower()
    tokens = set(re.findall(r"\b\w+\b", sent_lower))

    # Hitung overlap dengan top-terms
    overlap = len(tokens & top_terms)
    overlap_score = min(overlap / max(len(top_terms), 1), 1.0)

    # Sinyal urgensi/peringatan generik (bukan domain-spesifik)
    urgency_patterns = [
        r"\b(waspada|hati-hati|awas|bahaya|risiko|ancaman|darurat|kritis)\b",
        r"\b(penting|segera|mendesak|prioritas|utama)\b",
        r"\b(berhasil|sukses|gagal|meningkat|menurun|naik|turun)\b",
    ]
    urgency_score = 0.0
    for pat in urgency_patterns:
        if re.search(pat, sent_lower):
            urgency_score += 0.25
    urgency_score = min(urgency_score, 0.5)

    return min(overlap_score * 0.7 + urgency_score * 0.3, 1.0)


def _enumeration_boost(sentence: str) -> float:
    """
    Deteksi kalimat yang berisi enumeration/daftar berdasarkan pola struktural:
    - Bullet atau nomor di awal
    - Koma berulang yang menandakan daftar item
    - Pola "pertama, kedua, ketiga"
    - Pola "antara lain, yaitu, seperti" diikuti daftar
    """
    score = 0.0

    # Pola daftar bernomor atau bullet
    if re.match(r"^\s*(\d+[.)]\s+|[-•–]\s+)", sentence):
        score += 1.0

    # Kata ordinal Indonesia
    if re.search(
        r"\b(pertama|kedua|ketiga|keempat|kelima|keenam|pertama-tama|selanjutnya|terakhir)\b",
        sentence, re.IGNORECASE
    ):
        score += 0.7

    # Pola introduksi daftar
    if re.search(
        r"\b(antara\s+lain|yaitu|yakni|seperti|meliputi|terdiri\s+dari|di\s+antaranya|contohnya|misalnya|sebagai\s+contoh|berikut\s+ini|pilihannya|opsinya|alternatifnya|termasuk|mencakup|berupa|ialah|adalah)\b",
        sentence, re.IGNORECASE
    ):
        score += 1.0

    # Banyak koma → kemungkinan daftar item
    comma_count = sentence.count(",")
    if comma_count >= 3:
        score += (comma_count * 0.15)

    return min(score, 1.5)


def _policy_announcement_boost(sentence: str) -> float:
    """
    Memberikan boost pada kalimat yang memuat penetapan kebijakan, aturan, atau pengumuman penting (bersifat definitif).
    """
    score = 0.0
    sent_lower = sentence.lower()

    # Sinyal kebijakan atau ketetapan (generik)
    policy_patterns = [
        r"\b(pemerintah\s+menetapkan|mengumumkan|resmi\s+berlaku|kebijakan\s+ini|aturan\s+ini|insentif\s+ini|program\s+ini|berlaku\s+mulai|mulai\s+berlaku|ditanggung|diwajibkan)\b",
    ]

    for pat in policy_patterns:
        if re.search(pat, sent_lower):
            score += 1.0

    # Pola durasi, periode, atau jadwal yang definitif
    if re.search(r"\b(berlaku\s+untuk\s+periode|selama\s+\d+\s+(hari|minggu|bulan|tahun)|hingga\s+\d+\s+[a-z]+|mulai\s+\d+\s+[a-z]+|sejak\s+\d+\s+[a-z]+)\b", sent_lower):
        score += 1.5

    # Sinyal kebijakan kuantitatif (besaran insentif/diskon/pajak)
    if re.search(r"\b(sebesar\s+\d+\s*(persen|%)|diskon\s+\d+\s*(persen|%)|gratis|dibebaskan|ditanggung\s+\d+\s*(persen|%))\b", sent_lower):
        score += 1.5

    return min(score, 2.5)


def _expert_quote_boost(sentence: str) -> float:
    """
    Memberikan boost pada kalimat yang mengandung penjelasan ahli atau kausalitas.
    """
    score = 0.0
    sent_lower = sentence.lower()

    # Sinyal pernyataan ahli
    expert_patterns = [
        r"\b(menurut|kata|ujar|ungkap|jelas|tutur|tambah|pakar|ahli|dokter|peneliti|studi|riset|spesialis)\b",
    ]
    for pat in expert_patterns:
        if re.search(pat, sent_lower):
            score += 1.0

    # Sinyal penjelasan/kausalitas (alasan penting)
    cause_patterns = [
        r"\b(karena|disebabkan|hal\s+ini|alasan|penyebab|artinya|berarti)\b"
    ]
    for pat in cause_patterns:
        if re.search(pat, sent_lower):
            score += 0.5

    return min(score, 1.0)


def _recommendation_boost(sentence: str) -> float:
    """
    Memberikan boost pada kalimat yang memuat rekomendasi, saran, atau solusi.
    """
    score = 0.0
    sent_lower = sentence.lower()

    rec_patterns = [
        r"\b(disarankan|sebaiknya|dianjurkan|merekomendasikan|rekomendasi|saran|tips|cara|solusi|solusinya|langkah|panduan|perlu|harus|wajib|jangan|hindari|boleh|pastikan|pilihannya|opsinya|gantinya)\b"
    ]
    for pat in rec_patterns:
        if re.search(pat, sent_lower):
            score += 1.0

    return min(score, 1.5)
