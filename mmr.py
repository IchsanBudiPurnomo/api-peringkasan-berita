"""
mmr.py
Implementasi Maximum Marginal Relevance (MMR) untuk seleksi kalimat ringkasan.

MMR memilih kalimat yang:
1. Memiliki relevansi/kepentingan tinggi (dari LexRank score)
2. Tidak redundan terhadap kalimat yang sudah dipilih
3. Memastikan coverage dari seluruh bagian artikel (positional bins)

Formula MMR:
    score_MMR(s) = λ * importance(s) - (1 - λ) * max_sim(s, selected)

di mana:
    - importance(s) = LexRank score + boosting score
    - max_sim(s, selected) = cosine similarity maksimum antara s dan kalimat terpilih
    - λ (lambda) mengontrol trade-off relevance vs diversity
"""

import numpy as np
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessor import preprocess_for_tfidf, get_stopwords


# ---------------------------------------------------------------------------
# Konstanta
# ---------------------------------------------------------------------------
MMR_LAMBDA: float = 0.60    # > 0.5 → lebih condong ke relevance
MIN_SIMILARITY_PENALTY: float = 0.0   # batas bawah similarity untuk MMR


def select_sentences_mmr(
    sentences: List[str],
    scores: List[float],
    num_sentences: int,
) -> List[Tuple[int, str]]:
    """
    Memilih kalimat terbaik menggunakan algoritma MMR dengan coverage-aware
    selection.

    Parameters
    ----------
    sentences : List[str]
        Kalimat-kalimat original (belum di-stem).
    scores : List[float]
        Skor LexRank + boost per kalimat (output dari lexrank.score_sentences).
    num_sentences : int
        Jumlah kalimat yang akan dipilih untuk ringkasan.

    Returns
    -------
    List[Tuple[int, str]]
        Daftar (index_original, kalimat) yang dipilih, DIURUTKAN berdasarkan
        posisi asli kalimat dalam artikel (preserves logical order).
    """
    n = len(sentences)
    if n == 0:
        return []

    num_sentences = min(num_sentences, n)
    if num_sentences <= 0:
        return []

    # ------------------------------------------------------------------
    # 1. Bangun TF-IDF matrix untuk menghitung similarity antar kalimat
    # ------------------------------------------------------------------
    preprocessed = [preprocess_for_tfidf(s) for s in sentences]
    stopwords_id = list(get_stopwords())

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords_id,
        min_df=1,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
    except ValueError:
        # Vocabulary kosong: kembalikan kalimat dengan skor tertinggi
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        selected = sorted(ranked[:num_sentences])
        return [(i, sentences[i]) for i in selected]

    sim_matrix: np.ndarray = cosine_similarity(tfidf_matrix).tolist()

    # ------------------------------------------------------------------
    # 2. Coverage bins — pastikan kalimat dari tiap bagian artikel terwakili
    # ------------------------------------------------------------------
    # Bagi artikel menjadi bins berdasarkan num_sentences
    bin_targets = _compute_coverage_bins(n, num_sentences)

    # ------------------------------------------------------------------
    # 3. MMR iterative selection
    # ------------------------------------------------------------------
    remaining: set = set(range(n))
    selected_indices: List[int] = []

    # Seed: pilih kalimat dengan skor tertinggi sebagai kalimat pertama
    seed = max(remaining, key=lambda i: scores[i])
    selected_indices.append(seed)
    remaining.remove(seed)

    while len(selected_indices) < num_sentences and remaining:
        best_idx: int = -1
        best_mmr: float = -float("inf")

        for cand in remaining:
            # Relevance: normalized importance score
            relevance = scores[cand]

            # Redundancy: cosine similarity maks terhadap kalimat terpilih
            max_sim = max(
                sim_matrix[cand][sel] for sel in selected_indices
            )

            # Coverage bonus: bonus jika bin kalimat ini belum terwakili
            coverage_bonus = _coverage_bonus(
                cand, n, selected_indices, bin_targets)

            mmr = MMR_LAMBDA * relevance - \
                (1 - MMR_LAMBDA) * max_sim + coverage_bonus
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = cand

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    # ------------------------------------------------------------------
    # 4. Urutkan kembali berdasarkan posisi asli (preserves logical order)
    # ------------------------------------------------------------------
    selected_indices.sort()
    return [(i, sentences[i]) for i in selected_indices]


# ---------------------------------------------------------------------------
# Helper Privat
# ---------------------------------------------------------------------------

def _compute_coverage_bins(n_sentences: int, num_select: int) -> List[int]:
    """
    Membagi artikel menjadi segmen-segmen dan menentukan target coverage.
    Setiap bin idealnya diwakili oleh setidaknya 1 kalimat terpilih.

    Returns index representatif dari tiap bin (tengah bin).
    """
    if num_select <= 0 or n_sentences <= 0:
        return []

    # Jumlah bin = jumlah kalimat yang dipilih (minimal)
    num_bins = min(num_select, n_sentences)
    bin_size = n_sentences / num_bins

    targets: List[int] = []
    for b in range(num_bins):
        start = int(b * bin_size)
        end = int((b + 1) * bin_size)
        # Representatif bin = tengah
        mid = (start + end) // 2
        targets.append(min(mid, n_sentences - 1))

    return targets


def _coverage_bonus(
    cand_idx: int,
    n_total: int,
    selected: List[int],
    bin_targets: List[int],
) -> float:
    """
    Menghitung bonus coverage jika kalimat kandidat mewakili bin
    yang belum ada wakilnya di antara kalimat terpilih.
    """
    if not bin_targets or n_total <= 0:
        return 0.0

    num_bins = len(bin_targets)
    bin_size = n_total / num_bins

    # Tentukan bin kandidat
    cand_bin = int(cand_idx / max(bin_size, 1))
    cand_bin = min(cand_bin, num_bins - 1)

    # Tentukan bin yang sudah diwakili
    covered_bins: set = set()
    for sel in selected:
        sel_bin = int(sel / max(bin_size, 1))
        sel_bin = min(sel_bin, num_bins - 1)
        covered_bins.add(sel_bin)

    # Berikan bonus jika bin kandidat belum terwakili
    if cand_bin not in covered_bins:
        return 0.08  # bonus coverage ringan

    return 0.0


def compute_dynamic_summary_length(n_sentences: int) -> int:
    """
    Menghitung jumlah kalimat ringkasan secara dinamis berdasarkan
    panjang artikel.

    Skala (diperbesar agar ringkasan lebih utuh dan memuat banyak info):
    - <= 5  kalimat  → 3 kalimat
    - <= 10 kalimat  → 5 kalimat
    - <= 15 kalimat  → 7 kalimat
    - <= 20 kalimat  → 8 kalimat
    - <= 25 kalimat  → 10 kalimat
    - <= 35 kalimat  → 12 kalimat
    - <= 50 kalimat  → 14 kalimat
    - > 50  kalimat  → 16 kalimat (cap)

    Parameters
    ----------
    n_sentences : int
        Jumlah kalimat dalam artikel.

    Returns
    -------
    int
        Jumlah kalimat target untuk ringkasan.
    """
    if n_sentences <= 5:
        return 3
    elif n_sentences <= 10:
        return 5
    elif n_sentences <= 15:
        return 7
    elif n_sentences <= 20:
        return 8
    elif n_sentences <= 25:
        return 10
    elif n_sentences <= 35:
        return 12
    elif n_sentences <= 50:
        return 14
    else:
        return 16
