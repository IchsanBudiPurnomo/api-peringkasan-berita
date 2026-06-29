# visualize_full.py
"""
Visualisasi lengkap proses peringkasan berita, menampilkan SEMUA kalimat
di setiap tahap: scraping, preprocessing, TF-IDF, LexRank base score,
rincian boosting, seleksi MMR, rekonstruksi ringkasan, dan evaluasi.

PENTING: Skrip ini HANYA memvisualisasikan. Semua perhitungan inti
(LexRank scoring, seleksi MMR, rekonstruksi ringkasan) memanggil
LANGSUNG fungsi yang dipakai app.py/evaluasi.py/tuning.py:
    - lexrank.score_sentences()
    - mmr.select_sentences_mmr()
    - mmr.compute_dynamic_summary_length()
    - utils.reconstruct_summary()

Ini memastikan ringkasan yang ditampilkan di sini SELALU identik dengan
yang dihasilkan oleh API /summarize, termasuk saat parameter sedang
diubah oleh tuning.py (karena modul lexrank & mmr diimpor sebagai objek,
bukan menyalin nilai konstanta saat import).
"""

import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================================
# Impor modul yang sudah ada
# ============================================================================
from scraper import fetch_article
from preprocessor import (
    clean_text, segment_sentences, preprocess_for_tfidf, get_stopwords
)

# Impor modul sebagai objek (bukan simbol individual) agar nilai parameter
# yang sedang aktif ikut terbaca, termasuk jika diubah oleh tuning.py via
# setattr(lexrank_mod, "SIMILARITY_THRESHOLD", ...).
import lexrank as lexrank_mod
import mmr as mmr_mod

from lexrank import (
    score_sentences,
    _compute_title_similarity,
    _info_density_score,
    _generic_title_intent_score,
    _enumeration_boost,
    _expert_quote_boost,
    _recommendation_boost,
    _policy_announcement_boost,
)
from mmr import (
    select_sentences_mmr,
    compute_dynamic_summary_length,
    _compute_coverage_bins,
    _coverage_bonus,
)
from utils import reconstruct_summary, count_summary_sentences
from evaluasi import evaluate_rouge, calculate_5w1h_coverage


# ============================================================================
# Konfigurasi
# ============================================================================
URL = "https://news.detik.com/internasional/d-8467934/putin-dan-trump-teleponan-sampai-90-menit-bahas-perang-di-ukraina-dan-iran"

REFERENCE = (
    'Presiden Rusia Vladimir Putin teleponan dengan Presiden Amerika Serikat (AS) Donald Trump.'
    'Kedua pemimpin tersebut membahas perang di Iran dan Ukraina.'
    'Dilansir kantor berita AFP, Kamis (30/4/2026) ajudan Kremlin Yuri Ushakov mengatakan percakapan telepon itu berlangsung lebih dari 90 menit.'
    'Vladimir Putin menganggap keputusan Donald Trump untuk memperpanjang '
    'gencatan senjata dengan Iran sebagai keputusan yang tepat, karena hal '
    'ini akan memberi kesempatan pada negosiasi dan, secara keseluruhan, '
    'membantu menstabilkan situasi," ujarnya. '
    '"Menyoroti konsekuensi yang tak terhindarkan dan sangat merusak bukan '
    'hanya bagi Iran dan negara-negara tetangganya, tetapi juga bagi seluruh '
    'komunitas internasional, jika AS dan Israel kembali menggunakan aksi '
    'militer," kata Ushakov. '
    'Trump menambahkan bahwa Putin ingin membantu mengakhiri perang AS-Israel '
    'di Iran. '
    'Akan tetapi, dia telah mengatakan kepada pemimpin Rusia itu untuk '
    'mengakhiri invasi ke Ukraina terlebih dahulu.'
)


# ============================================================================
# Fungsi bantu — HANYA untuk keperluan tampilan rincian per-komponen.
# Tidak menentukan kalimat mana yang terpilih; itu murni tugas
# mmr.select_sentences_mmr().
# ============================================================================
def trace_mmr_selection(sentences, scores, num_sentences, tfidf_matrix_for_mmr=None):
    """
    Mereplikasi PERSIS loop internal mmr.select_sentences_mmr() (termasuk
    helper _compute_coverage_bins dan _coverage_bonus yang diimpor langsung
    dari mmr.py, bukan ditulis ulang), tetapi merekam skor MMR di setiap
    iterasi untuk ditampilkan.

    Mengembalikan:
        trace : list of dict, satu entri per iterasi terpilih, berisi
                index, skor MMR saat terpilih, relevance, max_sim,
                coverage_bonus, dan urutan iterasi.
        selected_indices : list[int] urutan index sesuai urutan TERPILIH
                            (bukan urutan posisi artikel).

    Catatan: TF-IDF di sini dihitung ulang dengan setup IDENTIK dengan yang
    dipakai mmr.py (ngram 1-2, stopword Indonesia, sublinear_tf) supaya
    sim_matrix-nya sama. Hasil akhir diverifikasi cocok dengan
    select_sentences_mmr() lewat assert di pemanggil.
    """
    n = len(sentences)
    num_sentences = min(num_sentences, n)
    if num_sentences <= 0 or n == 0:
        return [], []

    preprocessed = [preprocess_for_tfidf(s) for s in sentences]
    stopwords_id = list(get_stopwords())
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), stop_words=stopwords_id, min_df=1, sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(preprocessed)
    except ValueError:
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        selected = ranked[:num_sentences]
        trace = [{
            "iterasi": k + 1, "index": idx, "mmr_score": None,
            "relevance": scores[idx], "max_sim": None, "coverage_bonus": None,
        } for k, idx in enumerate(selected)]
        return trace, selected

    sim_matrix = cosine_similarity(tfidf_matrix).tolist()
    bin_targets = _compute_coverage_bins(n, num_sentences)

    remaining = set(range(n))
    selected_indices = []
    trace = []

    # Seed: kalimat dengan skor tertinggi (identik dengan mmr.py)
    seed = max(remaining, key=lambda i: scores[i])
    selected_indices.append(seed)
    remaining.remove(seed)
    trace.append({
        "iterasi": 1, "index": seed, "mmr_score": None,  # seed tidak melalui formula MMR
        "relevance": scores[seed], "max_sim": None, "coverage_bonus": None,
    })

    iterasi = 2
    while len(selected_indices) < num_sentences and remaining:
        best_idx, best_mmr = -1, -float("inf")
        best_detail = None

        for cand in remaining:
            relevance = scores[cand]
            max_sim = max(sim_matrix[cand][sel] for sel in selected_indices)
            coverage_bonus = _coverage_bonus(cand, n, selected_indices, bin_targets)
            mmr = mmr_mod.MMR_LAMBDA * relevance - (1 - mmr_mod.MMR_LAMBDA) * max_sim + coverage_bonus
            if mmr > best_mmr:
                best_mmr = mmr
                best_idx = cand
                best_detail = (relevance, max_sim, coverage_bonus)

        if best_idx == -1:
            break

        selected_indices.append(best_idx)
        remaining.remove(best_idx)
        relevance, max_sim, coverage_bonus = best_detail
        trace.append({
            "iterasi": iterasi, "index": best_idx, "mmr_score": best_mmr,
            "relevance": relevance, "max_sim": max_sim, "coverage_bonus": coverage_bonus,
        })
        iterasi += 1

    return trace, selected_indices


def compute_score_breakdown(sentences, title, tfidf_matrix, vectorizer):
    """
    Menghitung rincian komponen skor (base LexRank + tiap boost) per kalimat,
    murni untuk ditampilkan ke pengguna. Total di sini harus identik dengan
    nilai yang dikembalikan oleh lexrank.score_sentences(), karena memakai
    bobot W_* yang sama (dibaca dinamis dari lexrank_mod).
    """
    n = len(sentences)
    if n == 0:
        return [], set()

    sim_matrix = cosine_similarity(tfidf_matrix)
    adj_matrix = (sim_matrix >= lexrank_mod.SIMILARITY_THRESHOLD).astype(float)
    np.fill_diagonal(adj_matrix, 0)

    import networkx as nx
    if adj_matrix.sum() == 0:
        base_raw = [1.0 / n] * n
    else:
        graph = nx.from_numpy_array(adj_matrix)
        pr = nx.pagerank(
            graph,
            alpha=lexrank_mod.PAGERANK_DAMPING,
            max_iter=lexrank_mod.PAGERANK_MAX_ITER,
        )
        base_raw = [pr.get(i, 0.0) for i in range(n)]
    max_base = max(base_raw) or 1.0
    base_scores = [s / max_base for s in base_raw]

    title_sim = _compute_title_similarity(tfidf_matrix, vectorizer, title, n)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_sum = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
    top_indices = tfidf_sum.argsort()[-lexrank_mod.TOP_TERMS_K:][::-1]
    top_terms = {feature_names[i] for i in top_indices}

    def _positional_boost(idx, total):
        if total <= 1:
            return 0.0
        return lexrank_mod.W_POSITION * np.exp(-2.5 * (idx / (total - 1)))

    results = []
    for i, sent in enumerate(sentences):
        comp = {
            "index": i,
            "sentence": sent,
            "base": base_scores[i],
            "title": title_sim[i] * lexrank_mod.W_TITLE,
            "position": _positional_boost(i, n),
            "info_density": _info_density_score(sent) * lexrank_mod.W_INFO_DENSITY,
            "intent": _generic_title_intent_score(sent, top_terms) * lexrank_mod.W_TITLE_INTENT,
            "enumeration": _enumeration_boost(sent) * lexrank_mod.W_ENUMERATION,
            "expert": _expert_quote_boost(sent) * lexrank_mod.W_EXPERT,
            "recommendation": _recommendation_boost(sent) * lexrank_mod.W_RECOMMENDATION,
            "policy": _policy_announcement_boost(sent) * lexrank_mod.W_POLICY,
        }
        comp["total"] = sum(
            comp[k] for k in [
                "base", "title", "position", "info_density",
                "intent", "enumeration", "expert", "recommendation", "policy",
            ]
        )
        results.append(comp)
    return results, top_terms


# ============================================================================
# Visualisasi utama
# ============================================================================
def visualize_pipeline(url, reference):
    print("=" * 100)
    print("VISUALISASI LENGKAP PROSES PERINGKASAN BERITA")
    print(f"URL: {url}")
    print("=" * 100)

    # ----------------------------------------------------------------------
    # 1. Scraping
    # ----------------------------------------------------------------------
    print("\n[1. SCRAPING KONTEN ARTIKEL]")
    try:
        article = fetch_article(url)
    except Exception as e:
        print(f"Gagal scraping: {e}")
        sys.exit(1)

    title = article.get("title") or ""
    raw_text = article.get("text") or ""
    print(f"Judul                : {title}")
    print(f"Panjang teks mentah  : {len(raw_text):,} karakter")
    print("Cuplikan 300 karakter pertama:")
    print(raw_text[:300].replace("\n", " ") + "...")

    # ----------------------------------------------------------------------
    # 2. Preprocessing (sesuai preprocessor.clean_text + segment_sentences)
    # ----------------------------------------------------------------------
    print("\n[2. PREPROCESSING TEKS]")
    clean_article = clean_text(raw_text)
    sentences = segment_sentences(clean_article)
    n = len(sentences)
    print(f"Jumlah kalimat tersegmentasi : {n}")

    if n < 2:
        print("Artikel terlalu pendek untuk diringkas (< 2 kalimat). Berhenti.")
        sys.exit(1)

    print("\nSemua kalimat hasil segmentasi:")
    for i, sent in enumerate(sentences, start=1):
        print(f"  K{i:2d}: {sent}")

    # Contoh preprocessing untuk satu kalimat representatif
    example_idx = min(4, n - 1)
    example_sent = sentences[example_idx]
    print(f"\nContoh preprocessing (Kalimat K{example_idx + 1}):")
    print(f"  Kalimat asli        : {example_sent}")
    print(f"  Hasil preprocess_for_tfidf() (lowercase+stopword+stem yang dipakai sistem):")
    print(f"    {preprocess_for_tfidf(example_sent)}")

    # ----------------------------------------------------------------------
    # 3. TF-IDF Vectorization — replikasi setup yang dipakai lexrank.py/mmr.py
    #    (hanya untuk menampilkan bobot kata; LexRank asli dihitung ulang
    #    secara internal oleh score_sentences()).
    # ----------------------------------------------------------------------
    print("\n[3a. TF-IDF VECTORIZATION]")
    preprocessed = [preprocess_for_tfidf(s) for s in sentences]
    stopwords_id = list(get_stopwords())
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words=stopwords_id,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(preprocessed)
    feature_names = vectorizer.get_feature_names_out()
    print(f"Dimensi matriks TF-IDF  : {tfidf_matrix.shape} (kalimat x fitur)")
    print(f"Jumlah fitur (kata unik): {len(feature_names)}")

    print("\nBobot TF-IDF per kalimat (top-5 kata):")
    for i in range(n):
        vec = tfidf_matrix[i].toarray().flatten()
        top_idx = vec.argsort()[-5:][::-1]
        top_words = [(feature_names[j], vec[j]) for j in top_idx if vec[j] > 0]
        word_str = ", ".join(f"{w}:{v:.4f}" for w, v in top_words)
        print(f"  K{i + 1:2d}: {word_str}")

    # ----------------------------------------------------------------------
    # 4. LexRank — PANGGIL LANGSUNG score_sentences() dari lexrank.py
    # ----------------------------------------------------------------------
    print("\n[3b/4. LEXRANK + BOOSTING (via lexrank.score_sentences)]")
    print(f"Parameter aktif saat ini:")
    print(f"  SIMILARITY_THRESHOLD = {lexrank_mod.SIMILARITY_THRESHOLD}")
    print(f"  PAGERANK_DAMPING     = {lexrank_mod.PAGERANK_DAMPING}")
    print(f"  W_TITLE              = {lexrank_mod.W_TITLE}")
    print(f"  W_POSITION           = {lexrank_mod.W_POSITION}")
    print(f"  W_INFO_DENSITY       = {lexrank_mod.W_INFO_DENSITY}")
    print(f"  W_TITLE_INTENT       = {lexrank_mod.W_TITLE_INTENT}")
    print(f"  W_ENUMERATION        = {lexrank_mod.W_ENUMERATION}")
    print(f"  W_EXPERT             = {lexrank_mod.W_EXPERT}")
    print(f"  W_RECOMMENDATION     = {lexrank_mod.W_RECOMMENDATION}")
    print(f"  W_POLICY             = {lexrank_mod.W_POLICY}")

    # Skor FINAL yang benar-benar dipakai sistem (app.py memanggil ini juga).
    final_scores = score_sentences(sentences, title=title if title else None)

    # Rincian per-komponen, HANYA untuk ditampilkan (tidak memengaruhi seleksi).
    components, top_terms = compute_score_breakdown(sentences, title, tfidf_matrix, vectorizer)

    print(f"\nTop-{lexrank_mod.TOP_TERMS_K} terms TF-IDF artikel:")
    shown_terms = list(top_terms)[:15]
    print("  " + ", ".join(shown_terms) + (" ..." if len(top_terms) > 15 else ""))

    df_boost = pd.DataFrame([{
        "Kal": c["index"] + 1,
        "Base": c["base"],
        "Title": c["title"],
        "Pos": c["position"],
        "Expert": c["expert"],
        "Policy": c["policy"],
        "Rec": c["recommendation"],
        "Enum": c["enumeration"],
        "Info": c["info_density"],
        "Intent": c["intent"],
        "Total (breakdown)": c["total"],
        "Total (score_sentences)": final_scores[c["index"]],
    } for c in components])

    print("\nKomponen boosting per kalimat (semua):")
    print(df_boost.to_string(index=False, float_format="{:.6f}".format))

    # Sanity check: breakdown manual harus identik dengan fungsi resmi.
    max_diff = max(
        abs(c["total"] - final_scores[c["index"]]) for c in components
    )
    if max_diff > 1e-9:
        print(
            f"\n[PERINGATAN] Selisih breakdown vs score_sentences() = {max_diff:.2e}. "
            "Periksa apakah bobot W_* di atas sudah konsisten dengan lexrank.py."
        )

    # ----------------------------------------------------------------------
    # 5. Seleksi MMR — PANGGIL LANGSUNG select_sentences_mmr() dari mmr.py
    # ----------------------------------------------------------------------
    print("\n[5. SELEKSI KALIMAT DENGAN MMR]")
    target_length = compute_dynamic_summary_length(n)
    print(f"Target jumlah kalimat ringkasan : {target_length}")
    print(f"MMR_LAMBDA aktif (mmr.py)       : {mmr_mod.MMR_LAMBDA}")

    selected = select_sentences_mmr(sentences, final_scores, num_sentences=target_length)
    selected_indices_sorted_by_position = [idx for idx, _ in selected]

    # Jalankan trace untuk menangkap skor MMR di tiap iterasi (formula &
    # helper coverage yang dipakai SAMA PERSIS dengan mmr.py).
    trace, trace_selected_order = trace_mmr_selection(sentences, final_scores, target_length)

    # Verifikasi: himpunan kalimat yang terpilih trace harus identik dengan
    # himpunan yang dikembalikan select_sentences_mmr() asli.
    if set(trace_selected_order) != set(selected_indices_sorted_by_position):
        print(
            "\n[PERINGATAN] Trace MMR tidak menghasilkan himpunan kalimat yang "
            "sama dengan select_sentences_mmr() asli — tabel skor MMR di "
            "bawah TIDAK BISA dipercaya untuk kasus ini."
        )
    else:
        print("[OK] Trace MMR menghasilkan himpunan kalimat identik dengan "
              "select_sentences_mmr() asli.")

    # ------------------------------------------------------------------
    # Satu tabel ringkas: peringkat, kalimat, skor sebelum pengurangan
    # redundansi (LexRank+Boost), dan skor setelah (skor MMR/total).
    #
    # Untuk kalimat seed (peringkat 1), tidak ada kompetisi formula MMR
    # (max_sim & coverage_bonus = 0 karena belum ada kalimat lain untuk
    # dibandingkan), sehingga skor setelahnya secara matematis dihitung
    # sebagai λ × relevance — konsisten dengan formula MMR_LAMBDA * relevance
    # - (1-MMR_LAMBDA) * max_sim + coverage_bonus di mmr.py.
    # ------------------------------------------------------------------
    rows = []
    for rank, t in enumerate(trace, start=1):
        skor_sebelum = t["relevance"]
        if t["mmr_score"] is not None:
            skor_setelah = t["mmr_score"]
        else:
            skor_setelah = mmr_mod.MMR_LAMBDA * skor_sebelum
        rows.append({
            "Peringkat": rank,
            "Kalimat": t["index"] + 1,
            "Skor Sebelum (LexRank+Boost)": skor_sebelum,
            "Skor Setelah (MMR)": skor_setelah,
        })
    df_mmr_rank = pd.DataFrame(rows)

    print(f"\n[HASIL SELEKSI MMR] (target={target_length} kalimat, "
          f"MMR_LAMBDA={mmr_mod.MMR_LAMBDA})")
    print(df_mmr_rank.to_string(index=False, float_format="{:.6f}".format))

    # ----------------------------------------------------------------------
    # 6. Rekonstruksi Ringkasan — PANGGIL LANGSUNG reconstruct_summary()
    #    Catatan: hasil 'selected' SUDAH terurut berdasarkan posisi asli
    #    (logical order), bukan berdasarkan skor. Ini sengaja dipertahankan
    #    karena reconstruct_summary() di utils.py mengasumsikan urutan
    #    logis artikel agar ringkasan tetap koheren dan tidak melompat-lompat.
    # ----------------------------------------------------------------------
    print("\n[6. REKONSTRUKSI RINGKASAN]")
    summary = reconstruct_summary(selected)
    summary_sentence_count = count_summary_sentences(summary)

    print("Ringkasan akhir (urutan sesuai posisi asli di artikel — "
          "identik dengan output endpoint /summarize):")
    print("=" * 90)
    print(summary)
    print("=" * 90)
    print(f"Jumlah kalimat (count_summary_sentences) : {summary_sentence_count}")
    print(f"Jumlah kata                               : {len(summary.split())}")

    # ----------------------------------------------------------------------
    # 7. Evaluasi ROUGE & 5W1H
    # ----------------------------------------------------------------------
    print("\n[7. EVALUASI RINGKASAN]")
    rouge = evaluate_rouge(summary, reference)
    print("ROUGE Scores:")
    print(f"  ROUGE-1 : Precision={rouge['rouge1']['precision']:.6f}  "
          f"Recall={rouge['rouge1']['recall']:.6f}  F-measure={rouge['rouge1']['fmeasure']:.6f}")
    print(f"  ROUGE-2 : Precision={rouge['rouge2']['precision']:.6f}  "
          f"Recall={rouge['rouge2']['recall']:.6f}  F-measure={rouge['rouge2']['fmeasure']:.6f}")
    print(f"  ROUGE-L : Precision={rouge['rougeL']['precision']:.6f}  "
          f"Recall={rouge['rougeL']['recall']:.6f}  F-measure={rouge['rougeL']['fmeasure']:.6f}")

    cov = calculate_5w1h_coverage(summary, reference)
    print("\nCoverage 5W1H (Tingkat ketangkapan fakta):")
    valid_cov = []
    for k, v in cov.items():
        if v is not None:
            valid_cov.append(v)
            print(f"  {k.upper():<8}: {v * 100:.2f}%")
        else:
            print(f"  {k.upper():<8}: N/A (referensi tidak memiliki elemen ini)")
    if valid_cov:
        avg_cov = sum(valid_cov) / len(valid_cov)
        print(f"  RATA-RATA : {avg_cov * 100:.2f}%")

    print("\n" + "=" * 90)
    print("VISUALISASI SELESAI")


if __name__ == "__main__":
    visualize_pipeline(URL, REFERENCE)