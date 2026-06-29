"""
app.py
Flask REST API untuk sistem peringkasan berita Indonesia.

Endpoint:
    POST /summarize
        Input  : { "url": "https://..." }
        Output : {
            "title"                  : str,
            "clean_article"          : str,
            "summary_text"           : str,
            "summary_sentences"      : int,
            "summary_words"          : int,
            "total_sentences"        : int,
            "total_words"            : int,
            "total_pages"            : int,
            "error"                  : str | null
        }

Pipeline per request:
    URL → Scraping → Cleaning → Segmentasi → LexRank Scoring
    → MMR Selection → Rekonstruksi → (ROUGE Evaluasi) → Response JSON
"""

import re
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

from scraper import fetch_article
from preprocessor import clean_text, segment_sentences
from lexrank import score_sentences
from mmr import select_sentences_mmr, compute_dynamic_summary_length
from evaluasi import evaluate_rouge
from utils import reconstruct_summary, count_summary_sentences, validate_url

# ---------------------------------------------------------------------------
# Konfigurasi logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inisialisasi Flask
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)  # izinkan cross-origin request (untuk frontend Android/web)

# ---------------------------------------------------------------------------
# Konstanta Post-processing
# ---------------------------------------------------------------------------

# Pola deteksi lineup/susunan pemain
_LINEUP_KEYWORDS = [r"\bxi\s*:", r"susunan\s+pemain", r"starting\s*xi", r"line\s*up"]

# Pola item daftar bernomor (e.g. "1. Cahya Supriadi" atau "1) Budi")
_LIST_ITEM_PAT        = re.compile(r'^\d+\s*[\.\-\)]?\s*\w+', re.IGNORECASE)
_INLINE_LIST_ITEM_PAT = re.compile(r'(^|[\.:\n]\s*)\d{1,2}\.\s+[A-Z]', re.IGNORECASE)

# Rasio maksimum kalimat yang boleh masuk ke summary untuk artikel listicle/skuad.
# Contoh: artikel 35 kalimat → maks 35 * 0.65 ≈ 22 kalimat.
_MAX_LISTICLE_RATIO: float = 0.65


# ---------------------------------------------------------------------------
# Helper post-processing (module-level agar tidak re-compile tiap request)
# ---------------------------------------------------------------------------

def _is_lineup(text: str) -> bool:
    """Deteksi apakah kalimat mengandung pola susunan/lineup pemain."""
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in _LINEUP_KEYWORDS)


def _apply_listicle_postprocess(
    selected: list,
    sentences: list,
    original_count: int,
    target_length: int,
    is_squad: bool,
    is_lineup_present: bool,
    is_numbered_list: bool,
) -> list:
    """
    Menambahkan item daftar yang terlewat oleh MMR ke dalam summary,
    dengan batas atas kontekstual agar summary tidak membengkak tanpa kendali.

    Batas atas = maks antara target_length dan (original_count * _MAX_LISTICLE_RATIO).
    Artinya: ringkasan boleh lebih panjang dari normal untuk artikel listicle,
    tapi tidak boleh melebihi ~65% dari keseluruhan artikel.
    """
    max_allowed = max(target_length, int(original_count * _MAX_LISTICLE_RATIO))

    # --- Blok 1: Lineup & Skuad ---
    if is_lineup_present or is_squad:
        selected_indices = {idx for idx, _ in selected}
        for i, s in enumerate(sentences):
            if len(selected) >= max_allowed:
                break
            if i not in selected_indices:
                if _is_lineup(s):
                    selected.append((i, s))
                    selected_indices.add(i)
                elif is_squad and _LIST_ITEM_PAT.match(s):
                    selected.append((i, s))
                    selected_indices.add(i)
        selected.sort(key=lambda x: x[0])

    # --- Blok 2: Artikel listicle bernomor ---
    if is_numbered_list:
        selected_indices = {idx for idx, _ in selected}
        for i, s in enumerate(sentences):
            if len(selected) >= max_allowed:
                break
            if i not in selected_indices and _INLINE_LIST_ITEM_PAT.search(s):
                selected.append((i, s))
                selected_indices.add(i)
        selected.sort(key=lambda x: x[0])

    return selected


# ---------------------------------------------------------------------------
# Endpoint utama
# ---------------------------------------------------------------------------

@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Endpoint peringkasan berita.

    Request body (JSON):
        {
            "url"       : "https://...",        # wajib
            "reference" : "teks referensi..."   # opsional, untuk evaluasi ROUGE
        }

    Response (JSON):
        {
            "success"                : bool,
            "title"                  : str,
            "summary_text"           : str,
            "summary_sentences"      : int,
            "summary_words"          : int,
            "total_sentences"        : int,
            "total_words"            : int,
            "total_pages"            : int,
            "error"                  : str | null
        }
    """
    # ------------------------------------------------------------------
    # 1. Validasi input
    # ------------------------------------------------------------------
    data = request.get_json(silent=True)
    if not data:
        return _error("Request body harus berupa JSON.", 400)

    url: str = data.get("url", "").strip()
    if not url:
        return _error("Field 'url' wajib diisi.", 400)
    if not validate_url(url):
        return _error(f"URL tidak valid: '{url}'", 400)

    reference: str = data.get("reference", "").strip()  # opsional

    logger.info("Memproses URL: %s", url)

    # ------------------------------------------------------------------
    # 2. Web Scraping & Content Extraction
    # ------------------------------------------------------------------
    try:
        article_data = fetch_article(url)
    except ValueError as exc:
        logger.warning("Gagal scraping: %s", exc)
        return _error(str(exc), 422)

    raw_text: str       = article_data["text"]
    title: str          = article_data.get("title") or ""
    pagination_handled: bool = article_data.get("pagination_handled", False)
    processed_url: str  = article_data.get("processed_url", url)
    total_pages: int    = article_data.get("total_pages", 1)

    if pagination_handled:
        logger.info(
            "Judul: %s | Mode: Multi-Page (%d halaman) | Panjang: %d karakter",
            title, total_pages, len(raw_text),
        )
        logger.info("URL diubah (fetch): %s", processed_url)
    else:
        logger.info(
            "Judul: %s | Mode: Single Page | Panjang raw text: %d karakter",
            title, len(raw_text),
        )

    # ------------------------------------------------------------------
    # 3. Noise Removal & Cleaning
    # ------------------------------------------------------------------
    clean_article: str = clean_text(raw_text)

    if not clean_article.strip():
        return _error("Artikel tidak dapat dibersihkan. Periksa URL dan coba lagi.", 422)

    # ------------------------------------------------------------------
    # 4. Sentence Segmentation
    # ------------------------------------------------------------------
    sentences      = segment_sentences(clean_article)
    original_count = len(sentences)

    logger.info("Jumlah kalimat setelah segmentasi: %d", original_count)

    if original_count < 2:
        return _error(
            "Artikel terlalu pendek untuk diringkas (< 2 kalimat terdeteksi).", 422
        )

    # ------------------------------------------------------------------
    # 5. LexRank Scoring
    # ------------------------------------------------------------------
    scores = score_sentences(sentences, title=title if title else None)

    # ------------------------------------------------------------------
    # 6. Dynamic Summary Length + MMR Selection
    # ------------------------------------------------------------------
    target_length: int = compute_dynamic_summary_length(original_count)
    selected           = select_sentences_mmr(sentences, scores, num_sentences=target_length)

    # ------------------------------------------------------------------
    # 6b. Post-processing: pertahankan item daftar untuk artikel listicle
    #     dan artikel skuad/lineup dengan batas atas kontekstual.
    # ------------------------------------------------------------------
    text_combined_lower    = (title + " " + clean_article).lower()
    is_squad_article       = any(
        kw in text_combined_lower for kw in ["skuad", "pemain", "panggil", "daftar"]
    )
    is_numbered_list_article = len(_INLINE_LIST_ITEM_PAT.findall(clean_article)) >= 3
    has_lineup             = any(_is_lineup(s) for _, s in selected)

    if has_lineup or is_squad_article or is_numbered_list_article:
        selected = _apply_listicle_postprocess(
            selected        = selected,
            sentences       = sentences,
            original_count  = original_count,
            target_length   = target_length,
            is_squad        = is_squad_article,
            is_lineup_present = has_lineup,
            is_numbered_list  = is_numbered_list_article,
        )

    selected_units_count: int = len(selected)
    logger.info(
        "Target unit ringkasan: %d | Terpilih: %d | Batas listicle: %d",
        target_length,
        selected_units_count,
        max(target_length, int(original_count * _MAX_LISTICLE_RATIO)),
    )

    # ------------------------------------------------------------------
    # 7. Rekonstruksi Summary
    # ------------------------------------------------------------------
    summary: str       = reconstruct_summary(selected)
    summary_count: int = count_summary_sentences(summary)

    # ------------------------------------------------------------------
    # 8. Evaluasi ROUGE (opsional)
    # ------------------------------------------------------------------
    rouge_result = None
    if reference:
        rouge_result = evaluate_rouge(summary, reference)
        logger.info(
            "ROUGE-1 F: %.4f | ROUGE-2 F: %.4f | ROUGE-L F: %.4f",
            rouge_result["rouge1"]["fmeasure"],
            rouge_result["rouge2"]["fmeasure"],
            rouge_result["rougeL"]["fmeasure"],
        )

    # ------------------------------------------------------------------
    # 9. Susun response
    # ------------------------------------------------------------------
    response = {
        "success":           True,
        "title":             title,
        "summary_text":      summary,
        "summary_sentences": summary_count,
        "summary_words":     len(summary.split()),
        "total_sentences":   original_count,
        "total_words":       len(clean_article.split()),
        "total_pages":       total_pages,
        "error":             None,
    }

    if rouge_result:
        response["rouge"] = rouge_result

    return jsonify(response), 200


@app.route("/health", methods=["GET"])
def health():
    """Endpoint health check."""
    return jsonify({"status": "ok", "service": "Indonesian News Summarizer API"}), 200


# ---------------------------------------------------------------------------
# Helper privat
# ---------------------------------------------------------------------------

def _error(message: str, status_code: int):
    """Format respons error yang konsisten."""
    logger.error("Error %d: %s", status_code, message)
    return jsonify({
        "success":           False,
        "title":             None,
        "summary_text":      None,
        "summary_sentences": None,
        "summary_words":     None,
        "total_sentences":   None,
        "total_words":       None,
        "total_pages":       None,
        "error":             message,
    }), status_code


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
