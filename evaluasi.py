"""
evaluasi.py
Modul gabungan untuk mengevaluasi kualitas ringkasan sistem 
secara batch menggunakan dua pendekatan:
1. ROUGE (Precision, Recall, F-Measure)
2. Coverage 5W1H (Rule-Based Heuristic)
"""

import os
import re
import pandas as pd
import requests
from typing import Dict, Set
from tqdm import tqdm
from rouge_score import rouge_scorer

# ==============================================================================
# FUNGSI ROUGE (Juga digunakan oleh app.py)
# ==============================================================================

def evaluate_rouge(summary: str, reference: str) -> Dict[str, Dict[str, float]]:
    if not summary or not reference:
        return _empty_rouge()

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=False,
    )
    scores = scorer.score(reference, summary)

    return {
        "rouge1": {
            "precision": round(scores["rouge1"].precision, 4),
            "recall":    round(scores["rouge1"].recall, 4),
            "fmeasure":  round(scores["rouge1"].fmeasure, 4),
        },
        "rouge2": {
            "precision": round(scores["rouge2"].precision, 4),
            "recall":    round(scores["rouge2"].recall, 4),
            "fmeasure":  round(scores["rouge2"].fmeasure, 4),
        },
        "rougeL": {
            "precision": round(scores["rougeL"].precision, 4),
            "recall":    round(scores["rougeL"].recall, 4),
            "fmeasure":  round(scores["rougeL"].fmeasure, 4),
        },
    }

def _empty_rouge() -> Dict[str, Dict[str, float]]:
    empty = {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
    return {"rouge1": empty.copy(), "rouge2": empty.copy(), "rougeL": empty.copy()}


# ==============================================================================
# FUNGSI COVERAGE 5W1H
# ==============================================================================

def extract_who(text: str) -> Set[str]:
    words = re.findall(r'\b[A-Z][a-z]+\b', text)
    stopwords = {"Dan", "Di", "Ke", "Dari", "Yang", "Untuk", "Pada", "Dalam", "Itu", "Ini", "Sebagai", "Dengan"}
    return {w for w in words if w not in stopwords}

def extract_when(text: str) -> Set[str]:
    patterns = [
        r'\b(?:senin|selasa|rabu|kamis|jumat|sabtu|minggu)\b',
        r'\b(?:januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)\b',
        r'\b\d{4}\b',
        r'\b\d{1,2}:\d{2}\b',
        r'\b(?:wib|wita|wit)\b',
        r'\b(?:hari\s+ini|besok|kemarin|pagi|siang|sore|malam)\b'
    ]
    extracted = set()
    for p in patterns:
        matches = re.findall(p, text, re.IGNORECASE)
        extracted.update([m.lower() for m in matches])
    return extracted

def extract_where(text: str) -> Set[str]:
    matches = re.findall(r'\b(?:di|ke|dari)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text)
    return set(matches)

def extract_why(text: str) -> Set[str]:
    keywords = ["karena", "sebab", "dikarenakan", "lantaran", "akibat", "pasalnya", "alasan"]
    text_lower = text.lower()
    return {k for k in keywords if k in text_lower}

def extract_how(text: str) -> Set[str]:
    keywords = ["dengan", "melalui", "lewat", "secara", "menggunakan", "berupa"]
    text_lower = text.lower()
    return {k for k in keywords if k in text_lower}

def extract_what(text: str) -> Set[str]:
    words = re.findall(r'\b[a-z]{5,}\b', text.lower())
    stopwords = {
        "adalah", "merupakan", "bahwa", "untuk", "dalam", "dengan", "kepada", 
        "sebagai", "tersebut", "karena", "sehingga", "setelah", "menjadi"
    }
    return {w for w in words if w not in stopwords}

def calculate_5w1h_coverage(system_summary: str, reference_summary: str) -> Dict[str, float]:
    ref_elements = {
        "who": extract_who(reference_summary),
        "when": extract_when(reference_summary),
        "where": extract_where(reference_summary),
        "what": extract_what(reference_summary),
        "why": extract_why(reference_summary),
        "how": extract_how(reference_summary),
    }
    sys_elements = {
        "who": extract_who(system_summary),
        "when": extract_when(system_summary),
        "where": extract_where(system_summary),
        "what": extract_what(system_summary),
        "why": extract_why(system_summary),
        "how": extract_how(system_summary),
    }

    scores = {}
    for key in ref_elements:
        ref_set = ref_elements[key]
        sys_set = sys_elements[key]
        if not ref_set:
            scores[key] = None
        else:
            overlap = ref_set.intersection(sys_set)
            scores[key] = len(overlap) / len(ref_set)
    return scores


# ==============================================================================
# SCRIPT EVALUASI BATCH (GABUNGAN)
# ==============================================================================

def run_evaluation():
    API_URL = "http://127.0.0.1:5000/summarize"
    dataset_file = "dataset.xlsx"

    if not os.path.exists(dataset_file):
        print(f"File {dataset_file} tidak ditemukan di direktori saat ini.")
        return

    print(f"Memuat {dataset_file}...")
    try:
        df = pd.read_excel(dataset_file)
    except Exception as e:
        print(f"Gagal membaca excel: {e}")
        return

    # Deteksi kolom URL dan Referensi
    url_col, ref_col = None, None
    for c in df.columns:
        cl = str(c).lower()
        if 'url' in cl or 'link' in cl:
            url_col = c
        if 'reference' in cl or 'ringkasan' in cl or 'summary' in cl or 'referensi' in cl:
            ref_col = c

    if not url_col or not ref_col:
        print("Tidak dapat menemukan kolom URL atau Referensi secara otomatis.")
        return

    print(f"Menggunakan kolom URL: '{url_col}' dan kolom Referensi: '{ref_col}'")

    results = []
    # ROUGE aggregates
    total_r1_p, total_r1_r, total_r1_f = 0, 0, 0
    total_r2_p, total_r2_r, total_r2_f = 0, 0, 0
    total_rl_p, total_rl_r, total_rl_f = 0, 0, 0
    valid_count = 0

    print("Memulai evaluasi komprehensif (ROUGE & 5W1H)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        url = str(row[url_col]).strip()
        reference = str(row[ref_col]).strip()

        if pd.isna(row[url_col]) or not url.startswith("http"):
            continue

        try:
            payload = {"url": url, "reference": reference}
            resp = requests.post(API_URL, json=payload, timeout=60)
            
            if resp.status_code == 200:
                data = resp.json()
                if data.get("success") and "rouge" in data:
                    r = data["rouge"]
                    
                    r1_p, r1_r, r1_f = r["rouge1"]["precision"], r["rouge1"]["recall"], r["rouge1"]["fmeasure"]
                    r2_p, r2_r, r2_f = r["rouge2"]["precision"], r["rouge2"]["recall"], r["rouge2"]["fmeasure"]
                    rl_p, rl_r, rl_f = r["rougeL"]["precision"], r["rougeL"]["recall"], r["rougeL"]["fmeasure"]
                    
                    sys_summ = data.get("summary_text", "")
                    
                    # 5W1H Evaluation
                    cov = calculate_5w1h_coverage(sys_summ, reference)
                    valid_cov_scores = [v for v in cov.values() if v is not None]
                    avg_coverage = sum(valid_cov_scores) / len(valid_cov_scores) if valid_cov_scores else 0.0

                    results.append({
                        "url": url,
                        "title": data.get("title", ""),
                        "system_summary": sys_summ,
                        "reference_summary": reference,
                        "rouge1_precision": r1_p,
                        "rouge1_recall": r1_r,
                        "rouge1_fmeasure": r1_f,
                        "rouge2_precision": r2_p,
                        "rouge2_recall": r2_r,
                        "rouge2_fmeasure": r2_f,
                        "rougeL_precision": rl_p,
                        "rougeL_recall": rl_r,
                        "rougeL_fmeasure": rl_f,
                        "who_coverage": cov["who"],
                        "what_coverage": cov["what"],
                        "where_coverage": cov["where"],
                        "when_coverage": cov["when"],
                        "why_coverage": cov["why"],
                        "how_coverage": cov["how"],
                        "average_5w1h_coverage": avg_coverage
                    })

                    total_r1_p += r1_p; total_r1_r += r1_r; total_r1_f += r1_f
                    total_r2_p += r2_p; total_r2_r += r2_r; total_r2_f += r2_f
                    total_rl_p += rl_p; total_rl_r += rl_r; total_rl_f += rl_f
                    valid_count += 1
                else:
                    print(f"Gagal/Data tidak lengkap untuk {url}")
            else:
                print(f"Error HTTP {resp.status_code} untuk {url}")
        except Exception as e:
            print(f"Exception saat memproses {url}: {e}")

    if valid_count > 0:
        avg_r1_p, avg_r1_r, avg_r1_f = total_r1_p/valid_count, total_r1_r/valid_count, total_r1_f/valid_count
        avg_r2_p, avg_r2_r, avg_r2_f = total_r2_p/valid_count, total_r2_r/valid_count, total_r2_f/valid_count
        avg_rl_p, avg_rl_r, avg_rl_f = total_rl_p/valid_count, total_rl_r/valid_count, total_rl_f/valid_count

        out_df = pd.DataFrame(results)

        print("\n=== HASIL EVALUASI RATA-RATA GLOBAL ===")
        print(f"Total Artikel Berhasil Dievaluasi : {valid_count}\n")
        
        print("[1. ROUGE]")
        print(f"ROUGE-1: Precision: {avg_r1_p:.4f} | Recall: {avg_r1_r:.4f} | F-Measure: {avg_r1_f:.4f}")
        print(f"ROUGE-2: Precision: {avg_r2_p:.4f} | Recall: {avg_r2_r:.4f} | F-Measure: {avg_r2_f:.4f}")
        print(f"ROUGE-L: Precision: {avg_rl_p:.4f} | Recall: {avg_rl_r:.4f} | F-Measure: {avg_rl_f:.4f}")
        
        print("\n[2. COVERAGE 5W1H (Tingkat Ketangkapan Fakta)]")
        for col in ["who_coverage", "what_coverage", "where_coverage", "when_coverage", "why_coverage", "how_coverage", "average_5w1h_coverage"]:
            val = out_df[col].mean()
            if pd.notna(val):
                print(f"{col.replace('_', ' ').upper():<25} : {val * 100:.2f}%")

        # Format and save Excel
        out_file = "evaluasi_lengkap.xlsx"
        for col in out_df.select_dtypes(include=['float64', 'float32']).columns:
            out_df[col] = out_df[col].round(4)
        
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Evaluasi')
            ws = writer.sheets['Evaluasi']
            ws.freeze_panes = 'A2'
            
            from openpyxl.utils import get_column_letter
            for i, col in enumerate(out_df.columns):
                col_letter = get_column_letter(i + 1)
                max_len = max(
                    out_df[col].map(lambda x: len(str(x))).max() if not out_df[col].empty else 0,
                    len(str(col))
                )
                ws.column_dimensions[col_letter].width = min(max_len + 2, 60)

        print(f"\n[OK] Hasil evaluasi gabungan telah disimpan ke format rapi: {out_file}")
    else:
        print("Tidak ada artikel yang berhasil dievaluasi.")

if __name__ == "__main__":
    run_evaluation()
