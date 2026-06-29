from collections import Counter
from pathlib import Path
import importlib
import random
import re
import sys

import matplotlib.pyplot as plt


CURRENT_DIR = str(Path(__file__).resolve().parent)
ORIGINAL_SYS_PATH = sys.path.copy()
try:
    sys.path = [
        path for path in sys.path
        if path and str(Path(path).resolve()) != CURRENT_DIR
    ]
    WordCloud = importlib.import_module("wordcloud").WordCloud
finally:
    sys.path = ORIGINAL_SYS_PATH


# ======================
# Teks berita asli
# ======================
berita_asli = """
Presiden Rusia Vladimir Putin teleponan dengan Presiden Amerika Serikat (AS) Donald Trump.
Kedua pemimpin tersebut membahas perang di Iran dan Ukraina.
Dilansir kantor berita AFP, Kamis (30/4/2026) ajudan Kremlin Yuri Ushakov mengatakan percakapan telepon itu berlangsung lebih dari 90 menit.
Dia menyebut keduanya saling berterus terang dalam percakapan itu.
Terus terang dan profesional, kata Ushakov kepada wartawan, termasuk dari AFP, selama pengarahan melalui telepon.
Kata Ushakov, Rusia memberikan perhatian khusus pada situasi terkait Iran dan di Teluk Persia.
Putin, katanya, menganggap keputusan Donald Trump untuk memperpanjang gencatan senjata dengan Iran sebagai keputusan yang tepat.
Vladimir Putin menganggap keputusan Donald Trump untuk memperpanjang gencatan senjata dengan Iran sebagai keputusan yang tepat, karena hal ini akan memberi kesempatan pada negosiasi dan, secara keseluruhan, membantu menstabilkan situasi.
Kendati demikian, Putin menyoroti akibat dampak perang AS dengan Iran ini.
Katanya, perang ini bukan hanya merusak Iran tapi juga seluruh komunitas internasional.
Menyoroti konsekuensi yang tak terhindarkan dan sangat merusak bukan hanya bagi Iran dan negara-negara tetangganya, tetapi juga bagi seluruh komunitas internasional, jika AS dan Israel kembali menggunakan aksi militer.
Dia menambahkan bahwa Rusia berkomitmen penuh untuk memberikan setiap bantuan yang mungkin untuk upaya diplomatik terkait perang di Timur Tengah.
Di Washington, Trump mengatakan kepada wartawan bahwa ia telah melakukan percakapan yang sangat baik dengan Putin.
Meskipun ia mengatakan percakapan itu lebih berfokus pada perang Ukraina daripada Iran.
Trump menambahkan bahwa Putin ingin membantu mengakhiri perang AS-Israel di Iran.
"""

# ======================
# Teks ringkasan
# ======================
ringkasan = """
Presiden Rusia Vladimir Putin teleponan dengan Presiden Amerika Serikat (AS) Donald Trump.
Kedua pemimpin tersebut membahas perang di Iran dan Ukraina.
Dilansir kantor berita AFP, Kamis (30/4/2026) ajudan Kremlin Yuri Ushakov mengatakan percakapan telepon itu berlangsung lebih dari 90 menit.
Dia menyebut keduanya saling berterus terang dalam percakapan itu.
Terus terang dan profesional, kata Ushakov kepada wartawan, termasuk dari AFP, selama pengarahan melalui telepon.
Vladimir Putin menganggap keputusan Donald Trump untuk memperpanjang gencatan senjata dengan Iran sebagai keputusan yang tepat, karena hal ini akan memberi kesempatan pada negosiasi dan membantu menstabilkan situasi.
Meskipun ia mengatakan percakapan itu lebih berfokus pada perang Ukraina daripada Iran.
Trump menambahkan bahwa Putin ingin membantu mengakhiri perang AS-Israel di Iran.
"""


# ======================
# Konfigurasi tampilan
# ======================
OUTPUT_FILE = "wordcloud_perbandingan.png"

COLOR_PALETTE = [
    "#d9534f",  # red
    "#f28e2b",  # orange
    "#edc948",  # yellow
    "#59a14f",  # green
    "#4e79a7",  # blue
    "#76b7b2",  # teal
    "#af7aa1",  # purple
    "#bab0ac",  # gray
    "#ff9da7",  # pink
]

stopwords_id = {
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "pada", "itu",
    "ia", "dia", "bahwa", "sebagai", "akan", "lebih", "telah", "juga",
    "oleh", "karena", "dalam", "ini", "tersebut", "terhadap", "atau",
    "bagi", "sangat", "selama", "kepada", "kata", "katanya", "adalah",
    "agar", "atas", "bukan", "hanya", "namun", "kendati", "demikian",
    "hal", "setiap", "mungkin", "secara", "keseluruhan", "para", "pun",
    "tak", "saat", "telah", "sebuah", "seorang", "mereka", "keduanya",
    "termasuk", "melalui", "ujar", "ujarnya", "menurut", "antara",
}


def preprocess(text):
    """Membersihkan teks dan menghitung frekuensi kata penting."""
    text = text.lower()
    text = re.sub(r"as-israel", "as israel", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    filtered_words = [
        word for word in words
        if word not in stopwords_id and len(word) > 2
    ]
    return Counter(filtered_words)


def color_func(*args, **kwargs):
    """Memberi warna acak yang lembut seperti contoh word cloud."""
    return random.choice(COLOR_PALETTE)


def create_wordcloud(frequencies):
    return WordCloud(
        width=1100,
        height=650,
        background_color="white",
        prefer_horizontal=0.95,
        max_words=120,
        min_font_size=9,
        max_font_size=150,
        relative_scaling=0.55,
        collocations=False,
        random_state=42,
        color_func=color_func,
        margin=3,
    ).generate_from_frequencies(frequencies)


def generate_comparison_wordcloud(original_frequencies, summary_frequencies, output_file=None):
    wc_original = create_wordcloud(original_frequencies)
    wc_summary = create_wordcloud(summary_frequencies)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")

    axes[0].imshow(wc_original, interpolation="bilinear")
    axes[0].axis("off")
    axes[0].set_title("Word Cloud Berita Asli", fontsize=18, pad=16)

    axes[1].imshow(wc_summary, interpolation="bilinear")
    axes[1].axis("off")
    axes[1].set_title("Word Cloud Hasil Ringkasan", fontsize=18, pad=16)

    plt.tight_layout(pad=1.5)

    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Word cloud disimpan ke: {output_file}")

    plt.show()


def generate_wordcloud(frequencies, title, output_file=None):
    wc = WordCloud(
        width=1100,
        height=650,
        background_color="white",
        prefer_horizontal=0.95,
        max_words=120,
        min_font_size=9,
        max_font_size=150,
        relative_scaling=0.55,
        collocations=False,
        random_state=42,
        color_func=color_func,
        margin=3,
    ).generate_from_frequencies(frequencies)

    plt.figure(figsize=(11, 6.5), facecolor="white")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, pad=18)
    plt.tight_layout(pad=0)

    if output_file:
        plt.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"Word cloud disimpan ke: {output_file}")

    plt.show()


if __name__ == "__main__":
    frekuensi_berita_asli = preprocess(berita_asli)
    frekuensi_ringkasan = preprocess(ringkasan)

    generate_comparison_wordcloud(
        frekuensi_berita_asli,
        frekuensi_ringkasan,
        output_file=OUTPUT_FILE,
    )
