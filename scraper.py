"""
scraper.py
Modul untuk mengambil konten artikel berita dari URL menggunakan trafilatura.
Trafilatura secara otomatis mengekstrak konten utama artikel tanpa perlu
selector manual per situs berita.
"""

import re
import requests
import trafilatura
from html import unescape
from typing import Optional, Dict, List, Tuple, Set
from urllib.parse import urljoin, urlparse, urlencode, parse_qsl, urlunparse


# ---------------------------------------------------------------------------
# Header HTTP yang realistis agar tidak diblokir bot-protection
# ---------------------------------------------------------------------------
_HEADERS: Dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/webp,*/*;q=0.8"
    ),
}

_REQUEST_TIMEOUT: int = 20  # detik
_MAX_PAGINATION_PAGES: int = 12
_PAGINATION_HINT_RE = re.compile(
    r"(?:paging|pagination|pages?|pager|halaman|nav-page|page-nav|next|"
    r"selanjutnya|lanjut|berikutnya)",
    re.IGNORECASE,
)
_NEXT_TEXT_RE = re.compile(
    r"\b(?:next|selanjutnya|lanjut|berikutnya|halaman\s+berikut|next\s+page)\b|[›»]",
    re.IGNORECASE,
)
_PAGE_QUERY_KEY_RE = re.compile(
    r"^(?:page|p|pg|paged|halaman|hlm|page_num|page_number|pageNumber)$",
    re.IGNORECASE,
)


def _set_query_param(url: str, key: str, value: str) -> str:
    """Return URL dengan satu query parameter diganti/ditambahkan."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query[key] = value
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        parsed.params,
        urlencode(query),
        parsed.fragment,
    ))


def _candidate_single_page_urls(url: str) -> List[str]:
    """
    Buat kandidat URL single/all-page secara generik tanpa daftar domain.

    Kandidat hanya akan dipakai jika hasil ekstraksinya terbukti lebih lengkap,
    jadi aman untuk dicoba pada situs yang tidak mendukung parameter tersebut.
    """
    candidates = [
        _set_query_param(url, "page", "all"),
        _set_query_param(url, "single", "1"),
        _set_query_param(url, "output", "1"),
    ]
    deduped: List[str] = []
    for candidate in candidates:
        if candidate != url and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _fetch_html(url: str) -> str:
    """Download HTML dengan header dan error handling yang konsisten."""
    response = requests.get(url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.text


def _extract_main_text(html_content: str) -> Optional[str]:
    """Ekstrak teks utama artikel dari HTML memakai trafilatura."""
    return trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
        deduplicate=True,
    )


def _same_article_family(candidate_url: str, current_url: str) -> bool:
    """
    Filter ringan agar crawler pagination tidak melebar ke berita lain.
    Banyak situs memakai URL halaman 2 dengan path sama + query page=2,
    atau path artikel yang sama ditambah /2.
    """
    candidate = urlparse(candidate_url)
    current = urlparse(current_url)

    if candidate.netloc.lower() != current.netloc.lower():
        return False

    def canonical_article_path(path: str) -> str:
        path = path.rstrip("/")
        path = re.sub(r"/amp$", "", path)
        path = re.sub(r"/page/\d{1,2}$", "", path)
        path = re.sub(r"/\d{1,2}$", "", path)
        path = re.sub(r"(?:[-_])page(?:[-_])?\d{1,2}$", "", path, flags=re.IGNORECASE)
        path = re.sub(r"(?:[-_])\d{1,2}$", "", path)
        path = re.sub(r"-\d{6,}$", "", path)
        return path

    candidate_path = candidate.path.rstrip("/")
    current_path = current.path.rstrip("/")
    candidate_base = canonical_article_path(candidate_path)
    current_base = canonical_article_path(current_path)
    if candidate_path == current_path:
        return True

    # Bentuk umum: /judul-berita/2 atau /judul-berita?page=2.
    return candidate_base == current_base or candidate_path.startswith(current_base + "/")


def _page_number_from_url(url: str) -> Optional[int]:
    """Ambil nomor halaman dari query/path jika ada."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    for key, value in query.items():
        if not _PAGE_QUERY_KEY_RE.search(key):
            continue
        if value and value.isdigit():
            page_num = int(value)
            if 1 <= page_num <= _MAX_PAGINATION_PAGES:
                return page_num

    match = re.search(r"/(?:page/)?(\d{1,2})/?$", parsed.path.rstrip("/"), re.IGNORECASE)
    if match:
        page_num = int(match.group(1))
        if 1 <= page_num <= _MAX_PAGINATION_PAGES:
            return page_num

    match = re.search(r"(?:[-_])page(?:[-_])?(\d{1,2})/?$", parsed.path, re.IGNORECASE)
    if match:
        page_num = int(match.group(1))
        if 1 <= page_num <= _MAX_PAGINATION_PAGES:
            return page_num

    return None


def _strip_html(text: str) -> str:
    """Ubah HTML kecil seperti isi anchor menjadi teks polos."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_attrs(tag: str) -> Dict[str, str]:
    """Ambil atribut HTML dari satu tag secara ringan."""
    attrs: Dict[str, str] = {}
    for match in re.finditer(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*(["'])(.*?)\2""", tag, re.DOTALL):
        attrs[match.group(1).lower()] = unescape(match.group(3).strip())
    for match in re.finditer(r"""([a-zA-Z_:][-a-zA-Z0-9_:.]*)\s*=\s*([^\s"'=<>`]+)""", tag):
        attrs.setdefault(match.group(1).lower(), unescape(match.group(2).strip()))
    return attrs


def _pagination_context_score(tag: str, text: str) -> int:
    """Nilai indikasi bahwa sebuah link adalah bagian pagination."""
    attrs = _extract_attrs(tag)
    haystack = " ".join([
        text,
        attrs.get("rel", ""),
        attrs.get("aria-label", ""),
        attrs.get("title", ""),
        attrs.get("class", ""),
        attrs.get("id", ""),
    ])

    score = 0
    if _PAGINATION_HINT_RE.search(haystack):
        score += 2
    if _NEXT_TEXT_RE.search(haystack):
        score += 3
    if text.isdigit():
        score += 2
    if re.search(r"\bhalaman\s+\d{1,2}\b|\bpage\s+\d{1,2}\b", haystack, re.IGNORECASE):
        score += 2
    return score


def _extract_pagination_links(html_content: str, current_url: str) -> list[str]:
    """
    Mendeteksi link pagination secara dinamis tanpa daftar domain.

    Sumber sinyal:
    - <link rel="next" href="...">
    - anchor dengan rel/aria-label/title/class/id pagination
    - teks anchor berupa angka atau next/selanjutnya
    - URL yang mengandung nomor halaman pada query/path
    """
    page_urls: Set[Tuple[int, str]] = set()
    next_urls: List[str] = []

    def add_candidate(href: str, page_num: Optional[int] = None, is_next: bool = False) -> None:
        href = href.strip()
        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
            return

        full_url = urljoin(current_url, href)
        if not _same_article_family(full_url, current_url):
            return

        page_num = page_num or _page_number_from_url(full_url)
        if page_num and page_num > 1:
            page_urls.add((page_num, full_url))
        elif is_next and full_url not in next_urls:
            next_urls.append(full_url)

    for match in re.finditer(r"""<link\b[^>]*>""", html_content, re.DOTALL | re.IGNORECASE):
        tag = match.group(0)
        attrs = _extract_attrs(tag)
        if "next" in attrs.get("rel", "").lower():
            add_candidate(attrs.get("href", ""), is_next=True)

    anchor_re = re.compile(r"""<a\b([^>]*)>(.*?)</a>""", re.DOTALL | re.IGNORECASE)
    for match in anchor_re.finditer(html_content):
        attr_text = match.group(1)
        attrs = _extract_attrs(attr_text)
        href = attrs.get("href") or attrs.get("data-href") or attrs.get("data-url") or ""
        text = _strip_html(match.group(2)).lower()
        if not href:
            continue

        score = _pagination_context_score(attr_text, text)
        page_num = int(text) if text.isdigit() else _page_number_from_url(urljoin(current_url, href))
        if page_num and page_num > 1:
            score += 2

        if score >= 2:
            add_candidate(href, page_num=page_num, is_next=bool(_NEXT_TEXT_RE.search(text)))
                        
    # Urutkan berdasarkan nomor halaman, ambil satu URL pertama untuk setiap nomor.
    pages_by_number: Dict[int, str] = {}
    for page_num, page_url in sorted(page_urls, key=lambda x: x[0]):
        pages_by_number.setdefault(page_num, page_url)

    sorted_pages = list(pages_by_number.values())
    for next_url in next_urls:
        if next_url not in sorted_pages:
            sorted_pages.append(next_url)
    return sorted_pages[:_MAX_PAGINATION_PAGES - 1]


def _deduplicate_join_texts(texts: List[str]) -> str:
    """
    Gabungkan halaman artikel sambil membuang paragraf duplikat.
    Halaman berikut sering mengulang lead, judul, atau boilerplate artikel.
    """
    seen: Set[str] = set()
    paragraphs: List[str] = []

    for text in texts:
        for paragraph in re.split(r"\n+", text.strip()):
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if len(paragraph) < 30:
                continue
            key = re.sub(r"\W+", "", paragraph.lower())
            if key in seen:
                continue
            seen.add(key)
            paragraphs.append(paragraph)

    return "\n\n".join(paragraphs)


def _strip_repeated_title(text: str, title: Optional[str]) -> str:
    """Hapus judul yang ikut masuk di awal teks halaman artikel."""
    text = "\n".join(
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.strip().splitlines()
        if line.strip()
    )
    if not title:
        return text

    title_pattern = re.escape(re.sub(r"\s+", " ", title.strip()))
    text = re.sub(rf"^(?:{title_pattern})\s*", "", text, flags=re.IGNORECASE)
    return text.strip()


def _extract_title(html_content: str) -> Optional[str]:
    """Ekstrak judul dari metadata trafilatura, lalu fallback ke tag title."""
    metadata = trafilatura.extract_metadata(html_content)
    title: Optional[str] = None

    if metadata:
        title = metadata.title or metadata.sitename

    if not title:
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title = unescape(re.sub(r"\s+", " ", title_match.group(1))).strip()
            title = re.split(r"\s*[\|–\-—]\s*", title)[0].strip()

    return title


def fetch_article(url: str) -> Dict[str, Optional[str]]:
    """
    Mengambil dan mengekstrak konten artikel dari URL berita.

    Pipeline:
    1. Download HTML mentah via requests
    2. Deteksi pagination secara dinamis untuk mengambil halaman berikutnya jika ada
    3. Ekstrak konten utama menggunakan trafilatura dari setiap halaman
    4. Gabungkan seluruh konten teks halaman
    5. Ekstrak judul dari metadata trafilatura halaman pertama

    Parameters
    ----------
    url : str
        URL artikel berita yang akan diringkas.

    Returns
    -------
    dict dengan key:
        - "title"       : str | None — judul artikel
        - "text"        : str | None — konten artikel bersih gabungan
        - "url"         : str        — URL asli
        - "total_pages" : int        — jumlah total halaman yang dirayap

    Raises
    ------
    ValueError
        Jika gagal mengambil atau mengekstrak konten artikel.
    """
    # ------------------------------------------------------------------
    # 1. Download HTML asli, lalu coba URL single/all-page generik
    # ------------------------------------------------------------------
    try:
        original_html = _fetch_html(url)
    except requests.exceptions.RequestException as exc:
        raise ValueError(f"Gagal mengunduh URL '{url}': {exc}") from exc

    html_content = original_html
    fetch_url = url
    original_text = _extract_main_text(original_html) or ""
    extracted_text = original_text
    original_text_len = len(original_text.strip())
    best_processed_text_len = original_text_len

    for candidate_url in _candidate_single_page_urls(url):
        try:
            processed_html = _fetch_html(candidate_url)
            processed_text = _extract_main_text(processed_html) or ""
            processed_text_len = len(processed_text.strip())
            # Pakai versi all/single-page hanya jika isinya valid dan tidak lebih pendek.
            if processed_text_len > max(best_processed_text_len, 100):
                html_content = processed_html
                fetch_url = candidate_url
                extracted_text = processed_text
                best_processed_text_len = processed_text_len
        except requests.exceptions.RequestException:
            # Jika kandidat tidak didukung, lanjut dengan HTML terbaik yang sudah ada.
            continue

    # ------------------------------------------------------------------
    # 2. Deteksi pagination dinamis dari HTML asli dan HTML yang dipakai
    # ------------------------------------------------------------------
    next_page_urls: List[str] = []
    discovered = _extract_pagination_links(original_html, url)
    if html_content != original_html:
        discovered.extend(_extract_pagination_links(html_content, fetch_url))

    seen_urls = {url, fetch_url}
    for page_url in discovered:
        if page_url not in seen_urls:
            next_page_urls.append(page_url)
            seen_urls.add(page_url)

    # ------------------------------------------------------------------
    # 3. Ekstrak konten utama halaman pertama dengan trafilatura
    # ------------------------------------------------------------------
    if not extracted_text or len(extracted_text.strip()) < 100:
        raise ValueError(
            f"Trafilatura tidak dapat mengekstrak konten dari '{url}'. "
            "Pastikan URL valid dan dapat diakses."
        )

    # ------------------------------------------------------------------
    # 4. Ekstrak metadata (judul) dari trafilatura halaman pertama
    # ------------------------------------------------------------------
    title = _extract_title(html_content)

    # Fallback: cari <title> tag sederhana jika metadata kosong
    if not title:
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()
            # Hapus nama situs yang sering muncul setelah separator
            title = re.split(r"\s*[\|–\-—]\s*", title)[0].strip()

    # ------------------------------------------------------------------
    # 5. Crawl halaman berikutnya jika terdeteksi
    # ------------------------------------------------------------------
    all_texts = [_strip_repeated_title(extracted_text, title)]
    crawled_urls = {url, fetch_url}
    queued_urls = list(next_page_urls)

    while queued_urls and len(crawled_urls) < _MAX_PAGINATION_PAGES:
        next_url = queued_urls.pop(0)
        if next_url in crawled_urls:
            continue

        try:
            page_html = _fetch_html(next_url)
            crawled_urls.add(next_url)
            p_text = _extract_main_text(page_html)
            if p_text and len(p_text.strip()) > 50:
                all_texts.append(_strip_repeated_title(p_text, title))

            for page_url in _extract_pagination_links(page_html, next_url):
                if page_url not in crawled_urls and page_url not in queued_urls:
                    queued_urls.append(page_url)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Gagal mengunduh halaman tambahan '{next_url}': {e}")

    # Gabungkan semua teks halaman dengan pemisah baris baru ganda
    combined_text = _deduplicate_join_texts(all_texts)
    total_pages = max(1, len(crawled_urls) - (1 if fetch_url != url else 0))

    return {
        "title": title,
        "text": combined_text,
        "url": url,
        "processed_url": fetch_url,
        "pagination_handled": total_pages > 1 or (fetch_url != url),
        "total_pages": total_pages
    }
