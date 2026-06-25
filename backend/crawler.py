"""
ดึง URL รูปภาพทั้งหมดจากหน้าเว็บที่ผู้ใช้ระบุ
รองรับ <img src>, srcset, data-src (lazy-load) และ background-image ใน style

มี 2 โหมด:
  - โหมดปกติ (httpx)   : เร็ว ใช้กับเว็บ HTML ทั่วไป
  - โหมด render (Playwright): เปิดเบราว์เซอร์จริง รอ JS โหลดเสร็จ
    เหมาะกับเว็บ SPA / lazy-load ที่ใส่รูปด้วย JavaScript
    (ต้องติดตั้ง:  pip install playwright && python -m playwright install chromium)
"""
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_BG_URL_RE = re.compile(r"url\(\s*['\"]?(.*?)['\"]?\s*\)")
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


def _is_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith(_IMG_EXT)


def _best_srcset_url(srcset: str) -> str | None:
    """เลือก URL ความละเอียดสูงสุดจาก srcset (descriptor แบบ '840w' หรือ '2x')

    ถ้าไม่มี descriptor เลย จะคืนตัวแรกที่เจอ
    """
    best_url: str | None = None
    best_w = -1.0
    for part in srcset.split(","):
        tokens = part.strip().split()
        if not tokens:
            continue
        w = 0.0
        if len(tokens) > 1:
            try:
                w = float(tokens[1].lower().rstrip("wx"))
            except ValueError:
                w = 0.0
        if w > best_w:
            best_w, best_url = w, tokens[0]
    return best_url


def _extract_from_html(html: str, page_url: str, scope_selector: str | None = None) -> list[str]:
    """แยก image URL จาก HTML string

    scope_selector: ถ้าระบุ (เช่น ".site-main") จะดึงเฉพาะรูปที่อยู่ภายใต้ element
    ที่ตรงกับ CSS selector นั้นเท่านั้น (ถ้าไม่ตรงเลยจะคืน list ว่าง)
    """
    soup = BeautifulSoup(html, "html.parser")
    found: list[str] = []
    seen: set[str] = set()

    # เลือก "ขอบเขต" ที่จะค้นหารูป: ทั้งหน้า หรือเฉพาะภายใต้ selector
    if scope_selector:
        roots = soup.select(scope_selector)
    else:
        roots = [soup]

    def add(raw: str | None):
        if not raw:
            return
        raw = raw.strip()
        if raw.startswith("data:"):
            return
        absolute = urljoin(page_url, raw)
        if absolute in seen:
            return
        seen.add(absolute)
        found.append(absolute)

    for root in roots:
        for img in root.find_all("img"):
            # 1 tag = 1 URL: ใช้ src ก่อน ถ้าไม่มีค่อย fallback ไป srcset (เลือกความละเอียดสูงสุด)
            src = (img.get("src") or "").strip()
            if src:
                add(src)
            else:
                srcset = img.get("srcset") or img.get("data-srcset")
                if srcset:
                    add(_best_srcset_url(srcset))

        for src in root.find_all("source"):
            srcset = src.get("srcset")
            if srcset:
                for part in srcset.split(","):
                    add(part.strip().split(" ")[0])

        for tag in root.find_all(style=True):
            for m in _BG_URL_RE.findall(tag["style"]):
                add(m)

    return found


async def _fetch_html_httpx(page_url: str) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=20.0
    ) as client:
        resp = await client.get(page_url)
        resp.raise_for_status()
        return resp.text


async def _fetch_with_playwright(page_url: str) -> tuple[str, list[str]]:
    """เปิดหน้าเว็บด้วย Chromium จริง คืน (rendered_html, image_urls_from_network)

    เก็บ URL รูปจากทั้ง DOM และ network requests (เผื่อรูปถูกโหลดผ่าน JS)
    """
    network_imgs: list[str] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=USER_AGENT)
        page = await context.new_page()

        def on_response(resp):
            ctype = resp.headers.get("content-type", "")
            if ctype.startswith("image") or _is_image_url(resp.url):
                network_imgs.append(resp.url)

        page.on("response", on_response)
        await page.goto(page_url, wait_until="networkidle", timeout=45000)
        # เลื่อนหน้าเพื่อกระตุ้น lazy-load
        try:
            await page.evaluate(
                "async () => { for (let y=0; y<document.body.scrollHeight; y+=600)"
                " { window.scrollTo(0, y); await new Promise(r=>setTimeout(r,120)); } }"
            )
            await page.wait_for_timeout(800)
        except Exception:
            pass
        html = await page.content()
        await browser.close()
    return html, network_imgs


async def fetch_image_urls(
    page_url: str,
    limit: int | None = None,
    render_js: bool = False,
    scope_selector: str | None = None,
) -> list[str]:
    """คืน list ของ absolute image URL ที่เจอในหน้าเว็บ (ไม่ซ้ำ)

    render_js=True จะใช้ Playwright เปิดเบราว์เซอร์จริง (ถ้าติดตั้งไว้)
    limit=None หมายถึงไม่จำกัดจำนวนรูปต่อหน้า
    scope_selector: จำกัดให้ดึงเฉพาะรูปภายใต้ CSS selector นั้น (เช่น ".site-main")
    """
    scope = (scope_selector or "").strip() or None
    found: list[str] = []
    if render_js and PLAYWRIGHT_AVAILABLE:
        html, network_imgs = await _fetch_with_playwright(page_url)
        found.extend(_extract_from_html(html, page_url, scope))
        # network images อยู่นอก DOM จึงระบุขอบเขตไม่ได้ -> ใช้เฉพาะตอนไม่กำหนด scope
        if scope is None:
            found.extend(network_imgs)
    else:
        html = await _fetch_html_httpx(page_url)
        found.extend(_extract_from_html(html, page_url, scope))

    # dedup คงลำดับ
    seen: set[str] = set()
    unique = []
    for u in found:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    # กรองให้เหลือเฉพาะที่ดูเป็นไฟล์ภาพ (ถ้าหาไม่เจอเลย ค่อยคืนทั้งหมด)
    images = [u for u in unique if _is_image_url(u)]
    if not images:
        images = unique
    return images if limit is None else images[:limit]


async def download_image(url: str, max_bytes: int = 12_000_000) -> bytes | None:
    """ดาวน์โหลดไฟล์ภาพ คืน bytes (None ถ้าไม่ใช่ภาพ/ใหญ่เกิน/ผิดพลาด)"""
    try:
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, follow_redirects=True, timeout=20.0
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "image" not in ctype and not _is_image_url(url):
                return None
            content = resp.content
            if not content or len(content) > max_bytes:
                return None
            # ข้าม SVG (vector) ที่ OpenCV/CLIP decode เป็น raster ไม่ได้
            if "svg" in ctype or urlparse(url).path.lower().endswith(".svg"):
                return None
            return content
    except (httpx.HTTPError, ValueError):
        return None
