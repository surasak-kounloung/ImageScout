"""
FastAPI app: ค้นหาว่าภาพ query เกี่ยวข้องกับรูปภาพบนหน้าเว็บที่ระบุหรือไม่

2 โหมด:
  - inside  : หา "ภาพย่อยในภาพใหญ่" ด้วย SIFT feature matching (matcher.py)
  - similar : หา "ภาพที่หน้าตาคล้ายทั้งภาพ" ด้วย CLIP + FAISS (clip_search.py)

Endpoints:
  GET  /             -> เสิร์ฟหน้าเว็บ (frontend)
  GET  /api/capabilities -> บอกว่า Playwright / CLIP พร้อมใช้ไหม
  POST /api/search   -> รับไฟล์ภาพ + URL + ตัวเลือก แล้วคืนผลลัพธ์
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import clip_search
import crawler
import matcher

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Image Search (inside + similar)")

# จำกัดจำนวนงานดาวน์โหลด/ประมวลผลพร้อมกัน
_SEM = asyncio.Semaphore(8)
# จำกัดจำนวนหน้าเว็บที่ crawl พร้อมกัน (โดยเฉพาะตอน render ด้วย Playwright)
_PAGE_SEM = asyncio.Semaphore(4)


def _parse_urls(raw: str) -> list[str]:
    """แยกหลาย URL จาก textarea (ขึ้นบรรทัดใหม่ หรือคั่นด้วย comma/space) แบบไม่ซ้ำ"""
    parts = re.split(r"[\s,]+", raw.strip())
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not re.match(r"^https?://", p, re.I):
            p = "https://" + p
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


async def _collect_images(pages: list[str], render_js: bool, scope_selector: str | None = None):
    """crawl หลายหน้าแบบขนาน คืน (ordered_image_urls, image_url -> source_page, page_errors)"""
    async def crawl(page: str):
        async with _PAGE_SEM:
            try:
                imgs = await crawler.fetch_image_urls(
                    page, render_js=render_js, scope_selector=scope_selector
                )
                return page, imgs, None
            except Exception as e:  # noqa: BLE001
                return page, [], str(e)

    results = await asyncio.gather(*[crawl(p) for p in pages])

    image_to_page: dict[str, str] = {}
    ordered: list[str] = []
    page_errors: list[dict] = []
    for page, imgs, err in results:
        if err is not None:
            page_errors.append({"page": page, "error": err})
            continue
        for iu in imgs:
            if iu not in image_to_page:
                image_to_page[iu] = page
                ordered.append(iu)
    return ordered, image_to_page, page_errors


@app.get("/")
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon_ico():
    path = FRONTEND_DIR / "favicon.ico"
    if not path.exists():
        path = FRONTEND_DIR / "favicon.png"
    return FileResponse(path)


@app.get("/favicon.png", include_in_schema=False)
async def favicon_png():
    return FileResponse(FRONTEND_DIR / "favicon.png")


@app.get("/api/capabilities")
async def capabilities():
    return {
        "detector": matcher.DETECTOR_NAME,
        "playwright": crawler.PLAYWRIGHT_AVAILABLE,
        "clip": clip_search.availability(),
    }


async def _download_all(image_urls: list[str]) -> list[tuple[str, bytes]]:
    async def dl(u: str):
        async with _SEM:
            data = await crawler.download_image(u)
            return (u, data) if data is not None else None

    results = await asyncio.gather(*[dl(u) for u in image_urls])
    return [r for r in results if r is not None]


@app.post("/api/search")
async def search(
    image: UploadFile = File(...),
    urls: str = Form(...),                       # หลาย URL คั่นด้วยขึ้นบรรทัดใหม่/comma
    mode: str = Form("inside"),                  # "inside" | "similar"
    render_js: bool = Form(False),
    scope_selector: str = Form(""),              # CSS selector จำกัดขอบเขต เช่น ".site-main"
    # พารามิเตอร์โหมด inside (feature matching)
    ratio_test: float = Form(0.75),
    min_inliers: int = Form(10),
    # พารามิเตอร์โหมด similar (CLIP)
    similarity_threshold: float = Form(0.75),
):
    query_bytes = await image.read()

    pages = _parse_urls(urls)
    if not pages:
        return JSONResponse({"error": "กรุณาใส่ URL อย่างน้อย 1 รายการ"}, status_code=400)

    # ดึง URL รูปจากทุกหน้า (ขนาน) พร้อมจำว่ารูปไหนมาจากหน้าไหน
    image_urls, image_to_page, page_errors = await _collect_images(pages, render_js, scope_selector)

    if not image_urls:
        return JSONResponse(
            {
                "error": "ไม่พบรูปภาพในหน้าที่ระบุ (หรือดึงหน้าไม่สำเร็จทั้งหมด)",
                "page_errors": page_errors,
            },
            status_code=400,
        )

    base_meta = {
        "pages_requested": len(pages),
        "pages_failed": len(page_errors),
        "page_errors": page_errors,
        "total_image_urls": len(image_urls),
    }

    if mode == "similar":
        if not clip_search.CLIP_AVAILABLE:
            return JSONResponse(
                {"error": "โหมด similar ต้องติดตั้งโมดูลเสริมก่อน: "
                          "pip install -r requirements-clip.txt"},
                status_code=400,
            )
        candidates = await _download_all(image_urls)
        try:
            sims = await asyncio.to_thread(
                clip_search.search_similar,
                query_bytes, candidates, similarity_threshold,
            )
        except RuntimeError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        bytes_map = {u: b for u, b in candidates}
        matches = [
            {"url": s.url, "score": s.score, "source_page": image_to_page.get(s.url)}
            for s in sims
        ]
        hashes = {m["url"]: matcher.dhash(bytes_map[m["url"]]) for m in matches if m["url"] in bytes_map}
        matches = matcher.dedup_by_hash(matches, hashes)
        return {"mode": "similar", "scanned": len(candidates), "matches": matches, **base_meta}

    # ----- โหมด inside (default): feature matching -----
    params = matcher.MatchParams.clamp(ratio_test=ratio_test, min_inliers=min_inliers)
    query_feat = matcher.compute_features(query_bytes, params.max_dim)
    if query_feat is None or query_feat[2] is None:
        return JSONResponse(
            {"error": "อ่านภาพ query ไม่ได้ หรือภาพมีรายละเอียดน้อยเกินไป"},
            status_code=400,
        )

    async def process(img_url: str):
        async with _SEM:
            data = await crawler.download_image(img_url)
            if data is None:
                return None
            result = await asyncio.to_thread(
                matcher.match_in_image, query_feat, data, params
            )
            img_hash = matcher.dhash(data) if result.found else None
            return img_url, result, img_hash

    results = await asyncio.gather(*[process(u) for u in image_urls])

    matches = []
    hashes: dict[str, int | None] = {}
    scanned = 0
    for item in results:
        if item is None:
            continue
        scanned += 1
        img_url, res, img_hash = item
        if res.found:
            d = res.to_dict()
            d["url"] = img_url
            d["source_page"] = image_to_page.get(img_url)
            matches.append(d)
            hashes[img_url] = img_hash

    matches.sort(key=lambda m: m["inliers"], reverse=True)
    matches = matcher.dedup_by_hash(matches, hashes)
    return {
        "mode": "inside",
        "query_keypoints": len(query_feat[1]),
        "scanned": scanned,
        "matches": matches,
        **base_meta,
    }


def _dedup_grouped(matches: list[dict], hashes: dict, sort_key) -> list[dict]:
    """dedup แยกตามภาพ query แต่ละภาพ (ภาพเดียวกันตรงกับ query หลายภาพได้)"""
    by_q: dict[int, list[dict]] = {}
    for m in matches:
        by_q.setdefault(m["query_index"], []).append(m)
    out: list[dict] = []
    for qi in sorted(by_q):
        grp = sorted(by_q[qi], key=sort_key, reverse=True)
        out.extend(matcher.dedup_by_hash(grp, hashes))
    return out


async def _search_events(query_items, pages, mode, render_js, ratio_test, min_inliers,
                         similarity_threshold, scope_selector=None):
    """async generator: yield NDJSON events รายงานความคืบหน้าทีละขั้น

    query_items: list ของ (ชื่อไฟล์, bytes) ของภาพ query (สูงสุด 12 ภาพ)
    """
    def ev(d: dict) -> str:
        return json.dumps(d, ensure_ascii=False) + "\n"

    # ส่งรายการหน้าไปด้วย เพื่อให้ frontend คำนวณ "URL ที่ยังไม่เสร็จ" ได้เองถ้า stream หลุด
    yield ev({"stage": "start", "pages_total": len(pages), "pages": pages})

    # ---- 1) crawl ทุกหน้าแบบขนาน รายงานทีละหน้า ----
    image_to_page: dict[str, str] = {}
    ordered: list[str] = []
    page_errors: list[dict] = []

    async def crawl(page: str):
        async with _PAGE_SEM:
            try:
                imgs = await crawler.fetch_image_urls(
                    page, render_js=render_js, scope_selector=scope_selector
                )
                return page, imgs, None
            except Exception as e:  # noqa: BLE001
                return page, [], str(e)

    tasks = [asyncio.create_task(crawl(p)) for p in pages]
    try:
        pages_done = 0
        for fut in asyncio.as_completed(tasks):
            page, imgs, err = await fut
            pages_done += 1
            if err is not None:
                page_errors.append({"page": page, "error": err})
            else:
                for iu in imgs:
                    if iu not in image_to_page:
                        image_to_page[iu] = page
                        ordered.append(iu)
            yield ev({"stage": "crawl", "pages_done": pages_done, "pages_total": len(pages),
                      "images_found": len(ordered), "page": page,
                      **({"error": err} if err is not None else {})})
    finally:
        # client ตัดการเชื่อมต่อ/กดหยุด -> ยกเลิก task ที่ยังค้าง ไม่ให้วิ่งต่อในพื้นหลัง
        for t in tasks:
            t.cancel()

    base_meta = {
        "pages_requested": len(pages),
        "pages_failed": len(page_errors),
        "page_errors": page_errors,
        "total_image_urls": len(ordered),
    }
    # จำนวนรูปต่อหน้า ใช้ติดตามว่าหน้าไหนวิเคราะห์ครบแล้ว (สำหรับ partial results)
    page_image_counts: dict[str, int] = {}
    for iu in ordered:
        pg = image_to_page[iu]
        page_image_counts[pg] = page_image_counts.get(pg, 0) + 1
    yield ev({"stage": "collected", **base_meta, "page_image_counts": page_image_counts})

    if not ordered:
        yield ev({"stage": "done", "error": "ไม่พบรูปภาพในหน้าที่ระบุ", "matches": [], **base_meta})
        return

    # ===== โหมด similar =====
    if mode == "similar":
        if not clip_search.CLIP_AVAILABLE:
            yield ev({"stage": "done", "error": "โหมด similar ต้องติดตั้งโมดูลเสริม "
                      "(pip install -r requirements-clip.txt)", "matches": [], **base_meta})
            return

        yield ev({"stage": "embed_query"})
        qvecs = []  # (query_index, query_name, vector)
        for qi, (qname, qbytes) in enumerate(query_items):
            v = await asyncio.to_thread(clip_search.embed_image_bytes, qbytes)
            if v is not None:
                qvecs.append((qi, qname, v))
        if not qvecs:
            yield ev({"stage": "done", "error": "อ่านภาพ query ไม่ได้สักภาพ", "matches": [], **base_meta})
            return
        queries_meta = [{"index": qi, "name": qname} for qi, qname, _ in qvecs]

        # ประมวลผลทีละรูป (ดาวน์โหลด -> embed -> ให้คะแนนกับทุก query ทันที) เพื่อให้
        # ผลลัพธ์ทยอยส่งออกไประหว่างทาง — ถ้า stream หลุด frontend ยังมีผลบางส่วน
        total = len(ordered)
        done = 0
        scanned = 0
        found_count = 0
        matches: list[dict] = []
        hashes: dict[str, int | None] = {}

        async def dl(u: str):
            async with _SEM:
                return u, await crawler.download_image(u)

        dl_tasks = [asyncio.create_task(dl(u)) for u in ordered]
        try:
            for fut in asyncio.as_completed(dl_tasks):
                u, data = await fut
                done += 1
                found_here: list[dict] = []
                if data is not None:
                    scanned += 1
                    v = await asyncio.to_thread(clip_search.embed_image_bytes, data)
                    if v is not None:
                        for qi, qname, qv in qvecs:
                            score = clip_search.cosine(qv, v)
                            if score >= similarity_threshold:
                                found_here.append({"url": u, "score": round(score, 4),
                                                   "source_page": image_to_page.get(u),
                                                   "query_index": qi, "query_name": qname})
                        if found_here:
                            hashes[u] = matcher.dhash(data)
                matches.extend(found_here)
                found_count += len(found_here)
                yield ev({"stage": "analyze", "done": done, "total": total,
                          "found": found_count, "page": image_to_page.get(u),
                          **({"matches": found_here} if found_here else {})})
        finally:
            for t in dl_tasks:
                t.cancel()

        matches = _dedup_grouped(matches, hashes, sort_key=lambda m: m["score"])
        yield ev({"stage": "done", "mode": "similar", "scanned": scanned,
                  "matches": matches, "queries": queries_meta, **base_meta})
        return

    # ===== โหมด inside =====
    params = matcher.MatchParams.clamp(ratio_test=ratio_test, min_inliers=min_inliers)
    query_feats = []  # (query_index, query_name, features)
    for qi, (qname, qbytes) in enumerate(query_items):
        feat = matcher.compute_features(qbytes, params.max_dim)
        if feat is not None and feat[2] is not None:
            query_feats.append((qi, qname, feat))
    if not query_feats:
        yield ev({"stage": "done", "error": "อ่านภาพ query ไม่ได้สักภาพ หรือทุกภาพมีรายละเอียดน้อยเกินไป",
                  "matches": [], **base_meta})
        return
    queries_meta = [{"index": qi, "name": qname, "keypoints": len(feat[1])}
                    for qi, qname, feat in query_feats]
    yield ev({"stage": "query_ready", "queries_ready": len(query_feats),
              "keypoints": sum(len(feat[1]) for _, _, feat in query_feats)})

    total = len(ordered)
    done = 0
    found_count = 0
    matches: list[dict] = []
    hashes: dict[str, int | None] = {}

    def match_all_queries(data: bytes):
        """เทียบรูปจากเว็บกับภาพ query ทุกภาพ (รันใน thread เดียว)"""
        out = []
        for qi, qname, feat in query_feats:
            res = matcher.match_in_image(feat, data, params)
            if res.found:
                out.append((qi, qname, res))
        return out

    async def proc(u: str):
        async with _SEM:
            data = await crawler.download_image(u)
            if data is None:
                return u, [], None
            found = await asyncio.to_thread(match_all_queries, data)
            h = matcher.dhash(data) if found else None
            return u, found, h

    proc_tasks = [asyncio.create_task(proc(u)) for u in ordered]
    try:
        for fut in asyncio.as_completed(proc_tasks):
            u, found, h = await fut
            done += 1
            found_here: list[dict] = []
            for qi, qname, res in found:
                d = res.to_dict()
                d["url"] = u
                d["source_page"] = image_to_page.get(u)
                d["query_index"] = qi
                d["query_name"] = qname
                found_here.append(d)
                matches.append(d)
            if found_here:
                hashes[u] = h
                found_count += len(found_here)
            yield ev({"stage": "analyze", "done": done, "total": total, "found": found_count,
                      "page": image_to_page.get(u),
                      **({"matches": found_here} if found_here else {})})
    finally:
        for t in proc_tasks:
            t.cancel()

    matches = _dedup_grouped(matches, hashes, sort_key=lambda m: m["inliers"])
    yield ev({"stage": "done", "mode": "inside", "scanned": done,
              "matches": matches, "queries": queries_meta, **base_meta})


MAX_QUERY_IMAGES = 12


@app.post("/api/search/stream")
async def search_stream(
    images: list[UploadFile] = File(...),        # ภาพ query สูงสุด 12 ภาพ
    urls: str = Form(...),
    mode: str = Form("inside"),
    render_js: bool = Form(False),
    scope_selector: str = Form(""),              # CSS selector จำกัดขอบเขต เช่น ".site-main"
    ratio_test: float = Form(0.75),
    min_inliers: int = Form(10),
    similarity_threshold: float = Form(0.75),
):
    if len(images) > MAX_QUERY_IMAGES:
        return JSONResponse(
            {"error": f"อัปโหลดภาพได้สูงสุด {MAX_QUERY_IMAGES} ภาพ"}, status_code=400)

    query_items: list[tuple[str, bytes]] = []
    for i, img in enumerate(images):
        data = await img.read()
        if data:
            query_items.append((img.filename or f"ภาพที่ {i + 1}", data))
    if not query_items:
        return JSONResponse({"error": "กรุณาเลือกภาพอย่างน้อย 1 ภาพ"}, status_code=400)

    pages = _parse_urls(urls)
    if not pages:
        return JSONResponse({"error": "กรุณาใส่ URL อย่างน้อย 1 รายการ"}, status_code=400)

    gen = _search_events(query_items, pages, mode, render_js, ratio_test, min_inliers,
                         similarity_threshold, scope_selector)
    return StreamingResponse(gen, media_type="application/x-ndjson")


# เสิร์ฟไฟล์ static ของ frontend (styles.css, screenshots ฯลฯ)
# ต้อง mount ท้ายสุด เพื่อไม่ให้ทับ API routes / "/" / favicon ที่ประกาศไว้ก่อนหน้า
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
