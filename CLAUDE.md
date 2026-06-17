# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An image-search web app: upload one or more query images, point it at one or more web page URLs, and it crawls those pages, downloads every image found, and reports where the query appears. Two distinct modes back the same UI:

- **inside** — does the query image appear *inside* a larger image (e.g. a logo/product within a banner)? Uses OpenCV **SIFT** feature matching + RANSAC/Homography. Always available.
- **similar** — does an image look *similar overall* (same product, different angle)? Uses **CLIP** embeddings + **FAISS** cosine search. Optional — requires the heavy `requirements-clip.txt` deps; the app runs fine without them and only this mode is disabled.

Note: code comments, UI text, and error messages are in **Thai**. Keep new user-facing strings consistent with that.

## Commands

All commands run from `backend/`:

```powershell
pip install -r requirements.txt          # core deps (inside mode + crawling)
python -m playwright install chromium     # only if using render_js (JS-heavy sites)

# optional — only for "similar" mode:
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-clip.txt      # first similar run downloads ~350MB CLIP weights

python main.py        # serve at http://127.0.0.1:8000 (no reload)
python test_match.py  # offline matcher self-test (synthetic images, no network/fixtures)
```

There is no lint config, no test framework, and no build step (frontend is static, served by FastAPI). `test_match.py` is the only test — it asserts the SIFT matcher finds a cropped sub-image in its parent and exits non-zero on failure.

## Architecture

`backend/main.py` is the FastAPI app and the only orchestrator. The two analysis modules (`matcher.py`, `clip_search.py`) and the `crawler.py` are stateless libraries it calls into.

**Request flow** (the real one — the UI uses streaming):
- `POST /api/search/stream` is what `frontend/index.html` calls. It accepts up to `MAX_QUERY_IMAGES` (12) query images and returns **NDJSON** — one JSON event per line as work progresses. Event `stage` sequence: `start → crawl → collected → (query_ready | embed_query) → analyze → done`. Partial `matches` are emitted inside `analyze` events so the frontend still has results if the stream drops. `_search_events()` is the generator that drives this.
- `POST /api/search` is a simpler non-streaming variant (single image, returns all matches at once). Kept around but the UI doesn't use it.
- `GET /api/capabilities` reports which optional features are live (`detector` SIFT/ORB, `playwright`, `clip`) so the frontend can disable unavailable modes.

**Concurrency model:** two module-level `asyncio.Semaphore`s bound parallelism — `_SEM` (8) caps simultaneous image downloads/analysis, `_PAGE_SEM` (4) caps simultaneous page crawls. CPU-bound work (OpenCV, CLIP) is pushed off the event loop with `asyncio.to_thread`. There is intentionally **no cap on total image count** — large sites just take longer.

**Crawling (`crawler.py`):** `fetch_image_urls()` extracts images from `<img src/srcset/data-src/data-lazy-src>`, `<source srcset>`, and CSS `background-image`. `scope_selector` (a CSS selector like `.site-main`) restricts extraction to a subtree. `render_js=True` switches from the fast `httpx`+BeautifulSoup path to **Playwright** (real Chromium): it waits for networkidle, scrolls to trigger lazy-load, and also captures images seen in network responses. `download_image()` skips SVG (OpenCV/CLIP can't raster-decode it) and anything over 12MB.

**inside mode (`matcher.py`):** `compute_features()` (SIFT keypoints+descriptors) is computed once per query and reused across all target images. `match_in_image()` does BFMatcher kNN + Lowe ratio test, then RANSAC homography to confirm and locate a bounding box. Tuning lives in the `MatchParams` dataclass — pass params through, never mutate global state (multiple requests run concurrently). `MatchParams.clamp()` sanitizes UI-supplied values. If the OpenCV build lacks SIFT, it auto-falls back to **ORB** (`DETECTOR_NAME` reflects which).

**similar mode (`clip_search.py`):** lazy-loads the CLIP model on first use (`_ensure_model`). `embed_image_bytes()` returns an L2-normalized vector; the streaming path embeds images one at a time so it can report progress, then ranks via FAISS `IndexFlatIP` (inner product on normalized vectors = cosine). The FAISS index is built **in-memory per request** — not persisted. All CLIP/FAISS imports are wrapped in try/except so a missing install degrades gracefully via the `CLIP_AVAILABLE` flag.

**Dedup:** both modes call `matcher.dedup_by_hash()`, which collapses the same image served at different URLs/sizes using a perceptual **dHash** (Hamming distance ≤ 5). The kept item gains a `duplicates` count. Inputs must be pre-sorted best-first so the highest-scoring URL survives.

## Gotchas

- `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")` at the top of `clip_search.py` is required on Windows — torch and faiss link separate OpenMP runtimes and crash (OMP Error #15) without it. Don't remove it.
- The static-file mount `app.mount("/", StaticFiles(...))` in `main.py` must stay **last** — it's a catch-all that would shadow the API routes and `/favicon` if declared earlier.
- This is not a git repository.
