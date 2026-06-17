"""
ค้นหา "ภาพที่หน้าตาคล้ายกันทั้งภาพ" ด้วย CLIP embeddings + FAISS

ต่างจาก matcher.py (ที่หา "ภาพย่อยในภาพใหญ่"):
  - โหมดนี้แปลงทั้งภาพเป็นเวกเตอร์ความหมาย (CLIP) แล้ววัดความคล้าย (cosine)
  - เหมาะกับ "หารูปที่ดูคล้ายกันโดยรวม" เช่น สินค้าเดียวกันคนละมุม, สไตล์เดียวกัน

เป็นโมดูล "เสริม" — ถ้ายังไม่ได้ติดตั้ง torch/open_clip/faiss แอปหลักยังรันได้
(จะใช้ได้เฉพาะตอนผู้ใช้เลือกโหมด similar)

ติดตั้ง:  pip install -r requirements-clip.txt
"""
from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional

# Windows: torch และ faiss ต่างก็ลิงก์ OpenMP runtime คนละตัว ทำให้ชนกัน (OMP Error #15)
# ตั้งค่านี้ก่อน import เพื่อให้รันร่วมกันได้ (workaround มาตรฐานของ Intel OpenMP)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

try:
    import torch
    import open_clip
    import faiss
    from PIL import Image
    CLIP_AVAILABLE = True
    _IMPORT_ERROR = None
except ImportError as e:  # โมดูลเสริมยังไม่ถูกติดตั้ง
    CLIP_AVAILABLE = False
    _IMPORT_ERROR = str(e)


_MODEL = None
_PREPROCESS = None
_DEVICE = "cpu"
_MODEL_NAME = "ViT-B-32"
_PRETRAINED = "laion2b_s34b_b79k"


def availability() -> dict:
    return {"available": CLIP_AVAILABLE, "error": _IMPORT_ERROR}


def _ensure_model():
    """โหลดโมเดลครั้งแรก (lazy) — ครั้งแรกจะดาวน์โหลด weights ~350MB"""
    global _MODEL, _PREPROCESS
    if _MODEL is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            _MODEL_NAME, pretrained=_PRETRAINED
        )
        model.eval().to(_DEVICE)
        _MODEL = model
        _PREPROCESS = preprocess
    return _MODEL, _PREPROCESS


def embed_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """คืนเวกเตอร์ CLIP (normalize แล้ว) ของภาพ หรือ None ถ้าอ่านไม่ได้"""
    if not CLIP_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    model, preprocess = _ensure_model()
    tensor = preprocess(img).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype("float32")[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity ของเวกเตอร์ที่ normalize แล้ว (= dot product)"""
    return float(np.dot(a, b))


@dataclass
class SimilarResult:
    url: str
    score: float  # cosine similarity 0..1


def search_similar(
    query_bytes: bytes,
    candidates: list[tuple[str, bytes]],
    threshold: float = 0.75,
    top_k: int = 20,
) -> list[SimilarResult]:
    """หาภาพในรายการ candidates ที่คล้ายกับ query โดยรวม

    candidates: list ของ (url, image_bytes)
    ใช้ FAISS (IndexFlatIP) บนเวกเตอร์ที่ normalize แล้ว = cosine similarity
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError(
            "โมดูล CLIP/FAISS ยังไม่พร้อมใช้งาน: " + str(_IMPORT_ERROR)
            + " — ติดตั้งด้วย pip install -r requirements-clip.txt"
        )

    q = embed_image_bytes(query_bytes)
    if q is None:
        return []

    vectors = []
    urls = []
    for url, data in candidates:
        v = embed_image_bytes(data)
        if v is not None:
            vectors.append(v)
            urls.append(url)

    if not vectors:
        return []

    items = list(zip(urls, vectors))
    return rank_by_vectors(q, items, threshold=threshold, top_k=top_k)


def rank_by_vectors(
    query_vec: Optional[np.ndarray],
    items: list[tuple[str, np.ndarray]],
    threshold: float = 0.75,
    top_k: int = 50,
) -> list[SimilarResult]:
    """จัดอันดับความคล้ายด้วย FAISS จากเวกเตอร์ที่คำนวณไว้แล้ว

    แยกออกมาเพื่อให้ผู้เรียก (เช่น โหมด streaming) คำนวณ embedding ทีละรูป
    แล้วรายงานความคืบหน้าได้ ก่อนค่อยส่งมา rank รวดเดียว
    """
    if not CLIP_AVAILABLE:
        raise RuntimeError("โมดูล CLIP/FAISS ยังไม่พร้อมใช้งาน: " + str(_IMPORT_ERROR))
    if query_vec is None or not items:
        return []

    mat = np.vstack([v for _, v in items]).astype("float32")
    index = faiss.IndexFlatIP(mat.shape[1])  # inner product = cosine (normalize แล้ว)
    index.add(mat)

    k = min(top_k, len(items))
    scores, idxs = index.search(query_vec.reshape(1, -1).astype("float32"), k)

    out: list[SimilarResult] = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        s = float(score)
        if s >= threshold:
            out.append(SimilarResult(url=items[idx][0], score=round(s, 4)))
    return out
