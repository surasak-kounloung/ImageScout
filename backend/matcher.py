"""
Local feature matching: หาว่าภาพ query (เช่น โลโก้/สินค้า) ปรากฏอยู่ใน
ภาพปลายทางหรือไม่ พร้อมระบุตำแหน่ง (bounding box) และคะแนนความมั่นใจ

ใช้ SIFT เป็นหลัก (ทนต่อการย่อ-ขยาย-หมุน) และยืนยันผลด้วย
RANSAC + Homography เพื่อกรอง false positive

หมายเหตุ: ฟังก์ชัน match แบบรับ MatchParams เป็นพารามิเตอร์ (ไม่ใช้ global
mutable state) เพื่อให้ปลอดภัยเมื่อมีหลาย request พร้อมกัน
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class MatchParams:
    """พารามิเตอร์ปรับ "ความแม่น/ความไว" ของการค้นหา"""
    ratio_test: float = 0.75      # Lowe's ratio (ต่ำ = เข้มงวด, จับคู่น้อยลงแต่แม่นขึ้น)
    min_good_matches: int = 12    # จำนวนคู่ฟีเจอร์ขั้นต่ำก่อนลองหา homography
    min_inliers: int = 10         # inlier ขั้นต่ำหลัง RANSAC ถึงจะถือว่า "เจอ"
    max_dim: int = 2400           # ย่อภาพที่ใหญ่เกินไปก่อนประมวลผล (สูง = จับ instance เล็กในภาพใหญ่ได้ดีขึ้นแต่ช้าลง)

    @classmethod
    def clamp(cls, ratio_test=None, min_good_matches=None, min_inliers=None, max_dim=None):
        d = cls()
        if ratio_test is not None:
            d.ratio_test = float(min(0.95, max(0.4, ratio_test)))
        if min_good_matches is not None:
            d.min_good_matches = int(max(4, min_good_matches))
        if min_inliers is not None:
            d.min_inliers = int(max(4, min_inliers))
        if max_dim is not None:
            d.max_dim = int(min(2400, max(400, max_dim)))
        return d


@dataclass
class MatchResult:
    found: bool
    inliers: int
    good_matches: int
    confidence: float                      # 0..1 โดยประมาณ
    box: Optional[list[list[int]]] = None   # 4 มุมของกรอบที่เจอในภาพปลายทาง

    def to_dict(self) -> dict:
        return asdict(self)


def _load_gray(image_bytes: bytes) -> Optional[np.ndarray]:
    """ถอดรหัสเป็นภาพ grayscale คืน None ถ้าไม่ใช่ภาพที่ decode ได้

    กันเคส buffer ว่าง / ไฟล์เสีย / ฟอร์แมตที่ OpenCV ไม่รองรับ (เช่น SVG, AVIF)
    ซึ่ง cv2.imdecode อาจคืน None หรือ raise assertion (!buf.empty())
    """
    if not image_bytes:
        return None
    try:
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    except cv2.error:
        return None


def dhash(image_bytes: bytes, hash_size: int = 8) -> Optional[int]:
    """perceptual hash (difference hash) ใช้ดีดูป "รูปเดียวกันแต่คนละขนาด/URL"

    เพราะ resize เป็นขนาดคงที่ก่อนคำนวณ รูปเดียวกันคนละความละเอียดจะได้ค่าใกล้กันมาก
    """
    gray = _load_gray(image_bytes)
    if gray is None:
        return None
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_by_hash(items: list[dict], hashes: dict[str, Optional[int]], threshold: int = 5) -> list[dict]:
    """ยุบรายการที่เป็นภาพเดียวกัน (hamming distance ของ dHash <= threshold)

    items ควรเรียงจาก "ดีที่สุดก่อน" อยู่แล้ว เพื่อให้ตัวที่เก็บไว้คือตัวที่คะแนนสูงสุด
    คืน item เดิมแต่เพิ่ม key 'duplicates' = จำนวน URL อื่นที่เป็นภาพเดียวกัน
    """
    kept: list[dict] = []
    kept_hashes: list[int] = []
    for it in items:
        h = hashes.get(it.get("url"))
        if h is None:
            kept.append(it)
            continue
        dup_of = None
        for i, kh in enumerate(kept_hashes):
            if hamming(h, kh) <= threshold:
                dup_of = i
                break
        if dup_of is None:
            it["duplicates"] = 0
            kept.append(it)
            kept_hashes.append(h)
        else:
            kept[dup_of]["duplicates"] = kept[dup_of].get("duplicates", 0) + 1
    return kept


def _resize_max(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


# สร้าง detector ครั้งเดียวแล้วใช้ซ้ำ
try:
    _DETECTOR = cv2.SIFT_create()
    _NORM = cv2.NORM_L2
    DETECTOR_NAME = "SIFT"
except AttributeError:  # เผื่อ build ที่ไม่มี SIFT -> fallback เป็น ORB
    _DETECTOR = cv2.ORB_create(nfeatures=2000)
    _NORM = cv2.NORM_HAMMING
    DETECTOR_NAME = "ORB"


def compute_features(image_bytes: bytes, max_dim: int = 2400):
    """คำนวณ keypoints + descriptors ของภาพ (cache ฝั่ง query ได้)"""
    gray = _load_gray(image_bytes)
    if gray is None:
        return None
    gray = _resize_max(gray, max_dim)
    kp, desc = _DETECTOR.detectAndCompute(gray, None)
    return gray, kp, desc


def match_in_image(query_feat, target_bytes: bytes, params: MatchParams | None = None) -> MatchResult:
    """ตรวจว่า query (ที่ผ่าน compute_features แล้ว) อยู่ในภาพปลายทางไหม"""
    p = params or MatchParams()
    if query_feat is None:
        return MatchResult(False, 0, 0, 0.0)
    q_gray, q_kp, q_desc = query_feat
    if q_desc is None or len(q_kp) < 4:
        return MatchResult(False, 0, 0, 0.0)

    target = compute_features(target_bytes, p.max_dim)
    if target is None:
        return MatchResult(False, 0, 0, 0.0)
    t_gray, t_kp, t_desc = target
    if t_desc is None or len(t_kp) < 4:
        return MatchResult(False, 0, 0, 0.0)

    matcher = cv2.BFMatcher(_NORM)
    try:
        knn = matcher.knnMatch(q_desc, t_desc, k=2)
    except cv2.error:
        return MatchResult(False, 0, 0, 0.0)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < p.ratio_test * n.distance:
            good.append(m)

    if len(good) < p.min_good_matches:
        conf = round(len(good) / max(1, p.min_inliers), 3)
        return MatchResult(False, 0, len(good), min(conf, 0.99))

    # หา homography ด้วย RANSAC เพื่อยืนยันและหาตำแหน่ง
    src = np.float32([q_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([t_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or mask is None:
        return MatchResult(False, 0, len(good), 0.0)

    inliers = int(mask.sum())
    found = inliers >= p.min_inliers

    box = None
    if found:
        h, w = q_gray.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(corners, H)
        box = [[int(pt[0][0]), int(pt[0][1])] for pt in proj]

    confidence = min(1.0, inliers / 40.0)  # normalize คร่าว ๆ
    return MatchResult(found, inliers, len(good), round(confidence, 3), box)
