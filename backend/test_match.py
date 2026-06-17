"""
ทดสอบ matcher แบบ offline ด้วยภาพสังเคราะห์ (ไม่ต้องพึ่งไฟล์ตัวอย่างภายนอก)
รัน: python test_match.py
คาดหวัง: พบภาพย่อย (query) อยู่ในภาพใหญ่ (target) และ sanity query ใน query เอง
"""
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import cv2
import numpy as np

import matcher


def _encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_pair() -> tuple[bytes, bytes]:
    """สร้างภาพใหญ่ที่มีแพทเทิร์นแบบสุ่ม + crop เป็นภาพ query (subset)"""
    rng = np.random.default_rng(42)
    big = rng.integers(0, 255, (320, 320, 3), dtype=np.uint8)
    big = cv2.GaussianBlur(big, (3, 3), 0)
    query = big[80:200, 80:200].copy()
    return _encode_png(query), _encode_png(big)


def main():
    query, target = _make_pair()

    feat = matcher.compute_features(query)
    assert feat is not None and feat[2] is not None, "อ่านภาพ query ไม่ได้"
    print(f"query keypoints: {len(feat[1])}")

    res = matcher.match_in_image(feat, target)
    print("ผลการค้นหาภาพย่อยในภาพใหญ่:")
    print(f"  found        = {res.found}")
    print(f"  inliers      = {res.inliers}")
    print(f"  good_matches = {res.good_matches}")
    print(f"  confidence   = {res.confidence}")
    print(f"  box          = {res.box}")

    self_res = matcher.match_in_image(feat, query)
    print(f"\nsanity (query in itself) found = {self_res.found}, inliers = {self_res.inliers}")

    assert res.found, "ผิดคาด: ควรเจอภาพย่อยในภาพใหญ่"
    assert self_res.found, "ผิดคาด: query ต้องเจอใน query เอง"
    print("\nPASS ✅")


if __name__ == "__main__":
    main()
