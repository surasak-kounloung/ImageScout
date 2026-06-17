# Image Search — หาภาพย่อยในภาพใหญ่ + หาภาพที่คล้ายกัน

แอปค้นหารูปภาพบนหน้าเว็บที่ระบุ มี **2 โหมด**:

| โหมด | ทำอะไร | เทคนิค |
|------|--------|--------|
| **inside** (หาภาพย่อยในภาพใหญ่) | ภาพที่อัปโหลดปรากฏ "อยู่ข้างใน" รูปอื่น เช่น โลโก้/สินค้าในแบนเนอร์ | OpenCV **SIFT** + RANSAC/Homography |
| **similar** (หาภาพที่คล้ายกัน) | รูปที่หน้าตา "คล้ายกันทั้งภาพ" เช่น สินค้าเดียวกันคนละมุม | **CLIP** embeddings + **FAISS** |

ตัวอย่าง: อัปโหลดกล่อง `Neoclear` → ระบบไล่ดูรูปทั้งหมดในเว็บ → เจอว่า Neoclear อยู่ในแบนเนอร์ `button-meso`

## Tech Stack

| ส่วน | เทคโนโลยี |
|------|-----------|
| Backend / API | Python + FastAPI + Uvicorn |
| หาภาพย่อยในภาพ | OpenCV (opencv-contrib, SIFT) + NumPy |
| หาภาพคล้าย (เสริม) | PyTorch + open_clip + FAISS |
| Web crawling | httpx + BeautifulSoup (+ Playwright สำหรับเว็บ JS) |
| Frontend | HTML + JS (เสิร์ฟโดย FastAPI ไม่ต้องใช้ Node) |

## วิธีรัน

### 1. ติดตั้งหลัก (จำเป็น)

```powershell
cd backend
pip install -r requirements.txt

# ถ้าจะใช้โหมด render เว็บ JS ให้ติดตั้ง browser ของ Playwright ด้วย
python -m playwright install chromium
```

### 2. ติดตั้งโมดูลเสริม (เฉพาะถ้าจะใช้โหมด "หาภาพคล้าย")

```powershell
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-clip.txt
```
> ครั้งแรกที่ใช้โหมด similar จะดาวน์โหลด weights ของ CLIP (~350MB) อัตโนมัติ

### 3. รันเซิร์ฟเวอร์

```powershell
python main.py            # เปิด http://127.0.0.1:8000
python test_match.py      # (ตัวเลือก) ทดสอบ matcher offline ด้วยภาพสังเคราะห์
```

## การใช้งานบนหน้าเว็บ

1. เลือกโหมด: **หาภาพย่อยในภาพใหญ่** หรือ **หาภาพที่คล้ายกัน**
2. อัปโหลดภาพ query + ใส่ URL เว็บ (**ใส่ได้หลายลิงก์ บรรทัดละ 1 URL** ระบบจะ crawl ทุกหน้าแบบขนาน รวมรูปไม่ซ้ำ และบอกว่าแต่ละผลลัพธ์มาจากหน้าไหน)
3. (ตัวเลือก) ใส่ **CSS selector** ในช่อง "จำกัดขอบเขต" เช่น `.site-main` เพื่อดึงเฉพาะรูปที่อยู่ภายใต้ element นั้น (เว้นว่าง = ทั้งหน้า)
4. (ตัวเลือก) ติ๊ก "render ด้วยเบราว์เซอร์จริง" ถ้าเว็บโหลดรูปด้วย JavaScript
5. (ตัวเลือก) เปิด "ปรับความแม่น/ความไว" เพื่อปรับ slider
6. กด "ค้นหา"

## ปรับความแม่น / ความไว

**โหมด inside** (ปรับจาก UI หรือใน `matcher.py` → `MatchParams`):
- `min_inliers` — จำนวนจุดที่ผ่าน RANSAC ขั้นต่ำที่ถือว่า "เจอ" (น้อย = เจอง่าย, มาก = แม่น)
- `ratio_test` — ความเข้มงวดการจับคู่ฟีเจอร์ (ต่ำ = เข้มงวด)
- `max_dim` — ขนาดภาพสูงสุดก่อนย่อ (เร็ว/แม่น trade-off)

**โหมด similar** (ปรับจาก UI):
- `similarity_threshold` — cosine similarity ขั้นต่ำ (สูง = ต้องคล้ายมากจึงนับ)

## โครงสร้างโปรเจกต์

```
app-search-image/
├─ backend/
│  ├─ main.py                # FastAPI app + 2 โหมด + เสิร์ฟ frontend
│  ├─ matcher.py             # โหมด inside: SIFT + homography (รับ MatchParams)
│  ├─ clip_search.py         # โหมด similar: CLIP + FAISS (เสริม)
│  ├─ crawler.py             # ดึง URL รูป (httpx + Playwright)
│  ├─ test_match.py          # ทดสอบ offline
│  ├─ requirements.txt       # หลัก
│  └─ requirements-clip.txt  # เสริม (torch/clip/faiss)
├─ frontend/
│  ├─ index.html
│  └─ styles.css
```

## แสดงความคืบหน้าแบบ real-time

หน้าเว็บใช้ endpoint `POST /api/search/stream` ที่ส่งผลแบบ **streaming (NDJSON)** —
backend ส่ง event ความคืบหน้าออกมาเรื่อย ๆ ระหว่างทำงาน frontend จึงแสดง:
- แถบขั้นตอน (ดึงหน้าเว็บ → รวบรวมรูป → วิเคราะห์ภาพ → เสร็จสิ้น)
- progress bar + เปอร์เซ็นต์
- ตัวนับสด (ดึงหน้าเว็บ x/y, ดาวน์โหลด/วิเคราะห์ x/y, เจอแล้ว n)
- ตัวจับเวลาที่ใช้

ลำดับ event: `start → crawl → collected → (query_ready | download+embed_query) → analyze → done`

## หมายเหตุ / แนวทางต่อยอด

- **Playwright**: ใช้กับเว็บ SPA/lazy-load — เปิด Chromium จริง รอ JS โหลด แล้วเลื่อนหน้าเพื่อกระตุ้น lazy-load + ดักรูปจาก network ด้วย
- **FAISS**: ตอนนี้สร้าง index แบบ in-memory ต่อ 1 request หากต้องการค้นหาในคลังภาพขนาดใหญ่ ควรสร้าง index ถาวร (persist) แล้ว query ซ้ำได้
- **ไม่มีเพดานจำนวนรูป**: ระบบสแกนรูปทั้งหมดที่เจอ ไม่จำกัดต่อหน้าและไม่จำกัดผลรวม — เว็บที่มีรูปจำนวนมากจะใช้เวลานานขึ้น (โดยเฉพาะโหมด similar ที่รัน CLIP ทุกรูป) การทำงานยังขนานอยู่ (ดาวน์โหลด/วิเคราะห์พร้อมกันสูงสุด 8 งาน)
- ถ้า build OpenCV ไม่มี SIFT จะ fallback เป็น ORB อัตโนมัติ
- ความปลอดภัย: ควรเพิ่ม allow-list โดเมน / rate limit ก่อนนำขึ้น production
