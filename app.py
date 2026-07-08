# -*- coding: utf-8 -*-
"""
PhotoStudio — AI 去背 × 尺寸排版 Web App
=========================================
功能 1：護照大頭照去背 + 4x6 吋 (300 DPI) 白底排版對位
功能 2：LINE 貼圖自動去背 + 規範輸出 (透明 PNG、偶數畫布)

啟動方式：
    streamlit run app.py
"""

import io
import os
import math
import json
import zipfile
import gc

import numpy as np
import cv2
import requests
import streamlit as st
from PIL import Image, ImageDraw
from rembg import remove, new_session
from streamlit_image_coordinates import streamlit_image_coordinates

# ------------------------------------------------------------------
# 全域常數
# ------------------------------------------------------------------
DPI = 300  # 印刷標準解析度

# 4x6 吋畫布（直式）——300 DPI
CANVAS_4x6_PORTRAIT = (1200, 1800)   # 4 吋 x 6 吋
CANVAS_4x6_LANDSCAPE = (1800, 1200)  # 6 吋 x 4 吋

GRAY_CUT_LINE = (150, 150, 150)      # 中灰色裁切線


# ------------------------------------------------------------------
# 單位換算工具（皆以 300 DPI 為基準）
# ------------------------------------------------------------------
def cm_to_px(cm: float) -> int:
    """公分 → 像素 (300 DPI)"""
    return round(cm / 2.54 * DPI)


def mm_to_px(mm: float) -> int:
    """公厘 → 像素 (300 DPI)"""
    return round(mm / 25.4 * DPI)


def px_to_mm(px: float) -> float:
    """像素 → 公厘 (300 DPI)"""
    return px / DPI * 25.4


# ------------------------------------------------------------------
# 互動計數器（瀏覽 / 讚 / 分享）
# 用免費計數服務 abacus 永久保存（雲端重部署也不歸零）；
# 連不上時後備到本機 stats.json，本機開發仍可用。
# ------------------------------------------------------------------
COUNTER_BASE = "https://abacus.jasoncameron.dev"
COUNTER_NS = "photostudio-ginohung-v1"         # 計數命名空間（唯一）
STATS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "stats.json")
_STAT_KEYS = ("views", "likes", "shares")


def _api_get(key):
    r = requests.get(f"{COUNTER_BASE}/get/{COUNTER_NS}/{key}", timeout=2.5)
    return int(r.json().get("value", 0)) if r.status_code == 200 else 0


def _api_hit(key):
    r = requests.get(f"{COUNTER_BASE}/hit/{COUNTER_NS}/{key}", timeout=2.5)
    r.raise_for_status()
    return int(r.json().get("value", 0))


def _load_file():
    try:
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            s = json.load(f)
    except Exception:
        s = {}
    return {k: int(s.get(k, 0)) for k in _STAT_KEYS}


def _save_file(stats):
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f)
    except Exception:
        pass


@st.cache_data(ttl=30, show_spinner=False)
def _fetch_stats():
    return {k: _api_get(k) for k in _STAT_KEYS}


def load_stats():
    """讀取三個計數（優先計數服務並快取30秒，失敗則讀本機檔）。"""
    try:
        return dict(_fetch_stats())
    except Exception:
        return _load_file()


def bump_stat(key):
    """計數 +1，回傳新值（服務失敗則寫本機檔）。"""
    try:
        return _api_hit(key)
    except Exception:
        s = _load_file()
        s[key] = s.get(key, 0) + 1
        _save_file(s)
        return s[key]


# 相片實際尺寸（寬 x 高，像素）
# 1 吋照：2.8cm x 3.5cm
PHOTO_1INCH = (cm_to_px(2.8), cm_to_px(3.5))   # ≈ (331, 413)
# 2 吋照：3.5cm x 4.5cm
PHOTO_2INCH = (cm_to_px(3.5), cm_to_px(4.5))   # ≈ (413, 531)

# 各尺寸的「頭頂→下顎」規範（cm）。2 吋依台灣護照規定 3.2~3.6cm。
SIZE_SPECS = {
    "1inch": {"photo": PHOTO_1INCH, "head_min": 2.4, "head_max": 2.8,
              "head_default": 2.6, "label": "1 吋 (2.8×3.5cm)"},
    "2inch": {"photo": PHOTO_2INCH, "head_min": 3.2, "head_max": 3.6,
              "head_default": 3.4, "label": "2 吋 (3.5×4.5cm)"},
}


# ------------------------------------------------------------------
# rembg 模型：以 cache_resource 快取，避免每次互動重新載入
# ------------------------------------------------------------------
# 上傳圖片最長邊上限（HF Spaces 記憶體大，可放寬以保品質）
MAX_INPUT_SIDE = 2400

# 可選去背模型（顯示名 → rembg 模型代號）
REMBG_MODELS = {
    "髮絲級 birefnet-portrait（最佳、較慢）": "birefnet-portrait",
    "高品質 isnet（推薦、快）": "isnet-general-use",
    "人像 u2net_human_seg": "u2net_human_seg",
    "輕量 u2netp（最快省資源）": "u2netp",
}
# 預設模型可用環境變數覆寫（HF Spaces 記憶體大→設 birefnet-portrait；
# 1GB 的 Streamlit Cloud 不設→用輕量 u2netp 保持穩定）
DEFAULT_REMBG_MODEL = os.environ.get("REMBG_DEFAULT", "u2netp")
if DEFAULT_REMBG_MODEL not in REMBG_MODELS.values():
    DEFAULT_REMBG_MODEL = "u2netp"


@st.cache_resource(show_spinner=False)
def load_rembg_session(model_name: str):
    """載入並快取指定 rembg 去背模型（每個模型各自快取）。"""
    return new_session(model_name)


def remove_background(img: Image.Image,
                      model_name: str = DEFAULT_REMBG_MODEL,
                      matting: bool = False) -> Image.Image:
    """對 PIL 影像去背，回傳帶透明通道的 RGBA 影像。
    matting=True 啟用 alpha matting 細修髮絲邊緣（較慢、較吃資源）。"""
    session = load_rembg_session(model_name)
    if matting:
        out = remove(img, session=session, alpha_matting=True,
                     alpha_matting_foreground_threshold=250,
                     alpha_matting_background_threshold=5,
                     alpha_matting_erode_size=6).convert("RGBA")
    else:
        out = remove(img, session=session).convert("RGBA")
    gc.collect()
    return out


# ------------------------------------------------------------------
# 影像處理：等比縮放
# ------------------------------------------------------------------
def resize_cover(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """
    等比縮放並置中裁切，使影像「填滿」目標框（cover 模式）。
    適合大頭照，避免出現白邊，且主體置中。
    """
    src_w, src_h = img.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def resize_contain(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """等比縮放使影像「完整容納」於範圍內（contain 模式），不裁切。"""
    src_w, src_h = img.size
    scale = min(max_w / src_w, max_h / src_h)
    new_w = max(1, round(src_w * scale))
    new_h = max(1, round(src_h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def _limit_size(img: Image.Image, max_side: int = MAX_INPUT_SIDE) -> Image.Image:
    """把過大的圖等比縮到最長邊 max_side 以內，降低記憶體與運算量。"""
    w, h = img.size
    m = max(w, h)
    if m > max_side:
        s = max_side / m
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))),
                         Image.LANCZOS)
    return img


def open_normalized(uploaded_file) -> Image.Image:
    """
    開啟上傳檔並正規化色彩模式，並限制最大尺寸（省記憶體）。
    - TIFF 可能是 CMYK / 灰階 / LA / P（調色盤）等，統一轉成 rembg 可處理的模式。
    - 保留含透明度者為 RGBA，其餘轉為 RGB。
    """
    img = Image.open(uploaded_file)
    img.load()  # 先載入，避免延後讀取造成檔案指標問題
    icc = img.info.get("icc_profile")   # 原圖內嵌 ICC（若有）
    converted = False
    if img.mode in ("LA", "PA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        converted = True
    elif img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")  # CMYK / L / P 等一律轉 RGB
        converted = True
    img = _limit_size(img)
    # 只在「色彩模式沒被轉換」時保留 ICC，避免把 CMYK 的描述檔錯嵌到 RGB 資料
    if icc and not converted:
        img.info["icc_profile"] = icc
    return img


def flatten_to_white(rgba: Image.Image) -> Image.Image:
    """將 RGBA 影像合成到純白背景上，回傳 RGB 影像。"""
    white_bg = Image.new("RGB", rgba.size, (255, 255, 255))
    white_bg.paste(rgba, (0, 0), rgba)  # 以 alpha 作為遮罩
    return white_bg


# ------------------------------------------------------------------
# 人臉偵測（OpenCV YuNet）與護照大頭照合規定位
# ------------------------------------------------------------------
_YUNET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "face_detection_yunet_2023mar.onnx",
)
_YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
              "face_detection_yunet/face_detection_yunet_2023mar.onnx")


def _ensure_yunet():
    """YuNet 模型不存在時自動下載（雲端部署不必把二進位檔放進 repo）。"""
    if not os.path.exists(_YUNET_PATH):
        os.makedirs(os.path.dirname(_YUNET_PATH), exist_ok=True)
        r = requests.get(_YUNET_URL, timeout=60)
        r.raise_for_status()
        with open(_YUNET_PATH, "wb") as f:
            f.write(r.content)
    return _YUNET_PATH


@st.cache_resource(show_spinner=False)
def load_face_detector():
    """載入並快取 YuNet 人臉偵測器（首次會自動下載模型）。"""
    return cv2.FaceDetectorYN.create(_ensure_yunet(), "", (320, 320),
                                     score_threshold=0.6)


def detect_head_metrics(rgba: Image.Image):
    """
    偵測人臉並量測頭部關鍵位置（以原圖像素座標）。
    - 頭頂 (crown)：由去背 alpha 遮罩找臉部水平範圍內最上緣不透明列（含頭髮）。
    - 下顎 (chin)：由 YuNet 臉框底部推估。
    偵測不到人臉時回傳 None。
    回傳 dict：crown_y, chin_y, face_cx, eye_y。
    """
    rgb = np.array(rgba.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]

    det = load_face_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None or len(faces) == 0:
        return None

    # 取面積最大的臉
    face = max(faces, key=lambda f: f[2] * f[3])
    fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
    eye_y = (float(face[5]) + float(face[7])) / 2.0  # 兩眼 y 平均

    # 頭頂：在臉部水平範圍（略放寬）內，找 alpha 最上緣
    alpha = np.array(rgba.split()[-1])
    cx0 = max(0, int(fx - fw * 0.15))
    cx1 = min(w, int(fx + fw * 1.15))
    band = alpha[:, cx0:cx1] if cx1 > cx0 else alpha
    rows = np.where(band.max(axis=1) > 20)[0]
    crown_y = int(rows[0]) if len(rows) else int(fy)

    # 下顎：YuNet 臉框底通常落在下巴附近，略往下延伸補足
    chin_y = int(fy + fh * 1.05)
    chin_y = min(chin_y, h - 1)

    return {
        "crown_y": crown_y,
        "chin_y": chin_y,
        "face_cx": fx + fw / 2.0,
        "eye_y": eye_y,
    }


def id_guide_geom(spec, photo_size, top_ratio=0.32):
    """
    計算「固定的法規目標框」幾何——完全不受任何微調滑桿影響。
    - crown：固定的頭頂基準線（依建議頭高與 top_ratio 決定上緣留白）。
    - chin_min / chin_max：下顎的合法範圍帶（對應 head_min~head_max）。
    - chin_ref：建議頭高對應的下顎位置。
    """
    pw, ph = photo_size
    ref_head = cm_to_px(spec["head_default"])
    crown = (ph - ref_head) * top_ratio
    return {
        "crown": crown,
        "chin_min": crown + cm_to_px(spec["head_min"]),
        "chin_max": crown + cm_to_px(spec["head_max"]),
        "chin_ref": crown + ref_head,
        "cx": pw / 2.0,
        "head_ref": ref_head,
    }


def compose_id_photo(rgba, metrics, photo_size, head_cm, crown_ref,
                     corr_pct=100.0, vshift_mm=0.0, hshift_mm=0.0):
    """
    依人臉量測結果，將去背人像縮放並定位到單張證件照白底畫布。
    - head_cm：目標「頭頂→下顎」高度（cm）→ 決定人像縮放。
    - crown_ref：固定的頭頂基準 y（來自 id_guide_geom），未微調時頭頂對齊此線。
    - corr_pct：整體縮放校正（%）。vshift_mm/hshift_mm：上下/左右微調（正=下/右）。
    回傳 (單張 RGB 影像, 實際頭高cm)。輔助線幾何另由 id_guide_geom 提供（固定）。
    """
    pw, ph = photo_size
    target_head_px = cm_to_px(head_cm)
    corr = corr_pct / 100.0

    if metrics is not None:
        src_head = max(1.0, metrics["chin_y"] - metrics["crown_y"])
        crown_y = metrics["crown_y"]
        face_cx = metrics["face_cx"]
    else:
        # 無臉：粗估人物約佔畫面高 65%，置中
        src_head = rgba.size[1] * 0.65
        crown_y = rgba.size[1] * 0.10
        face_cx = rgba.size[0] / 2.0

    scale = (target_head_px / src_head) * corr

    W0, H0 = rgba.size
    resized = rgba.resize((max(1, round(W0 * scale)),
                           max(1, round(H0 * scale))), Image.LANCZOS)

    crown_s = crown_y * scale
    facecx_s = face_cx * scale
    vshift_px = mm_to_px(vshift_mm)
    hshift_px = mm_to_px(hshift_mm)

    paste_x = round(pw / 2.0 - facecx_s + hshift_px)
    paste_y = round(crown_ref - crown_s + vshift_px)

    canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
    canvas.paste(resized, (paste_x, paste_y), resized)

    return canvas, head_cm * corr


def _dashed_line(draw, p1, p2, fill, width=2, dash=14, gap=9):
    """畫虛線（PIL 無原生虛線）。"""
    x1, y1 = p1
    x2, y2 = p2
    total = math.hypot(x2 - x1, y2 - y1)
    if total == 0:
        return
    dx, dy = (x2 - x1) / total, (y2 - y1) / total
    pos = 0.0
    while pos < total:
        s = pos
        e = min(pos + dash, total)
        draw.line([(x1 + dx * s, y1 + dy * s), (x1 + dx * e, y1 + dy * e)],
                  fill=fill, width=width)
        pos += dash + gap


def draw_id_guides(photo_rgb: Image.Image, gg) -> Image.Image:
    """
    疊加「固定」的法規定位輔助線（僅供預覽，不會印出）：
    - 紅色頭頂基準線 + 頭部橢圓框（建議頭高）。
    - 綠色下顎合法範圍帶（3.2~3.6cm 對應區間）。
    - 灰色垂直中線。
    """
    base = photo_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    w, h = base.size
    red = (220, 30, 30, 230)
    green = (0, 160, 60, 230)

    crown = gg["crown"]
    cx = gg["cx"]
    ow = gg["head_ref"] * 0.78

    # 下顎合法範圍帶（半透明綠底 + 上下綠虛線）
    d.rectangle([0, gg["chin_min"], w, gg["chin_max"]], fill=(0, 160, 60, 40))
    _dashed_line(d, (0, gg["chin_min"]), (w, gg["chin_min"]), green, width=2)
    _dashed_line(d, (0, gg["chin_max"]), (w, gg["chin_max"]), green, width=2)

    # 頭頂基準線 + 建議頭部橢圓框
    _dashed_line(d, (0, crown), (w, crown), red, width=2)
    d.ellipse([cx - ow / 2, crown, cx + ow / 2, gg["chin_ref"]],
              outline=red, width=2)

    # 垂直中線
    _dashed_line(d, (cx, 0), (cx, h), (120, 120, 120, 180),
                 width=1, dash=10, gap=8)

    return Image.alpha_composite(base, overlay).convert("RGB")


# ------------------------------------------------------------------
# 功能 1：4x6 排版核心
# ------------------------------------------------------------------
def _grid_capacity(canvas_size, photo_size, edge_margin, spacing):
    """計算某畫布/相片方向下可排列的欄列數與總張數。"""
    cw, ch = canvas_size
    pw, ph = photo_size
    usable_w = cw - 2 * edge_margin
    usable_h = ch - 2 * edge_margin
    if pw > usable_w or ph > usable_h:
        return 0, 0, 0
    cols = int((usable_w + spacing) // (pw + spacing))
    rows = int((usable_h + spacing) // (ph + spacing))
    return cols, rows, cols * rows


def build_4x6_sheet(photo_rgb: Image.Image, photo_size, edge_margin, spacing):
    """
    在 4x6 白底畫布上，最大化排列相同尺寸的相片。
    會自動比較「直式 / 橫式畫布」，選擇可放最多張的方向。
    回傳 (排版後 RGB 影像, 總張數, 欄, 列)。
    """
    best = None  # (count, canvas_size, cols, rows)
    # 固定使用橫式 4x6 畫布（1 吋 / 2 吋皆橫向排版）
    for canvas_size in (CANVAS_4x6_LANDSCAPE,):
        cols, rows, count = _grid_capacity(canvas_size, photo_size, edge_margin, spacing)
        if best is None or count > best[0]:
            best = (count, canvas_size, cols, rows)

    count, canvas_size, cols, rows = best
    cw, ch = canvas_size
    pw, ph = photo_size

    sheet = Image.new("RGB", (cw, ch), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    if count == 0:
        return sheet, 0, 0, 0

    # 讓整個相片格陣列在畫布上置中，四周留白平均
    block_w = cols * pw + (cols - 1) * spacing
    block_h = rows * ph + (rows - 1) * spacing
    offset_x = (cw - block_w) // 2
    offset_y = (ch - block_h) // 2

    for r in range(rows):
        for c in range(cols):
            x = offset_x + c * (pw + spacing)
            y = offset_y + r * (ph + spacing)
            sheet.paste(photo_rgb, (x, y))
            # 極細中灰色裁切線
            draw.rectangle([x, y, x + pw - 1, y + ph - 1],
                           outline=GRAY_CUT_LINE, width=1)

    return sheet, count, cols, rows


# ------------------------------------------------------------------
# 功能 2：LINE 貼圖
# ------------------------------------------------------------------
LINE_MAX_W = 370
LINE_MAX_H = 320
LINE_PADDING = 10  # 四邊透明邊距


def make_even(n: int) -> int:
    """向上調整為最接近的偶數。"""
    return n if n % 2 == 0 else n + 1


def remove_by_colors(img, colors, tol):
    """
    滴管取色去背：把「與任一取樣色的 RGB 距離 < tol」的像素設為透明。
    - colors：取樣色清單 [(r,g,b), ...]，可多次點選累加。
    - tol：容差（顏色距離），越大移除越多。
    適合純色/單一背景（如綠底貼圖）。回傳 RGBA。
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.int32)   # int32：不溢位、比 float 省記憶體
    h, w = arr.shape[:2]
    alpha = np.full((h, w), 255, np.uint8)
    t2 = tol * tol                          # 比較平方距離，省去 sqrt 的大陣列
    rr, gg, bb = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    for (r, g, b) in colors:
        dist2 = (rr - r) ** 2 + (gg - g) ** 2 + (bb - b) ** 2
        alpha[dist2 < t2] = 0
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha, "L"))
    del arr, rr, gg, bb
    gc.collect()
    return rgba


def fit_to_line_canvas(rgba: Image.Image):
    """
    將去背後 RGBA 主體裁掉透明外框→等比縮放至 LINE 規範→四邊 10px 透明留白→
    畫布寬高補為偶數。回傳 (RGBA 畫布, info)。
    """
    bbox = rgba.getbbox()
    if bbox:
        rgba = rgba.crop(bbox)

    inner_w = LINE_MAX_W - 2 * LINE_PADDING
    inner_h = LINE_MAX_H - 2 * LINE_PADDING
    subject = resize_contain(rgba, inner_w, inner_h)
    sw, sh = subject.size

    canvas_w = make_even(sw + 2 * LINE_PADDING)
    canvas_h = make_even(sh + 2 * LINE_PADDING)

    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    canvas.paste(subject, ((canvas_w - sw) // 2, (canvas_h - sh) // 2), subject)

    info = {"subject_px": (sw, sh), "canvas_px": (canvas_w, canvas_h)}
    return canvas, info


def process_line_sticker(uploaded_img: Image.Image,
                         model_name: str = DEFAULT_REMBG_MODEL):
    """LINE 貼圖 AI 去背主流程：AI 去背→套 LINE 規範畫布。"""
    return fit_to_line_canvas(remove_background(uploaded_img, model_name))


def _foreground_mask(img_rgb, bg_colors, tol):
    """回傳前景布林遮罩（不接近任一背景色者為 True）。"""
    arr = np.asarray(img_rgb.convert("RGB")).astype(np.float32)
    fg = np.ones(arr.shape[:2], bool)
    for (r, g, b) in bg_colors:
        d = np.sqrt((arr[:, :, 0] - r) ** 2 +
                    (arr[:, :, 1] - g) ** 2 +
                    (arr[:, :, 2] - b) ** 2)
        fg &= (d >= tol)
    return fg


def _content_bands(profile, gutter_thresh, min_gap):
    """
    以投影剖面切出「內容帶」：長度 >= min_gap 且低於 gutter_thresh 的連續段視為
    溝縫（背景），溝縫之間即為一列（或一欄）內容。
    """
    n = len(profile)
    is_gap = profile < gutter_thresh
    gaps, start = [], None
    for i, v in enumerate(is_gap):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_gap:
                gaps.append((start, i))
            start = None
    if start is not None and n - start >= min_gap:
        gaps.append((start, n))

    bands, prev = [], 0
    for gs, ge in gaps:
        if gs > prev:
            bands.append((prev, gs))
        prev = ge
    if prev < n:
        bands.append((prev, n))
    return [(a, b) for (a, b) in bands if profile[a:b].max() > gutter_thresh]


def split_sticker_sheet(img_rgb, bg_colors, tol,
                        gutter_thresh=0.012, min_gap_ratio=0.008):
    """
    偵測純色背景「整版貼圖」的格狀排列，自動分割成多張個別貼圖。
    回傳 (stickers[(RGBA, info)...], cells[(x0,y0,x1,y1)...])，
    依閱讀順序（上→下、左→右）排列。
    """
    rgb = img_rgb.convert("RGB")
    fg = _foreground_mask(rgb, bg_colors, tol)
    H, W = fg.shape
    min_gap_r = max(4, int(H * min_gap_ratio))
    min_gap_c = max(4, int(W * min_gap_ratio))

    row_bands = _content_bands(fg.mean(axis=1), gutter_thresh, min_gap_r)
    col_bands = _content_bands(fg.mean(axis=0), gutter_thresh, min_gap_c)

    alpha = Image.fromarray((fg * 255).astype(np.uint8), "L")
    rgba_full = rgb.convert("RGBA")
    rgba_full.putalpha(alpha)

    stickers, cells = [], []
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            sub = fg[r0:r1, c0:c1]
            if sub.sum() < 50:          # 空格：跳過
                continue
            ys, xs = np.where(sub)
            bx0, bx1 = xs.min() + c0, xs.max() + c0 + 1
            by0, by1 = ys.min() + r0, ys.max() + r0 + 1
            crop = rgba_full.crop((bx0, by0, bx1, by1))
            stickers.append(fit_to_line_canvas(crop))
            cells.append((bx0, by0, bx1, by1))
    return stickers, cells


def guess_grid_counts(img_rgb, bg_colors, tol,
                      gutter_thresh=0.012, min_gap_ratio=0.008):
    """自動猜測欄數/列數（供預填，使用者可再修正）。回傳 (n_cols, n_rows)。"""
    rgb = img_rgb.convert("RGB")
    fg = _foreground_mask(rgb, bg_colors, tol)
    H, W = fg.shape
    rb = _content_bands(fg.mean(axis=1), gutter_thresh, max(4, int(H * min_gap_ratio)))
    cb = _content_bands(fg.mean(axis=0), gutter_thresh, max(4, int(W * min_gap_ratio)))
    return max(1, len(cb)), max(1, len(rb))


def sheet_bbox(img_rgb, bg_colors, tol):
    """整版前景外框 (x0,y0,x1,y1)；無前景時回整張。"""
    fg = _foreground_mask(img_rgb.convert("RGB"), bg_colors, tol)
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return (0, 0, img_rgb.width, img_rgb.height)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def draw_grid_lines(img_rgb, region, n_cols, n_rows):
    """依切框範圍與欄列數畫出等分紅色格線（即時預覽用，不裁切）。"""
    x0, y0, x1, y1 = region
    base = img_rgb.convert("RGB").copy()
    d = ImageDraw.Draw(base)
    for c in range(n_cols + 1):
        x = int(round(x0 + (x1 - x0) * c / n_cols))
        d.line([(x, y0), (x, y1)], fill=(230, 20, 20), width=3)
    for r in range(n_rows + 1):
        y = int(round(y0 + (y1 - y0) * r / n_rows))
        d.line([(x0, y), (x1, y)], fill=(230, 20, 20), width=3)
    return base


def split_sticker_grid(img_rgb, bg_colors, tol, n_cols, n_rows,
                       region=None, pad=0):
    """
    以「欄數 × 列數」把整版貼圖等分切割（適合規則格狀，最穩定）。
    region 指定切框範圍 (x0,y0,x1,y1)；None 時自動取整體前景外框。
    每格取前景 bbox（再向外擴 pad 像素）裁出個別貼圖。
    回傳 (stickers[(RGBA,info)], cells)。
    """
    rgb = img_rgb.convert("RGB")
    fg = _foreground_mask(rgb, bg_colors, tol)
    if region is None:
        x0, y0, x1, y1 = sheet_bbox(rgb, bg_colors, tol)
    else:
        x0, y0, x1, y1 = region
    if x1 <= x0 or y1 <= y0:
        return [], []
    cw = (x1 - x0) / n_cols
    ch = (y1 - y0) / n_rows

    alpha = Image.fromarray((fg * 255).astype(np.uint8), "L")
    rgba_full = rgb.convert("RGBA")
    rgba_full.putalpha(alpha)

    stickers, cells = [], []
    for r in range(n_rows):
        for c in range(n_cols):
            cx0, cx1 = int(round(x0 + c * cw)), int(round(x0 + (c + 1) * cw))
            cy0, cy1 = int(round(y0 + r * ch)), int(round(y0 + (r + 1) * ch))
            sub = fg[cy0:cy1, cx0:cx1]
            if sub.sum() < 50:          # 空格：跳過
                continue
            sy, sx = np.where(sub)
            bx0 = max(cx0, sx.min() + cx0 - pad)
            by0 = max(cy0, sy.min() + cy0 - pad)
            bx1 = min(cx1, sx.max() + cx0 + 1 + pad)
            by1 = min(cy1, sy.max() + cy0 + 1 + pad)
            crop = rgba_full.crop((bx0, by0, bx1, by1))
            stickers.append(fit_to_line_canvas(crop))
            cells.append((bx0, by0, bx1, by1))
    return stickers, cells


def draw_lines(img_rgb, xlines, ylines):
    """依「每條垂直線 x 座標」與「每條水平線 y 座標」畫紅色切線（可不等分）。"""
    xs = sorted(int(v) for v in xlines)
    ys = sorted(int(v) for v in ylines)
    base = img_rgb.convert("RGB").copy()
    d = ImageDraw.Draw(base)
    y0, y1 = ys[0], ys[-1]
    x0, x1 = xs[0], xs[-1]
    for x in xs:
        d.line([(x, y0), (x, y1)], fill=(230, 20, 20), width=3)
    for y in ys:
        d.line([(x0, y), (x1, y)], fill=(230, 20, 20), width=3)
    return base


def split_by_lines(img_rgb, bg_colors, tol, xlines, ylines, pad=0, boxes=None):
    """
    以指定的「垂直線 xlines、水平線 ylines」切割整版（可自由不等分）。
    每格預設取前景 bbox（再向外擴 pad 像素、夾在該格內）裁出。
    boxes：可選 dict {(r,c): (bx0,by0,bx1,by1)}，單獨覆寫某格的裁切框。
    回傳 (stickers[(RGBA,info)], cells)。
    """
    boxes = boxes or {}
    rgb = img_rgb.convert("RGB")
    fg = _foreground_mask(rgb, bg_colors, tol)
    alpha = Image.fromarray((fg * 255).astype(np.uint8), "L")
    rgba_full = rgb.convert("RGBA")
    rgba_full.putalpha(alpha)

    xs = sorted(int(v) for v in xlines)
    ys = sorted(int(v) for v in ylines)
    stickers, cells = [], []
    for r in range(len(ys) - 1):
        for c in range(len(xs) - 1):
            cx0, cx1 = xs[c], xs[c + 1]
            cy0, cy1 = ys[r], ys[r + 1]
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            if (r, c) in boxes:
                bx0, by0, bx1, by1 = boxes[(r, c)]
                bx0 = max(cx0, min(int(bx0), cx1 - 1))
                by0 = max(cy0, min(int(by0), cy1 - 1))
                bx1 = max(bx0 + 1, min(int(bx1), cx1))
                by1 = max(by0 + 1, min(int(by1), cy1))
            else:
                sub = fg[cy0:cy1, cx0:cx1]
                if sub.sum() < 50:          # 空格：跳過
                    continue
                sy, sx = np.where(sub)
                bx0 = max(cx0, sx.min() + cx0 - pad)
                by0 = max(cy0, sy.min() + cy0 - pad)
                bx1 = min(cx1, sx.max() + cx0 + 1 + pad)
                by1 = min(cy1, sy.max() + cy0 + 1 + pad)
            stickers.append(fit_to_line_canvas(rgba_full.crop((bx0, by0, bx1, by1))))
            cells.append((bx0, by0, bx1, by1))
    return stickers, cells


def draw_cells(img_rgb, cells):
    """在原圖上畫紅框標示每張貼圖的分割範圍（供預覽核對）。"""
    base = img_rgb.convert("RGB").copy()
    d = ImageDraw.Draw(base)
    for (x0, y0, x1, y1) in cells:
        d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(230, 20, 20), width=3)
    return base


# ------------------------------------------------------------------
# 下載用：影像 → BytesIO
# ------------------------------------------------------------------
def image_to_bytes(img: Image.Image, fmt: str, icc: bytes = None) -> bytes:
    """
    將 PIL 影像編碼為指定格式的 bytes（內嵌 300 DPI）。
    icc：原圖 ICC 描述檔（bytes），會原樣嵌回輸出以忠實保留色彩。
    （PDF 因 Pillow 不支援嵌 ICC，故不帶；色彩關鍵請用 TIFF/JPG。）
    """
    buf = io.BytesIO()
    fmt = fmt.upper()
    kw = {"icc_profile": icc} if icc else {}
    if fmt in ("JPG", "JPEG"):
        img.convert("RGB").save(buf, format="JPEG", quality=95,
                                dpi=(DPI, DPI), **kw)
    elif fmt == "TIFF":
        img.convert("RGB").save(buf, format="TIFF", dpi=(DPI, DPI),
                                compression="tiff_lzw", **kw)
    elif fmt == "PDF":
        img.convert("RGB").save(buf, format="PDF", resolution=DPI)
    elif fmt == "PNG":
        # optimize 壓縮，確保單張 LINE 貼圖遠小於 1MB
        img.save(buf, format="PNG", dpi=(DPI, DPI), optimize=True, **kw)
    else:
        raise ValueError(f"不支援的格式：{fmt}")
    buf.seek(0)
    return buf.getvalue()


# ==================================================================
# Streamlit UI
# ==================================================================
st.set_page_config(page_title="PhotoStudio 去背排版工具",
                   page_icon="📸", layout="wide")

# 全站字級放大
st.markdown("""
<style>
section.main p, section.main label, section.main li,
.stMarkdown, div[data-testid="stCaptionContainer"],
div[data-testid="stMarkdownContainer"] p { font-size: 1.12rem !important; }
div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
div[data-testid="stMetricLabel"] { font-size: 1.05rem !important; }
h1 { font-size: 2.5rem !important; }
h3 { font-size: 1.6rem !important; }
.stButton button, .stDownloadButton button { font-size: 1.1rem !important; }
</style>
""", unsafe_allow_html=True)

# ----- 計數初始化（每個 session 只讀一次；進站 +1 瀏覽）-----
if "stats" not in st.session_state:
    st.session_state["stats"] = load_stats()
if not st.session_state.get("counted_view"):
    st.session_state["counted_view"] = True
    st.session_state["stats"]["views"] = bump_stat("views")
_stats = st.session_state["stats"]

# 標題（左）＋ 互動統計（右上角）
_tcol, _scol = st.columns([2, 1])
with _tcol:
    st.title("📸 PhotoStudio — AI 去背 × 尺寸排版")
    st.caption("護照大頭照 4x6 排版　|　LINE 貼圖規範輸出　·　300 DPI 印刷標準")
with _scol:
    with st.container(border=True):
        st.markdown("**📊 互動統計**")
        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("👁️ 瀏覽", _stats["views"])
        _m2.metric("👍 讚", _stats["likes"])
        _m3.metric("🔗 分享", _stats["shares"])
        _b1, _b2 = st.columns(2)
        if _b1.button("👍 按讚", use_container_width=True):
            if st.session_state.get("has_liked"):
                st.toast("你已經按過讚了，謝謝！")
            else:
                st.session_state["has_liked"] = True
                st.session_state["stats"]["likes"] = bump_stat("likes")
                st.rerun()
        if _b2.button("🔗 分享", use_container_width=True):
            st.session_state["stats"]["shares"] = bump_stat("shares")
            st.session_state["show_share"] = True
            st.rerun()
        if st.session_state.get("show_share"):
            st.caption("複製分享：")
            st.code("https://huggingface.co/spaces/Ginohung/PhotoStudio",
                    language=None)

# 功能選擇
mode = st.radio(
    "選擇功能",
    ["護照大頭照 (4x6 排版)", "LINE 貼圖 (透明 PNG)"],
    horizontal=True,
)

_multi = not mode.startswith("護照")   # LINE 貼圖可一次上傳多張
uploaded = st.file_uploader(
    "上傳圖片（支援 JPG / PNG / TIFF）"
    + ("　·　可一次選多張" if _multi else ""),
    type=["jpg", "jpeg", "png", "tif", "tiff"],
    accept_multiple_files=_multi,
)

# 去背模型（AI 去背時使用；移到主畫面）
_mcol1, _mcol2 = st.columns([1, 1])
with _mcol1:
    _mlabels = list(REMBG_MODELS.keys())
    _default_idx = list(REMBG_MODELS.values()).index(DEFAULT_REMBG_MODEL)
    _model_label = st.selectbox("🪄 AI 去背模型（品質 / 速度）", _mlabels,
                                index=_default_idx, key="rembg_model_label")
    rembg_model = REMBG_MODELS[_model_label]
with _mcol2:
    st.caption("髮絲級最自然但較慢；isnet 品質好又快。"
               "（只有選「AI 去背」時會用到；首次使用該模型會先下載。）")

st.divider()

# ------------------------------------------------------------------
# 分支 1：護照大頭照
# ------------------------------------------------------------------
if mode.startswith("護照"):
    col_a, col_b = st.columns(2)
    with col_a:
        size_label = st.selectbox(
            "相片尺寸",
            [SIZE_SPECS["2inch"]["label"], SIZE_SPECS["1inch"]["label"]],
        )
    with col_b:
        out_fmt = st.selectbox("下載格式", ["JPG", "TIFF", "PDF"])
        if out_fmt == "PDF":
            st.caption("⚠️ PDF 不嵌 ICC；色彩關鍵請用 TIFF（無損）或 JPG。")
        else:
            st.caption("✅ 會原樣嵌回原圖 ICC，色彩忠實保留。")

    size_key = "1inch" if size_label.startswith("1") else "2inch"
    spec = SIZE_SPECS[size_key]
    photo_size = spec["photo"]

    bg_method = st.radio(
        "去背方式",
        ["AI 去背", "滴管取色去背（純色背景）",
         "已去背（透明PNG，用外部去背成品）", "不去背（相館已處理，直接排版）"],
        horizontal=True, key="pp_bg_method")
    is_no_bg = bg_method.startswith("不去背")
    is_dropper = bg_method.startswith("滴管")
    is_prebg = bg_method.startswith("已去背")

    if uploaded is None:
        st.info("請先於上方上傳一張人物照片。")

    # ---- 模式 C：不去背，直接把相館成品縮放排版 ----
    elif is_no_bg:
        src = open_normalized(uploaded)
        _icc = src.info.get("icc_profile")     # 保留原圖 ICC
        st.image(src, caption="原始圖片（相館已處理）", width=240)
        single = resize_cover(src.convert("RGB"), *photo_size)  # 縮放填滿相片框
        sheet, count, cols, rows = build_4x6_sheet(
            single, photo_size, edge_margin=0, spacing=0)
        pcol1, pcol2 = st.columns([1, 2])
        with pcol1:
            st.image(single, width=260, caption="單張")
        with pcol2:
            st.image(sheet, width=760, caption=f"4x6 排版（{count} 張）")
        st.info(f"共 {count} 張（{cols} 欄 × {rows} 列，橫式）。"
                "不去背模式：直接縮放至相片框排版，色彩數值不變。")
        data = image_to_bytes(sheet, out_fmt, icc=_icc)
        mime = {"JPG": "image/jpeg", "TIFF": "image/tiff",
                "PDF": "application/pdf"}[out_fmt]
        st.download_button(f"⬇️ 下載 4x6 排版（{out_fmt}）", data,
                           file_name=f"passport_4x6_{size_key}.{out_fmt.lower()}",
                           mime=mime)

    # ---- 模式 A/B：AI 去背 或 滴管取色去背（都走頭部定位流程）----
    else:
        src = open_normalized(uploaded)
        st.image(src, caption="原始圖片", width=240)

        # 滴管模式：先在原圖點背景取色
        pp_colors, pp_tol = [], 40
        if is_dropper:
            st.session_state.setdefault("pp_colors", {})
            pp_colors = st.session_state["pp_colors"].setdefault(
                uploaded.name, [])
            pp_tol = st.slider("容差（越大移除越多相近色）", 5, 150, 40, 1,
                               key="pp_tol")
            st.caption("👆 點原圖**背景**取色（可多次點不同區域累加）")
            _pk = streamlit_image_coordinates(src, width=340, key="pp_pick")
            if _pk and _pk.get("unix_time") != st.session_state.get("pp_pick_t"):
                st.session_state["pp_pick_t"] = _pk["unix_time"]
                _rw = _pk.get("width") or 340
                _rh = _pk.get("height") or 340
                _nx = min(max(int(_pk["x"] * src.width / _rw), 0), src.width - 1)
                _ny = min(max(int(_pk["y"] * src.height / _rh), 0), src.height - 1)
                pp_colors.append(tuple(int(c) for c in src.convert("RGB")
                                       .getpixel((_nx, _ny))))
            if pp_colors:
                st.caption("已取色：" +
                           "　".join("#%02X%02X%02X" % c for c in pp_colors))
            _dc1, _dc2 = st.columns(2)
            if _dc1.button("↩ 移除上一個取色") and pp_colors:
                pp_colors.pop()
                st.rerun()
            if _dc2.button("🗑 清除全部取色"):
                pp_colors.clear()
                st.rerun()

        # 已去背模式：用外部去背成品，只檢查是否真的有透明背景
        _has_alpha = (src.mode == "RGBA"
                      and int(np.asarray(src.split()[-1]).min()) < 250)
        if is_prebg:
            st.caption("用你上傳的**透明 PNG**（在 photogrid 等工具去背好的成品）；"
                       "本 App 不再去背，只做頭部定位與排版，完整保留你的去背品質。")
            if not _has_alpha:
                st.warning("⚠️ 這張圖沒有透明背景。請先用去背工具做成**透明 PNG** 再上傳，"
                           "或改選「AI 去背 / 滴管」。")

        # AI 模式：可選髮絲細修（alpha matting）
        pp_matting = False
        if not is_dropper and not is_prebg:
            pp_matting = st.checkbox(
                "🧵 髮絲細修（alpha matting，較慢、邊緣更柔）",
                value=False, key="pp_matting")

        # 第一階段：去背 + 人臉偵測（結果存 session_state，避免調滑桿時重跑）
        pp_key = (uploaded.name, bg_method, pp_matting)
        need_run = st.session_state.get("pp_key") != pp_key
        btn_label = "🚀 開始去背並偵測人臉" if need_run else "🔄 重新去背偵測"
        if st.button(btn_label, type="primary"):
            with st.spinner("去背與人臉偵測中，請稍候…"):
                if is_dropper:
                    rgba = (remove_by_colors(src, pp_colors, pp_tol)
                            if pp_colors else src.convert("RGBA"))
                elif is_prebg:
                    rgba = src.convert("RGBA")   # 直接用既有 alpha
                else:
                    rgba = remove_background(src, rembg_model, matting=pp_matting)
                metrics = detect_head_metrics(rgba)
            st.session_state["pp_rgba"] = rgba
            st.session_state["pp_metrics"] = metrics
            # 預覽用低解析度副本（加快調滑桿的即時預覽；下載時才用全解析度）
            _ps = min(1.0, 1100.0 / max(rgba.size))
            st.session_state["pp_rgba_prev"] = (rgba.resize(
                (max(1, round(rgba.size[0] * _ps)),
                 max(1, round(rgba.size[1] * _ps))), Image.LANCZOS)
                if _ps < 1.0 else rgba)
            st.session_state["pp_prev_scale"] = _ps
            st.session_state["pp_name"] = uploaded.name
            st.session_state["pp_key"] = pp_key
            st.session_state.pop("pp_dl", None)   # 清除舊的下載檔

        # 第二階段：已處理 → 顯示微調滑桿 + 即時預覽（不再重跑去背）
        if st.session_state.get("pp_key") == pp_key \
                and "pp_rgba" in st.session_state:
            rgba = st.session_state["pp_rgba"]                    # 全解析度（下載用）
            metrics = st.session_state["pp_metrics"]
            rgba_prev = st.session_state.get("pp_rgba_prev", rgba)  # 低解析度（預覽用）
            _pscale = st.session_state.get("pp_prev_scale", 1.0)
            metrics_prev = (None if metrics is None
                            else {k: v * _pscale for k, v in metrics.items()})

            if metrics is None:
                st.warning("⚠️ 未偵測到清晰正面人臉，改用預設置中定位。"
                           "請確認照片為正面、五官清晰，或用下方滑桿手動微調。")
            else:
                st.success("✅ 已偵測到人臉，可用下方滑桿即時微調定位。")

            gg = id_guide_geom(spec, photo_size)
            DISP_W = 340  # 單張預覽顯示寬（px，固定；勿超過欄寬以免抖動）

            # --- 狀態初始化（務必在建立 widget 之前）---
            lo_h, hi_h = spec["head_min"] - 0.4, spec["head_max"] + 0.4
            st.session_state.setdefault("pp_head", spec["head_default"])
            st.session_state["pp_head"] = min(hi_h, max(lo_h,
                                              st.session_state["pp_head"]))
            st.session_state.setdefault("pp_corr", 100)
            st.session_state.setdefault("pp_vshift", 0.0)
            st.session_state.setdefault("pp_hshift", 0.0)

            # --- 處理「上一輪在預覽圖上的點擊」→ 把臉部中心移到點擊處 ---
            # （必須在建立位移滑桿 widget 之前修改其 session_state）
            click = st.session_state.get("pp_click")
            if click and click.get("unix_time") != st.session_state.get("pp_click_t"):
                st.session_state["pp_click_t"] = click["unix_time"]
                rw = click.get("width") or DISP_W
                rh = click.get("height") or DISP_W
                # 顯示座標 → 相片像素座標
                px = click["x"] * photo_size[0] / rw
                py = click["y"] * photo_size[1] / rh
                head_now = st.session_state["pp_head"]
                corr_now = st.session_state["pp_corr"]
                actual_head_px = cm_to_px(head_now) * corr_now / 100.0
                hpx = px - photo_size[0] / 2.0                      # 臉部中心 x
                vpx = py - gg["crown"] - actual_head_px / 2.0        # 臉部中心 y
                st.session_state["pp_hshift"] = round(
                    min(20.0, max(-20.0, px_to_mm(hpx))), 2)
                st.session_state["pp_vshift"] = round(
                    min(20.0, max(-20.0, px_to_mm(vpx))), 2)

            st.markdown("**微調（即時預覽，不需重新去背）**")
            c1, c2 = st.columns(2)
            with c1:
                head_cm = st.slider(
                    f"目標頭高（頭頂→下顎，cm）　法規 {spec['head_min']}~{spec['head_max']}",
                    min_value=lo_h, max_value=hi_h, step=0.05, key="pp_head",
                )
                corr_pct = st.slider("整體縮放校正（%）", 70, 130,
                                     step=1, key="pp_corr")
            with c2:
                st.slider("上下位置（mm，正=下）", -20.0, 20.0,
                          step=0.1, key="pp_vshift")
                st.slider("左右位置（mm，正=右）", -20.0, 20.0,
                          step=0.1, key="pp_hshift")
            vshift = st.session_state["pp_vshift"]
            hshift = st.session_state["pp_hshift"]
            show_guide = st.checkbox("顯示定位輔助線（僅預覽，不會印出）", value=True)

            # 即時預覽用低解析度來源（快）；下載時才用全解析度重算
            single, actual_cm = compose_id_photo(
                rgba_prev, metrics_prev, photo_size, head_cm, gg["crown"],
                corr_pct=corr_pct, vshift_mm=vshift, hshift_mm=hshift,
            )

            # 合規檢查
            legal = spec["head_min"] <= actual_cm <= spec["head_max"]
            status = "✅ 符合規範" if legal else "❌ 超出法規範圍"
            st.caption(
                f"實際頭高約 **{actual_cm:.2f} cm**"
                f"（{status}；法規 {spec['head_min']}~{spec['head_max']} cm），"
                f"單張 {photo_size[0]}×{photo_size[1]} px @300DPI。"
                f"　🟩 綠帶=下顎合法範圍　🔴 紅線=頭頂基準"
            )

            preview = draw_id_guides(single, gg) if show_guide else single

            st.markdown("👆 **在下方預覽圖上點一下**，臉部中心就會移到該點"
                        "（之後可再用滑桿細調）")
            # 固定寬、全寬區塊（勿放進窄欄，否則 image-coordinates 會抖動）
            streamlit_image_coordinates(preview, width=DISP_W, key="pp_click")
            st.caption("單張定位預覽（可點擊）")

            # 相片邊對邊、零間距、零內邊距 → 內部裁切線對齊成整條直線（一刀裁）
            sheet, count, cols, rows = build_4x6_sheet(
                single, photo_size, edge_margin=0, spacing=0
            )
            # 預覽用縮小圖（顯示快很多；下載仍用全解析度）
            _disp = sheet.copy()
            _disp.thumbnail((760, 760), Image.LANCZOS)
            st.image(_disp, caption=f"4x6 排版（{count} 張）")

            st.info(f"共可排入 {count} 張（{cols} 欄 × {rows} 列，橫式，相片緊靠一刀裁）。")

            # 下載：按需產生，避免每次調滑桿都重新編碼整張大圖（大幅加快）
            mime = {"JPG": "image/jpeg", "TIFF": "image/tiff",
                    "PDF": "application/pdf"}[out_fmt]
            if st.button(f"🔽 產生下載檔（{out_fmt}，全解析度）", key="pp_gen",
                         type="primary"):
                with st.spinner("以全解析度產生檔案中…"):
                    single_full, _ = compose_id_photo(
                        rgba, metrics, photo_size, head_cm, gg["crown"],
                        corr_pct=corr_pct, vshift_mm=vshift, hshift_mm=hshift)
                    sheet_full, _, _, _ = build_4x6_sheet(
                        single_full, photo_size, edge_margin=0, spacing=0)
                    st.session_state["pp_dl"] = image_to_bytes(
                        sheet_full, out_fmt, icc=src.info.get("icc_profile"))
                    st.session_state["pp_dl_fmt"] = out_fmt
            if st.session_state.get("pp_dl") is not None \
                    and st.session_state.get("pp_dl_fmt") == out_fmt:
                st.download_button(
                    f"⬇️ 下載 4x6 排版（{out_fmt}）",
                    data=st.session_state["pp_dl"],
                    file_name=f"passport_4x6_{size_key}.{out_fmt.lower()}",
                    mime=mime,
                )
                st.caption("（以最後一次「產生下載檔」的畫面為準；再調整請重新產生）")

# ------------------------------------------------------------------
# 分支 2：LINE 貼圖
# ------------------------------------------------------------------
else:
    st.caption("規範：最大 370×320 px、透明背景、四邊 10px 留白、畫布寬高為偶數。"
               "自動等比縮放至規範內。")

    files = uploaded or []
    if not files:
        st.info("請先於上方上傳一或多張圖片（可一次選多張）。")
    else:
        method = st.radio(
            "去背方式",
            ["AI 自動去背（適合複雜背景）", "滴管取色去背（適合純色背景，可再次點選）"],
            horizontal=False,
        )

        # ==========================================================
        # 方式 A：AI 批次去背
        # ==========================================================
        if method.startswith("AI"):
            names_key = tuple(f.name for f in files)
            if st.button(f"🚀 批次 AI 去背並輸出全部（{len(files)} 張）",
                         type="primary"):
                results = []
                prog = st.progress(0.0, text="AI 去背中…")
                for i, f in enumerate(files):
                    sticker, info = process_line_sticker(
                        open_normalized(f), rembg_model)
                    results.append((f.name, sticker, info))
                    prog.progress((i + 1) / len(files),
                                  text=f"AI 去背中… {i + 1}/{len(files)}")
                prog.empty()
                st.session_state["line_batch"] = (names_key, results)

            batch = st.session_state.get("line_batch")
            if batch and batch[0] == names_key:
                results = batch[1]
                st.success(f"✅ 完成 {len(results)} 張，皆已套 LINE 規範"
                           "（透明 PNG、偶數畫布）。")
                gcols = st.columns(4)
                for i, (name, sticker, info) in enumerate(results):
                    with gcols[i % 4]:
                        st.image(sticker, caption=name, width=150)
                        base = os.path.splitext(name)[0]
                        st.download_button(
                            "⬇️ PNG", image_to_bytes(sticker, "PNG"),
                            file_name=f"{base}_sticker.png",
                            mime="image/png", key=f"dl_line_{i}",
                        )
                # 打包全部為 ZIP
                zbuf = io.BytesIO()
                with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for name, sticker, info in results:
                        base = os.path.splitext(name)[0]
                        zf.writestr(f"{base}_sticker.png",
                                    image_to_bytes(sticker, "PNG"))
                st.download_button("⬇️ 下載全部（ZIP）", zbuf.getvalue(),
                                   file_name="line_stickers.zip",
                                   mime="application/zip")

        # ==========================================================
        # 方式 B：滴管取色去背（可多次點選累加去背範圍）
        # ==========================================================
        else:
            names = [f.name for f in files]
            sel_name = st.selectbox("選擇要編輯的圖", names)
            sel_file = files[names.index(sel_name)]
            src = open_normalized(sel_file).convert("RGB")

            st.session_state.setdefault("line_colors", {})
            colors = st.session_state["line_colors"].setdefault(sel_name, [])

            tol = st.slider("容差（越大移除越多相近色）", 5, 150, 40, 1)

            DISP = 1400
            st.caption("👆 點原圖的**背景**取色去背（可多次點不同區域累加）")
            click = streamlit_image_coordinates(src, width=DISP,
                                                key="line_click")
            if click and click.get("unix_time") != \
                    st.session_state.get("line_click_t"):
                st.session_state["line_click_t"] = click["unix_time"]
                rw = click.get("width") or DISP
                rh = click.get("height") or DISP
                nx = min(max(int(click["x"] * src.width / rw), 0), src.width - 1)
                ny = min(max(int(click["y"] * src.height / rh), 0), src.height - 1)
                colors.append(tuple(int(c) for c in src.getpixel((nx, ny))))

            if colors:
                swatches = "　".join("#%02X%02X%02X" % c for c in colors)
                st.caption(f"已取色（{len(colors)}）：{swatches}")
            else:
                st.caption("尚未取色（目前顯示原圖）。")

            bc1, bc2, _ = st.columns([1, 1, 2])
            if bc1.button("↩ 移除上一個取色") and colors:
                colors.pop()
                st.rerun()
            if bc2.button("🗑 清除全部取色"):
                colors.clear()
                st.rerun()

            rgba_sheet = remove_by_colors(src, colors, tol) if colors \
                else src.convert("RGBA")

            # ------- 整版分割成個別貼圖（切割 + 底色檢視整合）-------
            st.divider()
            st.markdown("### 🔪 整版分割成個別貼圖")
            if not colors:
                st.info("請先在上方點原圖**背景**取色，才能分割。")
            else:
                st.caption("先填**欄數 × 列數**做等分，再到下方**逐條拉動每條切線**微調；"
                           "不確定可先按「自動猜」。")
                st.session_state.setdefault("line_cols", 5)
                st.session_state.setdefault("line_rows", 5)
                auto_box = sheet_bbox(src, colors, tol)

                bcol1, bcol2 = st.columns(2)
                if bcol1.button("🤖 自動猜欄列數"):
                    gc, gr = guess_grid_counts(src, colors, tol)
                    st.session_state["line_cols"] = int(gc)
                    st.session_state["line_rows"] = int(gr)
                    st.session_state.pop("line_sig", None)   # 觸發切線重算
                    st.rerun()
                reset_lines = bcol2.button("↺ 切線重設為自動等分")

                ic1, ic2 = st.columns(2)
                n_cols = int(ic1.number_input("欄數（直行）", 1, 30, key="line_cols"))
                n_rows = int(ic2.number_input("列數（橫排）", 1, 30, key="line_rows"))

                # 依（圖, 欄, 列）簽章初始化每條切線（等分）；變更或重設時重算
                sig = (sel_name, n_cols, n_rows)
                if reset_lines or st.session_state.get("line_sig") != sig:
                    x0, y0, x1, y1 = auto_box
                    st.session_state["xlines"] = [
                        int(round(x0 + (x1 - x0) * c / n_cols))
                        for c in range(n_cols + 1)]
                    st.session_state["ylines"] = [
                        int(round(y0 + (y1 - y0) * r / n_rows))
                        for r in range(n_rows + 1)]
                    st.session_state["line_sig"] = sig

                # ---- 直覺調整切線（整合底色檢視；預覽固定放大 1400px）----
                DZ = 1400
                m1, m2 = st.columns([1, 1])
                adj_mode = m1.radio(
                    "調整模式（點線附近即可移動該線）",
                    ["↕ 垂直線（左右移）", "↔ 水平線（上下移）"],
                    horizontal=True, key="adj_mode")
                bgname = m2.radio(
                    "檢視底色（切換看去背乾不乾淨）",
                    ["原圖", "白底", "淺藍底", "黑底", "洋紅底"],
                    horizontal=True, key="bg_view")
                is_v = adj_mode.startswith("↕")

                xlines = st.session_state["xlines"]
                ylines = st.session_state["ylines"]

                # 先處理大圖點擊（移動最近的線）
                clk = st.session_state.get("adj_click")
                if clk and clk.get("unix_time") != st.session_state.get("adj_click_t"):
                    st.session_state["adj_click_t"] = clk["unix_time"]
                    rw = clk.get("width") or DZ
                    rh = clk.get("height") or DZ
                    nx = clk["x"] * src.width / rw
                    ny = clk["y"] * src.height / rh
                    if is_v:
                        i = min(range(len(xlines)),
                                key=lambda k: abs(xlines[k] - nx))
                        xlines[i] = int(round(min(max(nx, 0), src.width)))
                    else:
                        i = min(range(len(ylines)),
                                key=lambda k: abs(ylines[k] - ny))
                        ylines[i] = int(round(min(max(ny, 0), src.height)))

                xlines = sorted(xlines)
                ylines = sorted(ylines)
                st.session_state["xlines"] = xlines
                st.session_state["ylines"] = ylines

                # 整合預覽：底色 + 去背結果 + 切線（可點圖移線）
                _bgmap = {"白底": (255, 255, 255), "淺藍底": (200, 225, 255),
                          "黑底": (0, 0, 0), "洋紅底": (255, 0, 255)}
                if bgname == "原圖":
                    base_view = src.convert("RGB")
                else:
                    base_view = Image.new("RGB", rgba_sheet.size, _bgmap[bgname])
                    base_view.paste(rgba_sheet, (0, 0), rgba_sheet)
                prev = draw_lines(base_view, xlines, ylines)
                st.caption(f"👆 點圖移動最近的{'垂直線' if is_v else '水平線'}；"
                           "切「檢視底色」看去背殘留（大圖固定放大 1400px）")
                streamlit_image_coordinates(prev, width=DZ, key="adj_click")

                cc1, cc2 = st.columns([1, 1])
                do_one = cc1.button("✂️ 依目前切線分割此圖", type="primary")
                do_all = False
                if len(files) > 1:
                    do_all = cc2.button(f"分割全部 {len(files)} 張"
                                        "（此圖用目前切線，其他自動等分）")

                if do_one or do_all:
                    out = []
                    with st.spinner("切割中…"):
                        if do_one:
                            sts, cells = split_by_lines(
                                src, colors, tol, xlines, ylines, pad=10)
                            out.append((os.path.splitext(sel_name)[0], src,
                                        sts, cells))
                        else:
                            for f in files:
                                im = open_normalized(f).convert("RGB")
                                if f is sel_file:
                                    sts, cells = split_by_lines(
                                        im, colors, tol, xlines, ylines, pad=10)
                                else:
                                    sts, cells = split_sticker_grid(
                                        im, colors, tol, n_cols, n_rows, pad=10)
                                out.append((os.path.splitext(f.name)[0], im,
                                            sts, cells))
                    st.session_state["line_split"] = {"tol": tol, "out": out}

                split = st.session_state.get("line_split")
                if split:
                    out = split["out"]
                    total = sum(len(o[2]) for o in out)
                    extra = f"（{len(out)} 個整版）" if len(out) > 1 else ""
                    st.success(f"✅ 共切出 {total} 張貼圖{extra}，"
                               "已依閱讀順序命名 _01、_02…")

                    # 第一個整版的分割框核對（放大）
                    base0, im0, sts0, cells0 = out[0]
                    st.image(draw_cells(im0, cells0), width=720,
                             caption=f"{base0}：切出 {len(sts0)} 格（紅框=每張範圍）")

                    # 縮圖網格（放大）
                    gcols = st.columns(6)
                    for i, (stk, info) in enumerate(sts0):
                        with gcols[i % 6]:
                            st.image(stk, caption=f"_{i + 1:02d}", width=110)

                    # 打包 ZIP（_01.._NN），並回報最大單檔
                    zbuf = io.BytesIO()
                    max_kb = 0.0
                    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
                        for base, im, sts, cells in out:
                            for i, (stk, info) in enumerate(sts):
                                png = image_to_bytes(stk, "PNG")
                                max_kb = max(max_kb, len(png) / 1024)
                                zf.writestr(f"{base}_{i + 1:02d}.png", png)
                    ok = "✅ 皆 < 1MB" if max_kb < 1024 else "⚠️ 有檔案超過 1MB"
                    st.caption(f"每張 PNG 最大約 {max_kb:.0f} KB（{ok}）。")
                    st.download_button(
                        f"⬇️ 下載全部 {total} 張（ZIP，_01～_{total:02d}）",
                        zbuf.getvalue(),
                        file_name="line_stickers_split.zip",
                        mime="application/zip",
                    )

st.divider()
st.caption("© PhotoStudio · 以 Streamlit + rembg + Pillow 打造")
