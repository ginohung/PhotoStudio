---
title: PhotoStudio 去背排版
emoji: 📸
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
license: mit
---

# 📸 PhotoStudio — AI 去背 × 尺寸排版

以 Streamlit 打造的影像處理 Web App，整合 AI 去背與尺寸排版：

- **護照大頭照**：AI 去背 → 白底 → 依台灣護照規範（頭頂→下顎 3.2~3.6cm）自動定位 → 4x6 吋 300 DPI 排版，可下載 JPG / TIFF / PDF。
- **LINE 貼圖**：AI 去背或**滴管取色去背**（適合純色背景）→ 套 LINE 規範（≤370×320、透明、偶數畫布）；支援**整版貼圖自動分割**成多張，輸出 `_01`、`_02`… ZIP。

## 去背模型

側欄可切換去背模型：`birefnet-portrait`（髮絲級最佳）、`isnet-general-use`（高品質快速）、`u2net_human_seg`、`u2netp`（最輕量）。
預設模型可用環境變數 `REMBG_DEFAULT` 設定；記憶體大的環境（如 HF Spaces）建議設為 `birefnet-portrait`。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

需 **Python 3.11 / 3.12**（rembg / onnxruntime 對 3.14 尚無 wheel）。

## 技術棧

Streamlit · rembg · Pillow · OpenCV (YuNet 人臉偵測) · NumPy
