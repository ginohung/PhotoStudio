# 📸 PhotoStudio — AI 去背 × 尺寸排版

以 Streamlit 打造的影像處理 Web App，整合 AI 去背與尺寸排版：

- **護照大頭照**：AI 去背 → 白底 → 依台灣護照規範（頭頂→下顎 3.2~3.6cm）自動定位 → 4x6 吋 300 DPI 排版，可下載 JPG / TIFF / PDF。
- **LINE 貼圖**：AI 去背或**滴管取色去背**（適合純色背景）→ 套 LINE 規範（≤370×320、透明、偶數畫布）；支援**整版貼圖自動分割**成多張（依欄列數 + 可調切線），輸出 `_01`、`_02`… ZIP。

## 本機執行

```bash
pip install -r requirements.txt
streamlit run app.py
```

需 **Python 3.11 或 3.12**（`rembg` / `onnxruntime` 對 3.14 尚無 wheel）。
第一次去背會自動下載 u2net 模型（約 170MB）。

## 技術棧

Streamlit · rembg (u2net) · Pillow · OpenCV (YuNet 人臉偵測) · NumPy

## 線上部署（Streamlit Community Cloud）

1. 將本資料夾推上 GitHub。
2. 到 https://share.streamlit.io → 連結 GitHub repo → 主檔案填 `app.py`。
3. Python 版本選 3.11 / 3.12。
