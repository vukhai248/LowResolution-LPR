"""
Streamlit Demo — ResTranOCR: Nhận dạng biển số xe độ phân giải thấp
===================================================================
Upload 1–5 frame ảnh biển số → mô hình dự đoán 7 ký tự biển số.
Nếu dưới 5 frame, ảnh sẽ được duplicate để đủ 5 frame.
"""

import sys
import os

# Đảm bảo project root nằm trong sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from models import ResTranOCR
from config import Config
from utils import decode_pred


# ─── Cấu hình trang ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LowRes LPR — Nhận dạng biển số",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS tuỳ chỉnh (dùng st.html để inject style) ───────────────────────────
st.html("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Header gradient ── */
.main-header {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 50%);
    animation: pulse 6s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 0.5; }
    50% { transform: scale(1.1); opacity: 1; }
}
.main-header h1 {
    color: #fff;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    position: relative;
    letter-spacing: -0.5px;
}
.main-header p {
    color: rgba(255, 255, 255, 0.7);
    font-size: 1rem;
    margin-top: 0.5rem;
    position: relative;
}

/* ── Upload area ── */
.upload-zone {
    background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
    border: 2px dashed rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
}
.upload-zone:hover {
    border-color: rgba(99, 102, 241, 0.8);
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.15);
}

/* ── Frame preview card ── */
.frame-card {
    background: linear-gradient(145deg, #1a1a2e, #252540);
    border-radius: 12px;
    padding: 0.75rem;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.frame-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}
.frame-label {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.5rem;
}
.frame-badge-original {
    display: inline-block;
    background: rgba(34, 197, 94, 0.2);
    color: #22c55e;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 4px;
}
.frame-badge-duplicated {
    display: inline-block;
    background: rgba(234, 179, 8, 0.2);
    color: #eab308;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 4px;
}

/* ── Result box ── */
.result-box {
    background: linear-gradient(145deg, #0f172a, #1e293b);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
}
.result-title {
    color: rgba(255, 255, 255, 0.6);
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 0.75rem;
}
.result-plate {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: 8px;
    background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
    animation: fadeInUp 0.6s ease-out;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Confidence bar ── */
.conf-container {
    display: flex;
    justify-content: center;
    gap: 12px;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.conf-char {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 0.5rem 0.75rem;
    min-width: 55px;
    text-align: center;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.conf-char-label {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
}
.conf-char-pct {
    font-size: 0.7rem;
    font-weight: 600;
    margin-top: 2px;
}
.conf-high { color: #22c55e; }
.conf-mid { color: #eab308; }
.conf-low { color: #ef4444; }

/* ── Info cards ── */
.info-card {
    background: linear-gradient(145deg, #1a1a2e, #252540);
    border-radius: 12px;
    padding: 1.25rem;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.info-card h4 {
    color: rgba(255, 255, 255, 0.9);
    font-size: 0.85rem;
    font-weight: 600;
    margin: 0 0 0.5rem 0;
}
.info-card p {
    color: rgba(255, 255, 255, 0.5);
    font-size: 0.8rem;
    margin: 0;
    line-height: 1.5;
}

/* ── Predict button ── */
div.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    font-weight: 700;
    font-size: 1.1rem;
    padding: 0.75rem 3rem;
    border: none;
    border-radius: 12px;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%);
}
div.stButton > button:active {
    transform: translateY(0px);
}

/* ── Footer ── */
.footer {
    text-align: center;
    color: rgba(255, 255, 255, 0.3);
    font-size: 0.75rem;
    margin-top: 3rem;
    padding: 1rem;
}
</style>
""")


# ─── Hàm tiện ích ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """Load mô hình ResTranOCR + weights đã train."""
    cfg = Config()
    model = ResTranOCR(
        label_len=cfg.label_len,
        num_classes=cfg.num_classes,
        embed_dim=cfg.embed_dim,
        ff_dim=cfg.ff_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        extractor_pretrained=False,
        freeze_extractor=False,
        drop_out=cfg.drop_out,
    )

    weight_path = os.path.join(PROJECT_ROOT, "weights", "ResTranOCR.pth")
    if not os.path.isfile(weight_path):
        st.error(f"⚠️ Không tìm thấy file weights tại: `{weight_path}`")
        st.stop()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Xử lý trường hợp DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg, device


def preprocess_image(pil_img: Image.Image, img_h: int = 32, img_w: int = 128) -> torch.Tensor:
    """Chuyển PIL Image → tensor đã normalize theo ImageNet."""
    transform = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(pil_img.convert("RGB"))


def build_5_frames(uploaded_images: list, img_h: int = 32, img_w: int = 128) -> tuple:
    """
    Nhận list PIL images (1–5), duplicate nếu thiếu → trả về tensor (1, 5, 3, H, W)
    và list thông tin mỗi frame (original hay duplicated).
    """
    n = len(uploaded_images)
    tensors = [preprocess_image(img, img_h, img_w) for img in uploaded_images]

    frame_info = [{"img": uploaded_images[i], "is_original": True, "source_idx": i} for i in range(n)]

    # Duplicate để đủ 5 frame
    while len(tensors) < 5:
        dup_idx = len(tensors) % n
        tensors.append(tensors[dup_idx].clone())
        frame_info.append({
            "img": uploaded_images[dup_idx],
            "is_original": False,
            "source_idx": dup_idx,
        })

    # Stack → (5, 3, H, W) → (1, 5, 3, H, W)
    batch = torch.stack(tensors, dim=0).unsqueeze(0)
    return batch, frame_info


def predict(model, batch: torch.Tensor, vocab: str, device: str):
    """Chạy inference, trả về predicted text + confidence mỗi ký tự."""
    batch = batch.to(device)
    with torch.no_grad():
        logits = model(batch)                          # (1, 7, 36)
    probs = F.softmax(logits, dim=2)                   # (1, 7, 36)
    pred_indices = probs.argmax(dim=2)                 # (1, 7)
    confidences = probs.max(dim=2).values              # (1, 7)

    pred_text = "".join(vocab[i] for i in pred_indices[0].cpu().tolist())
    char_confs = confidences[0].cpu().tolist()
    return pred_text, char_confs


# ─── Giao diện chính ─────────────────────────────────────────────────────────

# Header
st.html("""
<div class="main-header">
    <h1>Low-Resolution LPR — Nhận dạng biển số xe với độ phân giải thấp</h1>
</div>
""")

# Thông tin hướng dẫn
col_info1, col_info2, col_info3 = st.columns(3)
with col_info1:
    st.html("""
    <div class="info-card">
        <h4> Bước 1 — Upload ảnh</h4>
        <p>Tải lên 1–5 frame ảnh biển số xe. Hỗ trợ JPG, PNG, JPEG.</p>
    </div>
    """)
with col_info2:
    st.html("""
    <div class="info-card">
        <h4> Bước 2 — Tự động xử lý</h4>
        <p>Nếu dưới 5 frame, hệ thống sẽ tự duplicate ảnh để đủ 5 frame cho mô hình.</p>
    </div>
    """)
with col_info3:
    st.html("""
    <div class="info-card">
        <h4> Bước 3 — Dự đoán</h4>
        <p>Nhấn nút dự đoán để mô hình nhận dạng 7 ký tự biển số từ các frame.</p>
    </div>
    """)

st.html("<br>")

# Session state cho dynamic uploader key (dùng để reset uploader)
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Upload section: uploader + nút xoá cùng hàng
col_upload, col_clear_btn = st.columns([6, 1])
with col_upload:
    uploaded_files = st.file_uploader(
        "📷 Kéo thả hoặc chọn ảnh biển số xe (tối đa 5 frame • JPG, PNG, JPEG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key=f"plate_uploader_{st.session_state.uploader_key}",
    )
with col_clear_btn:
    st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Xoá tất cả"):
        st.session_state.uploader_key += 1
        st.rerun()

# Validate & hiển thị preview
if uploaded_files:
    if len(uploaded_files) > 5:
        st.warning("⚠️ Chỉ hỗ trợ tối đa **5 frame**. Chỉ 5 ảnh đầu tiên được sử dụng.")
        uploaded_files = uploaded_files[:5]

    pil_images = [Image.open(f) for f in uploaded_files]
    batch_tensor, frame_info = build_5_frames(pil_images)

    # Hiển thị 5 frame preview
    st.markdown("### 🖼️ Preview 5 Frame")
    cols = st.columns(5, gap="small")
    for i, (col, info) in enumerate(zip(cols, frame_info)):
        with col:
            st.image(
                info["img"],
                width="stretch",
                clamp=True,
            )
            badge_class = "frame-badge-original" if info["is_original"] else "frame-badge-duplicated"
            badge_text = "Original" if info["is_original"] else f"Duplicate #{info['source_idx']+1}"
            st.html(f"""
            <div style="text-align:center;">
                <div class="frame-label">Frame {i+1}</div>
                <div class="{badge_class}">{badge_text}</div>
            </div>
            """)

    st.html("<br>")

    # Nút dự đoán
    col_btn_left, col_btn_center, col_btn_right = st.columns([2, 1, 2])
    with col_btn_center:
        predict_clicked = st.button("🔍 Dự đoán biển số", use_container_width=True)  # button still supports this

    if predict_clicked:
        with st.spinner("⏳ Đang tải mô hình và xử lý..."):
            model, cfg, device = load_model()
            pred_text, char_confs = predict(model, batch_tensor, cfg.vocab, device)

        # Hiển thị kết quả
        avg_conf = sum(char_confs) / len(char_confs) * 100
        avg_color = '#22c55e' if avg_conf > 80 else '#eab308' if avg_conf > 50 else '#ef4444'

        st.html(f"""
        <div class="result-box">
            <div class="result-title">Kết quả nhận dạng biển số</div>
            <div class="result-plate">{pred_text}</div>
            <div style="color: rgba(255,255,255,0.5); font-size: 0.85rem; margin-top: 0.5rem;">
                Độ tin cậy trung bình: <strong style="color: {avg_color};">{avg_conf:.1f}%</strong>
            </div>
        </div>
        """)

        # Confidence per character
        conf_items = ""
        for ch, cf in zip(pred_text, char_confs):
            pct = cf * 100
            if pct >= 80:
                css_class = "conf-high"
            elif pct >= 50:
                css_class = "conf-mid"
            else:
                css_class = "conf-low"
            conf_items += f"""
            <div class="conf-char">
                <div class="conf-char-label">{ch}</div>
                <div class="conf-char-pct {css_class}">{pct:.1f}%</div>
            </div>
            """

        st.html(f'<div class="conf-container">{conf_items}</div>')

        # Thông tin bổ sung
        st.html("<br>")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.html(f"""
            <div class="info-card">
                <h4>🖥️ Device</h4>
                <p>{device.upper()}</p>
            </div>
            """)
        with c2:
            st.html(f"""
            <div class="info-card">
                <h4>📐 Input Size</h4>
                <p>{cfg.img_H} × {cfg.img_W} px • 5 frames</p>
            </div>
            """)
        with c3:
            st.html(f"""
            <div class="info-card">
                <h4>🔤 Vocabulary</h4>
                <p>{len(cfg.vocab)} classes (0-9, A-Z)</p>
            </div>
            """)

else:
    # Placeholder khi chưa upload
    st.html("""
    <div style="text-align: center; padding: 3rem 1rem; color: rgba(255,255,255,0.3);">
        <p style="font-size: 3rem; margin-bottom: 0.5rem;">📷</p>
        <p style="font-size: 1rem;">Hãy upload ảnh biển số xe để bắt đầu</p>
    </div>
    """)


# Footer
st.html("""
<div class="footer">
    <p>ResTranOCR — ResNet-50 + STN + Multi-Frame Attention Fusion + Transformer Encoder</p>
</div>
""")
