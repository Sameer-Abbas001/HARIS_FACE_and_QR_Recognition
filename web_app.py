"""
HARIS - High-Accuracy Recognition & Identity System
Web Application & Cloud Biometric Access Core
Creator & Lead Developer: Sameer Abbas
"""

import os
import sys
import time
import shutil
import platform
from datetime import datetime
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import streamlit as st

# Import HARIS Core Modules
import qr_generator
from create_classifier import train_classifer, train_all_classifiers

# Set Page Configuration
st.set_page_config(
    page_title="HARIS | Cloud Biometric & QR Access Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CLASSIFIERS_DIR = os.path.join(DATA_DIR, "classifiers")
QRCODES_DIR = os.path.join(DATA_DIR, "qrcodes")
NAMES_FILE = os.path.join(BASE_DIR, "nameslist.txt")
CASCADE_PATH = os.path.join(DATA_DIR, "haarcascade_frontalface_default.xml")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CLASSIFIERS_DIR, exist_ok=True)
os.makedirs(QRCODES_DIR, exist_ok=True)


# ==============================================================================
# --- Cyberpunk / Glassmorphic Custom Styling ---
# ==============================================================================
def inject_custom_styles():
    st.markdown("""
    <style>
    /* Dark Obsidian & Cyberpunk Theme */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #0f172a 0%, #0b1120 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }

    /* Glassmorphic Cyber Cards */
    .cyber-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: border-color 0.3s ease;
    }
    .cyber-card:hover {
        border-color: rgba(56, 189, 248, 0.6);
    }

    /* Glow Header */
    .neon-title {
        color: #38bdf8;
        font-weight: 800;
        font-size: 2.2rem;
        letter-spacing: 1px;
        text-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
        margin-bottom: 0px;
    }
    .neon-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: -6px;
        margin-bottom: 20px;
    }

    /* Metric Badges */
    .stat-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-blue {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    .badge-red {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }

    /* Access Banner */
    .access-granted-box {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8), rgba(4, 120, 87, 0.6));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 0 25px rgba(16, 185, 129, 0.4);
    }
    .access-denied-box {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8), rgba(153, 27, 27, 0.6));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.4);
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7, #0369a1);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid #38bdf8;
        border-radius: 8px;
        padding: 0.55rem 1.2rem;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35);
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0369a1, #0ea5e9);
        border-color: #7dd3fc;
        box-shadow: 0 6px 20px rgba(56, 189, 248, 0.5);
        transform: translateY(-1px);
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #080d1a !important;
        border-right: 1px solid rgba(56, 189, 248, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)


# ==============================================================================
# --- Helper Functions & State Management ---
# ==============================================================================
def get_registered_names():
    """Reads nameslist.txt and returns sorted list of unique usernames."""
    if not os.path.isfile(NAMES_FILE):
        return []
    try:
        with open(NAMES_FILE, "r") as f:
            content = f.read().strip()
            if not content:
                return []
            names = [n.strip() for n in content.split() if n.strip()]
            return sorted(list(set(names)))
    except Exception:
        return []


def save_registered_names(names_list):
    """Saves list of unique names back to nameslist.txt."""
    clean_names = sorted(list(set([n.strip() for n in names_list if n.strip()])))
    with open(NAMES_FILE, "w") as f:
        f.write(" ".join(clean_names))


def add_registered_name(name):
    """Adds a new name to nameslist.txt if not present."""
    current = get_registered_names()
    if name not in current:
        current.append(name)
        save_registered_names(current)


def remove_registered_name(name):
    """Removes a user and their files from the system."""
    current = get_registered_names()
    if name in current:
        current.remove(name)
        save_registered_names(current)

    # Clean user dataset
    user_data_path = os.path.join(DATA_DIR, name)
    if os.path.isdir(user_data_path):
        shutil.rmtree(user_data_path, ignore_errors=True)

    # Clean classifier
    clf_path = os.path.join(CLASSIFIERS_DIR, f"{name}_classifier.xml")
    if os.path.isfile(clf_path):
        try:
            os.remove(clf_path)
        except Exception:
            pass

    # Clean QR and Badges
    for f in [f"{name}_qr.png", f"{name}_badge.png"]:
        p = os.path.join(QRCODES_DIR, f)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except Exception:
                pass


def get_user_stats(name):
    """Returns dataset count, model status, and badge status for a user."""
    user_dir = os.path.join(DATA_DIR, name)
    frame_count = 0
    if os.path.isdir(user_dir):
        valid_exts = ('.jpg', '.jpeg', '.png')
        frame_count = len([f for f in os.listdir(user_dir) if f.lower().endswith(valid_exts) and "qr" not in f.lower() and "badge" not in f.lower()])

    clf_path = os.path.join(CLASSIFIERS_DIR, f"{name}_classifier.xml")
    model_exists = os.path.isfile(clf_path)
    model_size_mb = (os.path.getsize(clf_path) / (1024 * 1024)) if model_exists else 0

    badge_path = os.path.join(QRCODES_DIR, f"{name}_badge.png")
    badge_exists = os.path.isfile(badge_path)

    qr_path = os.path.join(QRCODES_DIR, f"{name}_qr.png")
    qr_exists = os.path.isfile(qr_path)

    return {
        "frames": frame_count,
        "model_exists": model_exists,
        "model_size_mb": model_size_mb,
        "badge_exists": badge_exists,
        "qr_exists": qr_exists,
    }


def synthesize_dataset_from_image(name, source_image, target_count=50):
    """
    Given a single source face photo, detects the face, normalizes it,
    and synthesizes 50 augmented training frames (slight rotations, brightness,
    flips, and contrast shifts) to build a robust LBPH dataset instantly.
    """
    user_dir = os.path.join(DATA_DIR, name)
    os.makedirs(user_dir, exist_ok=True)

    # Convert PIL to BGR OpenCV image
    img_bgr = cv2.cvtColor(np.array(source_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    detector = cv2.CascadeClassifier(CASCADE_PATH)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

    if len(faces) == 0:
        # Fallback: Use center crop if Haar cascade missed
        h, w = gray.shape
        min_dim = min(h, w)
        face_crop = img_bgr[(h - min_dim)//2:(h + min_dim)//2, (w - min_dim)//2:(w + min_dim)//2]
    else:
        # Pick largest detected face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        # Slight margin padding
        pad = int(w * 0.1)
        y1 = max(0, y - pad)
        y2 = min(img_bgr.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img_bgr.shape[1], x + w + pad)
        face_crop = img_bgr[y1:y2, x1:x2]

    # Save initial primary face photo
    primary_path = os.path.join(user_dir, f"0{name}.jpg")
    cv2.imwrite(primary_path, face_crop)

    # Synthesize remaining frames
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    w_crop, h_crop = face_pil.size

    for i in range(1, target_count):
        aug_img = face_pil.copy()

        # Slight rotation (-8 to +8 degrees)
        angle = np.random.uniform(-8, 8)
        aug_img = aug_img.rotate(angle, resample=Image.BICUBIC)

        # Brightness variation (0.85 to 1.15)
        bright_factor = np.random.uniform(0.85, 1.15)
        aug_img = ImageEnhance.Brightness(aug_img).enhance(bright_factor)

        # Contrast variation (0.9 to 1.1)
        contrast_factor = np.random.uniform(0.9, 1.1)
        aug_img = ImageEnhance.Contrast(aug_img).enhance(contrast_factor)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            aug_img = aug_img.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert back to OpenCV and save
        aug_bgr = cv2.cvtColor(np.array(aug_img), cv2.COLOR_RGB2BGR)
        save_path = os.path.join(user_dir, f"{i}{name}.jpg")
        cv2.imwrite(save_path, aug_bgr)

    return target_count


def run_biometric_verification(image, target_user=None, confidence_threshold=45):
    """
    Executes simultaneous LBPH Face Recognition and QR Code Decoding on a given frame.
    Renders Cyberpunk HUD overlay and returns structured diagnostic payload.
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    frame_h, frame_w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    qr_detector = cv2.QRCodeDetector()

    # Load active user models
    registered_names = get_registered_names()
    all_models = {}

    if target_user and target_user != "Auto-Detect":
        clf_p = os.path.join(CLASSIFIERS_DIR, f"{target_user}_classifier.xml")
        if os.path.isfile(clf_p):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(clf_p)
            all_models[target_user] = recognizer
    else:
        # Load all available classifiers for multi-user recognition
        for u in registered_names:
            clf_p = os.path.join(CLASSIFIERS_DIR, f"{u}_classifier.xml")
            if os.path.isfile(clf_p):
                try:
                    rec = cv2.face.LBPHFaceRecognizer_create()
                    rec.read(clf_p)
                    all_models[u] = rec
                except Exception:
                    pass

    # 1. QR Code Detection & Decoding
    qr_data = None
    qr_user = None
    qr_matched = False
    qr_color = (148, 163, 184)

    try:
        qr_val, qr_points, _ = qr_detector.detectAndDecode(img_bgr)
        if qr_val and qr_points is not None:
            qr_data = qr_val.strip()
            qr_user = qr_generator.parse_qr_data(qr_val)

            # Draw QR Bounding Polygon
            pts = qr_points[0].astype(int)
            for j in range(len(pts)):
                cv2.line(img_bgr, tuple(pts[j]), tuple(pts[(j + 1) % len(pts)]), (0, 242, 254), 3)

            qr_top = max(0, min(p[1] for p in pts) - 26)
            qr_left = min(p[0] for p in pts)
            cv2.rectangle(img_bgr, (qr_left, qr_top), (qr_left + 260, qr_top + 24), (15, 23, 42), -1)
            cv2.putText(img_bgr, f"QR: {qr_user or qr_data}", (qr_left + 6, qr_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 2)
    except Exception:
        pass

    # 2. Face Detection & Recognition
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(70, 70))
    face_matched = False
    best_user = None
    best_conf = 0
    box_coords = None

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        curr_best_user = None
        curr_best_conf = 0

        # Evaluate against available classifiers
        for uname, recognizer in all_models.items():
            try:
                _id, raw_conf = recognizer.predict(roi_gray)
                conf = max(0, min(100, 100 - int(raw_conf)))
                if conf > curr_best_conf:
                    curr_best_conf = conf
                    curr_best_user = uname
            except Exception:
                continue

        if curr_best_conf >= confidence_threshold:
            face_matched = True
            best_user = curr_best_user
            best_conf = curr_best_conf
            box_coords = (x, y, w, h)
            box_color = (0, 255, 128)  # Emerald Green
            label_text = f"VERIFIED: {curr_best_user.upper()} ({curr_best_conf}%)"
        else:
            box_color = (0, 165, 255)  # Cyber Orange
            label_text = f"UNKNOWN ({curr_best_conf}%)" if curr_best_user else "FACE DETECTED"

        # Cyber Corner Brackets
        d = 18
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), box_color, 1)
        # Top-Left
        cv2.line(img_bgr, (x, y), (x + d, y), box_color, 3)
        cv2.line(img_bgr, (x, y), (x, y + d), box_color, 3)
        # Top-Right
        cv2.line(img_bgr, (x + w, y), (x + w - d, y), box_color, 3)
        cv2.line(img_bgr, (x + w, y), (x + w, y + d), box_color, 3)
        # Bottom-Left
        cv2.line(img_bgr, (x, y + h), (x + d, y + h), box_color, 3)
        cv2.line(img_bgr, (x, y + h), (x, y + h - d), box_color, 3)
        # Bottom-Right
        cv2.line(img_bgr, (x + w, y + h), (x + w - d, y + h), box_color, 3)
        cv2.line(img_bgr, (x + w, y + h), (x + w, y + h - d), box_color, 3)

        # Label Banner
        cv2.rectangle(img_bgr, (x, y - 24), (x + w, y), (15, 23, 42), -1)
        cv2.putText(img_bgr, label_text, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 2)

    # Cross-match Face and QR credentials
    verified_user = None
    if qr_user and best_user and qr_user.lower() == best_user.lower():
        qr_matched = True
        verified_user = best_user
    elif face_matched and best_user:
        verified_user = best_user
    elif qr_user:
        verified_user = qr_user
        qr_matched = True

    # 3. Top Cyberpunk HUD Bar
    cv2.rectangle(img_bgr, (10, 10), (frame_w - 10, 68), (15, 23, 42), -1)
    cv2.rectangle(img_bgr, (10, 10), (frame_w - 10, 68), (56, 189, 248), 1)

    cv2.putText(img_bgr, "HARIS ACCESS CONTROL HUD", (24, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (56, 189, 248), 2)
    
    face_status_str = f"FACE: {best_user.upper() if face_matched else ('SCANNING' if faces else 'NO FACE')}"
    face_color = (0, 255, 128) if face_matched else (148, 163, 184)
    cv2.putText(img_bgr, face_status_str, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, face_color, 1)

    qr_status_str = f"QR: {qr_user.upper() if qr_user else 'NONE'}"
    qr_col = (0, 255, 0) if qr_matched else (148, 163, 184)
    cv2.putText(img_bgr, qr_status_str, (240, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.45, qr_col, 1)

    # Verification Decision Banner
    access_granted = (face_matched and (qr_matched or not qr_data)) or (qr_matched and qr_user in registered_names)

    if access_granted and verified_user:
        cv2.rectangle(img_bgr, (frame_w // 2 - 170, frame_h - 60), (frame_w // 2 + 170, frame_h - 15), (6, 78, 59), -1)
        cv2.rectangle(img_bgr, (frame_w // 2 - 170, frame_h - 60), (frame_w // 2 + 170, frame_h - 15), (16, 185, 129), 2)
        cv2.putText(img_bgr, "ACCESS GRANTED", (frame_w // 2 - 120, frame_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
        cv2.putText(img_bgr, f"IDENTITY: {verified_user.upper()}", (frame_w // 2 - 95, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (52, 211, 153), 1)

    # Convert BGR back to RGB for Streamlit rendering
    hud_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return {
        "hud_image": Image.fromarray(hud_rgb),
        "access_granted": access_granted,
        "verified_user": verified_user,
        "face_matched": face_matched,
        "face_confidence": best_conf,
        "qr_matched": qr_matched,
        "qr_data": qr_data,
        "qr_user": qr_user,
        "faces_detected": len(faces)
    }


# ==============================================================================
# --- Main Application Execution Flow ---
# ==============================================================================
def main():
    inject_custom_styles()
    registered_names = get_registered_names()

    # --- Sidebar Navigation ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 10px 0;">
            <h2 style="color: #38bdf8; margin: 0; font-size: 1.8rem;">⚡ HARIS</h2>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 0;">Biometric Access Intelligence Core</p>
            <span class="stat-badge badge-green" style="margin-top: 8px;">● AWS EC2 ONLINE</span>
        </div>
        <hr style="border-color: rgba(56, 189, 248, 0.2); margin: 15px 0;">
        """, unsafe_allow_html=True)

        menu = st.radio(
            "SYSTEM NAVIGATION",
            [
                "🎯 Live Access Core (Verify)",
                "📸 User Enrollment (Capture)",
                "🧠 Classifier Training Hub",
                "🪪 Digital ID & QR Studio",
                "👥 User Directory & DB",
                "☁️ AWS EC2 Telemetry & Docs"
            ],
            index=0
        )

        st.markdown("<hr style='border-color: rgba(56, 189, 248, 0.2); margin: 20px 0;'>", unsafe_allow_html=True)
        
        # Telemetry Stats in Sidebar
        trained_count = sum(1 for u in registered_names if os.path.isfile(os.path.join(CLASSIFIERS_DIR, f"{u}_classifier.xml")))
        st.markdown(f"""
        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.6;">
            <div>👥 <b>Enrolled Users:</b> {len(registered_names)}</div>
            <div>🧠 <b>Active Models:</b> {trained_count} / {len(registered_names)}</div>
            <div>🛡️ <b>Engine:</b> LBPH + Haar + QR</div>
            <div>🌐 <b>Server:</b> Headless EC2 (Port 8501)</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top: 35px; text-align: center; color: #64748b; font-size: 0.75rem;">
            Designed & Engineered by<br><b style="color: #94a3b8;">Sameer Abbas</b>
        </div>
        """, unsafe_allow_html=True)

    # Top Brand Ribbon
    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        st.markdown("""
        <div style="margin-bottom: 25px;">
            <h1 class="neon-title">⚡ HARIS ACCESS CONTROL CORE</h1>
            <p class="neon-sub">Dual Biometric Facial Recognition & Photo-Embedded QR Authentication Suite</p>
        </div>
        """, unsafe_allow_html=True)
    with col_t2:
        st.markdown(f"""
        <div style="text-align: right; margin-top: 10px;">
            <span class="stat-badge badge-blue">Cloud Ready</span>
            <span class="stat-badge badge-green">v2.5 Production</span>
        </div>
        """, unsafe_allow_html=True)


    # ==========================================================================
    # 1. LIVE ACCESS CORE (VERIFICATION)
    # ==========================================================================
    if menu == "🎯 Live Access Core (Verify)":
        st.markdown("### 🎯 Real-Time Biometric & QR Verification")
        st.write("Scan identity using your browser webcam or upload a surveillance snapshot. HARIS verifies facial landmarks and embedded QR tokens simultaneously.")

        col_left, col_right = st.columns([1.6, 1])

        with col_left:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            input_mode = st.radio("Select Verification Input Mode:", ["📷 Web Camera (Live)", "📁 Upload Image / Photo"], horizontal=True)

            img_input = None
            if input_mode == "📷 Web Camera (Live)":
                cam_pic = st.camera_input("Capture Face / QR Pass for Live Verification")
                if cam_pic is not None:
                    img_input = Image.open(cam_pic)
            else:
                uploaded_file = st.file_uploader("Upload Snapshot (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
                if uploaded_file is not None:
                    img_input = Image.open(uploaded_file)

            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("#### ⚙️ Verification Parameters")
            target_options = ["Auto-Detect"] + registered_names
            selected_target = st.selectbox("Target Profile to Validate Against:", target_options, index=0)
            threshold = st.slider("LBPH Confidence Cutoff Threshold (%):", min_value=20, max_value=85, value=45, step=5)
            st.caption("Auto-Detect evaluates against all enrolled biometric models simultaneously.")
            st.markdown("</div>", unsafe_allow_html=True)

        # Execution of Verification
        if img_input is not None:
            with st.spinner("⚡ Running Neural LBPH Face Classification & High-ECC QR Decoding..."):
                result = run_biometric_verification(
                    img_input,
                    target_user=selected_target if selected_target != "Auto-Detect" else None,
                    confidence_threshold=threshold
                )

            res_col1, res_col2 = st.columns([1.5, 1])

            with res_col1:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                st.markdown("#### 🖥️ Real-Time Cyberpunk HUD Output")
                st.image(result["hud_image"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                st.markdown("#### 📋 Verification Telemetry")

                if result["access_granted"]:
                    st.markdown(f"""
                    <div class="access-granted-box">
                        <h2 style="margin: 0; font-size: 1.6rem; color: #34d399;">✅ ACCESS GRANTED</h2>
                        <p style="margin: 5px 0 0 0; font-size: 1.1rem; font-weight: 600;">IDENTITY: {result['verified_user'].upper()}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="access-denied-box">
                        <h2 style="margin: 0; font-size: 1.6rem; color: #f87171;">❌ ACCESS DENIED</h2>
                        <p style="margin: 5px 0 0 0; font-size: 0.95rem;">Confidence below threshold or unregistered token.</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
                st.write(f"**👤 Matched Identity:** `{result['verified_user'] or 'Unknown'}`")
                st.write(f"**🧠 Face Confidence:** `{result['face_confidence']}%` (Cutoff: `{threshold}%`)")
                st.write(f"**🪪 QR Token Status:** `{'VERIFIED' if result['qr_matched'] else ('DETECTED: ' + str(result['qr_user']) if result['qr_user'] else 'NOT DETECTED')}`")
                st.write(f"**🕒 Timestamp:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
                st.markdown("</div>", unsafe_allow_html=True)

                # Show Registered Reference Photo and Badge if matched
                if result["verified_user"]:
                    ref_photo = qr_generator.get_user_photo_path(result["verified_user"])
                    badge_p = os.path.join(QRCODES_DIR, f"{result['verified_user']}_badge.png")

                    t1, t2 = st.tabs(["Reference Photo", "Access Pass Badge"])
                    with t1:
                        if ref_photo and os.path.isfile(ref_photo):
                            st.image(ref_photo, caption=f"Enrolled Portrait: {result['verified_user']}", width=160)
                        else:
                            st.info("No reference photo on disk.")
                    with t2:
                        if os.path.isfile(badge_p):
                            st.image(badge_p, caption=f"Digital ID Badge: {result['verified_user']}", use_container_width=True)
                        else:
                            st.info("Badge not yet generated.")

                st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================================
    # 2. USER ENROLLMENT (DATASET CAPTURE & AUTO SETUP)
    # ==========================================================================
    elif menu == "📸 User Enrollment (Capture)":
        st.markdown("### 📸 Biometric User Enrollment")
        st.write("Register a new user identity into HARIS. The engine will capture the portrait, automatically extract face landmarks, synthesize a 50-frame dataset, compile the LBPH neural model, and mint a photo-embedded QR ID pass.")

        c1, c2 = st.columns([1.2, 1])

        with c1:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("#### 1. Identity Details")
            new_name = st.text_input("Username / Personnel ID (alphanumeric, e.g. JohnDoe):", value="").strip()

            st.markdown("#### 2. Face Portrait Capture")
            enr_mode = st.radio("Capture Method:", ["📷 Web Camera Snapshot", "📁 Upload Face Image File"], horizontal=True)

            source_img = None
            if enr_mode == "📷 Web Camera Snapshot":
                snapped = st.camera_input("Take Portrait Snapshot")
                if snapped:
                    source_img = Image.open(snapped)
            else:
                up_file = st.file_uploader("Upload Clear Portrait Image", type=["jpg", "jpeg", "png"])
                if up_file:
                    source_img = Image.open(up_file)

            enroll_btn = st.button("🚀 Enroll & Generate Biometric Profile", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
            st.markdown("#### ℹ️ Automated Pipeline Details")
            st.markdown("""
            - **⚡ 50-Frame Dataset Synthesis**: Automatically detects the face and applies rotational, illumination, and contrast variations to build a 50-frame training dataset.
            - **🧠 Neural LBPH Model**: Compiles an optimized XML classifier specifically tuned for the user.
            - **🪪 Photo-Embedded QR**: Embeds the user portrait inside a High-Error-Correction QR code (Level H, 30% recovery).
            - **💳 Cybernetic ID Badge**: Renders a complete 540x780 holographic access badge ready for printing or digital display.
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        if enroll_btn:
            if not new_name:
                st.error("Please enter a valid Username / Personnel ID.")
            elif source_img is None:
                st.error("Please provide a portrait photograph via Web Camera or File Upload.")
            else:
                clean_name = "".join(c for c in new_name if c.isalnum() or c in ("_", "-"))
                if not clean_name:
                    st.error("Username must contain alphanumeric characters.")
                else:
                    prog_bar = st.progress(0, text="Initializing Enrollment...")

                    # 1. Synthesize 50 Face Dataset Frames
                    prog_bar.progress(25, text=f"Detecting face and synthesizing 50 training frames for {clean_name}...")
                    frames_saved = synthesize_dataset_from_image(clean_name, source_img, target_count=50)

                    # 2. Add to registered names list
                    prog_bar.progress(50, text=f"Updating registered user database...")
                    add_registered_name(clean_name)

                    # 3. Generate QR and Badge
                    prog_bar.progress(70, text=f"Minting Photo-Embedded QR Code and Cybernetic Badge...")
                    qr_p = qr_generator.generate_photo_qr(clean_name)
                    badge_p = qr_generator.generate_id_badge(clean_name)

                    # 4. Train LBPH Classifier
                    prog_bar.progress(90, text=f"Compiling LBPH Facial Biometric Classifier XML...")
                    try:
                        train_classifer(clean_name)
                    except Exception as e:
                        st.warning(f"Notice on classifier training: {e}")

                    prog_bar.progress(100, text="Enrollment Complete!")
                    st.success(f"🎉 Successfully enrolled user '{clean_name}'! 50 frames captured, QR badge minted, and model trained.")

                    # Show freshly minted badges
                    b1, b2 = st.columns(2)
                    with b1:
                        if os.path.isfile(qr_p):
                            st.image(qr_p, caption="Generated Photo-Embedded QR Code", width=240)
                            with open(qr_p, "rb") as f:
                                st.download_button("📥 Download QR Code", f, file_name=f"{clean_name}_qr.png", mime="image/png")
                    with b2:
                        if os.path.isfile(badge_p):
                            st.image(badge_p, caption="Generated Access Badge Card", width=280)
                            with open(badge_p, "rb") as f:
                                st.download_button("📥 Download Access Badge", f, file_name=f"{clean_name}_badge.png", mime="image/png")


    # ==========================================================================
    # 3. CLASSIFIER TRAINING HUB
    # ==========================================================================
    elif menu == "🧠 Classifier Training Hub":
        st.markdown("### 🧠 LBPH Classifier Training Center")
        st.write("Train and compile Local Binary Pattern Histograms (LBPH) facial recognition models from captured user datasets.")

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            train_all_btn = st.button("🚀 Train All Enrolled Users (Batch)", use_container_width=True)
        with col_btn2:
            target_single = st.selectbox("Or select specific user to train:", registered_names if registered_names else ["No users enrolled"])
            train_single_btn = st.button("🎯 Train Selected User", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if train_all_btn:
            if not registered_names:
                st.warning("No users registered to train.")
            else:
                p_bar = st.progress(0, text="Training models...")
                successes = 0
                for idx, uname in enumerate(registered_names):
                    p_bar.progress(int((idx / len(registered_names)) * 100), text=f"Training classifier for '{uname}'...")
                    try:
                        train_classifer(uname)
                        successes += 1
                    except Exception as e:
                        st.error(f"Error training {uname}: {e}")
                p_bar.progress(100, text="Batch training finished!")
                st.success(f"Successfully trained {successes} of {len(registered_names)} models!")

        if train_single_btn and target_single and target_single != "No users enrolled":
            with st.spinner(f"Compiling model for {target_single}..."):
                try:
                    train_classifer(target_single)
                    st.success(f"Model for '{target_single}' successfully compiled and saved to data/classifiers/{target_single}_classifier.xml!")
                except Exception as e:
                    st.error(f"Failed to train model for {target_single}: {e}")

        # Status Table of all models
        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Enrolled Users & Classifier Model Registry")

        if not registered_names:
            st.info("No users registered yet. Go to 'User Enrollment' to add profiles.")
        else:
            table_data = []
            for u in registered_names:
                stats = get_user_stats(u)
                table_data.append({
                    "Username": u,
                    "Dataset Frames": stats["frames"],
                    "Classifier Model": "✅ Active & Ready" if stats["model_exists"] else "❌ Missing / Not Trained",
                    "Model File Size": f"{stats['model_size_mb']:.2f} MB" if stats["model_exists"] else "N/A",
                    "QR & Badge": "✅ Minted" if stats["badge_exists"] else "❌ Not Minted"
                })
            st.dataframe(table_data, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================================
    # 4. DIGITAL ID & QR STUDIO
    # ==========================================================================
    elif menu == "🪪 Digital ID & QR Studio":
        st.markdown("### 🪪 Digital ID Badge & QR Code Studio")
        st.write("Inspect, generate, and export high-resolution biometric access passes and photo-embedded QR codes.")

        if not registered_names:
            st.warning("No users registered in system yet.")
        else:
            sel_user = st.selectbox("Select Enrolled User:", registered_names)

            col_qr, col_badge = st.columns([1, 1.2])

            qr_path = os.path.join(QRCODES_DIR, f"{sel_user}_qr.png")
            badge_path = os.path.join(QRCODES_DIR, f"{sel_user}_badge.png")

            with col_qr:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                st.markdown("#### 🔲 Photo-Embedded QR Code")
                if os.path.isfile(qr_path):
                    st.image(qr_path, caption=f"Payload: HARIS:USER:{sel_user}", width=260)
                    with open(qr_path, "rb") as f:
                        st.download_button("📥 Download QR Code (PNG)", f, file_name=f"{sel_user}_qr.png", mime="image/png", use_container_width=True)
                else:
                    st.warning("QR code not found for this user.")
                    if st.button("Generate QR Code"):
                        qr_generator.generate_photo_qr(sel_user)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            with col_badge:
                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                st.markdown("#### 🪪 Holographic ID Access Badge")
                if os.path.isfile(badge_path):
                    st.image(badge_path, caption=f"HARIS Biometric Pass - {sel_user}", width=340)
                    with open(badge_path, "rb") as f:
                        st.download_button("📥 Download Access Badge (PNG)", f, file_name=f"{sel_user}_badge.png", mime="image/png", use_container_width=True)
                else:
                    st.warning("ID Badge not found for this user.")
                    if st.button("Generate ID Badge"):
                        qr_generator.generate_id_badge(sel_user)
                        st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================================
    # 5. USER DIRECTORY & DB MANAGEMENT
    # ==========================================================================
    elif menu == "👥 User Directory & DB":
        st.markdown("### 👥 User Database & Profiles Management")
        st.write("Browse, inspect, and manage biometric identities enrolled within HARIS.")

        if not registered_names:
            st.info("No enrolled users found.")
        else:
            for uname in registered_names:
                stats = get_user_stats(uname)
                photo_p = qr_generator.get_user_photo_path(uname)

                st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
                c_img, c_info, c_action = st.columns([1, 2.5, 1])

                with c_img:
                    if photo_p and os.path.isfile(photo_p):
                        st.image(photo_p, width=120)
                    else:
                        st.write("👤 *No Photo*")

                with c_info:
                    st.markdown(f"### {uname.upper()}")
                    st.write(f"📁 **Dataset Frames:** `{stats['frames']}`")
                    st.write(f"🧠 **Classifier:** `{'Active (' + str(round(stats['model_size_mb'], 2)) + ' MB)' if stats['model_exists'] else 'Missing'}`")
                    st.write(f"🪪 **Digital Badge:** `{'Available' if stats['badge_exists'] else 'Not generated'}`")

                with c_action:
                    st.write("")
                    st.write("")
                    if st.button(f"🗑️ Delete {uname}", key=f"del_{uname}", use_container_width=True):
                        remove_registered_name(uname)
                        st.toast(f"User {uname} successfully deleted.")
                        st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)


    # ==========================================================================
    # 6. AWS EC2 TELEMETRY & CLOUD DOCS
    # ==========================================================================
    elif menu == "☁️ AWS EC2 Telemetry & Docs":
        st.markdown("### ☁️ AWS EC2 Cloud Infrastructure & System Telemetry")
        st.write("Live system status, cloud environment diagnostics, and deployment orchestration guidelines.")

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("#### 🖥️ Server & Runtime Diagnostics")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.metric("Operating System", platform.system(), f"{platform.release()} ({platform.machine()})")
        with col_s2:
            st.metric("Python Runtime", platform.python_version(), f"{platform.python_implementation()}")
        with col_s3:
            st.metric("OpenCV Version", cv2.__version__, "Contrib Headless")
        with col_s4:
            st.metric("Streamlit Version", st.__version__, "Port 8501")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='cyber-card'>", unsafe_allow_html=True)
        st.markdown("#### 🚀 Direct AWS EC2 Deployment Architecture")
        st.markdown("""
        **1. Security Group Configuration (Inbound Rules)**:
        - Port `22` (SSH): For administrative shell access.
        - Port `8501` (Custom TCP): HARIS Web Dashboard (`http://<EC2-PUBLIC-IP>:8501`).
        - Port `80` (HTTP) / `443` (HTTPS): For optional reverse-proxy routing via Nginx.

        **2. Launching via Systemd Service**:
        HARIS includes an automated service unit `haris.service` so that on EC2 instance boot or unexpected crash, the system automatically starts and remains online 24/7.
        ```bash
        sudo systemctl status haris
        sudo systemctl restart haris
        ```

        **3. Launching via Docker**:
        ```bash
        docker compose up -d --build
        ```
        """)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

