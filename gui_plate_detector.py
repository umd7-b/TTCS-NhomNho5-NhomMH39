import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import re
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# PHẦN LOGIC CỐT LÕI AI
# ==============================================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (64, 64)
_hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

DEFAULT_CONFIDENCE_THRESHOLD = 0.95
MAX_DETECTIONS = 5

# 30 ký tự biển số VN (bỏ I, J, O, Q, R, W)
CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','K','L',
    'M','N','P','S','T','U','V','X','Y','Z'
]
NUM_CLASSES = len(CLASS_NAMES)  # 30

def extract_hog_features(img_bgr):
    """Trích HOG cho plate detector — dùng CLAHE + bilateral filter."""
    if img_bgr is None or img_bgr.size == 0: return None
    if img_bgr.shape[0] < 5 or img_bgr.shape[1] < 5: return None
    resized = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return _hog.compute(gray).flatten()

def extract_hog_char(img_bgr):
    """Trích HOG cho OCR ký tự — giữ tỉ lệ + padding + CLAHE."""
    if img_bgr is None or img_bgr.size == 0: return None
    h, w = img_bgr.shape[:2]
    if h < 5 or w < 5: return None

    scale = min(64 / w, 64 / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    canvas = np.zeros((64, 64, 3), dtype=np.uint8)
    rsz = cv2.resize(img_bgr, (nw, nh))
    xo = (64 - nw) // 2
    yo = (64 - nh) // 2
    canvas[yo:yo+nh, xo:xo+nw] = rsz

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return _hog.compute(gray).flatten()


# ==============================================================================
# KIẾN TRÚC MODEL (ĐỒNG BỘ VỚI TRAINING)
# ==============================================================================

class PlateDetectorANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
    def forward(self, x): return self.network(x)


class OCRNet(nn.Module):
    """Model nhận diện ký tự — đồng bộ với train_ocr.py (SiLU + narrower to match weights)"""
    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),       nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.35),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x): return self.net(x)


# ==============================================================================
# LOAD MODEL
# ==============================================================================

def load_model_and_scaler(model_path="final_plate_model.pth", scaler_path="scaler.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy file model '{model_path}' hoặc scaler '{scaler_path}'.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    input_dim = _hog.getDescriptorSize()
    model = PlateDetectorANN(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, scaler

def load_ocr_model(model_path="ocr_model_best.pth", scaler_path="ocr_scaler.pkl"):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"⚠️ Không tìm thấy OCR model ({model_path}) hoặc scaler ({scaler_path}). OCR sẽ bị tắt.")
        return None, None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = OCRNet(_hog.getDescriptorSize()).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, scaler


# ==============================================================================
# PHÁT HIỆN BIỂN SỐ
# ==============================================================================

def predict_plate_with_confidence(model, scaler, img_crop):
    """Dự đoán biển số VÀ trả về confidence."""
    features = extract_hog_features(img_crop)
    if features is None: return -1, 0.0
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item()


def is_plate_candidate(img_crop):
    """Bộ lọc heuristic 7 tầng - CẢI TIẾN."""
    if img_crop is None or img_crop.size == 0:
        return False

    h, w = img_crop.shape[:2]
    if h < 12 or w < 30:
        return False

    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)

    # TẦNG 1: ASPECT RATIO — nới rộng hơn cho biển 2 hàng
    aspect = w / float(h)
    if aspect < 0.8 or aspect > 5.0:
        return False

    # TẦNG 2: MÀU NỀN SÁNG
    bright_ratio = float(np.sum(hsv[:, :, 2] > 130)) / (h * w)
    if bright_ratio < 0.25:
        return False

    # TẦNG 3: NỀN TRẮNG hoặc VÀNG hoặc XANH/ĐỎ (biển đặc biệt)
    white_mask  = cv2.inRange(hsv, (0,   0, 150), (180, 70, 255))
    yellow_mask = cv2.inRange(hsv, (15, 60, 120), (40, 255, 255))
    blue_mask   = cv2.inRange(hsv, (90, 50, 100), (130, 255, 255))
    red_mask1   = cv2.inRange(hsv, (0,  70, 100), (10, 255, 255))
    red_mask2   = cv2.inRange(hsv, (160, 70, 100), (180, 255, 255))
    valid_bg = (white_mask > 0) | (yellow_mask > 0) | (blue_mask > 0) | (red_mask1 > 0) | (red_mask2 > 0)
    valid_bg_ratio = float(np.sum(valid_bg)) / (h * w)
    if valid_bg_ratio < 0.20:
        return False

    # TẦNG 4: LOẠI NỀN ĐẤT/ĐƯỜNG
    dirt_mask  = cv2.inRange(hsv, (5,  20, 40),  (25, 150, 180))
    asph_mask  = cv2.inRange(hsv, (0,   0, 30),  (180, 40, 130))
    dirt_ratio = float(np.sum((dirt_mask > 0) | (asph_mask > 0))) / (h * w)
    if dirt_ratio > 0.40:
        return False

    # TẦNG 5: ĐỘ TƯƠNG PHẢN
    std_val = float(np.std(gray.astype(np.float32)))
    if std_val < 22 or std_val > 100:
        return False

    # TẦNG 6: MẬT ĐỘ CẠNH
    median_val = float(np.median(gray))
    lo = max(0,   int(0.5 * median_val))
    hi = min(255, int(1.3 * median_val))
    edges = cv2.Canny(gray, lo, hi)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    if edge_density < 0.02 or edge_density > 0.55:
        return False

    # TẦNG 7: CẤU TRÚC NGANG
    row_means = np.mean(gray.astype(np.float32), axis=1)
    row_std   = float(np.std(row_means))
    if row_std < 6.0:
        return False

    return True


def post_nms_filter(img, boxes, scores):
    """Lọc sau NMS — CẢI TIẾN: hỗ trợ biển xanh/đỏ."""
    valid_boxes  = []
    valid_scores = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = roi.shape[:2]

        white_mask  = cv2.inRange(hsv, (0,   0, 150), (180, 70, 255))
        yellow_mask = cv2.inRange(hsv, (15, 60, 120), (40, 255, 255))
        blue_mask   = cv2.inRange(hsv, (90, 50, 100), (130, 255, 255))
        red_mask1   = cv2.inRange(hsv, (0,  70, 100), (10, 255, 255))
        red_mask2   = cv2.inRange(hsv, (160, 70, 100), (180, 255, 255))

        bg_ratio = float(np.sum((white_mask > 0) | (yellow_mask > 0) |
                                (blue_mask > 0) | (red_mask1 > 0) | (red_mask2 > 0))) / (h_roi * w_roi)

        # Kiểm tra có tương phản giữa nền và chữ
        std_val = float(np.std(gray.astype(np.float32)))

        if bg_ratio >= 0.20 and std_val >= 20:
            valid_boxes.append((x1, y1, x2, y2))
            valid_scores.append(scores[i])

    return valid_boxes, valid_scores


def non_max_suppression_scored(boxes, scores, overlapThresh=0.3):
    """NMS có tính đến confidence score."""
    if len(boxes) == 0: return [], []
    boxes = np.array(boxes, dtype="float")
    scores = np.array(scores, dtype="float")

    pick = []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs[:last]]
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int"), scores[pick]


# ==============================================================================
# TÁCH KÝ TỰ TỪ BIỂN SỐ (CẢI TIẾN)
# ==============================================================================

def segment_characters(plate_bgr):
    """
    Tách ký tự từ ảnh biển số VN — CẢI TIẾN.
    Dùng CLAHE trước threshold, thêm nhiều threshold levels.
    """
    h, w = plate_bgr.shape[:2]
    if h < 10 or w < 20:
        return []

    # Scale lên nếu ảnh nhỏ
    target_h = 100  # Tăng từ 80 → 100
    scale = max(1.0, target_h / h)
    if scale > 1.0:
        plate_bgr = cv2.resize(plate_bgr,
                               (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_CUBIC)
        h, w = plate_bgr.shape[:2]

    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE trước blur
    gray_clahe = _clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

    # Phát hiện biển 2 hàng
    split_y = _detect_two_rows(gray, h, w)

    # Tạo nhiều ảnh binary
    thresholds = _make_thresholds(blur, gray_clahe)

    best_result = []
    best_score = -1

    for th_raw in thresholds:
        th_clean = _clean_binary(th_raw, h, w)

        if split_y is not None:
            top_chars = _find_char_contours(th_clean[:split_y, :],
                                            plate_bgr[:split_y, :],
                                            split_y, w, row_mode=True)
            bot_chars = _find_char_contours(th_clean[split_y:, :],
                                            plate_bgr[split_y:, :],
                                            h - split_y, w, row_mode=True)
            top_chars.sort(key=lambda t: t[0])
            bot_chars.sort(key=lambda t: t[0])
            combined = [(cx, roi) for cx, _cy, _cw, _ch, roi in top_chars]
            combined += [(cx + 10000, roi) for cx, _cy, _cw, _ch, roi in bot_chars]
        else:
            chars = _find_char_contours(th_clean, plate_bgr, h, w, row_mode=False)
            chars.sort(key=lambda t: t[0])
            combined = [(cx, roi) for cx, _cy, _cw, _ch, roi in chars]

        n = len(combined)
        if 7 <= n <= 9:
            score = n + 10  # Ưu tiên cao nhất
        elif 4 <= n <= 6:
            score = n
        elif n > 9:
            score = max(0, 9 - (n - 9))
        else:
            score = 0

        if score > best_score:
            best_score = score
            best_result = combined

    # Giới hạn 9 ký tự
    if len(best_result) > 9:
        scored = [(key, roi, roi.shape[0] * roi.shape[1]) for key, roi in best_result]
        scored.sort(key=lambda t: -t[2])
        scored = scored[:9]
        scored.sort(key=lambda t: t[0])
        best_result = [(key, roi) for key, roi, _ in scored]

    return best_result


def _detect_two_rows(gray, h, w):
    """Phát hiện biển 2 hàng — CẢI TIẾN."""
    ar = w / float(h)
    if ar > 3.5 or ar < 0.8:
        return None

    proj = np.sum(gray < 100, axis=1).astype(float)
    # Smooth projection
    kernel = np.ones(max(1, h // 20)) / max(1, h // 20)
    proj_smooth = np.convolve(proj, kernel, mode='same')

    s1 = int(h * 0.28)
    s2 = int(h * 0.72)
    if s2 <= s1:
        return None

    region = proj_smooth[s1:s2]
    if len(region) == 0:
        return None

    min_idx = np.argmin(region) + s1
    if proj_smooth[min_idx] > w * 0.35:
        return None

    return min_idx


def _make_thresholds(blur, gray):
    """Tạo nhiều ảnh binary — CẢI TIẾN: thêm nhiều biến thể hơn."""
    results = []

    # === POLARITY 1: BINARY_INV (biển nền sáng, chữ tối) ===
    _, th_inv_otsu = cv2.threshold(blur, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(th_inv_otsu)

    th_inv_gauss = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 19, 8)
    results.append(th_inv_gauss)

    th_inv_mean = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 10)
    results.append(th_inv_mean)

    # Thêm adaptive với block size khác (MỚI)
    th_inv_gauss2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 25, 6)
    results.append(th_inv_gauss2)

    # === POLARITY 2: BINARY (biển nền tối, chữ sáng) ===
    _, th_norm_otsu = cv2.threshold(blur, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(th_norm_otsu)

    th_norm_gauss = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 19, 8)
    results.append(th_norm_gauss)

    # === POLARITY 3: CLAHE-based threshold (MỚI) ===
    _, th_clahe = cv2.threshold(gray, 0, 255,
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    results.append(th_clahe)

    return results


def _clean_binary(th, h, w):
    """Xóa viền + morphological ops — CẢI TIẾN."""
    th2 = th.copy()
    by = max(2, int(h * 0.05))
    bx = max(2, int(w * 0.02))
    th2[:by, :] = 0; th2[-by:, :] = 0
    th2[:, :bx] = 0; th2[:, -bx:] = 0

    # Close trước (nối nét đứt)
    k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, k1, iterations=1)
    # Open (loại noise nhỏ)
    k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    th2 = cv2.morphologyEx(th2, cv2.MORPH_OPEN, k2, iterations=1)
    return th2


def _find_char_contours(th, plate_bgr, h, w, row_mode=False):
    """Tìm contours hợp lệ là ký tự — CẢI TIẾN."""
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    chars = []

    for c in cnts:
        cx, cy, cw, ch2 = cv2.boundingRect(c)

        if row_mode:
            if ch2 < h * 0.25 or ch2 > h * 0.98: continue
        else:
            if ch2 < h * 0.18 or ch2 > h * 0.92: continue

        # Aspect ratio: cho phép rộng hơn (W, M rộng; 1 rất hẹp)
        char_ar = cw / float(ch2)
        if char_ar < 0.06 or char_ar > 1.3: continue

        # Kích thước tối thiểu
        if cw < 3 or ch2 < 6: continue

        # Solidity
        area = cv2.contourArea(c)
        if cw * ch2 > 0 and area / (cw * ch2) < 0.08: continue

        # Cắt ROI — padding
        pad = max(2, int(ch2 * 0.1))
        y1 = max(0, cy - pad)
        y2 = min(plate_bgr.shape[0], cy + ch2 + pad)
        x1 = max(0, cx - pad)
        x2 = min(plate_bgr.shape[1], cx + cw + pad)
        roi = plate_bgr[y1:y2, x1:x2]
        if roi.size > 0:
            chars.append((cx, cy, cw, ch2, roi))

    return chars


# ==============================================================================
# ĐỌC KÝ TỰ (OCR) — CẢI TIẾN: TOP-K VOTING
# ==============================================================================

def read_plate_text(ocr_model, ocr_scaler, plate_bgr, min_conf=0.30):
    """Tách ký tự → classify với top-k voting → sửa nhầm lẫn → format."""
    if ocr_model is None or ocr_scaler is None:
        return "", []

    chars = segment_characters(plate_bgr)
    if len(chars) < 4:
        return "", []

    texts, confs = [], []
    ocr_model.eval()
    for _sort_key, roi in chars:
        feat = extract_hog_char(roi)
        if feat is None:
            texts.append("?"); confs.append(0.0); continue
        fs = ocr_scaler.transform(feat.reshape(1, -1))
        t = torch.tensor(fs, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(ocr_model(t), dim=1)
            topk_conf, topk_pred = torch.topk(probs, 3, dim=1)

        c = topk_conf[0, 0].item()
        p = topk_pred[0, 0].item()

        if c >= min_conf:
            texts.append(CLASS_NAMES[p])
        else:
            texts.append("?")
        confs.append(c)

    # Sửa nhầm lẫn OCR theo vị trí
    texts = _correct_ocr_confusions(texts)

    return _format_plate("".join(texts)), confs


def _correct_ocr_confusions(texts):
    """
    Sửa lỗi nhầm lẫn phổ biến — CẢI TIẾN.
    """
    if len(texts) < 4:
        return texts

    letter_to_digit = {
        'B': '8', 'D': '0', 'G': '6', 'S': '5', 'Z': '2',
        'T': '7', 'A': '4', 'E': '3', 'L': '1', 'C': '0',
    }
    digit_to_letter = {
        '0': 'D', '8': 'B', '6': 'G', '5': 'S', '2': 'Z',
        '7': 'T', '4': 'A', '3': 'E', '1': 'L',
    }

    corrected = list(texts)

    # Vị trí 0, 1: phải là số (mã tỉnh)
    for i in [0, 1]:
        if i < len(corrected) and corrected[i].isalpha():
            corrected[i] = letter_to_digit.get(corrected[i], corrected[i])

    # Vị trí 2: phải là chữ (series)
    if len(corrected) > 2 and corrected[2].isdigit():
        corrected[2] = digit_to_letter.get(corrected[2], corrected[2])

    # Từ vị trí 3 trở đi: phải là số (trừ series 2 chữ)
    start_digits = 3
    if len(corrected) > 3 and corrected[3].isalpha():
        start_digits = 4

    for i in range(start_digits, len(corrected)):
        if corrected[i].isalpha():
            corrected[i] = letter_to_digit.get(corrected[i], corrected[i])

    return corrected


def _format_plate(raw):
    """Format chuỗi OCR → dạng biển số VN chuẩn."""
    s = raw.replace("?", "").upper()
    if len(s) < 4:
        return raw

    # 30A-12345
    m = re.match(r'^(\d{2})([A-Z])(\d{5})$', s)
    if m: return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    # 30A1-2345 (biển 2 hàng)
    m = re.match(r'^(\d{2})([A-Z])(\d)(\d{4})$', s)
    if m: return f"{m.group(1)}{m.group(2)}{m.group(3)}-{m.group(4)}"

    # 30A-1234
    m = re.match(r'^(\d{2})([A-Z])(\d{4})$', s)
    if m: return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    # 30AB-12345
    m = re.match(r'^(\d{2})([A-Z]{2})(\d{4,5})$', s)
    if m: return f"{m.group(1)}{m.group(2)}-{m.group(3)}"

    return raw


# ==============================================================================
# PHẦN GIAO DIỆN NGƯỜI DÙNG (GUI) VỚI TKINTER
# ==============================================================================

class PlateDetectorGUI:
    def __init__(self, root, model, scaler, ocr_model=None, ocr_scaler=None):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.ocr_model = ocr_model
        self.ocr_scaler = ocr_scaler

        self.root.title("Trình Nhận Diện & Đọc Biển Số Xe VN (HOG + ANN)")
        self.root.geometry("1000x750")
        self.root.configure(bg="#f0f0f0")

        self.img_original = None
        self.img_tk = None
        self.scale_factor = 1.0
        self.rect_id = None
        self.start_x = None
        self.start_y = None

        # --- Giao diện điều khiển ---
        top_frame = tk.Frame(self.root, pady=8, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

        self.btn_load = tk.Button(top_frame, text="📁 Tải Ảnh", command=self.load_image,
                                   bg="#e0e0e0", font=("Arial", 10, "bold"), padx=10)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_auto = tk.Button(top_frame, text="🔍 Nhận diện Tự động", command=self.auto_detect,
                                   bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=10)
        self.btn_auto.pack(side=tk.LEFT, padx=5)

        self.lbl_status = tk.Label(top_frame, text="Vui lòng tải ảnh để bắt đầu.",
                                    font=("Helvetica", 11), fg="blue", bg="#f0f0f0")
        self.lbl_status.pack(side=tk.LEFT, padx=20)

        # --- Slider Confidence Threshold ---
        slider_frame = tk.Frame(self.root, pady=5, bg="#f0f0f0")
        slider_frame.pack(side=tk.TOP, fill=tk.X, padx=10)

        tk.Label(slider_frame, text="Ngưỡng Confidence:", font=("Arial", 9), bg="#f0f0f0").pack(side=tk.LEFT, padx=5)

        self.confidence_var = tk.DoubleVar(value=DEFAULT_CONFIDENCE_THRESHOLD)
        self.confidence_slider = tk.Scale(
            slider_frame, from_=0.50, to=0.999, resolution=0.005,
            orient=tk.HORIZONTAL, variable=self.confidence_var,
            length=200, font=("Arial", 9), bg="#f0f0f0",
            highlightthickness=0
        )
        self.confidence_slider.pack(side=tk.LEFT, padx=5)

        self.lbl_threshold = tk.Label(slider_frame, text=f"({DEFAULT_CONFIDENCE_THRESHOLD*100:.0f}%)",
                                       font=("Arial", 10, "bold"), fg="#333", bg="#f0f0f0")
        self.lbl_threshold.pack(side=tk.LEFT, padx=5)
        self.confidence_slider.config(command=self._update_threshold_label)

        # --- Label kết quả biển số ---
        self.lbl_result = tk.Label(self.root, text="Biển số: —",
                                   font=("Courier New", 20, "bold"),
                                   fg="#1a237e", bg="#f0f0f0")
        self.lbl_result.pack(side=tk.BOTTOM, pady=8)

        lbl_instructions = tk.Label(self.root,
            text="Bấm '🔍 Nhận diện Tự động' để quét, hoặc kéo chuột để khoanh vùng thủ công.",
            justify=tk.LEFT, bg="#f0f0f0", font=("Arial", 9))
        lbl_instructions.pack(side=tk.BOTTOM, pady=3)

        # Canvas hiển thị
        self.canvas = tk.Canvas(self.root, cursor="cross", bg="#DDDDDD")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def _update_threshold_label(self, val):
        self.lbl_threshold.config(text=f"({float(val)*100:.0f}%)")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return

        try:
            self.img_original = cv2.imread(file_path)
            if self.img_original is None: raise ValueError("Không thể đọc file ảnh.")

            self.root.update_idletasks()
            canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            img_h, img_w = self.img_original.shape[:2]

            self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = int(img_w * self.scale_factor), int(img_h * self.scale_factor)

            display_img = cv2.resize(self.img_original, (new_w, new_h))
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))

            self.canvas.delete("all")
            self.canvas.create_image(canvas_w/2, canvas_h/2, anchor=tk.CENTER, image=self.img_tk)

            self.lbl_status.config(text="Ảnh đã tải. Sẵn sàng!", fg="black")
            self.lbl_result.config(text="Biển số: —")
            self.rect_id = None

        except Exception as e:
            messagebox.showerror("Lỗi Tải Ảnh", f"Đã xảy ra lỗi: {e}")

    # =====================================================
    # NHẬN DIỆN TỰ ĐỘNG
    # =====================================================
    def auto_detect(self):
        if self.img_original is None:
            messagebox.showwarning("Nhắc nhở", "Bạn chưa tải ảnh kìa!")
            return

        threshold = self.confidence_var.get()
        self.lbl_status.config(text=f"Đang phân tích (ngưỡng {threshold*100:.0f}%)...", fg="orange")
        self.lbl_result.config(text="Biển số: …")
        self.root.update()

        self.canvas.delete("auto_box")

        # Selective Search
        ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        ss.setBaseImage(self.img_original)
        ss.switchToSelectiveSearchFast()
        rects = ss.process()

        detected_boxes = []
        detected_scores = []
        img_h, img_w = self.img_original.shape[:2]
        img_area = img_h * img_w

        for (x, y, w, h) in rects:
            if w < 50 or h < 15: continue
            if w > img_w * 0.5 or h > img_h * 0.4: continue
            box_area = w * h
            if box_area < 1500: continue
            if box_area / img_area > 0.15: continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.8 or aspect_ratio > 5.0: continue

            if y + h > img_h * 0.95 and h < img_h * 0.08:
                continue
            if y < img_h * 0.02 and h < img_h * 0.06:
                continue

            roi = self.img_original[y:y+h, x:x+w]
            if not is_plate_candidate(roi):
                continue

            pred, confidence = predict_plate_with_confidence(self.model, self.scaler, roi)

            if pred == 1 and confidence >= threshold:
                detected_boxes.append([x, y, x + w, y + h])
                detected_scores.append(confidence)

        # NMS + post filter
        if len(detected_boxes) > 0:
            final_boxes, final_scores = non_max_suppression_scored(
                detected_boxes, detected_scores, overlapThresh=0.2
            )
            final_boxes_list, final_scores_list = post_nms_filter(
                self.img_original,
                [(b[0], b[1], b[2], b[3]) for b in final_boxes],
                list(final_scores)
            )
            final_boxes = np.array(final_boxes_list) if final_boxes_list else np.array([])
            final_scores = np.array(final_scores_list)
        else:
            final_boxes, final_scores = np.array([]), np.array([])

        if len(final_boxes) > MAX_DETECTIONS:
            sorted_indices = np.argsort(final_scores)[::-1][:MAX_DETECTIONS]
            final_boxes = final_boxes[sorted_indices]
            final_scores = final_scores[sorted_indices]

        if len(final_boxes) == 0:
            self.lbl_status.config(text=f"Không tìm thấy biển số nào (ngưỡng {threshold*100:.0f}%).", fg="red")
            self.lbl_result.config(text="Biển số: —")
            return

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_start_x = (canvas_w - self.img_tk.width()) / 2
        img_start_y = (canvas_h - self.img_tk.height()) / 2

        all_plate_texts = []
        for i, (x1, y1, x2, y2) in enumerate(final_boxes):
            score = final_scores[i]

            plate_roi = self.img_original[y1:y2, x1:x2]
            plate_text, char_confs = read_plate_text(
                self.ocr_model, self.ocr_scaler, plate_roi)

            has_text = len(plate_text.replace("?", "").replace("-", "").replace(" ", "")) >= 4

            if score >= 0.98:
                color = "#00CC00"
            elif score >= 0.95:
                color = "#66CC00"
            else:
                color = "#FFAA00"

            c_x1 = x1 * self.scale_factor + img_start_x
            c_y1 = y1 * self.scale_factor + img_start_y
            c_x2 = x2 * self.scale_factor + img_start_x
            c_y2 = y2 * self.scale_factor + img_start_y

            self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2,
                                         outline=color, width=3, tags="auto_box")

            if has_text:
                label_text = f"{plate_text}  ({score*100:.1f}%)"
                all_plate_texts.append(plate_text)
            else:
                label_text = f"Biển Số ({score*100:.1f}%)"

            self.canvas.create_text(c_x1, c_y1 - 5,
                text=label_text,
                fill=color, font=("Helvetica", 11, "bold"), anchor=tk.SW, tags="auto_box")

        if all_plate_texts:
            self.lbl_result.config(text="Biển số: " + "  |  ".join(all_plate_texts))
            self.lbl_status.config(text=f"Tìm thấy {len(final_boxes)} biển số, đọc được {len(all_plate_texts)} biển!", fg="green")
        else:
            self.lbl_result.config(text="Biển số: (chưa đọc được ký tự)")
            self.lbl_status.config(text=f"Tìm thấy {len(final_boxes)} biển số!", fg="green")

    # =====================================================
    # KÉO CHUỘT THỦ CÔNG
    # =====================================================
    def on_button_press(self, event):
        if self.img_original is None: return
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.rect_id: self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y,
                                                     self.start_x, self.start_y,
                                                     outline="cyan", width=2)

    def on_mouse_drag(self, event):
        if not self.rect_id: return
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if not self.rect_id: return

        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
        img_start_x = (canvas_w - self.img_tk.width()) / 2
        img_start_y = (canvas_h - self.img_tk.height()) / 2

        x1_c, y1_c, x2_c, y2_c = self.canvas.coords(self.rect_id)

        orig_x1 = int((min(x1_c, x2_c) - img_start_x) / self.scale_factor)
        orig_y1 = int((min(y1_c, y2_c) - img_start_y) / self.scale_factor)
        orig_x2 = int((max(x1_c, x2_c) - img_start_x) / self.scale_factor)
        orig_y2 = int((max(y1_c, y2_c) - img_start_y) / self.scale_factor)

        h, w = self.img_original.shape[:2]
        orig_x1, orig_y1 = max(0, orig_x1), max(0, orig_y1)
        orig_x2, orig_y2 = min(w, orig_x2), min(h, orig_y2)

        if (orig_x2 - orig_x1) < 10 or (orig_y2 - orig_y1) < 10:
            self.lbl_status.config(text="Vùng chọn quá nhỏ!", fg="orange")
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return

        img_crop = self.img_original[orig_y1:orig_y2, orig_x1:orig_x2]
        prediction, confidence = predict_plate_with_confidence(self.model, self.scaler, img_crop)

        plate_text, char_confs = read_plate_text(self.ocr_model, self.ocr_scaler, img_crop)

        if prediction == 1:
            if plate_text:
                avg_ocr = np.mean(char_confs) if char_confs else 0
                self.lbl_status.config(
                    text=f"BIỂN SỐ  det={confidence*100:.1f}%  ocr={avg_ocr*100:.1f}%",
                    fg="green")
                self.lbl_result.config(text=f"Biển số: {plate_text}")
            else:
                self.lbl_status.config(
                    text=f"DỰ ĐOÁN: BIỂN SỐ (Confidence: {confidence*100:.1f}%) — chưa đọc được ký tự",
                    fg="green")
                self.lbl_result.config(text="Biển số: (?)")
            self.canvas.itemconfig(self.rect_id, outline="green")
        else:
            self.lbl_status.config(text=f"DỰ ĐOÁN: NỀN/RÁC (Confidence: {confidence*100:.1f}%)", fg="red")
            self.lbl_result.config(text="Biển số: —")
            self.canvas.itemconfig(self.rect_id, outline="red")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    try:
        print(f"⚙️ Đang chạy trên: {device}")

        model, scaler = load_model_and_scaler()
        print("✅ Plate detector loaded.")

        ocr_model, ocr_scaler = load_ocr_model()
        if ocr_model is not None:
            print("✅ OCR model loaded.")
        else:
            print("⚠️ OCR model không khả dụng — chỉ phát hiện biển số, không đọc ký tự.")

        root = tk.Tk()
        app = PlateDetectorGUI(root, model, scaler, ocr_model, ocr_scaler)
        root.mainloop()
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")