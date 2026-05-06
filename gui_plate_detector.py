import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import re
import pickle
import torch
from plate_settings import MAX_DETECTIONS, MAX_IMAGE_SIDE_FOR_SEARCH, CLASS_NAMES
from plate_models import PlateDetectorANN, OCRNet
from plate_features import extract_hog_features, extract_hog_char, infer_plate_type, pad_plate_feature, MAX_HOG_LEN, HOG_CHAR_LEN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def load_model_and_scaler(model_path='final_plate_model.pth', scaler_path='scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Không tìm thấy file model '{model_path}' hoặc scaler '{scaler_path}'.")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    input_dim = int(getattr(scaler, 'n_features_in_', MAX_HOG_LEN))
    model = PlateDetectorANN(input_dim=input_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return (model, scaler)

def load_ocr_model(model_path='ocr_model_best.pth', scaler_path='ocr_scaler.pkl'):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f'⚠️ Không tìm thấy OCR model ({model_path}) hoặc scaler ({scaler_path}). OCR sẽ bị tắt.')
        return (None, None)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    model = OCRNet(HOG_CHAR_LEN).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return (model, scaler)

def predict_plate_with_confidence(model, scaler, img_crop):
    plate_type = infer_plate_type(img_crop)
    features = extract_hog_features(img_crop, plate_type=plate_type)
    features = pad_plate_feature(features)
    if features is None:
        return (-1, 0.0)
    features_scaled = scaler.transform(features.reshape(1, -1))
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(features_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return (predicted.item(), confidence.item())

def _generate_contour_proposals(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, th_base = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernels = [cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)), cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7)), cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))]
    boxes = []
    seen = set()
    for kernel in kernels:
        th = cv2.morphologyEx(th_base, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw < 30 or bh < 10:
                continue
            if bw > w * 0.75 or bh > h * 0.5:
                continue
            area = bw * bh
            if area < 400 or area / img_area > 0.25:
                continue
            aspect = bw / float(max(1, bh))
            if 0.5 <= aspect <= 6.5:
                qkey = (x // 6, y // 6, bw // 6, bh // 6)
                if qkey not in seen:
                    seen.add(qkey)
                    boxes.append((x, y, bw, bh))
    return boxes

def _generate_mser_proposals(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = h * w
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    mser = cv2.MSER_create()
    mser.setMinArea(200)
    mser.setMaxArea(int(img_area * 0.15))
    regions, _ = mser.detectRegions(gray)
    boxes = []
    seen = set()
    for region in regions:
        x, y, bw, bh = cv2.boundingRect(region)
        if bw < 25 or bh < 8:
            continue
        if bw > w * 0.6 or bh > h * 0.4:
            continue
        area = bw * bh
        if area < 300 or area / img_area > 0.2:
            continue
        aspect = bw / float(max(1, bh))
        if 0.5 <= aspect <= 6.5:
            qkey = (x // 6, y // 6, bw // 6, bh // 6)
            if qkey not in seen:
                seen.add(qkey)
                boxes.append((x, y, bw, bh))
    merged = []
    if boxes:
        boxes_np = np.array(boxes)
        for bx, by, bw, bh in boxes:
            found = False
            for mi in range(len(merged)):
                mx, my, mw, mh = merged[mi]
                cy1, cy2 = (by, by + bh)
                my1, my2 = (my, my + mh)
                y_overlap = max(0, min(cy2, my2) - max(cy1, my1))
                if y_overlap > min(bh, mh) * 0.3:
                    nx1 = min(mx, bx)
                    ny1 = min(my, by)
                    nx2 = max(mx + mw, bx + bw)
                    ny2 = max(my + mh, by + bh)
                    merged[mi] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                    found = True
                    break
            if not found:
                merged.append((bx, by, bw, bh))
    result = []
    for x, y, bw, bh in merged:
        area = bw * bh
        aspect = bw / float(max(1, bh))
        if area > 500 and 0.5 <= aspect <= 6.5:
            result.append((x, y, bw, bh))
    return result

def _prepare_candidate_boxes(img_bgr):
    h, w = img_bgr.shape[:2]
    img_area = h * w
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img_bgr)
    ss.switchToSelectiveSearchFast()
    ss_rects = ss.process()
    candidates = []
    seen = set()
    for x, y, bw, bh in ss_rects:
        if bw < 30 or bh < 10:
            continue
        if bw > w * 0.75 or bh > h * 0.5:
            continue
        area = bw * bh
        if area < 400 or area / img_area > 0.25:
            continue
        aspect = bw / float(max(1, bh))
        if aspect < 0.5 or aspect > 6.5:
            continue
        qx, qy = (x // 4, y // 4)
        qw, qh = (bw // 4, bh // 4)
        key = (qx, qy, qw, qh)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((x, y, bw, bh))
    for x, y, bw, bh in _generate_mser_proposals(img_bgr):
        qx, qy = (x // 4, y // 4)
        qw, qh = (bw // 4, bh // 4)
        key = (qx, qy, qw, qh)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((x, y, bw, bh))

    def rank_key(box):
        bx, by, bw, bh = box
        area = bw * bh
        aspect = bw / float(max(1, bh))
        center_y = by + bh * 0.5
        center_dist = abs(center_y / max(1.0, h) - 0.55)
        return (-min(area / max(1.0, img_area), 0.2), abs(aspect - 3.0), center_dist)
    candidates.sort(key=rank_key)
    return candidates[:2000]

def _batch_predict_boxes(model, scaler, img_bgr, boxes_xyxy):
    feats = []
    valid_boxes = []
    for x1, y1, x2, y2 in boxes_xyxy:
        roi = img_bgr[y1:y2, x1:x2]
        plate_type = infer_plate_type(roi)
        feat = extract_hog_features(roi, plate_type=plate_type)
        feat = pad_plate_feature(feat)
        if feat is None:
            continue
        feats.append(feat)
        valid_boxes.append((x1, y1, x2, y2))
    if not feats:
        return ([], [])
    feat_arr = np.asarray(feats, dtype=np.float32)
    feat_scaled = scaler.transform(feat_arr)
    feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(feat_tensor), dim=1).detach().cpu().numpy()
    plate_scores = probs[:, 1]
    return (valid_boxes, plate_scores.tolist())

def is_plate_candidate(img_crop):
    if img_crop is None or img_crop.size == 0:
        return False
    h, w = img_crop.shape[:2]
    if h < 12 or w < 30:
        return False
    gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV)
    aspect = w / float(h)
    if aspect < 0.8 or aspect > 5.0:
        return False
    bright_ratio = float(np.sum(hsv[:, :, 2] > 90)) / (h * w)
    if bright_ratio < 0.15:
        return False
    white_mask = cv2.inRange(hsv, (0, 0, 80), (180, 80, 255))
    yellow_mask = cv2.inRange(hsv, (15, 60, 100), (40, 255, 255))
    blue_mask = cv2.inRange(hsv, (90, 50, 80), (130, 255, 255))
    red_mask1 = cv2.inRange(hsv, (0, 70, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 70, 80), (180, 255, 255))
    valid_bg = (white_mask > 0) | (yellow_mask > 0) | (blue_mask > 0) | (red_mask1 > 0) | (red_mask2 > 0)
    valid_bg_ratio = float(np.sum(valid_bg)) / (h * w)
    if valid_bg_ratio < 0.15:
        return False
    dirt_mask = cv2.inRange(hsv, (5, 20, 40), (25, 150, 180))
    asph_mask = cv2.inRange(hsv, (0, 0, 20), (180, 40, 120))
    dirt_ratio = float(np.sum((dirt_mask > 0) | (asph_mask > 0))) / (h * w)
    if dirt_ratio > 0.65:
        return False
    std_val = float(np.std(gray.astype(np.float32)))
    if std_val < 15 or std_val > 100:
        return False
    median_val = float(np.median(gray))
    lo = max(0, int(0.5 * median_val))
    hi = min(255, int(1.3 * median_val))
    edges = cv2.Canny(gray, lo, hi)
    edge_density = float(np.sum(edges > 0)) / (h * w)
    if edge_density < 0.02 or edge_density > 0.55:
        return False
    row_means = np.mean(gray.astype(np.float32), axis=1)
    row_std = float(np.std(row_means))
    if row_std < 6.0:
        return False
    return True

def suppress_landscape_vehicle_false_positives(img_bgr, boxes, scores):
    if boxes is None or len(boxes) == 0:
        return (boxes, scores)
    h, w_img = img_bgr.shape[:2]
    if h <= 0 or w_img <= 0:
        return (boxes, scores)
    boxes = np.asarray(boxes)
    scores = np.asarray(scores, dtype=float)
    landscape = w_img / float(max(1, h)) >= 1.12
    if not landscape:
        return (boxes, scores)
    if len(boxes) >= 2:
        return (boxes, scores)
    items = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = [int(v) for v in boxes[i]]
        bw, bh = (max(1, x2 - x1), max(1, y2 - y1))
        aspect = bw / float(bh)
        cy = (y1 + y2) * 0.5 / float(max(1, h))
        cx = (x1 + x2) * 0.5 / float(max(1, w_img))
        items.append((i, aspect, cy, cx, float(scores[i])))
    keep = np.ones(len(items), dtype=bool)
    for j, (_, aspect, cy, cx, sc) in enumerate(items):
        if cy < 0.42 and aspect < 2.25 and (sc < 0.925):
            keep[j] = False
    if not np.any(keep):
        keep[:] = True
    return (boxes[keep], scores[keep])

def post_nms_filter(img, boxes, scores):
    valid_boxes = []
    valid_scores = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = roi.shape[:2]
        white_mask = cv2.inRange(hsv, (0, 0, 80), (180, 80, 255))
        yellow_mask = cv2.inRange(hsv, (15, 60, 100), (40, 255, 255))
        blue_mask = cv2.inRange(hsv, (90, 50, 80), (130, 255, 255))
        red_mask1 = cv2.inRange(hsv, (0, 70, 80), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 70, 80), (180, 255, 255))
        bg_ratio = float(np.sum((white_mask > 0) | (yellow_mask > 0) | (blue_mask > 0) | (red_mask1 > 0) | (red_mask2 > 0))) / (h_roi * w_roi)
        std_val = float(np.std(gray.astype(np.float32)))
        if bg_ratio >= 0.15 and std_val >= 15:
            valid_boxes.append((x1, y1, x2, y2))
            valid_scores.append(scores[i])
    return (valid_boxes, valid_scores)

def non_max_suppression_scored(boxes, scores, overlapThresh=0.2):
    if len(boxes) == 0:
        return ([], [])
    boxes = np.array(boxes, dtype='float')
    scores = np.array(scores, dtype='float')
    pick = []
    x1, y1, x2, y2 = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    adjusted_scores = scores - area / np.max(area) * 0.02
    idxs = np.argsort(adjusted_scores)
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
        inter = w * h
        union = area[i] + area[idxs[:last]] - inter
        iou = inter / union
        iom = inter / np.minimum(area[i], area[idxs[:last]])
        suppress_idx = np.where((iou > overlapThresh) | (iom > 0.6))[0]
        idxs = np.delete(idxs, np.concatenate(([last], suppress_idx)))
    return (boxes[pick].astype('int'), scores[pick])

def _refine_plate_crop(plate_bgr):
    h_orig, w_orig = plate_bgr.shape[:2]
    if h_orig < 20 or w_orig < 40:
        return (plate_bgr, 0, 0)
    ref_h = 120
    sc = ref_h / h_orig
    img_ref = cv2.resize(plate_bgr, (int(w_orig * sc), ref_h))
    h, w = img_ref.shape[:2]
    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    max_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < h * w * 0.3:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        rx, ry, rw, rh = cv2.boundingRect(approx)
        if 0.5 < rw / rh < 6.0:
            if area > max_area:
                max_area = area
                best_rect = (rx, ry, rw, rh)
    if best_rect:
        rx, ry, rw, rh = best_rect
        orig_rx = int(rx / sc)
        orig_ry = int(ry / sc)
        orig_rw = int(rw / sc)
        orig_rh = int(rh / sc)
        px = max(1, int(orig_rw * 0.02))
        py = max(1, int(orig_rh * 0.02))
        x1, y1 = (max(0, orig_rx + px), max(0, orig_ry + py))
        x2, y2 = (min(w_orig, orig_rx + orig_rw - px), min(h_orig, orig_ry + orig_rh - py))
        return (plate_bgr[y1:y2, x1:x2], x1, y1)
    bx, by = (int(w_orig * 0.05), int(h_orig * 0.07))
    return (plate_bgr[by:-by, bx:-bx], bx, by)

def segment_characters(plate_bgr):
    h_orig, w_orig = plate_bgr.shape[:2]
    if h_orig < 8 or w_orig < 8:
        return []
    target_h = 100
    sc = max(1.0, target_h / h_orig)
    if sc > 1.0:
        img_bgr = cv2.resize(plate_bgr, (int(w_orig * sc), int(h_orig * sc)), interpolation=cv2.INTER_CUBIC)
    else:
        img_bgr = plate_bgr.copy()
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray_cl = _clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_cl, (3, 3), 0)
    thresholds = []
    _, th1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresholds.append(th1)
    th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 8)
    thresholds.append(th2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    thresholds.append(th3)
    th3b = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 6)
    thresholds.append(th3b)
    _, th4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresholds.append(th4)
    best_chars = []
    for th in thresholds:
        th_clean = th.copy()
        by = max(2, int(h * 0.05))
        bx = max(2, int(w * 0.03))
        th_clean[:by, :] = 0
        th_clean[-by:, :] = 0
        th_clean[:, :bx] = 0
        th_clean[:, -bx:] = 0
        k1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_CLOSE, k1, iterations=1)
        k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        th_clean = cv2.morphologyEx(th_clean, cv2.MORPH_OPEN, k2, iterations=1)
        cnts, _ = cv2.findContours(th_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        chars = []
        for c in cnts:
            cx, cy, cw, ch2 = cv2.boundingRect(c)
            if ch2 < h * 0.15 or ch2 > h * 0.95:
                continue
            if cw < 4 or ch2 < 8:
                continue
            char_ar = cw / float(ch2)
            if char_ar > 1.0 and ch2 < h * 0.4:
                continue
            area_ratio = cv2.contourArea(c) / float(cw * ch2) if cw * ch2 > 0 else 0
            if char_ar > 0.8 and ch2 < h * 0.3 and (area_ratio < 0.35):
                continue
            edge_margin = max(5, int(w * 0.06))
            at_edge = cx < edge_margin or cx + cw > w - edge_margin
            if at_edge and char_ar < 0.12:
                continue
            if ch2 > h * 0.8 and char_ar < 0.1:
                continue
            edge_margin_y = max(4, int(h * 0.08))
            at_edge_y = cy < edge_margin_y or cy + ch2 > h - edge_margin_y
            if at_edge and at_edge_y:
                if ch2 < h * 0.3 or area_ratio < 0.3 or char_ar > 0.85:
                    continue
            if char_ar < 0.05 or char_ar > 1.3:
                continue
            area = cv2.contourArea(c)
            if cw * ch2 > 0 and area / (cw * ch2) < 0.15:
                continue
            pad = max(2, int(ch2 * 0.1))
            y1, y2 = (max(0, cy - pad), min(h, cy + ch2 + pad))
            x1, x2 = (max(0, cx - pad), min(w, cx + cw + pad))
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                orig_cx = int(cx / sc)
                orig_cy = int(cy / sc)
                orig_cw = int(cw / sc)
                orig_ch = int(ch2 / sc)
                chars.append((orig_cx, orig_cy, orig_cw, orig_ch, roi, cx, cy, cw, ch2))
        n = len(chars)
        score = n + 100 if 7 <= n <= 9 else n
        best_score_val = len(best_chars) + 100 if 7 <= len(best_chars) <= 9 else len(best_chars)
        if score > best_score_val:
            best_chars = chars
    if len(best_chars) >= 3:
        heights = [c[8] for c in best_chars]
        median_h = sorted(heights)[len(heights) // 2]
        best_chars = [c for c in best_chars if c[8] >= median_h * 0.55]
    if len(best_chars) < 2:
        return [(c[0], c[0], c[1], c[2], c[3], c[4]) for c in best_chars]
    ys = sorted(set((c[6] for c in best_chars)))
    avg_h = np.mean([c[8] for c in best_chars])
    y_centers = sorted([c[6] + c[8] / 2.0 for c in best_chars])
    max_gap = 0
    split_y = None
    if len(ys) >= 2:
        for i in range(len(y_centers) - 1):
            gap = y_centers[i + 1] - y_centers[i]
            if gap > max_gap:
                max_gap = gap
                split_y = (y_centers[i] + y_centers[i + 1]) / 2.0
    if split_y is not None and max_gap > avg_h * 0.5:
        top_row = sorted([c for c in best_chars if c[6] + c[8] / 2.0 < split_y], key=lambda t: t[5])
        bot_row = sorted([c for c in best_chars if c[6] + c[8] / 2.0 >= split_y], key=lambda t: t[5])
        final_chars = [(c[0], c[0], c[1], c[2], c[3], c[4]) for c in top_row]
        final_chars += [(c[0] + 10000, c[0], c[1], c[2], c[3], c[4]) for c in bot_row]
        return final_chars
    else:
        best_chars.sort(key=lambda t: t[5])
        return [(c[0], c[0], c[1], c[2], c[3], c[4]) for c in best_chars]

def _score_char_segments(chars_data):
    if not chars_data:
        return -1.0
    n = len(chars_data)
    if n < 3:
        return -0.5
    widths = [max(1, c[3]) for c in chars_data]
    heights = [max(1, c[4]) for c in chars_data]
    ratios = [w / float(h) for w, h in zip(widths, heights)]
    median_h = float(np.median(heights))
    h_var = float(np.std(heights)) / max(1.0, median_h)
    plausible = 0
    for r in ratios:
        if 0.1 <= r <= 1.2:
            plausible += 1
    plausible_ratio = plausible / float(max(1, n))
    count_score = 1.0 - min(1.0, abs(n - 7) / 6.0)
    score = 1.7 * count_score + 1.2 * plausible_ratio + 0.8 * max(0.0, 1.0 - h_var)
    return float(score)

def _select_best_plate_crop(plate_bgr):
    if plate_bgr is None or plate_bgr.size == 0:
        return (plate_bgr, [], 0, 0)
    base_chars = segment_characters(plate_bgr)
    base_score = _score_char_segments(base_chars)
    refined_crop, off_x, off_y = _refine_plate_crop(plate_bgr)
    if refined_crop is None or refined_crop.size == 0:
        return (plate_bgr, base_chars, 0, 0)
    ref_chars = segment_characters(refined_crop)
    ref_score = _score_char_segments(ref_chars)
    if ref_score >= base_score + 0.2 and len(ref_chars) >= max(3, len(base_chars) - 2):
        return (refined_crop, ref_chars, off_x, off_y)
    return (plate_bgr, base_chars, 0, 0)

def read_plate_text(ocr_model, ocr_scaler, plate_bgr, min_conf=0.3):
    if ocr_model is None or ocr_scaler is None:
        return ('', [], [])
    _, chars_data, offset_x, offset_y = _select_best_plate_crop(plate_bgr)
    if len(chars_data) < 4:
        return ('', [], [])
    results = []
    ocr_model.eval()
    for _sort_key, cx, cy, cw, ch2, roi in chars_data:
        feat = extract_hog_char(roi)
        if feat is None:
            results.append((_sort_key, '?', 0.0, cx, cy, cw, ch2))
            continue
        aug_rois = [roi]
        h_r, w_r = roi.shape[:2]
        if h_r > 10 and w_r > 6:
            p = 2
            if h_r > p * 2 + 4 and w_r > p * 2 + 4:
                aug_rois.append(roi[p:-p, p:-p])
        vote_scores = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        for aug_roi in aug_rois:
            f = extract_hog_char(aug_roi)
            if f is None:
                continue
            fs = ocr_scaler.transform(f.reshape(1, -1))
            t = torch.tensor(fs, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.softmax(ocr_model(t), dim=1).cpu().numpy()[0]
            vote_scores += probs
        vote_scores /= max(1, len(aug_rois))
        p = int(np.argmax(vote_scores))
        c = float(vote_scores[p])
        ch = CLASS_NAMES[p] if c >= min_conf else '?'
        results.append((_sort_key, ch, c, cx, cy, cw, ch2))
    top_row = [(sk, ch, cf, ocx, ocy, ocw, och2) for sk, ch, cf, ocx, ocy, ocw, och2 in results if sk < 10000]
    bot_row = [(sk, ch, cf, ocx, ocy, ocw, och2) for sk, ch, cf, ocx, ocy, ocw, och2 in results if sk >= 10000]
    is_two_rows = len(bot_row) > 0
    if is_two_rows:
        MAX_TOP = 4
        while len(top_row) > MAX_TOP:
            min_idx = min(range(len(top_row)), key=lambda i: top_row[i][2])
            top_row.pop(min_idx)
        MAX_BOT = 5
        while len(bot_row) > MAX_BOT:
            min_idx = min(range(len(bot_row)), key=lambda i: bot_row[i][2])
            bot_row.pop(min_idx)
        results = top_row + bot_row
    else:
        while len(results) > 9:
            min_idx = min(range(len(results)), key=lambda i: results[i][2])
            results.pop(min_idx)
    texts = [ch for _, ch, _, _, _, _, _ in results]
    confs = [cf for _, _, cf, _, _, _, _ in results]
    bboxes_relative_to_plate = [(ocx + offset_x, ocy + offset_y, ocw, och2) for _, _, _, ocx, ocy, ocw, och2 in results]
    texts = _correct_ocr_confusions(texts)
    return (_format_plate(''.join(texts)), confs, bboxes_relative_to_plate)

def _correct_ocr_confusions(texts):
    if len(texts) < 4:
        return texts
    letter_to_digit = {'B': '8', 'D': '0', 'G': '6', 'S': '5', 'Z': '2', 'T': '7', 'A': '4', 'E': '3', 'L': '1', 'C': '0', 'U': '0', 'N': '4', 'H': '4'}
    digit_to_letter = {'0': 'D', '8': 'B', '6': 'G', '5': 'S', '2': 'Z', '7': 'T', '4': 'A', '3': 'E', '1': 'L'}
    valid_series = set('ABCDEFGHKLMNPSTUVXYZ')
    corrected = list(texts)
    for i in [0, 1]:
        if i < len(corrected) and corrected[i].isalpha():
            corrected[i] = letter_to_digit.get(corrected[i], corrected[i])
    if len(corrected) > 2 and corrected[2].isdigit():
        corrected[2] = digit_to_letter.get(corrected[2], corrected[2])
    if len(corrected) > 2 and corrected[2].isalpha() and (corrected[2] not in valid_series):
        close_map = {'F': 'E', 'I': 'L', 'J': 'L', 'O': 'D', 'Q': 'D', 'R': 'B', 'W': 'V'}
        corrected[2] = close_map.get(corrected[2], corrected[2])
    start_digits = 3
    if len(corrected) > 3 and corrected[3].isalpha():
        start_digits = 4
    for i in range(start_digits, len(corrected)):
        if corrected[i].isalpha():
            corrected[i] = letter_to_digit.get(corrected[i], corrected[i])
    return corrected

def _format_plate(raw):
    s = raw.replace('?', '').upper()
    if len(s) < 4:
        return raw
    m = re.match('^(\\d{2})([A-Z]{1,2})(\\d{5})$', s)
    if m:
        return f'{m.group(1)}{m.group(2)}-{m.group(3)[:3]}.{m.group(3)[3:]}'
    m = re.match('^(\\d{2})([A-Z]{1,2})(\\d{4})$', s)
    if m:
        return f'{m.group(1)}{m.group(2)}-{m.group(3)}'
    m = re.match('^(\\d{2})([A-Z])(\\d)(\\d{4})$', s)
    if m:
        return f'{m.group(1)}{m.group(2)}{m.group(3)}.{m.group(4)}'
    return raw

class PlateDetectorGUI:

    # ── Design tokens (Light, soft, harmonious) ──
    C_BG         = '#f5f6fa'      # warm light gray background
    C_SURFACE    = '#ffffff'      # white cards
    C_ELEVATED   = '#ffffff'      # header
    C_BORDER     = '#e4e7f0'      # soft border
    C_ACCENT     = '#6366f1'      # indigo accent
    C_ACCENT_H   = '#818cf8'      # lighter indigo hover
    C_ACCENT_BG  = '#eef2ff'      # very faint indigo background
    C_TEXT       = '#1e293b'      # dark slate text
    C_TEXT2      = '#94a3b8'      # muted text
    C_SUCCESS    = '#10b981'      # emerald green
    C_SUCCESS_BG = '#ecfdf5'      # faint green bg
    C_WARN       = '#f59e0b'      # warm amber
    C_DANGER     = '#ef4444'      # soft red
    C_DANGER_H   = '#fca5a5'      # light red hover
    C_CANVAS_BG  = '#f0f1f6'      # canvas background

    def __init__(self, root, model, scaler, ocr_model=None, ocr_scaler=None):
        self.root = root
        self.model = model
        self.scaler = scaler
        self.ocr_model = ocr_model
        self.ocr_scaler = ocr_scaler

        # ── Theme & Window ──
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.root.title('Nhận diện biển số xe')
        self.root.geometry('1200x880')
        self.root.minsize(980, 720)
        self.root.configure(fg_color=self.C_BG)

        self.img_original = None
        self.img_tk = None
        self.scale_factor = 1.0
        self.rect_id = None
        self.start_x = None
        self.start_y = None
        self.fixed_confidence_threshold = 0.85

        # ═══════════════════════════════════════════
        # HEADER
        # ═══════════════════════════════════════════
        header = ctk.CTkFrame(self.root, fg_color=self.C_SURFACE, corner_radius=0, height=64,
                              border_width=0)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)

        # Soft accent gradient line at top
        accent_line = ctk.CTkFrame(header, fg_color=self.C_ACCENT, height=3, corner_radius=0)
        accent_line.pack(side=tk.TOP, fill=tk.X)

        header_inner = ctk.CTkFrame(header, fg_color='transparent')
        header_inner.pack(fill=tk.BOTH, expand=True, padx=28)

        # App icon circle
        ctk.CTkLabel(header_inner, text='  VP  ',
                     font=ctk.CTkFont(size=13, weight='bold'),
                     text_color='#ffffff',
                     fg_color=self.C_ACCENT, corner_radius=10
                     ).pack(side=tk.LEFT, padx=(0, 14), pady=14)

        ctk.CTkLabel(header_inner, text='Nhận diện biển số xe',
                     font=ctk.CTkFont(family='Segoe UI', size=20, weight='bold'),
                     text_color=self.C_TEXT).pack(side=tk.LEFT, pady=14)

        ctk.CTkLabel(header_inner, text='HOG + ANN  ·  OCR',
                     font=ctk.CTkFont(size=11),
                     text_color=self.C_TEXT2).pack(side=tk.LEFT, padx=14, pady=14)

        # Device badge
        device_text = 'CUDA' if 'cuda' in str(device) else 'CPU'
        badge_fg = self.C_SUCCESS if 'cuda' in str(device) else self.C_WARN
        badge_bg = self.C_SUCCESS_BG if 'cuda' in str(device) else '#fffbeb'
        ctk.CTkLabel(header_inner, text=f'  {device_text}  ',
                     font=ctk.CTkFont(size=10, weight='bold'),
                     text_color=badge_fg,
                     fg_color=badge_bg, corner_radius=8
                     ).pack(side=tk.RIGHT, padx=4, pady=14)

        # Header bottom border
        ctk.CTkFrame(self.root, fg_color=self.C_BORDER, height=1, corner_radius=0
                     ).pack(side=tk.TOP, fill=tk.X)

        # ═══════════════════════════════════════════
        # TOOLBAR
        # ═══════════════════════════════════════════
        toolbar_card = ctk.CTkFrame(self.root, fg_color=self.C_SURFACE,
                                    corner_radius=16, border_width=1,
                                    border_color=self.C_BORDER)
        toolbar_card.pack(side=tk.TOP, fill=tk.X, padx=24, pady=(16, 0))

        toolbar_inner = ctk.CTkFrame(toolbar_card, fg_color='transparent')
        toolbar_inner.pack(fill=tk.X, padx=18, pady=14)

        btn_font = ctk.CTkFont(family='Segoe UI', size=13, weight='bold')

        self.btn_load = ctk.CTkButton(
            toolbar_inner, text='Tải Ảnh', command=self.load_image,
            width=130, height=42,
            fg_color=self.C_BG, hover_color=self.C_BORDER,
            border_color=self.C_BORDER, border_width=1,
            font=btn_font, corner_radius=22, text_color=self.C_TEXT)
        self.btn_load.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_auto = ctk.CTkButton(
            toolbar_inner, text='Nhận Diện Tự Động', command=self.auto_detect,
            width=200, height=42,
            fg_color=self.C_ACCENT, hover_color=self.C_ACCENT_H,
            font=btn_font, corner_radius=22, text_color='#ffffff')
        self.btn_auto.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_clear = ctk.CTkButton(
            toolbar_inner, text='Xoá', command=self.clear_detections,
            width=90, height=42,
            fg_color='#fef2f2', hover_color=self.C_DANGER_H,
            border_color='#fecaca', border_width=1,
            font=btn_font, corner_radius=22, text_color=self.C_DANGER)
        self.btn_clear.pack(side=tk.LEFT, padx=(0, 20))

        # Separator dot
        ctk.CTkLabel(toolbar_inner, text='·',
                     font=ctk.CTkFont(size=20), text_color=self.C_BORDER
                     ).pack(side=tk.LEFT, padx=(0, 12))

        self.lbl_status = ctk.CTkLabel(
            toolbar_inner, text='Tải ảnh để bắt đầu  ·  Ngưỡng 85%',
            font=ctk.CTkFont(family='Segoe UI', size=12), text_color=self.C_TEXT2)
        self.lbl_status.pack(side=tk.LEFT, padx=4)

        # ═══════════════════════════════════════════
        # RESULT BANNER
        # ═══════════════════════════════════════════
        self.result_frame = ctk.CTkFrame(
            self.root, fg_color=self.C_SURFACE, corner_radius=16,
            height=76, border_width=1, border_color=self.C_BORDER)
        self.result_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=24, pady=(0, 20))
        self.result_frame.pack_propagate(False)

        result_inner = ctk.CTkFrame(self.result_frame, fg_color='transparent')
        result_inner.pack(fill=tk.BOTH, expand=True, padx=22, pady=10)

        self.lbl_result_tag = ctk.CTkLabel(
            result_inner, text='  BIỂN SỐ  ',
            font=ctk.CTkFont(size=10, weight='bold'),
            text_color=self.C_ACCENT, fg_color=self.C_ACCENT_BG, corner_radius=8)
        self.lbl_result_tag.pack(side=tk.LEFT, padx=(0, 18))

        self.lbl_result = ctk.CTkLabel(
            result_inner, text='- - -',
            font=ctk.CTkFont(family='Consolas', size=32, weight='bold'),
            text_color=self.C_ACCENT)
        self.lbl_result.pack(side=tk.LEFT, padx=4)

        # ═══════════════════════════════════════════
        # HINT
        # ═══════════════════════════════════════════
        ctk.CTkLabel(self.root,
                     text='Nhận Diện Tự Động để quét toàn bộ, hoặc kéo chuột khoanh vùng thủ công',
                     font=ctk.CTkFont(size=11), text_color=self.C_TEXT2
                     ).pack(side=tk.BOTTOM, pady=(0, 6))

        # ═══════════════════════════════════════════
        # CANVAS
        # ═══════════════════════════════════════════
        canvas_card = ctk.CTkFrame(self.root, fg_color=self.C_SURFACE,
                                   corner_radius=16, border_width=1,
                                   border_color=self.C_BORDER)
        canvas_card.pack(fill=tk.BOTH, expand=True, padx=24, pady=(12, 8))

        self.canvas = tk.Canvas(canvas_card, cursor='cross', bg=self.C_CANVAS_BG,
                                highlightthickness=1, highlightbackground=self.C_BORDER)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind('<ButtonPress-1>', self.on_button_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_button_release)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image Files', '*.jpg *.jpeg *.png *.bmp')])
        if not file_path:
            return
        try:
            self.img_original = cv2.imread(file_path)
            if self.img_original is None:
                raise ValueError('Không thể đọc file ảnh.')
            self.root.update_idletasks()
            canvas_w, canvas_h = (self.canvas.winfo_width(), self.canvas.winfo_height())
            img_h, img_w = self.img_original.shape[:2]
            self.scale_factor = min(canvas_w / img_w, canvas_h / img_h)
            new_w, new_h = (int(img_w * self.scale_factor), int(img_h * self.scale_factor))
            display_img = cv2.resize(self.img_original, (new_w, new_h))
            img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            self.img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
            self.canvas.delete('all')
            self.canvas.create_image(canvas_w / 2, canvas_h / 2, anchor=tk.CENTER, image=self.img_tk)
            self.lbl_status.configure(text='Ảnh đã tải. Sẵn sàng!', text_color=self.C_ACCENT)
            self.lbl_result.configure(text='- - -')
            self.result_frame.configure(border_color=self.C_BORDER)
            self.rect_id = None
        except Exception as e:
            messagebox.showerror('Lỗi Tải Ảnh', f'Đã xảy ra lỗi: {e}')

    def auto_detect(self):
        if self.img_original is None:
            messagebox.showwarning('Nhắc nhở', 'Bạn chưa tải ảnh kìa!')
            return
        threshold = self.fixed_confidence_threshold
        self.lbl_status.configure(text=f'Đang phân tích (ngưỡng {threshold * 100:.0f}%)...', text_color=self.C_WARN)
        self.lbl_result.configure(text='...')
        self.result_frame.configure(border_color=self.C_BORDER)
        self.root.update()
        self.canvas.delete('auto_box')
        self.canvas.delete('auto_char_box')
        img_h, img_w = self.img_original.shape[:2]
        largest_side = max(img_h, img_w)
        search_scale = min(1.0, MAX_IMAGE_SIDE_FOR_SEARCH / float(max(1, largest_side)))
        if search_scale < 1.0:
            search_img = cv2.resize(self.img_original, (int(img_w * search_scale), int(img_h * search_scale)), interpolation=cv2.INTER_AREA)
        else:
            search_img = self.img_original
        proposal_wh = _prepare_candidate_boxes(search_img)
        if len(proposal_wh) == 0:
            self.lbl_status.configure(text='Không tạo được candidate box từ ảnh.', text_color=self.C_DANGER)
            self.lbl_result.configure(text='- - -')
            return
        if search_scale < 1.0:
            inv_scale = 1.0 / search_scale
            proposal_xyxy = [[int(x * inv_scale), int(y * inv_scale), int((x + w) * inv_scale), int((y + h) * inv_scale)] for x, y, w, h in proposal_wh]
        else:
            proposal_xyxy = [[x, y, x + w, y + h] for x, y, w, h in proposal_wh]
        filtered_xyxy = []
        for x1, y1, x2, y2 in proposal_xyxy:
            x1 = max(0, min(img_w - 1, x1))
            y1 = max(0, min(img_h - 1, y1))
            x2 = max(0, min(img_w, x2))
            y2 = max(0, min(img_h, y2))
            if x2 - x1 < 30 or y2 - y1 < 12:
                continue
            roi = self.img_original[y1:y2, x1:x2]
            if is_plate_candidate(roi):
                filtered_xyxy.append([x1, y1, x2, y2])
        if not filtered_xyxy:
            self.lbl_status.configure(text='Không còn candidate hợp lệ sau bước lọc.', text_color=self.C_DANGER)
            self.lbl_result.configure(text='- - -')
            return
        model_boxes, model_scores = _batch_predict_boxes(self.model, self.scaler, self.img_original, filtered_xyxy)
        detected_boxes = []
        detected_scores = []
        if model_boxes:
            dynamic_threshold = max(0.7, float(threshold) - 0.15)
            for i, box in enumerate(model_boxes):
                score = model_scores[i]
                if score >= threshold:
                    detected_boxes.append(box)
                    detected_scores.append(score)
            if not detected_boxes:
                for i, box in enumerate(model_boxes):
                    score = model_scores[i]
                    if score >= dynamic_threshold:
                        detected_boxes.append(box)
                        detected_scores.append(score)
        if len(detected_boxes) > 0:
            final_boxes, final_scores = non_max_suppression_scored(detected_boxes, detected_scores, overlapThresh=0.2)
            final_boxes_list, final_scores_list = post_nms_filter(self.img_original, [(b[0], b[1], b[2], b[3]) for b in final_boxes], list(final_scores))
            final_boxes = np.array(final_boxes_list) if final_boxes_list else np.array([])
            final_scores = np.array(final_scores_list)
            if len(final_boxes) > 0:
                final_boxes, final_scores = suppress_landscape_vehicle_false_positives(self.img_original, final_boxes, final_scores)
        else:
            final_boxes, final_scores = (np.array([]), np.array([]))
        if len(final_boxes) > MAX_DETECTIONS:
            sorted_indices = np.argsort(final_scores)[::-1][:MAX_DETECTIONS]
            final_boxes = final_boxes[sorted_indices]
            final_scores = final_scores[sorted_indices]
        if len(final_boxes) == 0:
            self.lbl_status.configure(text=f'Không tìm thấy biển số nào (ngưỡng {threshold * 100:.0f}%).', text_color=self.C_DANGER)
            self.lbl_result.configure(text='- - -')
            return
        canvas_w, canvas_h = (self.canvas.winfo_width(), self.canvas.winfo_height())
        img_start_x = (canvas_w - self.img_tk.width()) / 2
        img_start_y = (canvas_h - self.img_tk.height()) / 2
        img_h, img_w = self.img_original.shape[:2]
        all_plate_texts = []
        verified_results = []
        for i, (x1, y1, x2, y2) in enumerate(final_boxes):
            score = final_scores[i]
            pad_x = int((x2 - x1) * 0.04)
            pad_y = int((y2 - y1) * 0.04)
            px1 = max(0, x1 - pad_x)
            py1 = max(0, y1 - pad_y)
            px2 = min(img_w, x2 + pad_x)
            py2 = min(img_h, y2 + pad_y)
            plate_roi = self.img_original[py1:py2, px1:px2]
            plate_text, char_confs, char_bboxes = read_plate_text(self.ocr_model, self.ocr_scaler, plate_roi)
            clean_text = plate_text.replace('?', '').replace('-', '').replace(' ', '').replace('.', '')
            has_text = len(clean_text) >= 4
            is_valid_plate = False
            if has_text and char_confs:
                avg_conf = np.mean(char_confs)
                n_question = plate_text.count('?')
                has_structure = len(clean_text) >= 4 and clean_text[0].isdigit() and clean_text[1].isdigit() and any((c.isalpha() for c in clean_text[2:4]))
                if avg_conf >= 0.4 and n_question <= 2:
                    if has_structure or avg_conf >= 0.6:
                        is_valid_plate = True
            if is_valid_plate:
                verified_results.append((x1, y1, x2, y2, score, plate_text, char_confs, char_bboxes, px1, py1))
            elif score >= 0.97 and has_text:
                verified_results.append((x1, y1, x2, y2, score, plate_text, char_confs, char_bboxes, px1, py1))
        if len(verified_results) == 0:
            self.lbl_status.configure(text='Không tìm thấy biển số hợp lệ nào.', text_color=self.C_DANGER)
            self.lbl_result.configure(text='- - -')
            return
        for x1, y1, x2, y2, score, plate_text, char_confs, char_bboxes, px1, py1 in verified_results:
            has_text = len(plate_text.replace('?', '').replace('-', '').replace(' ', '').replace('.', '')) >= 4
            if score >= 0.98:
                color = self.C_ACCENT
            elif score >= 0.95:
                color = self.C_SUCCESS
            else:
                color = self.C_WARN
            c_x1 = x1 * self.scale_factor + img_start_x
            c_y1 = y1 * self.scale_factor + img_start_y
            c_x2 = x2 * self.scale_factor + img_start_x
            c_y2 = y2 * self.scale_factor + img_start_y
            self.canvas.create_rectangle(c_x1, c_y1, c_x2, c_y2, outline=color, width=3, tags='auto_box')
            if has_text:
                label_text = f' {plate_text} ({score * 100:.1f}%) '
                all_plate_texts.append(plate_text)
                for cbx, cby, cbw, cbh in char_bboxes:
                    char_orig_x1 = px1 + cbx
                    char_orig_y1 = py1 + cby
                    char_orig_x2 = px1 + cbx + cbw
                    char_orig_y2 = py1 + cby + cbh
                    char_canvas_x1 = char_orig_x1 * self.scale_factor + img_start_x
                    char_canvas_y1 = char_orig_y1 * self.scale_factor + img_start_y
                    char_canvas_x2 = char_orig_x2 * self.scale_factor + img_start_x
                    char_canvas_y2 = char_orig_y2 * self.scale_factor + img_start_y
                    self.canvas.create_rectangle(char_canvas_x1, char_canvas_y1, char_canvas_x2, char_canvas_y2, outline='yellow', width=1, tags='auto_char_box')
            else:
                label_text = f' Biển Số ({score * 100:.1f}%) '
            if c_y1 < 25:
                text_y = c_y1 + 2
                text_anchor = tk.NW
            else:
                text_y = c_y1 - 2
                text_anchor = tk.SW
            t_id = self.canvas.create_text(c_x1, text_y, text=label_text, fill='white', font=('Arial', 11, 'bold'), anchor=text_anchor, tags='auto_box')
            t_bbox = self.canvas.bbox(t_id)
            bg_id = self.canvas.create_rectangle(t_bbox, fill=color, outline='white', tags='auto_box')
            self.canvas.tag_lower(bg_id, t_id)
        if all_plate_texts:
            res_str = ' | '.join(all_plate_texts)
            self.lbl_status.configure(text=f'Tìm được {len(verified_results)} biển số', text_color=self.C_SUCCESS)
            self.lbl_result.configure(text=res_str)
            self.result_frame.configure(border_color=self.C_ACCENT)
        else:
            self.lbl_status.configure(text=f'Tìm được {len(verified_results)} biển số', text_color=self.C_SUCCESS)
            self.lbl_result.configure(text='- - -')

    def clear_detections(self):
        if self.img_original is None:
            return
        self.canvas.delete('auto_box')
        self.canvas.delete('auto_char_box')
        self.canvas.delete('manual_char_box')
        if self.rect_id:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
        self.lbl_status.configure(text='Đã xoá các vùng nhận diện.', text_color=self.C_TEXT2)
        self.lbl_result.configure(text='- - -')
        self.result_frame.configure(border_color=self.C_BORDER)

    def on_button_press(self, event):
        if self.img_original is None:
            return
        self.start_x, self.start_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='cyan', width=2)

    def on_mouse_drag(self, event):
        if not self.rect_id:
            return
        cur_x, cur_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        if not self.rect_id:
            return
        self.canvas.delete('manual_char_box')
        canvas_w, canvas_h = (self.canvas.winfo_width(), self.canvas.winfo_height())
        img_start_x = (canvas_w - self.img_tk.width()) / 2
        img_start_y = (canvas_h - self.img_tk.height()) / 2
        x1_c, y1_c, x2_c, y2_c = self.canvas.coords(self.rect_id)
        orig_x1 = int((min(x1_c, x2_c) - img_start_x) / self.scale_factor)
        orig_y1 = int((min(y1_c, y2_c) - img_start_y) / self.scale_factor)
        orig_x2 = int((max(x1_c, x2_c) - img_start_x) / self.scale_factor)
        orig_y2 = int((max(y1_c, y2_c) - img_start_y) / self.scale_factor)
        h, w = self.img_original.shape[:2]
        orig_x1, orig_y1 = (max(0, orig_x1), max(0, orig_y1))
        orig_x2, orig_y2 = (min(w, orig_x2), min(h, orig_y2))
        if orig_x2 - orig_x1 < 10 or orig_y2 - orig_y1 < 10:
            self.lbl_status.configure(text='Vùng chọn quá nhỏ!', text_color=self.C_WARN)
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        img_crop = self.img_original[orig_y1:orig_y2, orig_x1:orig_x2]
        prediction, confidence = predict_plate_with_confidence(self.model, self.scaler, img_crop)
        plate_text, char_confs, char_bboxes = read_plate_text(self.ocr_model, self.ocr_scaler, img_crop)
        if prediction == 1:
            if plate_text:
                avg_ocr = np.mean(char_confs) if char_confs else 0
                self.lbl_status.configure(text=f'BIỂN SỐ  det={confidence * 100:.1f}%  ocr={avg_ocr * 100:.1f}%', text_color=self.C_SUCCESS)
                self.lbl_result.configure(text=plate_text)
                self.result_frame.configure(border_color=self.C_ACCENT)
                if char_bboxes:
                    for cbx, cby, cbw, cbh in char_bboxes:
                        char_orig_x1 = orig_x1 + cbx
                        char_orig_y1 = orig_y1 + cby
                        char_orig_x2 = orig_x1 + cbx + cbw
                        char_orig_y2 = orig_y1 + cby + cbh
                        char_canvas_x1 = char_orig_x1 * self.scale_factor + img_start_x
                        char_canvas_y1 = char_orig_y1 * self.scale_factor + img_start_y
                        char_canvas_x2 = char_orig_x2 * self.scale_factor + img_start_x
                        char_canvas_y2 = char_orig_y2 * self.scale_factor + img_start_y
                        self.canvas.create_rectangle(char_canvas_x1, char_canvas_y1, char_canvas_x2, char_canvas_y2, outline='yellow', width=1, tags='manual_char_box')
                label_text = f' {plate_text} '
            else:
                self.lbl_status.configure(text=f'BIỂN SỐ (Confidence: {confidence * 100:.1f}%) - chưa đọc được ký tự', text_color=self.C_SUCCESS)
                self.lbl_result.configure(text='(?)')
                self.result_frame.configure(border_color=self.C_ACCENT)
                label_text = ' Biển Số '
            self.canvas.itemconfig(self.rect_id, outline='green')
            c_x1, c_y1, _, _ = self.canvas.coords(self.rect_id)
            if c_y1 < 25:
                text_y, text_anchor = (c_y1 + 2, tk.NW)
            else:
                text_y, text_anchor = (c_y1 - 2, tk.SW)
            t_id = self.canvas.create_text(c_x1, text_y, text=label_text, fill='white', font=('Arial', 12, 'bold'), anchor=text_anchor, tags='manual_char_box')
            t_bbox = self.canvas.bbox(t_id)
            bg_id = self.canvas.create_rectangle(t_bbox, fill='green', outline='white', tags='manual_char_box')
            self.canvas.tag_lower(bg_id, t_id)
        else:
            self.lbl_status.configure(text=f'NỀN/RÁC (Confidence: {confidence * 100:.1f}%)', text_color=self.C_DANGER)
            self.lbl_result.configure(text='- - -')
            self.result_frame.configure(border_color=self.C_BORDER)
            self.canvas.itemconfig(self.rect_id, outline='red')
if __name__ == '__main__':
    try:
        print(f'⚙️ Đang chạy trên: {device}')
        print('💾 Đang tải model phát hiện biển số (plate detector)... Vui lòng chờ.')
        model, scaler = load_model_and_scaler()
        print('✅ Plate detector loaded.')
        print('💾 Đang tải model nhận diện ký tự (OCR)... Vui lòng chờ.')
        ocr_model, ocr_scaler = load_ocr_model()
        if ocr_model is not None:
            print('✅ OCR model loaded.')
        else:
            print('⚠️ OCR model không khả dụng — chỉ phát hiện biển số, không đọc ký tự.')
        print('🎨 Đang khởi tạo giao diện người dùng (GUI)...')
        root = ctk.CTk()
        app = PlateDetectorGUI(root, model, scaler, ocr_model, ocr_scaler)
        print('✅ Giao diện đã sẵn sàng.')
        root.mainloop()
    except Exception as e:
        import traceback
        print('❌ Đã xảy ra lỗi nghiêm trọng khi khởi động:')
        traceback.print_exc()
        temp_root = tk.Tk()
        temp_root.withdraw()
        messagebox.showerror('Lỗi Khởi Động', f'Đã xảy ra lỗi:\n\n{e}\n\nChi tiết đã được in ra console. Vui lòng kiểm tra.')