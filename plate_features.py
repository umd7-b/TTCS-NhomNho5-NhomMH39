import cv2
import numpy as np
import random
from plate_settings import IMG_SIZE_RECT, IMG_SIZE_SQ, IMG_SIZE_OCR, ASPECT_RATIO_THRESHOLD, MAX_HOG_LEN
_hog_rect = cv2.HOGDescriptor((IMG_SIZE_RECT[0], IMG_SIZE_RECT[1]), (16, 16), (8, 8), (8, 8), 9)
_hog_sq = cv2.HOGDescriptor((IMG_SIZE_SQ[0], IMG_SIZE_SQ[1]), (16, 16), (8, 8), (8, 8), 9)
_hog_ocr = cv2.HOGDescriptor((IMG_SIZE_OCR, IMG_SIZE_OCR), (16, 16), (8, 8), (8, 8), 9)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
HOG_CHAR_LEN = _hog_ocr.getDescriptorSize()

def infer_plate_type(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return 'rect'
    h, w = img_bgr.shape[:2]
    aspect = w / float(max(1, h))
    return 'sq' if aspect < ASPECT_RATIO_THRESHOLD else 'rect'

def extract_hog_features(img_bgr, plate_type='rect'):
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.shape[0] < 5 or img_bgr.shape[1] < 5:
        return None
    is_sq = plate_type == 'sq'
    target_size = IMG_SIZE_SQ if is_sq else IMG_SIZE_RECT
    resized = cv2.resize(img_bgr, target_size)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    hog_obj = _hog_sq if is_sq else _hog_rect
    return hog_obj.compute(gray).flatten()

def pad_plate_feature(features):
    if features is None:
        return np.zeros(MAX_HOG_LEN, dtype=np.float32)
    feat_len = len(features)
    if feat_len == MAX_HOG_LEN:
        return features
    if feat_len > MAX_HOG_LEN:
        return features[:MAX_HOG_LEN]
    padded = np.zeros(MAX_HOG_LEN, dtype=np.float32)
    padded[:feat_len] = features
    return padded

def extract_hog_char(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if h < 5 or w < 5:
        return None
    scale = min(IMG_SIZE_OCR / w, IMG_SIZE_OCR / h)
    nw, nh = (max(1, int(w * scale)), max(1, int(h * scale)))
    canvas = np.zeros((IMG_SIZE_OCR, IMG_SIZE_OCR, 3), dtype=np.uint8)
    rsz = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    xo = (IMG_SIZE_OCR - nw) // 2
    yo = (IMG_SIZE_OCR - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = rsz
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return _hog_ocr.compute(gray).flatten()