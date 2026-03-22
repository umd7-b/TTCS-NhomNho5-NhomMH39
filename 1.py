"""
BÁO CÁO TUẦN 5 - THỰC TẬP CƠ SỞ
Đề tài: Thuật toán ANN để nhận diện biển số xe
Nhóm 39 | PTIT - Khoa CNTT1

BƯỚC 1: Phân loại có phải biển số xe Việt Nam không
  - Label 1 = Có biển số xe (crop đúng vùng biển số)
  - Label 0 = Không phải biển số xe (crop ngẫu nhiên vùng khác)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import json
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ============================================================
# BƯỚC 1: TIỀN XỬ LÝ ẢNH
# ============================================================

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    return img, gray, blurred, binary, contours


def visualize_preprocessing(image_path):
    img, gray, blurred, binary, _ = preprocess_image(image_path)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); axes[0].set_title("Ảnh gốc (RGB)")
    axes[1].imshow(gray,    cmap='gray');                 axes[1].set_title("Grayscale")
    axes[2].imshow(blurred, cmap='gray');                 axes[2].set_title("Gaussian Blur")
    axes[3].imshow(binary,  cmap='gray');                 axes[3].set_title("Threshold (Otsu)")
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    plt.savefig("preprocessing_steps.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu: preprocessing_steps.png")


# ============================================================
# BƯỚC 2: CROP & CHUẨN HÓA
# ============================================================

IMG_WIDTH  = 64
IMG_HEIGHT = 64

def crop_license_plate(image_path, bbox):
    img = cv2.imread(image_path)
    x, y, w, h = [int(v) for v in bbox]
    pad = 4
    x1 = max(0, x - pad);  y1 = max(0, y - pad)
    x2 = min(img.shape[1], x + w + pad)
    y2 = min(img.shape[0], y + h + pad)
    return img[y1:y2, x1:x2]


def normalize_plate(plate_crop):
    h, w   = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return None, None
    scale  = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w, new_h = int(max(1, w * scale)), int(max(1, h * scale))
    resized = cv2.resize(plate_crop, (new_w, new_h))
    canvas  = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    x_off   = (IMG_WIDTH  - new_w) // 2
    y_off   = (IMG_HEIGHT - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    gray       = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    normalized = gray.astype(np.float32) / 255.0
    return normalized.flatten(), normalized


def augment_image(img):
    """Augment ảnh: Chỉ tạo 2 bản (Gốc và Đảo màu) để train cho lẹ!"""
    augments = [img]

    # Đảo màu (Invert Colors) -> Giúp model học nhận diện Biển Xanh/Đỏ
    inverted = cv2.bitwise_not(img)
    augments.append(inverted)
    
    return augments


# ============================================================
# BƯỚC 2 MỚI: LOAD DATASET NHỊ PHÂN
# Label 1 = Có biển số xe VN
# Label 0 = Không phải biển số xe
# ============================================================

def generate_negative_crop(img, plate_bbox, num_neg=2):
    h_img, w_img = img.shape[:2]
    px, py, pw, ph = [int(v) for v in plate_bbox]
    results = []
    attempts = 0

    # Đảm bảo kích thước crop hợp lệ
    cw_min = 20
    cw_max = max(cw_min + 1, min(pw * 2, w_img // 2))
    ch_min = 10
    ch_max = max(ch_min + 1, min(ph * 2, h_img // 2))

    while len(results) < num_neg and attempts < 50:
        attempts += 1
        cw = random.randint(cw_min, cw_max)
        ch = random.randint(ch_min, ch_max)

        # Đảm bảo vị trí crop không vượt khỏi ảnh
        cx_max = max(0, w_img - cw)
        cy_max = max(0, h_img - ch)
        if cx_max == 0 or cy_max == 0:
            continue

        cx = random.randint(0, cx_max)
        cy = random.randint(0, cy_max)

        # Kiểm tra không overlap với biển số
        no_overlap_x = (cx + cw < px) or (cx > px + pw)
        no_overlap_y = (cy + ch < py) or (cy > py + ph)

        if no_overlap_x or no_overlap_y:
            crop = img[cy:cy+ch, cx:cx+cw]
            if crop.size > 0:
                results.append(crop)

    return results


def load_dataset_binary(img_dir, annotation_file, augment=False):
    """
    Load dataset nhị phân:
      label 1 = crop đúng vùng biển số xe VN  → POSITIVE
      label 0 = crop ngẫu nhiên vùng khác     → NEGATIVE
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    # Chỉ lấy annotation biển số (category 3 và 4)
    PLATE_CATS = {3, 4}
    image_map  = {img['id']: img['file_name'] for img in coco['images']}
    class_names = ['khong_phai_bsx', 'co_bsx']

    X, y, skipped = [], [], 0

    for ann in coco['annotations']:
        cat_id    = ann['category_id']
        if cat_id not in PLATE_CATS:
            continue

        file_name  = image_map.get(ann['image_id'])
        image_path = os.path.join(img_dir, file_name) if file_name else None
        if not image_path or not os.path.exists(image_path):
            skipped += 1
            continue

        img = cv2.imread(image_path)
        if img is None:
            skipped += 1
            continue

        # ✅ POSITIVE: crop vùng biển số
        plate_crop = crop_license_plate(image_path, ann['bbox'])
        if plate_crop.size == 0:
            skipped += 1
            continue

        pos_samples = augment_image(plate_crop) if augment else [plate_crop]
        for s in pos_samples:
            flat, _ = normalize_plate(s)
            if flat is not None:
                X.append(flat)
                y.append(1)

        # ❌ NEGATIVE: crop vùng ngẫu nhiên không phải biển số (Chỉ sinh 1ảnh và KHÔNG AUGMENT để giảm bộ nhớ chạy)
        neg_crops = generate_negative_crop(img, ann['bbox'], num_neg=1)
        for neg in neg_crops:
            # Luôn không augment ảnh negative để giảm tải
            neg_samples = [neg]
            for s in neg_samples:
                flat, _ = normalize_plate(s)
                if flat is not None:
                    X.append(flat)
                    y.append(0)

    pos_count = sum(y)
    neg_count = len(y) - pos_count
    print(f"Loaded: {len(X)} samples | "
          f"Positive (BSX): {pos_count} | "
          f"Negative: {neg_count} | "
          f"Skipped: {skipped}")
    return np.array(X), np.array(y), class_names


# ============================================================
# BƯỚC 3: XÂY DỰNG ANN BẰNG NUMPY THUẦN
# ============================================================

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def to_categorical(y, num_classes):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

def build_ann_model(input_dim, num_classes):
    """
    Kiến trúc: input → 256 → 128 → 64 → 2
    output: [P(không phải BSX), P(có BSX)]
    """
    layer_sizes = [input_dim, 256, 128, 64, num_classes]
    params = {}
    for l in range(1, len(layer_sizes)):
        n_in  = layer_sizes[l - 1]
        n_out = layer_sizes[l]
        params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        params[f'b{l}'] = np.zeros((1, n_out))
    num_layers = len(layer_sizes) - 1
    print(f"ANN_BienSoXe: {layer_sizes}")
    print(f"Tổng số layers: {num_layers}")
    return params, num_layers


# ============================================================
# BƯỚC 4: HUẤN LUYỆN
# ============================================================

def forward_pass(X, params, num_layers):
    cache = {'A0': X}
    A = X
    for l in range(1, num_layers + 1):
        Z = A @ params[f'W{l}'] + params[f'b{l}']
        cache[f'Z{l}'] = Z
        A = softmax(Z) if l == num_layers else relu(Z)
        cache[f'A{l}'] = A
    return A, cache


def compute_loss(y_pred, y_true):
    m = y_true.shape[0]
    y_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.sum(y_true * np.log(y_clip)) / m


def backward_pass(y_true, params, cache, num_layers):
    grads = {}
    m  = y_true.shape[0]
    dA = cache[f'A{num_layers}'] - y_true
    for l in reversed(range(1, num_layers + 1)):
        A_prev = cache[f'A{l-1}']
        grads[f'dW{l}'] = (A_prev.T @ dA) / m
        grads[f'db{l}'] = np.sum(dA, axis=0, keepdims=True) / m
        if l > 1:
            dA = (dA @ params[f'W{l}'].T) * relu_grad(cache[f'Z{l-1}'])
    return grads


def init_adam(params):
    adam = {}
    for key in params:
        adam[f'm_{key}'] = np.zeros_like(params[key])
        adam[f'v_{key}'] = np.zeros_like(params[key])
    return adam


def update_adam(params, grads, adam, t,
                lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    num_layers = len([k for k in params if k.startswith('W')])
    for l in range(1, num_layers + 1):
        for p in ['W', 'b']:
            key = f'{p}{l}'; g = grads[f'd{p}{l}']
            adam[f'm_{key}'] = beta1 * adam[f'm_{key}'] + (1 - beta1) * g
            adam[f'v_{key}'] = beta2 * adam[f'v_{key}'] + (1 - beta2) * g**2
            m_hat = adam[f'm_{key}'] / (1 - beta1**t)
            v_hat = adam[f'v_{key}'] / (1 - beta2**t)
            params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return params, adam


def train_model(params, num_layers, X_train, y_train, X_val, y_val,
                num_classes, epochs=50, batch_size=32):
    y_train_cat = to_categorical(y_train, num_classes)
    y_val_cat   = to_categorical(y_val,   num_classes)

    adam    = init_adam(params)
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    best_val_loss = float('inf')
    best_params   = None
    wait = 0; lr_wait = 0; t = 0
    lr   = 0.001; min_lr = 1e-6

    for epoch in range(epochs):
        idx  = np.random.permutation(len(X_train))
        X_s, y_s = X_train[idx], y_train_cat[idx]

        for start in range(0, len(X_s), batch_size):
            Xb = X_s[start:start+batch_size]
            yb = y_s[start:start+batch_size]
            t += 1
            y_pred, cache = forward_pass(Xb, params, num_layers)
            grads = backward_pass(yb, params, cache, num_layers)
            params, adam = update_adam(params, grads, adam, t, lr)

        yp_tr, _ = forward_pass(X_train, params, num_layers)
        yp_vl, _ = forward_pass(X_val,   params, num_layers)
        tr_loss  = compute_loss(yp_tr, y_train_cat)
        vl_loss  = compute_loss(yp_vl, y_val_cat)
        tr_acc   = np.mean(np.argmax(yp_tr, 1) == y_train)
        vl_acc   = np.mean(np.argmax(yp_vl, 1) == y_val)

        history['loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['accuracy'].append(tr_acc)
        history['val_accuracy'].append(vl_acc)

        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"loss: {tr_loss:.4f} | acc: {tr_acc:.4f} | "
              f"val_loss: {vl_loss:.4f} | val_acc: {vl_acc:.4f} | "
              f"lr: {lr:.6f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_params   = {k: v.copy() for k, v in params.items()}
            np.save("best_model.npy", best_params)
            wait = 0; lr_wait = 0
            print(f"  >> Lưu model tốt nhất (val_loss: {vl_loss:.4f})")
        else:
            wait    += 1
            lr_wait += 1
            if lr_wait >= 5:
                lr = max(lr * 0.5, min_lr)
                lr_wait = 0
                print(f"  >> Giảm learning rate: {lr:.6f}")
            if wait >= 10:
                print(f"\nEarly stopping tại epoch {epoch+1}")
                break

    return best_params, history


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['accuracy'],     label='Train Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Val Accuracy',   linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history['loss'],     label='Train Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Val Loss',   linewidth=2)
    ax2.set_title('Model Loss', fontsize=14)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu: training_history.png")


# ============================================================
# BƯỚC 5: ĐÁNH GIÁ
# ============================================================

def evaluate_model(params, num_layers, X_test, y_test,
                   num_classes, class_names):
    y_test_cat      = to_categorical(y_test, num_classes)
    y_pred_probs, _ = forward_pass(X_test, params, num_layers)
    y_pred          = np.argmax(y_pred_probs, axis=1)
    test_loss       = compute_loss(y_pred_probs, y_test_cat)
    test_accuracy   = np.mean(y_pred == y_test)

    print(f"\n{'='*50}")
    print(f"Test Loss     : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*50}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Nhận diện biển số xe', fontsize=14)
    plt.ylabel('Nhãn thực tế'); plt.xlabel('Nhãn dự đoán')
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Đã lưu: confusion_matrix.png")
    return test_accuracy, y_pred


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    TRAIN_DIR = "dataset/train"; TRAIN_ANN = "dataset/train/_annotations.coco.json"
    VALID_DIR = "dataset/valid"; VALID_ANN = "dataset/valid/_annotations.coco.json"
    TEST_DIR  = "dataset/test";  TEST_ANN  = "dataset/test/_annotations.coco.json"

    print("=" * 60)
    print("BƯỚC 1: PHÂN LOẠI CÓ/KHÔNG PHẢI BIỂN SỐ XE VN")
    print("=" * 60)

    print("\n[Bước 2] Load dataset nhị phân (augment train)...")
    X_train, y_train, class_names = load_dataset_binary(TRAIN_DIR, TRAIN_ANN, augment=True)
    X_val,   y_val,   _           = load_dataset_binary(VALID_DIR, VALID_ANN, augment=False)
    X_test,  y_test,  _           = load_dataset_binary(TEST_DIR,  TEST_ANN,  augment=False)

    num_classes = 2          # 0: không phải BSX, 1: có BSX
    input_dim   = X_train.shape[1]
    print(f"\nTrain: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    print("\n[Bước 3] Xây dựng ANN (input → 256 → 128 → 64 → 2)...")
    params, num_layers = build_ann_model(input_dim, num_classes)

    print("\n[Bước 4] Bắt đầu huấn luyện...")
    best_params, history = train_model(
        params, num_layers,
        X_train, y_train,
        X_val,   y_val,
        num_classes, epochs=50, batch_size=32
    )
    plot_training_history(history)

    print("\n[Bước 5] Đánh giá trên tập Test...")
    accuracy, y_pred = evaluate_model(
        best_params, num_layers,
        X_test, y_test, num_classes, class_names
    )

    np.save("ann_biensoxe_b1.npy", best_params)
    print("\nĐã lưu model: ann_biensoxe_b1.npy")
    print("Hoàn thành Bước 1!")