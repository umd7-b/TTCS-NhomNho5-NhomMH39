"""
BƯỚC 2: NHẬN DIỆN KÝ TỰ TRÊN BIỂN SỐ XE (OCR)
  - Đầu vào: Ảnh từng ký tự đã được cắt từ biển số (thư mục chars/)
  - Đầu ra : Dự đoán ký tự (0-9, A-Z trừ I,J,O,Q,R,W = 30 class)
  - Sử dụng ANN thuần NumPy + HOG features (giống Bước 1)
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# ========== CẤU HÌNH ==========
IMG_SIZE   = 64          # Kích thước ảnh resize (64x64)
NUM_CLASSES = 30         # 0-9 (10) + 20 chữ cái biển số VN

# Mapping 30 class ký tự biển số Việt Nam
CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','K','L',
    'M','N','P','S','T','U','V','X','Y','Z'
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# ========== XỬ LÝ ẢNH ==========
def load_and_preprocess(image_path):
    """Đọc ảnh, resize, trích xuất HOG features"""
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Resize giữ tỉ lệ + padding
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w = int(max(1, w * scale))
    new_h = int(max(1, h * scale))
    resized = cv2.resize(img, (new_w, new_h))
    
    # Đặt vào canvas đen 64x64
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    # Chuyển grayscale + equalizeHist
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    # Trích xuất HOG features
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(gray).flatten()
    
    return features

def augment_char_image(img):
    """Augment ảnh ký tự: noise, sáng/tối, blur, đảo màu"""
    augments = [img]
    
    # Gaussian noise
    noise = np.random.normal(0, 15, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augments.append(noisy)
    
    # Tối đi
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_dark = hsv.copy()
    hsv_dark[:,:,2] = np.clip(hsv_dark[:,:,2] * 0.7, 0, 255)
    augments.append(cv2.cvtColor(hsv_dark.astype(np.uint8), cv2.COLOR_HSV2BGR))
    
    # Sáng lên
    hsv_bright = hsv.copy()
    hsv_bright[:,:,2] = np.clip(hsv_bright[:,:,2] * 1.3, 0, 255)
    augments.append(cv2.cvtColor(hsv_bright.astype(np.uint8), cv2.COLOR_HSV2BGR))
    
    # Blur nhẹ
    augments.append(cv2.GaussianBlur(img, (3, 3), 0))
    
    # Đảo màu (giúp nhận diện biển xanh/đỏ)
    augments.append(cv2.bitwise_not(img))
    
    return augments

def preprocess_augmented(img_bgr):
    """Preprocess một ảnh BGR đã augment thành HOG features"""
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return None
    
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    new_w = int(max(1, w * scale))
    new_h = int(max(1, h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h))
    
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    x_off = (IMG_SIZE - new_w) // 2
    y_off = (IMG_SIZE - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(gray).flatten()
    return features

# ========== LOAD DATASET ==========
def load_char_dataset(data_dir, augment=False, max_per_class=None):
    """
    Load dataset ký tự từ thư mục chars/train hoặc chars/val.
    Cấu trúc: data_dir/0/, data_dir/1/, ..., data_dir/Z/
    """
    X, y = [], []
    class_counts = {}
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"  ⚠️ Không tìm thấy thư mục: {class_dir}")
            continue
        
        class_idx = CLASS_TO_IDX[class_name]
        files = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if max_per_class and len(files) > max_per_class:
            files = files[:max_per_class]
        
        count = 0
        for fname in files:
            fpath = os.path.join(class_dir, fname)
            
            if augment:
                # Đọc ảnh gốc và augment
                img = cv2.imread(fpath)
                if img is None:
                    continue
                aug_imgs = augment_char_image(img)
                for aug_img in aug_imgs:
                    feat = preprocess_augmented(aug_img)
                    if feat is not None:
                        X.append(feat)
                        y.append(class_idx)
                        count += 1
            else:
                feat = load_and_preprocess(fpath)
                if feat is not None:
                    X.append(feat)
                    y.append(class_idx)
                    count += 1
        
        class_counts[class_name] = count
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"\n📊 Loaded {len(X)} samples từ {data_dir}")
    print(f"   Phân bố theo class:")
    for name in CLASS_NAMES:
        c = class_counts.get(name, 0)
        bar = '█' * min(50, c // 50)
        print(f"   {name:>2s}: {c:>5d} {bar}")
    
    return X, y

# ========== MÔ HÌNH ANN ==========
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

def build_ocr_model(input_dim, num_classes):
    """
    Xây dựng ANN cho OCR ký tự:
    input → 512 → 256 → 128 → 30 (output)
    Mạng sâu hơn vì 30 class phức tạp hơn 2 class
    """
    layer_sizes = [input_dim, 512, 256, 128, num_classes]
    params = {}
    for l in range(1, len(layer_sizes)):
        n_in  = layer_sizes[l - 1]
        n_out = layer_sizes[l]
        # He initialization
        params[f'W{l}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
        params[f'b{l}'] = np.zeros((1, n_out))
    
    num_layers = len(layer_sizes) - 1
    print(f"\n🧠 ANN_OCR model: {layer_sizes}")
    print(f"   Tổng số layers: {num_layers}")
    total_params = sum(p.size for p in params.values())
    print(f"   Tổng số parameters: {total_params:,}")
    return params, num_layers

def forward_pass(X, params, num_layers):
    cache = {'A0': X}
    A = X
    for l in range(1, num_layers + 1):
        Z = A @ params[f'W{l}'] + params[f'b{l}']
        cache[f'Z{l}'] = Z
        A = softmax(Z) if l == num_layers else relu(Z)
        cache[f'A{l}'] = A
    return A, cache

def compute_loss(y_pred, y_true, params=None, lambda_reg=0.01):
    m = y_true.shape[0]
    y_clip = np.clip(y_pred, 1e-15, 1 - 1e-15)
    loss = -np.sum(y_true * np.log(y_clip)) / m
    
    if params is not None:
        l2_cost = 0
        num_layers = len([k for k in params if k.startswith('W')])
        for l in range(1, num_layers + 1):
            l2_cost += np.sum(np.square(params[f'W{l}']))
        loss += (lambda_reg / (2 * m)) * l2_cost
    
    return loss

def backward_pass(y_true, params, cache, num_layers, lambda_reg=0.01):
    grads = {}
    m  = y_true.shape[0]
    dA = cache[f'A{num_layers}'] - y_true
    for l in reversed(range(1, num_layers + 1)):
        A_prev = cache[f'A{l-1}']
        grads[f'dW{l}'] = (A_prev.T @ dA) / m + (lambda_reg / m) * params[f'W{l}']
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

# ========== HUẤN LUYỆN ==========
def train_ocr_model(params, num_layers, X_train, y_train, X_val, y_val,
                    epochs=80, batch_size=64):
    """Huấn luyện model OCR ký tự với Adam optimizer"""
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_val_cat   = to_categorical(y_val,   NUM_CLASSES)
    
    adam    = init_adam(params)
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
    
    best_val_loss = float('inf')
    best_val_acc  = 0.0
    best_params   = None
    wait = 0
    lr_wait = 0
    t = 0
    lr = 0.001
    min_lr = 1e-6
    
    print(f"\n{'='*70}")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Acc':>7} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>10}")
    print(f"{'='*70}")
    
    for epoch in range(epochs):
        # Shuffle training data
        idx = np.random.permutation(len(X_train))
        X_shuffled = X_train[idx]
        y_shuffled = y_train_cat[idx]
        
        # Mini-batch training
        for start in range(0, len(X_shuffled), batch_size):
            Xb = X_shuffled[start:start+batch_size]
            yb = y_shuffled[start:start+batch_size]
            t += 1
            y_pred, cache = forward_pass(Xb, params, num_layers)
            grads = backward_pass(yb, params, cache, num_layers)
            params, adam = update_adam(params, grads, adam, t, lr)
        
        # Tính metrics
        yp_train, _ = forward_pass(X_train, params, num_layers)
        yp_val, _   = forward_pass(X_val,   params, num_layers)
        
        train_loss = compute_loss(yp_train, y_train_cat, params=params, lambda_reg=0.01)
        val_loss   = compute_loss(yp_val,   y_val_cat,   params=params, lambda_reg=0.01)
        train_acc  = np.mean(np.argmax(yp_train, 1) == y_train)
        val_acc    = np.mean(np.argmax(yp_val, 1)   == y_val)
        
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        
        # In kết quả
        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_params   = {k: v.copy() for k, v in params.items()}
            np.save("ocr_model_best.npy", best_params)
            wait = 0
            lr_wait = 0
            marker = " ✅ BEST"
        else:
            wait += 1
            lr_wait += 1
            if lr_wait >= 5:
                lr = max(lr * 0.5, min_lr)
                lr_wait = 0
                marker = f" ↘️ lr={lr:.1e}"
            if wait >= 15:
                print(f"\n⏹️  Early stopping tại epoch {epoch+1}")
                break
        
        print(f"{epoch+1:>5d} | {train_loss:>8.4f} | {train_acc:>6.2%} | "
              f"{val_loss:>9.4f} | {val_acc:>7.2%} | {lr:>10.6f}{marker}")
    
    print(f"\n🏆 Best validation: loss={best_val_loss:.4f}, acc={best_val_acc:.2%}")
    return best_params, history

# ========== ĐÁNH GIÁ ==========
def evaluate_ocr(params, num_layers, X_test, y_test):
    """Đánh giá model và vẽ confusion matrix"""
    y_test_cat      = to_categorical(y_test, NUM_CLASSES)
    y_pred_probs, _ = forward_pass(X_test, params, num_layers)
    y_pred          = np.argmax(y_pred_probs, axis=1)
    test_loss       = compute_loss(y_pred_probs, y_test_cat)
    test_acc        = np.mean(y_pred == y_test)
    
    print(f"\n{'='*60}")
    print(f"  📋 KẾT QUẢ ĐÁNH GIÁ OCR KÝ TỰ")
    print(f"{'='*60}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix - Nhận diện ký tự biển số xe', fontsize=16)
    plt.ylabel('Nhãn thực tế', fontsize=12)
    plt.xlabel('Nhãn dự đoán', fontsize=12)
    plt.tight_layout()
    plt.savefig("ocr_confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Đã lưu: ocr_confusion_matrix.png")
    
    return test_acc, y_pred

def plot_ocr_history(history):
    """Vẽ biểu đồ training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['accuracy'],     label='Train Accuracy', linewidth=2)
    ax1.plot(history['val_accuracy'], label='Val Accuracy',   linewidth=2)
    ax1.set_title('OCR Character Recognition - Accuracy', fontsize=14)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['loss'],     label='Train Loss', linewidth=2)
    ax2.plot(history['val_loss'], label='Val Loss',   linewidth=2)
    ax2.set_title('OCR Character Recognition - Loss', fontsize=14)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ocr_training_history.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Đã lưu: ocr_training_history.png")

# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("  BƯỚC 2: HUẤN LUYỆN MÔ HÌNH OCR KÝ TỰ BIỂN SỐ XE")
    print("=" * 60)
    
    TRAIN_DIR = "chars/train"
    VAL_DIR   = "chars/val"
    
    # === Kiểm tra thư mục ===
    if not os.path.isdir(TRAIN_DIR):
        print(f"❌ Không tìm thấy thư mục {TRAIN_DIR}")
        print("   Hãy chạy script.py trước để trích xuất ký tự từ dataset2!")
        exit(1)
    
    # === Load dữ liệu ===
    print("\n[1/5] 📂 Load dữ liệu training (có augment)...")
    X_train, y_train = load_char_dataset(TRAIN_DIR, augment=True)
    
    print("\n[2/5] 📂 Load dữ liệu validation...")
    X_val, y_val = load_char_dataset(VAL_DIR, augment=False)
    
    if len(X_train) == 0 or len(X_val) == 0:
        print("❌ Không có đủ dữ liệu! Kiểm tra lại thư mục chars/")
        exit(1)
    
    print(f"\n📊 Tổng quan dữ liệu:")
    print(f"   Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
    print(f"   Val  : {X_val.shape[0]:,} samples, {X_val.shape[1]} features")
    print(f"   Classes: {NUM_CLASSES}")
    
    # === Xây dựng model ===
    input_dim = X_train.shape[1]
    print(f"\n[3/5] 🧠 Xây dựng ANN model (input={input_dim} → 512 → 256 → 128 → {NUM_CLASSES})...")
    params, num_layers = build_ocr_model(input_dim, NUM_CLASSES)
    
    # === Huấn luyện ===
    print(f"\n[4/5] 🏋️ Bắt đầu huấn luyện OCR...")
    best_params, history = train_ocr_model(
        params, num_layers,
        X_train, y_train,
        X_val, y_val,
        epochs=80,
        batch_size=64
    )
    
    # Lưu model
    np.save("ocr_model.npy", best_params)
    print("\n💾 Đã lưu model: ocr_model.npy")
    print("💾 Đã lưu model tốt nhất: ocr_model_best.npy")
    
    # Vẽ biểu đồ
    plot_ocr_history(history)
    
    # === Đánh giá ===
    print(f"\n[5/5] 📋 Đánh giá trên tập Validation...")
    accuracy, y_pred = evaluate_ocr(best_params, num_layers, X_val, y_val)
    
    print(f"\n{'='*60}")
    print(f"  ✅ HOÀN THÀNH BƯỚC 2!")
    print(f"  📁 Model đã lưu: ocr_model.npy, ocr_model_best.npy")
    print(f"  📊 Biểu đồ: ocr_training_history.png, ocr_confusion_matrix.png")
    print(f"  🎯 Accuracy: {accuracy*100:.2f}%")
    print(f"{'='*60}")
