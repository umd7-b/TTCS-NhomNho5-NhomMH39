import numpy as np
import cv2
import os
import random
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from plate_models import PlateDetectorANN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATASET_DIR = 'dataset_plate'
IMG_SIZE = (80, 64)
BATCH_SIZE = 64
EPOCHS = 150
LEARNING_RATE = 0.001
PATIENCE = 25
LABEL_SMOOTHING = 0.1
_hog = cv2.HOGDescriptor((80, 64), (16, 16), (8, 8), (8, 8), 9)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def extract_hog_features(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    if img_bgr.shape[0] < 5 or img_bgr.shape[1] < 5:
        return None
    resized = cv2.resize(img_bgr, IMG_SIZE)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return _hog.compute(gray).flatten()

def check_overlap_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xx1, yy1 = (max(x1, x2), max(y1, y2))
    xx2, yy2 = (min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2))
    if xx1 >= xx2 or yy1 >= yy2:
        return 0.0
    inter_area = (xx2 - xx1) * (yy2 - yy1)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area

def augment_plate(img_crop):
    augmented = []
    h, w = img_crop.shape[:2]
    if h < 10 or w < 10:
        return augmented
    alpha = random.uniform(0.6, 1.4)
    beta = random.randint(-40, 40)
    bright = cv2.convertScaleAbs(img_crop, alpha=alpha, beta=beta)
    augmented.append(bright)
    ksize = random.choice([3, 5])
    augmented.append(cv2.GaussianBlur(img_crop, (ksize, ksize), 0))
    noise = np.random.normal(0, 15, img_crop.shape).astype(np.float32)
    noisy = np.clip(img_crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    augmented.append(noisy)
    angle = random.uniform(-7, 7)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    augmented.append(cv2.warpAffine(img_crop, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    margin = max(1, int(min(w, h) * 0.08))
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[random.randint(0, margin), random.randint(0, margin)], [w - random.randint(0, margin), random.randint(0, margin)], [w - random.randint(0, margin), h - random.randint(0, margin)], [random.randint(0, margin), h - random.randint(0, margin)]])
    M_persp = cv2.getPerspectiveTransform(src_pts, dst_pts)
    augmented.append(cv2.warpPerspective(img_crop, M_persp, (w, h), borderMode=cv2.BORDER_REPLICATE))
    shear = random.uniform(-0.15, 0.15)
    M_shear = np.float32([[1, shear, 0], [0, 1, 0]])
    augmented.append(cv2.warpAffine(img_crop, M_shear, (w, h), borderMode=cv2.BORDER_REPLICATE))
    hsv = cv2.cvtColor(img_crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(-10, 10), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(-30, 30), 0, 255)
    augmented.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    return augmented

def generate_random_negatives(img, bboxes, img_h, img_w, count=3):
    negatives = []
    attempts = 0
    while len(negatives) < count and attempts < 80:
        attempts += 1
        rw = random.randint(40, 200)
        rh = random.randint(15, 100)
        rx = random.randint(0, max(1, img_w - rw))
        ry = random.randint(0, max(1, img_h - rh))
        random_box = [rx, ry, rw, rh]
        is_overlap = any((check_overlap_iou(random_box, b) > 0.05 for b in bboxes))
        if not is_overlap:
            bg_crop = img[ry:ry + rh, rx:rx + rw]
            feat = extract_hog_features(bg_crop)
            if feat is not None:
                negatives.append(feat)
    return negatives

def load_dataset_from_json(mode='train'):
    X, Y = ([], [])
    folder_path = os.path.join(DATASET_DIR, mode)
    json_path = os.path.join(folder_path, '_annotations.coco.json')
    if not os.path.exists(json_path):
        print(f'⚠️ Không tìm thấy file JSON tại: {json_path}')
        return (np.array([]), np.array([]))
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images_info = {img['id']: {'file_name': img['file_name'], 'w': img['width'], 'h': img['height']} for img in coco['images']}
    annotations = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann['bbox'])
    print(f'\n--- Đang tải tập {mode.upper()}: {len(images_info)} ảnh ---')
    total = len(images_info)
    for idx, (img_id, info) in enumerate(images_info.items()):
        if (idx + 1) % 500 == 0:
            print(f'  Đang xử lý: {idx + 1}/{total}...')
        img_path = os.path.join(folder_path, info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            continue
        bboxes = annotations.get(img_id, [])
        img_h, img_w = (info['h'], info['w'])
        for bbox in bboxes:
            x, y, w, h = [int(v) for v in bbox]
            x, y = (max(0, x), max(0, y))
            plate_crop = img[y:y + h, x:x + w]
            feat = extract_hog_features(plate_crop)
            if feat is not None:
                X.append(feat)
                Y.append(1)
                if mode == 'train':
                    aug_crops = augment_plate(plate_crop)
                    for aug_crop in aug_crops:
                        aug_feat = extract_hog_features(aug_crop)
                        if aug_feat is not None:
                            X.append(aug_feat)
                            Y.append(1)
        rand_negs = generate_random_negatives(img, bboxes, img_h, img_w, count=8 if mode == 'train' else 3)
        for feat in rand_negs:
            X.append(feat)
            Y.append(0)
    X, Y = (np.array(X, dtype=np.float32), np.array(Y, dtype=np.int64))
    pos_count = np.sum(Y == 1)
    neg_count = np.sum(Y == 0)
    print(f' -> Tập {mode}: {pos_count} Biển số | {neg_count} Nền/Rác | Tổng: {len(Y)}')
    print(f'    Tỉ lệ Pos:Neg = 1:{neg_count / max(pos_count, 1):.1f}')
    return (X, Y)

class FocalLoss(nn.Module):

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
if __name__ == '__main__':
    print(f'⚙️ Đang chạy trên: {device}')
    print('=' * 60)
    print('\n📦 BƯỚC 1: Tải và xử lý dữ liệu...')
    print('⚠️ Hard Negative Mining sẽ tốn thời gian, vui lòng chờ...\n')
    X_train_raw, y_train = load_dataset_from_json('train')
    X_val_raw, y_val = load_dataset_from_json('valid')
    X_test_raw, y_test = load_dataset_from_json('test')
    if len(X_train_raw) == 0:
        print('❌ Lỗi dữ liệu! Kết thúc.')
        exit()
    print('\n📐 BƯỚC 2: Chuẩn hóa features...')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(y_val))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    print('\n🧠 BƯỚC 3: Khởi tạo mô hình...')
    model = PlateDetectorANN(input_dim=X_train.shape[1]).to(device)
    total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'   Tổng parameters: {total_params:,}')
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    total_samples = n_pos + n_neg
    weight_for_0 = total_samples / (2.0 * n_neg)
    weight_for_1 = total_samples / (2.0 * n_pos)
    class_weights = torch.tensor([weight_for_0, weight_for_1], dtype=torch.float32).to(device)
    print(f'   Class weights: Nền={weight_for_0:.3f}, Biển số={weight_for_1:.3f}')
    focal_alpha = torch.tensor([0.25, 0.75], dtype=torch.float32).to(device)
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-06)
    print(f'\n🚀 BƯỚC 4: Bắt đầu huấn luyện ({EPOCHS} epochs max, patience={PATIENCE})...\n')
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    for ep in range(EPOCHS):
        model.train()
        tr_loss, tr_correct, tr_total = (0, 0, 0)
        for bX, bY in train_loader:
            bX, bY = (bX.to(device), bY.to(device))
            if random.random() < 0.5:
                lam = np.random.beta(0.4, 0.4)
                idx = torch.randperm(bX.size(0)).to(device)
                bX_mixed = lam * bX + (1 - lam) * bX[idx]
                bY_a, bY_b = (bY, bY[idx])
                optimizer.zero_grad()
                outputs = model(bX_mixed)
                loss = lam * criterion(outputs, bY_a) + (1 - lam) * criterion(outputs, bY_b)
            else:
                optimizer.zero_grad()
                outputs = model(bX)
                loss = criterion(outputs, bY)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_correct += (outputs.argmax(1) == bY).sum().item()
            tr_total += bY.size(0)
        scheduler.step()
        avg_tr_loss = tr_loss / len(train_loader)
        tr_acc = tr_correct / tr_total * 100
        model.eval()
        val_loss, val_correct, val_total = (0, 0, 0)
        with torch.no_grad():
            for vX, vY in val_loader:
                vX, vY = (vX.to(device), vY.to(device))
                v_out = model(vX)
                v_loss = criterion(v_out, vY)
                val_loss += v_loss.item()
                val_correct += (v_out.argmax(1) == vY).sum().item()
                val_total += vY.size(0)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total * 100
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            save_marker = ' ✅'
        else:
            epochs_no_improve += 1
            save_marker = ''
        if (ep + 1) % 5 == 0 or save_marker:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {ep + 1:3d}/{EPOCHS} | Train Loss: {avg_tr_loss:.4f} Acc: {tr_acc:.1f}% | Val Loss: {avg_val_loss:.4f} Acc: {val_acc:.1f}% | LR: {current_lr:.6f}{save_marker}')
        if epochs_no_improve >= PATIENCE:
            print(f'\n⏹️ Early stopping tại epoch {ep + 1}')
            break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print('✅ Đã khôi phục model tốt nhất.')
    print('\n' + '=' * 60)
    print('📊 BƯỚC 5: ĐÁNH GIÁ TRÊN TẬP TEST:')
    print('=' * 60)
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test).to(device)
        test_outputs = model(test_tensor)
        test_preds = test_outputs.argmax(1).cpu().numpy()
        test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
        avg_conf_plate = np.mean(test_probs[y_test == 1, 1]) if np.any(y_test == 1) else 0
        avg_conf_bg = np.mean(test_probs[y_test == 0, 0]) if np.any(y_test == 0) else 0
    print(classification_report(y_test, test_preds, target_names=['Nền/Rác', 'Biển số']))
    cm = confusion_matrix(y_test, test_preds)
    print('Ma trận nhầm lẫn (Confusion Matrix):')
    print(f'  Đúng Nền: {cm[0][0]:4d} | Sai (Nền→Biển): {cm[0][1]:4d}')
    print(f'  Sai (Biển→Nền): {cm[1][0]:4d} | Đúng Biển: {cm[1][1]:4d}')
    print(f'\n  Confidence trung bình:')
    print(f'    Biển số đúng: {avg_conf_plate * 100:.1f}%')
    print(f'    Nền đúng:     {avg_conf_bg * 100:.1f}%')
    torch.save(model.state_dict(), 'final_plate_model.pth')
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f'\n✅ Đã lưu model và scaler thành công!')
    print('   → final_plate_model.pth')
    print('   → scaler.pkl')