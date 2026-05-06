import os, sys, random, pickle, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
from plate_models import OCRNet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print('⚠️  tqdm không khả dụng — pip install tqdm để có progress bar')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIRS = ['dataset_ocr']
IMG_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 200
LR_MAX = 0.003
PATIENCE = 35
TARGET_PER_CLASS = 2500
MAX_OVERSAMPLE_RATIO = 15
FOCAL_ALPHA = 1.0
FOCAL_GAMMA = 2.0
MIXUP_PROB = 0.4
MIXUP_ALPHA = 0.4
SWA_START_FRAC = 0.75
SWA_LR = 0.0005
CHECKPOINT_EVERY = 10
TTA_N_AUG = 5
CLASS_NAMES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N', 'P', 'S', 'T', 'U', 'V', 'X', 'Y', 'Z']
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
_hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, smoothing=0.05, num_classes=NUM_CLASSES):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.num_classes = num_classes

    def forward(self, logits, targets):
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smoothing / (self.num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        prob = log_prob.exp()
        p_t = (prob * smooth_targets).sum(dim=1)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = -(smooth_targets * log_prob).sum(dim=1)
        return (focal_weight * loss).mean()

def extract_hog(img_bgr):
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if h < 5 or w < 5:
        return None
    scale = min(IMG_SIZE / w, IMG_SIZE / h)
    nw, nh = (max(1, int(w * scale)), max(1, int(h * scale)))
    canvas = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    rsz = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    xo = (IMG_SIZE - nw) // 2
    yo = (IMG_SIZE - nh) // 2
    canvas[yo:yo + nh, xo:xo + nw] = rsz
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = _clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    return _hog.compute(gray).flatten()

def elastic_distortion(img, alpha=20, sigma=4):
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(np.random.uniform(-1, 1, (h, w)).astype(np.float32), (0, 0), sigma) * alpha
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.clip(x + dx, 0, w - 1).astype(np.float32)
    map_y = np.clip(y + dy, 0, h - 1).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def random_erasing(img, prob=0.5, sl=0.02, sh=0.25):
    if random.random() > prob:
        return img
    h, w = img.shape[:2]
    area = h * w
    for _ in range(20):
        erase_area = random.uniform(sl, sh) * area
        aspect = random.uniform(0.3, 3.0)
        eh = int(round((erase_area * aspect) ** 0.5))
        ew = int(round((erase_area / aspect) ** 0.5))
        if eh >= h or ew >= w:
            continue
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)
        fill = np.random.randint(100, 220, 3, dtype=np.uint8)
        img = img.copy()
        img[ey:ey + eh, ex:ex + ew] = fill
        return img
    return img

def augment_char(img_bgr):
    aug = []
    h, w = img_bgr.shape[:2]
    if h < 5 or w < 5:
        return aug
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    aug.append(cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.6, 1.4) + random.randint(-40, 40), 0, 255)
    aug.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR))
    noise = np.random.normal(0, random.uniform(8, 20), img_bgr.shape).astype(np.float32)
    aug.append(np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8))
    aug.append(cv2.GaussianBlur(img_bgr, (random.choice([3, 5]), random.choice([3, 5])), 0))
    aug.append(cv2.bitwise_not(img_bgr))
    shear = random.uniform(-0.2, 0.2)
    M2 = np.float32([[1, shear, 0], [0, 1, 0]])
    aug.append(cv2.warpAffine(img_bgr, M2, (w, h), borderMode=cv2.BORDER_REPLICATE))
    if w > 10 and h > 10:
        margin = max(1, int(min(w, h) * 0.1))
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[random.randint(0, margin), random.randint(0, margin)], [w - random.randint(0, margin), random.randint(0, margin)], [w - random.randint(0, margin), h - random.randint(0, margin)], [random.randint(0, margin), h - random.randint(0, margin)]])
        try:
            Mp = cv2.getPerspectiveTransform(src, dst)
            aug.append(cv2.warpPerspective(img_bgr, Mp, (w, h), borderMode=cv2.BORDER_REPLICATE))
        except Exception:
            pass
    kernel = np.ones((2, 2), np.uint8)
    if random.random() > 0.5:
        aug.append(cv2.erode(img_bgr, kernel, iterations=1))
    else:
        aug.append(cv2.dilate(img_bgr, kernel, iterations=1))
    hsv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2[:, :, 0] = np.clip(hsv2[:, :, 0] + random.randint(-15, 15), 0, 179)
    hsv2[:, :, 1] = np.clip(hsv2[:, :, 1] * random.uniform(0.5, 1.5), 0, 255)
    aug.append(cv2.cvtColor(hsv2.astype(np.uint8), cv2.COLOR_HSV2BGR))
    aug.append(elastic_distortion(img_bgr, alpha=random.uniform(10, 25), sigma=random.uniform(3, 5)))
    aug.append(random_erasing(img_bgr, prob=1.0, sl=0.02, sh=0.2))
    return aug

def cache_images(file_list):
    cache = {}
    failed = 0
    for fp in file_list:
        img = cv2.imread(fp)
        if img is None:
            failed += 1
            continue
        cache[fp] = img
    if failed > 0:
        print(f'  ⚠️  Không đọc được {failed} file ảnh (bị hỏng hoặc sai định dạng)')
    return cache

def load_and_split_datasets(data_dirs, test_size=0.15, val_size=0.15, target=TARGET_PER_CLASS):
    split_paths = {'train': [], 'val': [], 'test': []}
    split_labels = {'train': [], 'val': [], 'test': []}
    split_map = {'train': 'train', 'valid': 'val', 'test': 'test'}
    for ddir in data_dirs:
        if not os.path.exists(ddir):
            print(f'  ⚠️  Thư mục không tồn tại: {ddir}')
            continue
        for folder_name, split_key in split_map.items():
            split_dir = os.path.join(ddir, folder_name)
            if not os.path.isdir(split_dir):
                continue
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                flist = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                split_paths[split_key].extend(flist)
                split_labels[split_key].extend([cls_idx] * len(flist))
    total = sum(len(v) for v in split_paths.values())
    if total == 0:
        print('❌ Không tìm thấy dữ liệu nào!')
        return (None, None, None)
    print(f'\nTổng ảnh gốc: {total:,}')
    for split_key, display in [('train', 'TRAIN'), ('val', 'VAL'), ('test', 'TEST')]:
        print(f'  {display}: {len(split_paths[split_key]):,} ảnh')
    print('\nPhân bố theo class (train):')
    from collections import Counter
    train_counts = Counter(split_labels['train'])
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        n = train_counts.get(cls_idx, 0)
        bar = '█' * min(40, max(1, n // max(1, len(split_paths['train']) // (40 * NUM_CLASSES))))
        print(f'  {cls_name:>2}: {n:5d}  {bar}')
    all_paths = split_paths['train'] + split_paths['val'] + split_paths['test']
    print(f'\n📂 Đang cache {len(all_paths):,} ảnh vào RAM...')
    t0 = time.time()
    img_cache = cache_images(all_paths)
    print(f'  ✅ Cache hoàn tất trong {time.time() - t0:.1f}s ({len(img_cache):,}/{len(all_paths):,} ảnh thành công)')

    def process_split(paths, labels, do_augment=False, split_name=''):
        X, y = ([], [])
        paths_by_class = defaultdict(list)
        for p, l in zip(paths, labels):
            paths_by_class[l].append(p)
        for cls_idx, cls_paths in sorted(paths_by_class.items()):
            cls_name = CLASS_NAMES[cls_idx] if cls_idx < NUM_CLASSES else '?'
            orig_feats = []
            for fp in cls_paths:
                img = img_cache.get(fp)
                if img is None:
                    continue
                feat = extract_hog(img)
                if feat is not None:
                    orig_feats.append((img, feat))
            if not orig_feats:
                continue
            for img, feat in orig_feats:
                X.append(feat)
                y.append(cls_idx)
            if do_augment:
                n_orig = len(orig_feats)
                actual_target = min(target, n_orig * MAX_OVERSAMPLE_RATIO)
                needed = max(0, actual_target - n_orig)
                aug_count = 0
                while aug_count < needed:
                    img, _ = random.choice(orig_feats)
                    for aug_img in augment_char(img):
                        if aug_count >= needed:
                            break
                        feat = extract_hog(aug_img)
                        if feat is not None:
                            X.append(feat)
                            y.append(cls_idx)
                            aug_count += 1
                n_total = n_orig + aug_count
                print(f'    {cls_name:>2}: {n_orig:4d} gốc → {n_total:5d} mẫu (×{n_total / max(n_orig, 1):.1f})')
        return (np.array(X, dtype=np.float32), np.array(y, dtype=np.int64))
    print(f'\n  Đang xử lý tập TRAIN (kèm oversampling)...')
    train_data = process_split(split_paths['train'], split_labels['train'], do_augment=True, split_name='train')
    print(f'\n  Đang xử lý tập VAL...')
    val_data = process_split(split_paths['val'], split_labels['val'], do_augment=False, split_name='val')
    print(f'  Đang xử lý tập TEST...')
    test_data = process_split(split_paths['test'], split_labels['test'], do_augment=False, split_name='test')
    return (train_data, val_data, test_data)

def predict_with_tta(model, X_test_raw, scaler, n_aug=TTA_N_AUG):
    model.eval()
    X_test_sc = scaler.transform(X_test_raw)
    all_probs = []
    with torch.no_grad():
        t_in = torch.tensor(X_test_sc, dtype=torch.float32).to(device)
        probs = torch.softmax(model(t_in), dim=1).cpu().numpy()
        all_probs.append(probs)
        for _ in range(n_aug):
            noise = np.random.normal(0, 0.01, X_test_sc.shape).astype(np.float32)
            t_noisy = torch.tensor(X_test_sc + noise, dtype=torch.float32).to(device)
            probs_aug = torch.softmax(model(t_noisy), dim=1).cpu().numpy()
            all_probs.append(probs_aug)
    avg_probs = np.mean(all_probs, axis=0)
    return avg_probs.argmax(axis=1)

def analyze_confusions(cm, class_names, top_n=10):
    n = len(class_names)
    per_class_acc = []
    for i in range(n):
        total = cm[i].sum()
        acc = cm[i, i] / total if total > 0 else 0.0
        per_class_acc.append((class_names[i], acc, total))
    pairs = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], class_names[i], class_names[j]))
    pairs.sort(reverse=True)
    report_lines = []
    report_lines.append('=' * 55)
    report_lines.append(f'TOP {top_n} CẶP NHẦM LẪN NHIỀU NHẤT')
    report_lines.append('=' * 55)
    report_lines.append(f"{'Thực tế':>10} → {'Dự đoán':<10}  {'Số lần':>8}")
    report_lines.append('-' * 40)
    for count, true_cls, pred_cls in pairs[:top_n]:
        report_lines.append(f'{true_cls:>10} → {pred_cls:<10}  {count:>8}')
    report_lines.append('\n' + '=' * 55)
    report_lines.append('PER-CLASS ACCURACY (5 class tệ nhất)')
    report_lines.append('=' * 55)
    per_class_acc_sorted = sorted(per_class_acc, key=lambda x: x[1])
    for cls_name, acc, total in per_class_acc_sorted[:5]:
        bar = '▓' * int(acc * 20)
        report_lines.append(f'  {cls_name:>2}: {acc:6.2%}  {bar}  (n={total})')
    text = '\n'.join(report_lines)
    print(text)
    with open('ocr_confusion_pairs.txt', 'w', encoding='utf-8') as f:
        f.write('=' * 55 + '\n')
        f.write(f'TẤT CẢ CẶP NHẦM LẪN\n')
        f.write('=' * 55 + '\n')
        for count, true_cls, pred_cls in pairs:
            f.write(f'{true_cls} → {pred_cls}: {count}\n')
        f.write('\n' + text)
    return per_class_acc
if __name__ == '__main__':
    print(f'⚙️  Device: {device}')
    print('=' * 65)
    print('\n📦 [1/6] Khám phá và load dữ liệu...')
    result = load_and_split_datasets(DATA_DIRS, test_size=0.15, val_size=0.15, target=TARGET_PER_CLASS)
    if result is None:
        print('❌ Không thu thập được dữ liệu!')
        sys.exit(1)
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = result
    if len(y_train) == 0:
        print('❌ Tập train rỗng!')
        sys.exit(1)
    print(f'\n  Train: {len(y_train):,}  |  Val: {len(y_val):,}  |  Test: {len(y_test):,}')
    print(f'  HOG dim: {X_train_raw.shape[1]}')
    print('\n📐 [2/6] Chuẩn hoá features (StandardScaler)...')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    train_dl = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0, pin_memory=device.type == 'cuda')
    val_dl = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=BATCH_SIZE, num_workers=0)
    print('\n🧠 [3/6] Khởi tạo model, loss, optimiser...')
    model = OCRNet(X_train.shape[1]).to(device)
    total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'  Parameters: {total_params:,}')
    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LR_MAX / 25.0, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, steps_per_epoch=len(train_dl), epochs=EPOCHS, pct_start=0.15, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)
    swa_model = AveragedModel(model)
    swa_start = max(1, int(EPOCHS * SWA_START_FRAC))
    swa_sched = SWALR(optimizer, swa_lr=SWA_LR, anneal_epochs=10)
    swa_active = False
    print(f'\n🚀 [4/6] Training (max {EPOCHS} epochs, patience={PATIENCE} theo val_acc)...')
    print(f"{'Ep':>5} | {'LR':>8} | {'TrLoss':>7} | {'TrAcc':>6} | {'VlLoss':>7} | {'VlAcc':>6} | GradNorm")
    print('-' * 70)
    best_val_acc = 0.0
    best_state = None
    no_improve = 0
    history = {'tr_loss': [], 'vl_loss': [], 'tr_acc': [], 'vl_acc': [], 'lr': []}
    swa_started = False
    os.makedirs('checkpoints', exist_ok=True)
    for ep in range(1, EPOCHS + 1):
        if ep == swa_start and (not swa_started):
            print(f'\n  🔄 Bắt đầu SWA phase từ epoch {ep}')
            swa_started = True
        model.train()
        tr_loss = tr_correct = tr_total = 0
        grad_norms = []
        iterator = tqdm(train_dl, desc=f'Ep {ep:3d}', leave=False) if HAS_TQDM else train_dl
        for bx, by in iterator:
            bx, by = (bx.to(device), by.to(device))
            optimizer.zero_grad()
            if random.random() < MIXUP_PROB:
                lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
                idx = torch.randperm(bx.size(0), device=device)
                bx_mix = lam * bx + (1 - lam) * bx[idx]
                by_a, by_b = (by, by[idx])
                out = model(bx_mix)
                loss = lam * criterion(out, by_a) + (1 - lam) * criterion(out, by_b)
                pred_label = by_a if lam >= 0.5 else by_b
            else:
                out = model(bx)
                loss = criterion(out, by)
                pred_label = by
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(gn.item() if hasattr(gn, 'item') else float(gn))
            optimizer.step()
            if not swa_started:
                scheduler.step()
            tr_loss += loss.item()
            tr_correct += (out.argmax(1) == pred_label).sum().item()
            tr_total += by.size(0)
        if swa_started:
            swa_model.update_parameters(model)
            swa_sched.step()
        model.eval()
        vl_loss = vl_correct = vl_total = 0
        with torch.no_grad():
            for bx, by in val_dl:
                bx, by = (bx.to(device), by.to(device))
                out = model(bx)
                loss = criterion(out, by)
                vl_loss += loss.item()
                vl_correct += (out.argmax(1) == by).sum().item()
                vl_total += by.size(0)
        avg_tr_loss = tr_loss / len(train_dl)
        avg_vl_loss = vl_loss / max(len(val_dl), 1)
        tr_acc = tr_correct / max(tr_total, 1)
        vl_acc = vl_correct / max(vl_total, 1)
        avg_gn = np.mean(grad_norms)
        cur_lr = optimizer.param_groups[0]['lr']
        history['tr_loss'].append(avg_tr_loss)
        history['vl_loss'].append(avg_vl_loss)
        history['tr_acc'].append(tr_acc)
        history['vl_acc'].append(vl_acc)
        history['lr'].append(cur_lr)
        marker = ''
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            marker = ' ✅'
            torch.save(best_state, 'ocr_model_best.pth')
        else:
            no_improve += 1
        if ep % 5 == 0 or marker:
            print(f'{ep:5d} | {cur_lr:8.2e} | {avg_tr_loss:7.4f} | {tr_acc:6.2%} | {avg_vl_loss:7.4f} | {vl_acc:6.2%} | {avg_gn:.3f}{marker}')
        if ep % CHECKPOINT_EVERY == 0:
            ckpt_path = f'checkpoints/ocr_ep{ep:04d}.pth'
            torch.save({'epoch': ep, 'model_state': {k: v.cpu() for k, v in model.state_dict().items()}, 'optimizer_state': optimizer.state_dict(), 'best_val_acc': best_val_acc, 'history': history}, ckpt_path)
        if no_improve >= PATIENCE:
            print(f'\n⏹️  Early stopping tại epoch {ep} (val_acc không tăng {PATIENCE} epoch liên tiếp)')
            break
    if swa_started:
        print('\n⚙️  Cập nhật BatchNorm cho SWA model...')
        update_bn(train_dl, swa_model, device=device)
        swa_state = {k: v.cpu().clone() for k, v in swa_model.module.state_dict().items()}
        torch.save(swa_state, 'ocr_model_swa.pth')
        print('  ✅ Đã lưu: ocr_model_swa.pth')
    with open('ocr_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f'\n💾 Đã lưu: ocr_model_best.pth  |  ocr_scaler.pkl')
    print(f'   Best Val Acc: {best_val_acc:.2%}')
    print('\n📊 [5/6] Đánh giá tập Test...')
    if len(X_test_raw) > 0 and best_state is not None:
        eval_model = OCRNet(X_train.shape[1]).to(device)
        eval_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        eval_model.eval()
        with torch.no_grad():
            t_in = torch.tensor(scaler.transform(X_test_raw), dtype=torch.float32).to(device)
            t_pred_std = eval_model(t_in).argmax(1).cpu().numpy()
        print(f'  Đang chạy TTA (n_aug={TTA_N_AUG})...')
        t_pred_tta = predict_with_tta(eval_model, X_test_raw, scaler, n_aug=TTA_N_AUG)
        acc_std = (t_pred_std == y_test).mean()
        acc_tta = (t_pred_tta == y_test).mean()
        print(f'\n  Accuracy (standard): {acc_std:.2%}')
        print(f"  Accuracy (TTA ×{TTA_N_AUG + 1}):  {acc_tta:.2%}  ({('↑' if acc_tta > acc_std else '=')}{abs(acc_tta - acc_std):.2%})")
        print('\n  Classification Report:')
        print(classification_report(y_test, t_pred_tta, target_names=CLASS_NAMES, zero_division=0))
        cm = confusion_matrix(y_test, t_pred_tta)
        print('\n')
        per_class = analyze_confusions(cm, CLASS_NAMES)
        print('  → Đã lưu: ocr_confusion_pairs.txt')
        fig, ax = plt.subplots(figsize=(16, 13))
        vmax = np.percentile(cm[cm > 0], 95) if (cm > 0).any() else 1
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmax=vmax)
        ax.set_title('Confusion Matrix — OCR v2 (TTA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Dự đoán', fontsize=11)
        ax.set_ylabel('Thực tế', fontsize=11)
        ticks = np.arange(NUM_CLASSES)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(CLASS_NAMES, fontsize=8)
        ax.set_yticklabels(CLASS_NAMES, fontsize=8)
        thresh = cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                if cm[i, j] > 0:
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6, color='white' if cm[i, j] > thresh else 'black')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('ocr_confusion_matrix.png', dpi=150)
        plt.close()
        print('  → Đã lưu: ocr_confusion_matrix.png')
    else:
        print('  Không có tập test.')
    print('\n📈 [6/6] Vẽ training history...')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['tr_acc'], label='Train')
    axes[0].plot(history['vl_acc'], label='Val')
    if swa_started and best_val_acc > 0:
        axes[0].axvline(swa_start, color='r', linestyle='--', alpha=0.5, label='SWA start')
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([max(0, min(history['tr_acc'] + history['vl_acc']) - 0.05), 1.0])
    axes[1].plot(history['tr_loss'], label='Train')
    axes[1].plot(history['vl_loss'], label='Val')
    axes[1].set_title('Focal Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(history['lr'], color='orange')
    axes[2].set_title('Learning Rate (OneCycleLR)')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('LR')
    axes[2].grid(True, alpha=0.3)
    if swa_started:
        axes[2].axvline(swa_start, color='r', linestyle='--', alpha=0.5, label='SWA start')
        axes[2].legend()
    plt.tight_layout()
    plt.savefig('ocr_training_history.png', dpi=150)
    plt.close()
    print('  → Đã lưu: ocr_training_history.png')
    print('\n✅ Hoàn thành Train OCR v2!')
    print('   Đầu ra:')
    print('     ocr_model_best.pth    — dùng trong inference code')
    if swa_started:
        print('     ocr_model_swa.pth    — thử dùng cái này nếu best không tốt hơn')
    print('     ocr_scaler.pkl')
    print('     ocr_confusion_matrix.png')
    print('     ocr_confusion_pairs.txt')
    print('     ocr_training_history.png')