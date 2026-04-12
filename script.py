# script.py
import os, cv2

# Map class_id → tên ký tự
CLASS_MAP = {
    0:'0', 1:'1', 2:'2', 3:'3', 4:'4',
    5:'5', 6:'6', 7:'7', 8:'8', 9:'9',
    10:'A', 11:'B', 12:'C', 13:'D', 14:'E',
    15:'F', 16:'G', 17:'H', 18:'K', 19:'L',
    20:'M', 21:'N', 22:'P', 23:'S', 24:'T',
    25:'U', 26:'V', 27:'X', 28:'Y', 29:'Z'
}

def convert(img_dir, label_dir, out_dir):
    print(f"--- Đang xử lý: {label_dir} ---")
    os.makedirs(out_dir, exist_ok=True)
    
    # Tạo thư mục cho từng ký tự
    for name in CLASS_MAP.values():
        os.makedirs(f"{out_dir}/{name}", exist_ok=True)

    count = 0
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
    
    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        
        # Bỏ qua nếu file nhãn trống
        if os.path.getsize(label_path) == 0:
            continue

        # Tìm file ảnh tương ứng (thử các định dạng phổ biến)
        img_found = False
        for ext in ['.jpg', '.JPG', '.png', '.png', '.jpeg']:
            tmp_img_file = label_file.replace('.txt', ext)
            tmp_img_path = os.path.join(img_dir, tmp_img_file)
            if os.path.exists(tmp_img_path):
                img_path = tmp_img_path
                img_found = True
                break
        
        if not img_found:
            # print(f"⚠️ Không tìm thấy ảnh cho nhãn: {label_file}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        H, W = img.shape[:2]

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])

                # YOLO → pixel coordinates
                x1 = int((cx - bw/2) * W)
                y1 = int((cy - bh/2) * H)
                x2 = int((cx + bw/2) * W)
                y2 = int((cy + bh/2) * H)

                # Clamp để tránh out of bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if cls_id in CLASS_MAP:
                    char_name = CLASS_MAP[cls_id]
                    save_path = os.path.join(out_dir, char_name, f"{count}.jpg")
                    cv2.imwrite(save_path, crop)
                    count += 1
                else:
                    print(f"⚠️ Class ID {cls_id} không nằm trong CLASS_MAP (File: {label_file})")

    print(f"✅ Thành công! Đã trích xuất {count} ảnh ký tự vào thư mục: {out_dir}")

# Chạy convert cho train và val của dataset2
if __name__ == "__main__":
    # Xử lý tập Train
    convert(
        img_dir   = "dataset2/images/train",
        label_dir = "dataset2/labels/train",
        out_dir   = "chars/train"
    )
    
    # Xử lý tập Val
    convert(
        img_dir   = "dataset2/images/val",
        label_dir = "dataset2/labels/val",
        out_dir   = "chars/val"
    )
1