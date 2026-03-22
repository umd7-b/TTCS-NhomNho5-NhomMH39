import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import os

IMG_WIDTH = 64
IMG_HEIGHT = 64
MODEL_PATH = 'ann_biensoxe_b1.npy'
CLASS_NAMES = ["Không phải Biển Số", "Là Biển Số Xe"]

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

def forward_pass(X, params, num_layers):
    A = X
    for l in range(1, num_layers + 1):
        Z = A @ params[f'W{l}'] + params[f'b{l}']
        A = softmax(Z) if l == num_layers else relu(Z)
    return A

try:
    model_params = np.load(MODEL_PATH, allow_pickle=True).item()
    num_layers = len([k for k in model_params if k.startswith('W')])
except Exception as e:
    model_params = None
    num_layers = 0
    print(f"Cannot load model: {e}")

def normalize_plate(plate_crop):
    h, w = plate_crop.shape[:2]
    if h == 0 or w == 0:
        return None
    scale = min(IMG_WIDTH / w, IMG_HEIGHT / h)
    new_w, new_h = int(max(1, w * scale)), int(max(1, h * scale))
    resized = cv2.resize(plate_crop, (new_w, new_h))
    canvas = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    x_off = (IMG_WIDTH - new_w) // 2
    y_off = (IMG_HEIGHT - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
    features = hog.compute(gray).flatten()
    
    return features

def auto_detect_plate(cv_img, model_params, num_layers):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Tìm kiếm các đốm có dấu hiệu giống biển số (Hỗ trợ nhạy hơn cho biển VN)
    # 1. Dò tìm trên ảnh Gốc (Chuyên trị Biển Trắng, chữ Đen)
    plates_white = plate_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 15))
    
    # 2. Dò tìm trên ảnh Đảo Màu (Chuyên trị Biển ĐỎ, Biển XANH có chữ trắng)
    gray_inverted = cv2.bitwise_not(gray)
    plates_color = plate_cascade.detectMultiScale(gray_inverted, scaleFactor=1.05, minNeighbors=3, minSize=(30, 15))
    
    # 3. Dò tìm bằng phương pháp bám viền (Contour) - Vét máng mọi hình chữ nhật còn sót
    bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(bfilter, 30, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    
    plates = []
    if len(plates_white) > 0:
        for p in plates_white: plates.append(p)
    if len(plates_color) > 0:
        for p in plates_color: plates.append(p)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        plates.append((x, y, w, h))
        
    candidates = []
    
    for (x, y, w, h) in plates:
        aspect_ratio = w / float(max(1, h))
        
        # Vuông (~1.3 - 1.5), Dài (~4.0 - 5.0). Thả lỏng min size một chút cho ảnh chụp xa
        if 1.1 < aspect_ratio < 6.0 and w > 40 and h > 15:
            pad = 2
            rx1 = max(0, x - pad)
            ry1 = max(0, y - pad)
            rx2 = min(cv_img.shape[1], x + w + pad)
            ry2 = min(cv_img.shape[0], y + h + pad)
            
            plate_crop = cv_img[ry1:ry2, rx1:rx2]
            if plate_crop.size == 0: continue
            
            plate_crop_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
            flat = normalize_plate(plate_crop_bgr)
            
            if flat is not None:
                X = np.array([flat])
                probs = forward_pass(X, model_params, num_layers)
                
                prob_is_plate = probs[0][1] * 100
                pred_idx = np.argmax(probs[0])
                
                if pred_idx == 1 and prob_is_plate > 50.0:
                    candidates.append({
                        "box": (rx1, ry1, rx2, ry2),
                        "conf": prob_is_plate
                    })
                    
    candidates = sorted(candidates, key=lambda k: k['conf'], reverse=True)
    final_boxes = []
    
    def iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        return interArea / float(((boxA[2]-boxA[0])*(boxA[3]-boxA[1])) + ((boxB[2]-boxB[0])*(boxB[3]-boxB[1])) - interArea)

    for c in candidates:
        keep = True
        for f in final_boxes:
            if iou(c['box'], f['box']) > 0.1:
                keep = False
                break
        if keep:
            final_boxes.append(c)
            
    return final_boxes

class LicensePlateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện loại biển số xe")
        self.root.geometry("800x600")
        
        self.frame_top = tk.Frame(root)
        self.frame_top.pack(side=tk.TOP, fill=tk.X, pady=10)
        
        self.btn_load = tk.Button(self.frame_top, text="Tải ảnh lên", command=self.load_image, font=("Arial", 12))
        self.btn_load.pack(side=tk.LEFT, padx=10)
        
        self.lbl_info = tk.Label(self.frame_top, text="Kéo thả vùng HOẶC nhấn MÀU XANH để tự tìm", font=("Arial", 11))
        self.lbl_info.pack(side=tk.LEFT, padx=10)
        
        self.btn_predict = tk.Button(self.frame_top, text="Dự đoán (Kéo Box)", command=self.predict, font=("Arial", 12), state=tk.DISABLED)
        self.btn_predict.pack(side=tk.RIGHT, padx=5)

        self.btn_auto = tk.Button(self.frame_top, text="Tự động Truy Lùng", command=self.auto_detect, font=("Arial", 12, "bold"), state=tk.DISABLED, bg='lightgreen')
        self.btn_auto.pack(side=tk.RIGHT, padx=5)
        
        self.lbl_result = tk.Label(root, text="", font=("Arial", 16, "bold"), fg="blue")
        self.lbl_result.pack(pady=5)
        
        self.canvas = tk.Canvas(root, cursor="cross", bg="darkgray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
        self.rect = None
        self.rects = []
        self.start_x = None
        self.start_y = None
        
        self.cv_img = None
        self.tk_img = None
        self.scale = 1.0
        self.img_x_offset = 0
        self.img_y_offset = 0
        
    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
            
        self.cv_img = cv2.imread(file_path)
        if self.cv_img is None:
            messagebox.showerror("Lỗi", "Không thể đọc ảnh!")
            return
            
        self.cv_img = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)
        
        self.root.update_idletasks()
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10: canvas_w = 780
        if canvas_h < 10: canvas_h = 480
        
        h, w = self.cv_img.shape[:2]
        self.scale = min(canvas_w/max(1, w), canvas_h/max(1, h))
        new_w, new_h = int(w * self.scale), int(h * self.scale)
        
        resized = cv2.resize(self.cv_img, (new_w, new_h))
        self.tk_img = ImageTk.PhotoImage(image=Image.fromarray(resized))
        
        self.canvas.delete("all")
        self.img_x_offset = 0
        self.img_y_offset = 0
        
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.rect = None
        for r in self.rects: self.canvas.delete(r)
        self.rects = []
        
        self.lbl_result.config(text="")
        self.btn_predict.config(state=tk.NORMAL)
        self.btn_auto.config(state=tk.NORMAL)
        
    def on_button_press(self, event):
        if self.tk_img is None: return
        self.start_x = event.x
        self.start_y = event.y
        if self.rect:
            self.canvas.delete(self.rect)
        for r in self.rects:
            self.canvas.delete(r)
        self.rects = []
        
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red', width=2)
        
    def on_move_press(self, event):
        if not self.rect: return
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)
        
    def on_button_release(self, event):
        pass
        
    def predict(self):
        if self.cv_img is None or self.rect is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng vẽ khung biển số trước!")
            return
            
        if model_params is None:
            messagebox.showerror("Lỗi", f"Mô hình không tải được (kiểm tra {MODEL_PATH})!")
            return
            
        coords = self.canvas.coords(self.rect)
        if len(coords) != 4:
            return
            
        x1, y1, x2, y2 = coords
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        
        if x2 - x1 < 5 or y2 - y1 < 5:
            messagebox.showwarning("Cảnh báo", "Khung quét quá nhỏ!")
            return
            
        real_x1 = int((x1 - self.img_x_offset) / self.scale)
        real_y1 = int((y1 - self.img_y_offset) / self.scale)
        real_x2 = int((x2 - self.img_x_offset) / self.scale)
        real_y2 = int((y2 - self.img_y_offset) / self.scale)
        
        pad = 4
        h, w = self.cv_img.shape[:2]
        rx1 = max(0, real_x1 - pad)
        ry1 = max(0, real_y1 - pad)
        rx2 = min(w, real_x2 + pad)
        ry2 = min(h, real_y2 + pad)
        
        plate_crop = self.cv_img[ry1:ry2, rx1:rx2]
        if plate_crop.size == 0:
            return
            
        plate_crop_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
        flat = normalize_plate(plate_crop_bgr)
        if flat is None:
            messagebox.showerror("Lỗi", "Lỗi chuẩn hóa ảnh cropped!")
            return
            
        X = np.array([flat])
        probs = forward_pass(X, model_params, num_layers)
        pred_idx = np.argmax(probs[0])
        pred_label = CLASS_NAMES[pred_idx]
        conf = probs[0][pred_idx] * 100
        
        color = "green" if pred_idx == 1 else "red"
        self.lbl_result.config(text=f"Dự đoán thủ công: {pred_label} (Xác suất: {conf:.2f}%)", fg=color)

    def auto_detect(self):
        if self.cv_img is None:
            return
        if model_params is None:
            messagebox.showerror("Lỗi", "Không tải được mô hình numpy!")
            return
            
        detected_plates = auto_detect_plate(self.cv_img, model_params, num_layers)
        
        for r in self.rects: self.canvas.delete(r)
        self.rects = []
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None
            
        if not detected_plates:
            self.lbl_result.config(text="Tự động: KHÔNG TÌM THẤY VÙNG NÀO LÀ BIỂN SỐ!", fg="orange")
            return
            
        for p in detected_plates:
            rx1, ry1, rx2, ry2 = p["box"]
            conf = p["conf"]
            
            cx1 = int(rx1 * self.scale) + self.img_x_offset
            cy1 = int(ry1 * self.scale) + self.img_y_offset
            cx2 = int(rx2 * self.scale) + self.img_x_offset
            cy2 = int(ry2 * self.scale) + self.img_y_offset
            
            r = self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline='lime', width=4)
            self.rects.append(r)
            
            t = self.canvas.create_text(cx1, cy1 - 5, text=f"{conf:.1f}%", fill="lime", font=("Arial", 12, "bold"), anchor="sw")
            self.rects.append(t)
        
        self.lbl_result.config(text=f"Đã tìm thấy {len(detected_plates)} biển số xe!", fg="green")

if __name__ == "__main__":
    if model_params is None:
        print("Lỗi: Không tải được mô hình numpy.")
    root = tk.Tk()
    app = LicensePlateApp(root)
    root.mainloop()
