# main_gui.py  — YOLO-Seg + tính diện tích + nút Chụp ảnh (save RAW & Annotated)
import os, time, threading, tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
from ultralytics import YOLO
from area_calc import AreaCalculator, extract_union_mask

# ===== CẤU HÌNH =====
MODEL_PATH = "best_yolo_seg_309.onnx"
CAM_INDEX  = 1
CONF_THRES = 0.5
IMG_SIZE   = 640
CALIB_PATH = "calibration.json"
SHOW_MASKS = True
FORCE_GPU  = True   # cố gắng dùng GPU qua ONNXRuntime nếu có, lỗi thì về CPU
# ====================

running = False
last_frame = None       # ảnh đã vẽ (annotated) – để hiển thị & chụp
last_raw_frame = None   # ảnh gốc từ camera – để chụp
distance_m, area_m2, fps_value = 0.30, 0.0, 0.0

def force_gpu_session(model: YOLO):
    """Ép ORT dùng CUDA nếu có (không dùng PyTorch CUDA)."""
    if not FORCE_GPU:
        return
    try:
        dummy = np.zeros((32, 32, 3), dtype=np.uint8)
        _ = model.predict(dummy, conf=0.01, imgsz=32, verbose=False)  # khởi tạo predictor/backend
        backend = getattr(model.predictor, "model", None)
        sess = getattr(backend, "session", None)
        if sess is not None:
            print("Session BEFORE:", sess.get_providers())
            sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])
            print("Session AFTER :", sess.get_providers())
    except Exception as e:
        print("Không ép được GPU, dùng CPU. Lý do:", e)

def set_distance():
    global distance_m
    try:
        cm = float(entry_distance.get())
        distance_m = max(0.0, cm / 100.0)
    except Exception:
        messagebox.showerror("Lỗi", "Vui lòng nhập số cm hợp lệ!")

def capture_image():
    """Lưu ảnh hiện tại: annotated + raw vào thư mục captures/"""
    if last_frame is None:
        messagebox.showwarning("Chưa sẵn sàng", "Chưa có khung hình để chụp.")
        return
    ts = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs("captures", exist_ok=True)
    ann_path = os.path.join("captures", f"cap_{ts}_ann.jpg")
    raw_path = os.path.join("captures", f"cap_{ts}_raw.jpg")
    try:
        cv2.imwrite(ann_path, last_frame.copy())
        if last_raw_frame is not None:
            cv2.imwrite(raw_path, last_raw_frame.copy())
            messagebox.showinfo("Đã lưu", f"Đã lưu:\n{ann_path}\n{raw_path}")
        else:
            messagebox.showinfo("Đã lưu", f"Đã lưu:\n{ann_path}")
    except Exception as e:
        messagebox.showerror("Lỗi lưu ảnh", str(e))

def video_worker():
    global running, last_frame, last_raw_frame, area_m2, fps_value
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được camera!")
        running = False
        return

    model = YOLO(MODEL_PATH, task="segment")
    # tắt summary dùng 'names' nếu có
    if hasattr(model.predictor, "write_results"):
        model.predictor.write_results = lambda *a, **k: ""

    # cố gắng chuyển session sang GPU (nếu có)
    force_gpu_session(model)

    calc = None
    prev_t = time.time()

    while running:
        ok, frame = cap.read()
        if not ok:
            continue
        last_raw_frame = frame  # lưu ảnh gốc để chụp

        if calc is None or calc.frame_width != frame.shape[1]:
            calc = AreaCalculator(frame.shape[1], calib_path=CALIB_PATH)

        # infer (không truyền device=0 để tránh PyTorch kiểm tra CUDA)
        r0 = model.predict(frame, conf=CONF_THRES, imgsz=IMG_SIZE, verbose=False)[0]

        # vẽ
        annotated = r0.plot(labels=False, boxes=False, masks=SHOW_MASKS)

        # tính diện tích từ mask union
        mask = extract_union_mask(r0)
        area_m2 = calc.area_m2(mask, distance_m) if distance_m > 0 else 0.0

        # fps
        now = time.time()
        fps_value = 1.0 / (now - prev_t) if now > prev_t else 0.0
        prev_t = now

        # overlay text
        cv2.putText(annotated, f"FPS:{fps_value:.1f}", (10, 30), 0, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Dist:{distance_m*100:.1f}cm", (10, 65), 0, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Area:{area_m2:.4f}m^2", (10, 98), 0, 0.8, (255, 255, 255), 2)

        last_frame = annotated

    cap.release()

def update_ui():
    if last_frame is not None:
        rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        imtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        video_label.imtk = imtk
        video_label.configure(image=imtk)
        lbl_area.config(text=f"{area_m2:.4f} m²")
        lbl_fps.config(text=f"{fps_value:.1f} FPS")
    root.after(30, update_ui)

def start():
    global running
    if running:
        return
    set_distance()
    running = True
    threading.Thread(target=video_worker, daemon=True).start()

def stop():
    global running
    running = False

def close():
    stop()
    root.destroy()

# ============= TKINTER UI =============
root = tk.Tk()
root.title("YOLO-Seg Area (GPU/CPU) + Capture")

# thanh điều khiển
ctrl = ttk.Frame(root, padding=6)
ctrl.pack(side=tk.TOP, fill=tk.X)

ttk.Label(ctrl, text="Khoảng cách (cm):").pack(side=tk.LEFT)
entry_distance = ttk.Entry(ctrl, width=8)
entry_distance.insert(0, "20")
entry_distance.pack(side=tk.LEFT, padx=4)

ttk.Button(ctrl, text="Cập nhật", command=set_distance).pack(side=tk.LEFT, padx=4)
ttk.Button(ctrl, text="Start", command=start).pack(side=tk.LEFT, padx=6)
ttk.Button(ctrl, text="Stop", command=stop).pack(side=tk.LEFT)
ttk.Button(ctrl, text="Chụp ảnh", command=capture_image).pack(side=tk.LEFT, padx=6)  # <-- nút mới

# info
info = ttk.Frame(root, padding=6)
info.pack(side=tk.TOP, fill=tk.X)
ttk.Label(info, text="Diện tích:").pack(side=tk.LEFT)
lbl_area = ttk.Label(info, text="0.0000 m²", width=16)
lbl_area.pack(side=tk.LEFT, padx=6)
ttk.Label(info, text="Tốc độ:").pack(side=tk.LEFT)
lbl_fps = ttk.Label(info, text="0.0 FPS", width=10)
lbl_fps.pack(side=tk.LEFT, padx=6)

# video
video_label = ttk.Label(root)
video_label.pack(side=tk.TOP, padx=6, pady=6)

root.protocol("WM_DELETE_WINDOW", close)
root.after(30, update_ui)
root.mainloop()
