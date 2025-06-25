from ultralytics import YOLO
import cv2
print("OpenCV version:", cv2.__version__)
print("cv2.imshow exists:", hasattr(cv2, "imshow"))

model = YOLO("runs/train/exp/weights/best.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak dapat membuka kamera.")
    exit()
else:
    print("Kamera berhasil dibuka.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera.")
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Realtime", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()