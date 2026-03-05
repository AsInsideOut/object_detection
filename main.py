import cv2
from ultralytics import YOLO


def run_detection(video_source, output_name="result.mp4"):
    model = YOLO("yolo11n.pt")

    cap = cv2.VideoCapture(video_source)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    print(f"Detecting {w}x{h}...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model.track(frame, persist=True, verbose=False, classes=[0, 2, 3, 5, 7])

        if results[0].boxes:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                obj_w = x2 - x1
                obj_h = y2 - y1

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                display_text = f"{label} {obj_w}x{obj_h}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2

                text_pos = (x1, y1 - 10 if y1 > 20 else y1 + 20)

                cv2.putText(frame, display_text, text_pos, font, font_scale, (0, 0, 0), thickness + 1)  # Тень
                cv2.putText(frame, display_text, text_pos, font, font_scale, (0, 0, 255), thickness)  # Основной цвет

        out.write(frame)
        cv2.imshow("Detection Process", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Done!!")


run_detection("video.mp4")

# TG: hyeplet231
# TT: heyr1xx