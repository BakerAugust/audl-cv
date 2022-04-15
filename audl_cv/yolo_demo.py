import torch
import torchvision
import cv2

INPUT = "data/possession_clips/2021-09-03-RAL-DC-12-1738-1786.mp4"

cap = cv2.VideoCapture(INPUT)
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        results = model([frame[..., ::-1]])  # BGR to RGB ?
        for idx, row in results.pandas().xyxy[0].iterrows():
            # Add rectangle
            cv2.rectangle(
                frame,
                (int(row["xmin"]), int(row["ymin"])),
                (int(row["xmax"]), int(row["ymax"])),
                color=(255, 0, 0),
            )
            # Add label
            cv2.putText(  # img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]
                img=frame,
                text=row["name"],
                org=(int(row["xmax"]), int(row["ymax"] - 10)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.9,
                color=(36, 255, 12),
                thickness=2,
            )

        cv2.imshow("frame", frame)
        cv2.waitKey(25)
    else:
        break

cap.release()
cv2.destroyAllWindows()
