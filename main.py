import cv2

from ultralytics import YOLO
import numpy as np
import time

model = YOLO("yolov8x.pt")
cap = cv2.VideoCapture("data/0625.mp4")
assert cap.isOpened(), "Error reading video file"

names = model.names
people_spent_time = {}
spent_time = []

frame_count = 0
fps = 30

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
        
    tracks = model.track(im0, persist=True, show=False, classes=[0], device=0)
    # drawing bbox and tracking id on image
    start_time = time.time()

    boxes = tracks[0].boxes

    # get bbox cordinates
    xyxy = boxes.xyxy.cpu()
    # get tracking ids
    track_ids = boxes.id.detach().cpu().numpy()
    # get class
    cls = boxes.cls.cpu().tolist()
    # get confidence
    conf = boxes.conf

    # loop through each bbox
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        track_id = int(track_ids[i])
        # if track id is not in people_spent_time, add it, else not do anything
        if track_id not in people_spent_time:
            people_spent_time[track_id] = {"start": frame_count, "end": frame_count}
        # calculate time spent
        timestamp = (people_spent_time[track_id]["end"] - people_spent_time[track_id]["start"]) / fps
        # draw bbox and tracking id on image
        cv2.rectangle(im0, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(im0, f"{names[int(cls[i])]} #{track_id}: {timestamp:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # update frame_count, peopeple's spent time
    for spent_track_id in people_spent_time:
        if spent_track_id in track_ids:
            people_spent_time[spent_track_id]["end"] = frame_count

    # break if q is pressed
    if cv2.waitKey(1) == ord("q"):
        break
    
    # calculate fps
    FPS = 1.0 / (time.time() - start_time)

    # put fps on image
    im0 = cv2.putText(im0, f"FPS: {int(FPS)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    # update frame_count
    frame_count += 1
    cv2.imshow("YOLOv8", im0)

cap.release()
cv2.destroyAllWindows()