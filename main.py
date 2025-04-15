import cv2
from lane_utils import canny, region_of_interest, average_slope_intercept, display_lines

input_video_path = "solidWhiteRight.mp4"
output_video_path = "output.mp4"

cap = cv2.VideoCapture(input_video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    edges = canny(rgb)
    roi = region_of_interest(edges)
    lines = cv2.HoughLinesP(roi, 2, 3.14 / 180, 100, minLineLength=40, maxLineGap=5)
    left, right = average_slope_intercept(frame, lines)
    overlay = display_lines(frame, left, right)
    combined = cv2.addWeighted(frame, 0.8, overlay, 1, 1)
    out.write(combined)

cap.release()
out.release()
print(f"✅ Done! Saved to {output_video_path}")
