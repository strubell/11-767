
import cv2, os, sys

def gstreamer_pipeline(capture_width=1280, capture_height=720, 
                       display_width=1280, display_height=720,
                       framerate=60, flip_method=0):
  return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )

HEIGHT=1280
WIDTH=1920
center = (WIDTH / 2, HEIGHT / 2)
M = cv2.getRotationMatrix2D(center, 180, 1.0)

nano = False
if nano:
  cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
else:
  # Start Camera
  cam = cv2.VideoCapture(0)
  cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)  # 3280
  cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT) # 2464



if cam.isOpened():
  val, img = cam.read()
  if val:
    cv2.imwrite('output.png', img)
    #cv2.imwrite('output.png', cv2.warpAffine(img, M, (WIDTH, HEIGHT)))
