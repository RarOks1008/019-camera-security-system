import cv2
import time
import datetime

camera = cv2.VideoCapture(0)
face_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
body_c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

SHOW_VIDEO = True
AFTER_DETECTION = 30

detected = False
started = False
stop_time = None

framesize = (int(camera.get(3)), int(camera.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

while True:
    _, frame = camera.read()

    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_c.detectMultiScale(grayscale, 1.1, 5)
    bodies = body_c.detectMultiScale(grayscale, 1.1, 5)
    #Middle value is best kept between 1.1 to 1.6
    #Closer to 1.1 for shorter but better match, and closer to 1.6 for faster algorithm

    if len(faces) + len(bodies) > 0:
        if detected:
            started = False
        else:
            detected = True
            currr = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            outp = cv2.VideoWriter(f"{currr}.mp4", fourcc, 20.0, framesize)
    elif detected:
        if started:
            if time.time() - stop_time >= AFTER_DETECTION:
                detected = False
                started = False
                outp.release()
        else:
            started = True
            stop_time = time.time()
    if detected:
        outp.write(frame)

    if SHOW_VIDEO:
        cv2.imshow("My Camera", frame)

    if cv2.waitKey(1) == ord('q'):
        break

outp.release()
camera.release()
cv2.destroyAllWindows()