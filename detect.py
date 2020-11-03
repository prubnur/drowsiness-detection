
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2


def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)


def eye_aspect_ratio(eye):
    # vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # horizontal
    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    # return ear
    return ear

def mouth_aspect_ratio(mouth):
    # vertical
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[3], mouth[9])
    C = dist.euclidean(mouth[4], mouth[8])

    # horizontal
    D = dist.euclidean(mouth[0], mouth[6])

    mar = (A + B + C) / (2.0 * D)
    # return mar
    return mar

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=str, default="",
                help="path alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.20
EYE_AR_BLINK_FRAMES = 3
EYE_AR_CONSEC_FRAMES = 48
EYE_BLINK_THRESH = 0.30
YAWN_THRESH = 0.90

# frame counter and blink/yawn counters
BFRAMECOUNTER = 0
COUNTER = 0
ALARM_ON = False
BLINK_COUNT = 0
BLINK_FLAG = 0
YAWN_COUNT = 0
YAWN_FLAG = 0

# initialize dlib's face detector (HOG-based)
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# get eyes and mouth indexes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
# loop over frames from the video stream
while True:
    # resize and convert frame to greyscale
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect
    rects = detector(gray, 0)

    # for each face
    for rect in rects:
        # get coordinates using predictor and convert to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # use mouth, left eye and right eye coordinates to get EAR and MAR
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # check for blink and update count using a blink flag
        if ear < EYE_BLINK_THRESH:
            BFRAMECOUNTER += 1
        else:
            if BFRAMECOUNTER > EYE_AR_BLINK_FRAMES:
                BLINK_COUNT += 1
            BFRAMECOUNTER = 0

        mouth = shape[mStart:mEnd]

        mar = mouth_aspect_ratio(mouth)

        # check for yawn and update count using a yawn flag
        if mar > YAWN_THRESH:
            YAWN_FLAG += 1
        elif YAWN_FLAG != 0:
            YAWN_FLAG = 0
            YAWN_COUNT += 1

        # draw eye hulls
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        # display metrics on frame
        cv2.putText(frame, "E: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "M: {:.2f}".format(mar), (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "B: {:}".format(BLINK_COUNT), (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Y: {:}".format(YAWN_COUNT), (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # if ear is less than threshold increment frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            
            # if eyes closed for long sound alarm
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # if the alarm is not on, turn it on
                if not ALARM_ON:
                    ALARM_ON = True
                    # start new thread to play alarm file in the background
                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm,
                                   args=(args["alarm"],))
                        t.deamon = True
                        t.start()

        # reset counter
        else:
            COUNTER = 0
            ALARM_ON = False

    # show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # press 'q' to exit (breaks loop)
    if key == ord("q"):
        break
# exit
cv2.destroyAllWindows()
vs.stop()
