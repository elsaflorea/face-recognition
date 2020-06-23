from imutils.video import FPS
from imutils.video import VideoStream
from datetime import datetime
import time
import ffmpeg
import numpy as np
import pickle
import cv2
import os
import pymysql.cursors
import smtplib

from configparser import ConfigParser
config = ConfigParser()
config.read('config.ini')


fmt = '%Y-%m-%d %H:%M:%S'
last_date = datetime.strptime(datetime.now().strftime(fmt), fmt)

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([config.get('face_recognition', 'detector'), "deploy.prototxt"])
modelPath = os.path.sep.join([config.get('face_recognition', 'detector'), "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(config.get('face_recognition', 'embedding_model'))

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(config.get('face_recognition', 'recognizer'), "rb").read())
le = pickle.loads(open(config.get('face_recognition', 'le'), "rb").read())

# initialize the ImageHub object
vs = VideoStream(0).start()

# initialize the video stream, then allow the camera sensor to warm up
print("[INFO] starting video stream...")
time.sleep(2.0)

# start the FPS throughput estimator
fps = FPS().start()

gmail_user = config.get('gmail', 'user')
gmail_password = config.get('gmail', 'password')

sent_from = gmail_user
to = [config.get('gmail', 'to')]
subject = 'Miscare neautorizata detectata'

# Connect to the database
host = config.get('database', 'host')
user = config.get('database', 'user')
password = config.get('database', 'pass')
db = config.get('database', 'db_name')

db_connection = pymysql.connect(host, user, password, db, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

process = (
    ffmpeg
        .input('pipe:', r='6')
        .output(f"rtmp://{config.get('streaming_server', 'host')}/live/{config.get('streaming_server', 'name')}",
                vcodec='libx264', pix_fmt='yuv420p', preset='veryfast',
                r='20', g='50', video_bitrate='1.4M', maxrate='2M', bufsize='2M', segment_time='6',
                format='flv')
        .run_async(pipe_stdin=True)
)

# start looping over all the frames
while True:
    # receive RPi name and frame from the RPi and acknowledge
    # the receipt
    frame = vs.read()

    # resize the frame to have a maximum width of 400 pixels, then
    # grab the frame dimensions and construct a blob
    frame = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA)
    # imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)

    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections
        if confidence > config.getfloat('face_recognition', 'confidence'):
            # compute the (x, y)-coordinates of the bounding box for
            # the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            # construct a blob for the face ROI, then pass the blob
            # through our face embedding model to obtain the 128-d
            # quantification of the face
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                             (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]

            # draw the bounding box of the face along with the
            # associated probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            identify_color = (0, 0, 255)
            check_date = datetime.strptime(datetime.now().strftime(fmt), fmt)
            timeDiff = (check_date - last_date).seconds

            if name != 'unknown':
                identify_color = (0, 255, 0)

                if timeDiff > 60:
                    try:
                        with db_connection.cursor() as cursor:
                            # Create a new record
                            sql = "UPDATE `location_stats` SET `value` = 0 WHERE name = 'state' "
                            cursor.execute(sql)
                        db_connection.commit()
                    finally:
                        pass

                    last_date = datetime.strptime(datetime.now().strftime(fmt), fmt)
            else:
                if timeDiff > 60:
                    try:
                        body = f'Miscare neautorizata la {check_date}'
                        email_text = f'From: {sent_from} \nTo: {", ".join(to)} \nSubject: {subject} \n\n{body}'

                        print(email_text)
                        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                        server.ehlo()
                        server.login(gmail_user, gmail_password)
                        server.sendmail(sent_from, to, email_text)
                        server.close()
                    finally:
                        pass
                    last_date = datetime.strptime(datetime.now().strftime(fmt), fmt)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          identify_color, 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, identify_color, 2)

    # update the FPS counter
    fps.update()

    ret2, frame2 = cv2.imencode('.png', frame)
    process.stdin.write(frame2.tobytes())

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
vs.stop()
