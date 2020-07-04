# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("-f", "--face", type=str, default="face_detector", help="path to face detector model directory")
argument_parser.add_argument("-m", "--model", type=str, default="mask_detector.model", help="path to trained face mask detector model")
arguments = vars(argument_parser.parse_args())

CONFIDENCE_THRESHOLD = 0.5

# Load face detector
prototxt_path = os.path.sep.join([arguments["face"], "deploy.prototxt"])
weights_path  = os.path.sep.join([arguments["face"], "res10_300x300_ssd_iter_140000_fp16.caffemodel"])
face_net 	  = cv2.dnn.readNet(prototxt_path, weights_path)

# Load face mask detector
mask_net = load_model(arguments["model"])

# Video stream start
video_stream = VideoStream(src=0).start()
time.sleep(2.0)

def detect_faces_and_predict_mask(frame, face_net, mask_net):
	(h, w) = frame.shape[:2]
	blob   = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
	
	face_net.setInput(blob)
	detections = face_net.forward()
	
	face_roi_list 	   = []
	face_location_list = []
	prediction_list    = []

	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		
		if confidence > CONFIDENCE_THRESHOLD:
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			
			(startX, startY, endX, endY) = box.astype("int")
			
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) 	 = (min(w - 1, endX), min(h - 1, endY))

			face_roi = frame[startY:endY, startX:endX]
			face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
			face_roi = cv2.resize(face_roi, (224, 224))
			face_roi = img_to_array(face_roi)
			face_roi = preprocess_input(face_roi)
			face_roi = np.expand_dims(face_roi, axis=0)
			
			face_roi_list.append(face_roi)
			face_location_list.append((startX, startY, endX, endY))

	if len(face_roi_list) > 0:
		prediction_list = mask_net.predict(face_roi_list)
            
	return (face_location_list, prediction_list)

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = video_stream.read()
	frame = imutils.resize(frame, width=400)
	# detect faces in the frame and determine if they are wearing a
	# face mask or not
	(locs, preds) = detect_and_predict_mask(frame, face_net, mask_net)

	# loop over the detected face locations and their corresponding
	# locations
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred
		# determine the class label and color we'll use to draw
		# the bounding box and text
		label = "With Mask" if mask > withoutMask else "No Mask"
		color = (0, 190, 0) if label == "With Mask" else (0, 0, 255)
		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		# display the label and bounding box rectangle on the output
		# frame
		cv2.putText(img=frame, text=label, org=(startX, startY - 10), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=color, thickness=2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
	# show the output frame
	cv2.imshow("Doug M. - Face Mask Detector", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
video_stream.stop()
