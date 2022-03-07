# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import dlib
from PIL import Image

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


# load the image, convert it to grayscale, and blur it slightly
width = 19.05
image = cv2.imread("ej.JPG")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)
# perform edge detection, then perform a dilation + erosion to
# close gaps in between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# find contours in the edge map

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)


cnts = imutils.grab_contours(cnts)


# sort the contours from left-to-right and initialize the
# 'pixels per metric' calibration variable
#(cnts, _) = contours.sort_contours(cnts)
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)
(cnts, boundingBoxes) = sort_contours(cnts, method="top-to-bottom")


	# return the list of sorted contours and bounding boxes

pixelsPerMetric = None

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
faces = detector(gray)

for face in faces:

    landmarks = predictor(gray, face)
    myPoints = []

    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        myPoints.append([x,y])
        cv2.circle(image,(x,y),2,(50,50,225),cv2.FILLED)


# loop over the contours individually
for c in cnts:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 1000:
		continue
	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	myPoints = np.array(myPoints, dtype="int")
	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left
	# order, then draw the outline of the rotated bounding
	# box
	box = perspective.order_points(box)

	#face

	# loop over the original points and draw them
	#for (x, y) in box:
		#cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		# unpack the ordered bounding box, then compute the midpoint
		# between the top-left and top-right coordinates, followed by
		# the midpoint between bottom-left and bottom-right coordinates

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)
	ltl = myPoints[36].astype(int)
	ltr = myPoints[39].astype(int)
	rtl = myPoints[42].astype(int)
	rtr = myPoints[45].astype(int)

		# compute the midpoint between the top-left and top-right points,
		# followed by the midpoint between the top-righ and bottom-right
		# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
	cv2.circle(orig, (int(ltl[0]), int(ltl[1])), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(ltr[0]), int(ltr[1])), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(rtl[0]), int(rtl[1])), 2, (255, 0, 0), -1)
	cv2.circle(orig, (int(rtr[0]), int(rtr[1])), 2, (255, 0, 0), -1)
		# draw lines between the midpoints
	print(ltl)

	cv2.line(orig, (int(ltl[0]), int(ltl[1])), (int(ltr[0]), int(ltr[1])),
			 (255, 0, 255), 2)
	cv2.line(orig, (int(rtl[0]), int(rtl[1])), (int(rtr[0]), int(rtr[1])),
			 (255, 0, 255), 2)
	cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(55, 0, 55), 2)

			# compute the Euclidean distance between the midpoints
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
	edB = dist.euclidean((ltl[0], ltl[1]), (ltr[0], ltr[1]))
	erdB = dist.euclidean((rtl[0], rtl[1]), (rtr[0], rtr[1]))
			# if the pixels per metric has not been initialized, then
			# compute it as the ratio of pixels to supplied metric

	if pixelsPerMetric is None:
			pixelsPerMetric = dB / width

			# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric
	edimB = edB / pixelsPerMetric
	erdimB = erdB / pixelsPerMetric

			# draw the object sizes on the image
	cv2.putText(orig, "{:.1f}mm".format(dimA),
						(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}mm".format(dimB),
						(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
						0.65, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}mm".format(edimB),
						(int(ltr[0]), int(ltr[1]+10)), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}mm".format(erdimB),
						(int(rtr[0]), int(rtr[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX,
						0.5, (255, 255, 255), 2)


			# show the output image
	if erdimB < 25:
		print(str(erdimB) + " WITH FAS")
	else:
		print(str(erdimB) + " WITHOUT FAS")
	cv2.imshow("Contour", orig)


	cv2.waitKey(0)

