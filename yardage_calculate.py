# import the necessary packages

from scipy.spatial import distance as dist
import numpy as np
import argparse
import cv2
from imutils import perspective
from imutils import contours
import imutils
import os
import math
import decimal


# function to get the midpoint of two points

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# function to get the x, y coordinates of the center of a given image

def get_coords(template):

	# get the size of the template

	w, h = template.shape[:-1]

	# find the template in the overall image

	res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
	threshold = .8

	loc = np.where(res >= threshold)

	# loop through all the places where the image has been found 
	# and add those coordinates to a list

	coord_list = []

	for pt in zip(*loc[::-1]):  # Switch columns and rows

		# adjusting to find the scenter of the image

		centerx = int(pt[0] + w/2)
		centery = int(pt[1] + h/2)
		center_coord = (centerx,centery)

		# debug to circle found templates - cv2.circle(image,centercoord,3,(0, 0, 255))

		newcoord = True

		# check that we haven't already found this marker
		# (sometimes the template match returns multiple results for the same marker)

		for (exist_x,exist_y) in coord_list:

			# measure how far this new marker is from the ones we've already found

			coord_diff = abs(exist_x - centerx) + abs(exist_y - centery)

			# if it's less than 20 total pixels away, we'll say it's not a new marker

			if coord_diff < 20:
				print("Duplicate found")
				newcoord = False

		# add new coordinate to the list

		if newcoord:
			coord_list.append(center_coord)


	return coord_list, w, h



# define colors of the scale measure and the green measurement

length_param_lower = [0,0,230]
length_param_upper = [10,10,255]


through_green_lower = [123,213,130]
through_green_upper = [158,246,163]


green_lower = [153,240,159]
green_upper = [157,245,163]



# get template images of markers for measurements

teemarker_template = cv2.imread('teemarker.png')
teedistance_lt_template = cv2.imread('distance_lt.png')
teedistance_rt_template = cv2.imread('distance_rt.png')
sprinkler_template = cv2.imread('sprinkler.png')


# initialize a list to track the green depths

green_length_list = []


# load each image in the "holes" folder, one at a time

file_list = os.listdir("holes")

for file in file_list:

	# macOS, skip .DS_Store...find a better way to do this (e.g. only pngs?)

	if file[0] == ".":
		continue

	image_path = "holes/" + file

	image = cv2.imread(image_path)

	# create NumPy arrays from the color boundaries defined earlier

	lower = np.array(length_param_lower, dtype = "uint8")
	upper = np.array(length_param_upper, dtype = "uint8")

	# find the red color of the scale measure and apply a mask

	mask = cv2.inRange(image, lower, upper)

	# blur mask for contour detection

	gray = cv2.GaussianBlur(mask, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# find contours in the edge map

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	try:
		c = cnts[0] # there should be only one contour
	except:
		print("Scale distance unable to be found")


	# compute the bounding box of the scale measure

	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the scale measure such that they appear
	# in top-left, top-right, bottom-right, and bottom-left

	box = perspective.order_points(box)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates

	(tl, tr, br, bl) = box

	# compute the Euclidean distance between top left/right and top/bottom left

	dA = dist.euclidean(tl,tr)
	dB = dist.euclidean(tl,bl)


	# the scale length is the longer of these

	scale = max(dA,dB)

	# define yards per pixel: currently using a 100 foot scale length 

	ypp = 100 / 3 / scale 

	print("yards per pixel is ",ypp)


	# measure the size of the entire hole to scale text
	# define the color range of fairway and green as numpy array 

	lower = np.array(through_green_lower, dtype = "uint8")
	upper = np.array(through_green_upper, dtype = "uint8")

	# find a mask with only the colors of the fairway and green

	mask = cv2.inRange(image, lower, upper)

	# blur mask for contour detection

	gray = cv2.GaussianBlur(mask, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# find contours in the edge map

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)


	# iterate through the contours to get the bounding box that
	# encompasses the entire fairway and green

	max_x = max_y = 0
	min_x = min_y = 6000


	for c in cnts:
		(x,y,w,h) = cv2.boundingRect(c)

		min_x, max_x = min(x, min_x), max(x+w, max_x)
		min_y, max_y = min(y, min_y), max(y+h, max_y)

	# define corners of the bounding box

	tl = (min_x,max_y)
	tr = (max_x,max_y)
	bl = (min_x,min_y)
	br = (max_x,min_y)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates

	
	(tltrX, tltrY) = mdpt1 = midpoint(tl, tr)
	(blbrX, blbrY) = mdpt2 = midpoint(bl, br)

	# determine the length of the hole compared to the total image height

	hole_length = dist.euclidean(mdpt1, mdpt2)

	print("Hole length:", hole_length)

	image_height = image.shape[0]

	print("Image height:", image_height)

	# define a factor to scale text size according to hole height
	# we need par 3 text to be smaller and par 5 text to be bigger,
	# otherwise when we scale holes up at the end the texts will be
	# wildly different size (and par 3s will get swallowed by big text)

	text_size = hole_length / image_height

	text_size = math.sqrt(text_size)

	text_size = round(text_size,2)

	print("Text size:", text_size)


	# now get the green midpoint

	# create NumPy arrays from the color boundaries of the green

	lower = np.array(green_lower, dtype = "uint8")
	upper = np.array(green_upper, dtype = "uint8")


	# find the red color of the scale measure and apply the mask

	mask = cv2.inRange(image, lower, upper)

	# blur mask for contour detection

	gray = cv2.GaussianBlur(mask, (7, 7), 0)

	# perform edge detection, then perform a dilation + erosion to
	# close gaps in between object edges

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# find contours in the edge map

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	try:
		c = cnts[0] # should be the only contour found
	except:
		print("Green could not be found")


	# get bounding box of the green

	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
	box = np.array(box, dtype="int")

	# order the points in the contour such that they appear
	# in top-left, top-right, bottom-right, and bottom-left

	box = perspective.order_points(box)

	# unpack the ordered bounding box, then compute the midpoint
	# between the top-left and top-right coordinates, followed by
	# the midpoint between bottom-left and bottom-right coordinates

	(tl, tr, br, bl) = box
	(tltrX, tltrY) = mdpt1 = midpoint(tl, tr)
	(blbrX, blbrY) = mdpt2 = midpoint(bl, br)


	# find the length of the bounding box and convert to yards
	# this is the green depth

	green_length = dist.euclidean(mdpt1, mdpt2) * ypp

	green_length = round(green_length)

	green_center = midpoint(mdpt1, mdpt2)


	# find all the tee markers, carry markers ("distance left" and "distance right"
	# depending on which way the arrow faces), and sprinkler markers

	tee_coords, tee_width, tee_height = get_coords(teemarker_template)

	distance_lt_coords, distance_lt_width, distance_lt_height = get_coords(teedistance_lt_template)

	distance_rt_coords, distance_rt_width, distance_rt_height = get_coords(teedistance_rt_template)

	sprinkler_coords, sprinkler_width, sprinkler_height = get_coords(sprinkler_template)


	# loop through carry markers and measure distance from each tee marker

	for point in distance_lt_coords:
		dist_list = []

		# loop over tee markers and measure distance from each

		for tee in tee_coords:
			distance = dist.euclidean(tee,point) * ypp
			distance = int(distance)
			dist_list.append(distance)

		print("distances: ",dist_list)

		# count number of tees

		tee_num = len(dist_list)

		# measure how big the label will be so we can center properly

		(label_width, label_height), baseline = cv2.getTextSize(str(distance), cv2.FONT_HERSHEY_SIMPLEX,text_size, 2)

		# calculate the total label height

		totalheight = (32 * (tee_num-1) * text_size) 

		# declare x and y coordinates to place the text

		x = int(point[0] - (80 * (text_size + 0.1)))
		y = int(point[1] - totalheight/2 + distance_lt_height/2 + baseline)

		# declare an increment for each new tee distance (so that they stack vertically)

		yinc = int(32 * text_size)

		# now for each distance found, write it on the image next to the marker

		for distance in dist_list:

			cv2.putText(image, str(distance), (x , y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)

			y += yinc



	# loop through carry markers and measure distance from each tee marker

	for point in distance_rt_coords:
		dist_list = []

		# loop over tee markers and measure distance from each

		for tee in tee_coords:
			distance = dist.euclidean(tee,point) * ypp
			distance = int(distance)
			dist_list.append(distance)

		print("distances: ",dist_list)

		# count number of tees

		tee_num = len(dist_list)

		# measure how big the label will be so we can center properly

		(label_width, label_height), baseline = cv2.getTextSize(str(distance), cv2.FONT_HERSHEY_SIMPLEX,text_size, 2)

		# calculate the total label height

		totalheight = (32 * (tee_num-1) * text_size) 

		# declare x and y coordinates to place the text

		x = int(point[0] + (14 * (text_size + 0.1)))
		y = int(point[1] - totalheight/2 + distance_lt_height/2 + baseline)

		# declare an increment for each new tee distance (so that they stack vertically)

		yinc = int(32 * text_size)

		# now for each distance found, write it on the image next to the marker

		for distance in dist_list:

			cv2.putText(image, str(distance), (x , y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)

			y += yinc



	# loop through the "sprinkler" markers and get distances to the center of the green

	for point in sprinkler_coords:
		distance = dist.euclidean(point,green_center) * ypp
		distance = int(distance)

		# measuring the size of the label we are adding to properly center the distance

		(label_width, label_height), baseline = cv2.getTextSize(str(distance), cv2.FONT_HERSHEY_SIMPLEX,text_size, 2)

		# declare x and y points to place the text so that it is centered

		x = int(point[0] - label_width/2 + sprinkler_width/2)
		y = int(point[1] + 30)

		# write the distance to the center of the green under this marker

		cv2.putText(image, str(distance), (x,y), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), 2)

	
	# write the output image to a folder called "yardages" in our current directory

	output_path = 'yardages/' + file

	# create a list of green depths to be added on later

	length_text = file + " green length is " + str(green_length)
	green_length_list.append(length_text)


	# save the image 

	cv2.imwrite(output_path, image)


# print out all the green lengths 

for green in green_length_list:
	print(green)



