import os
import cv2
import glob
import numpy as np
from random import shuffle

drawing = False
mode = True
ix, iy = -1,-1


done_list = []

frame_count = 0
with open("dataset.txt", "r") as f:
	for line in f:
		line = line.strip().split(",")
		frame_count = int(line[0])
		f_name = line[1]
		done_list.append(f_name)

if frame_count != 0:
	frame_count += 1

file = open("dataset.txt", 'a')
#mouse callback function
def draw_circle(event, x, y, flags, param):
	global ix,iy,drawing,mode 
	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix,iy = x,y
	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == True:
				cv2.rectangle(img, (ix,iy),(x,y),(0,255,0),3)
	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), 3)
			coordinate = [ix,iy,x,y]
			width = abs(x-ix) + 20
			height = abs(y-iy) + 20
			file.write(str(param["k"]) + "," + str(imga)+ "," + str(width) + "," + str(height) + "," +str(ix)+","+str(iy)+","+str(x)+","+str(y)+"\n")


for r, d, f in os.walk("data"):

	for data in d:
		frames = os.listdir("data/"+data)
		frames = frames[-5:]

		shuffle(frames)
		
		for imga in frames:

			if imga in done_list:
				continue

			img = cv2.imread("data/"+data+"/"+imga)
			clone = np.copy(img)

			cv2.namedWindow('image')
			cv2.setMouseCallback('image', draw_circle, param={"k": frame_count})

			while(1):
				cv2.imshow('image',img )
				img = np.copy(clone)
				k = cv2.waitKey(1) & 0xFF
				if k == ord('m'):
					break

				if k == ord('q'):
					cv2.destroyAllWindows()
					file.close()
					exit(0)
			frame_count += 1

cv2.destroyAllWindows()
file.close()