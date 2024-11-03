import cv2

# function to display the coordinates of points clicked on the image
def click_event(event, x, y, flags, params):

	# checking for left mouse clicks
	if event == cv2.EVENT_LBUTTONDOWN:

		# display coordinates on Shell
		print(x, ' ', y)

		# display coordinates on image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img, str(x) + ',' +
					str(y), (x,y), font,
					1, (255, 0, 0), 2)
		cv2.imshow('image', img)

	# checking for right mouse clicks	
	if event==cv2.EVENT_RBUTTONDOWN:

		# display coordinates on the Shell
		print(x, ' ', y)

		# displaying the coordinates
		# on the image window
		font = cv2.FONT_HERSHEY_SIMPLEX
		b = img[y, x, 0]
		g = img[y, x, 1]
		r = img[y, x, 2]
		cv2.putText(img, str(b) + ',' +
					str(g) + ',' + str(r),
					(x,y), font, 1,
					(255, 255, 0), 2)
		cv2.imshow('image', img)

if __name__=="__main__":

	# read image
    # img = cv2.imread('', 1)
	# img = cv2.imread('', 1)

	# display image
    cv2.imshow('image', img)

	# setting mouse handler for the image
    cv2.setMouseCallback('image', click_event)

    cv2.waitKey(0)

    cv2.destroyAllWindows()