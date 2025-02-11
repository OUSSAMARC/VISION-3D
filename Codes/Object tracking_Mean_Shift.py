import numpy as np
import cv2

roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    # If the left mouse button was clicked, record the starting ROI coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # If the left mouse button was released, record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True

cap = cv2.VideoCapture('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/VOT-Ball.mp4')

# Take the first frame of the video
ret, frame = cap.read()
# Load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)

# Keep looping until the 'q' key is pressed
while True:
    # Display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # If the ROI is defined, draw it
    if roi_defined:
        # Draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    # Else, reset the image
    else:
        frame = clone.copy()
    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r, c, h, w)
# Set up the ROI for tracking
roi = frame[c:c + w, r:r + h]
# Conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# Computation mask of the histogram:
# Pixels with S<30, V<20, or V>235 are ignored
mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
# Marginal histogram of the Hue component
#roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
# Compute histogram for Hue and Saturation channels
roi_hist = cv2.calcHist([hsv_roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
# Histogram values are normalized to [0,255]
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria: either 10 iterations or move by less than 1 pixel
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cpt = 1
while True:
    ret, frame = cap.read()
    if ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Backproject the model histogram roi_hist onto the current image hsv
        #dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        dst = cv2.calcBackProject([hsv], [0, 1], roi_hist, [0, 180, 0, 256], 1)
        dst = cv2.GaussianBlur(dst, (5, 5), 0)  # Smoothing for robustness
        
        # Apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw a blue rectangle on the current image
        r, c, h, w = track_window
        frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)
        cv2.imshow('Sequence', frame_tracked)

        cv2.imshow("Backprojection", dst)
        hue_channel, saturation_channel, value_channel = cv2.split(hsv)
        cv2.imshow("Hue Channel", hue_channel)
        cv2.imshow("S Channel", saturation_channel)
        
        """
		# Updating the histogram
        x, y, w, h = track_window
        updated_roi = hsv[y:y+h, x:x+w]
        updated_mask = cv2.inRange(updated_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
        updated_hist = cv2.calcHist([updated_roi], [0], updated_mask, [180], [0, 180])
        cv2.normalize(updated_hist, updated_hist, 0, 255, cv2.NORM_MINMAX)
        roi_hist = updated_hist
        """
        
        if cv2.waitKey(100) & 0xFF == ord('u'):  # Press 'u' to save
            
            cv2.imwrite('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/Sequence_hs.png', frame_tracked)
            cv2.imwrite("C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/Backprojection_hs.png", dst)
            cv2.imwrite("C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/h.png", hue_channel)
            cv2.imwrite("C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/s.png", saturation_channel)
            print("Images saved successfully !")
            
        if cv2.waitKey(100) & 0xFF == ord('s'):  # Press 's' to stop
            break

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png' % cpt, frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()
