import cv2
import numpy as np

roi_defined = False

def define_ROI(event, x, y, flags, param):
    global r, c, w, h, roi_defined
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2 - r)
        w = abs(c2 - c)
        r = min(r, r2)
        c = min(c, c2)
        roi_defined = True
    return r, c, h, w

def Reference(r, c, h, w, frame, first_frame=1):
    ref_point = (r + h // 2, c + w // 2)

    H, W, _ = frame.shape
    
    if first_frame == 1 :
       image = np.zeros((H, W,3), dtype=np.float32)
       image[r:r + h, c:c + w ] = frame[r:r + h, c:c + w ]
    else : image = frame
    


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = cv2.magnitude(grad_x, grad_y)
    orientation = cv2.phase(grad_x*100, grad_y, angleInDegrees=True)

    mask = (magnitude >= 100) & (magnitude < 300)  # Keep magnitudes between 100 and 150
    orientation_masked = orientation.copy()
    magnitude_masked = magnitude.copy()
    orientation_masked[~mask] = 0  # Set insignificant gradients to zero
    magnitude_masked[~mask] = 0

    return ref_point, magnitude_masked, orientation_masked

def calculate_r_table(edge_img, orientation_img, ref_point):
    r_table = {}
    h, w = edge_img.shape
    for y in range(h):
        for x in range(w):
            if edge_img[y, x] > 0:
                orientation = int(orientation_img[y, x])
                vector = (ref_point[0] - y, ref_point[1] - x)
                if orientation not in r_table:
                    r_table[orientation] = []
                r_table[orientation].append(vector)
    return r_table

def hough_transform(edge_img, orientation_img, r_table):
    h, w = edge_img.shape
    accumulator = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            if orientation_img[y, x] > 0:
                orientation = int(orientation_img[y, x])
                if orientation in r_table:
                    for dy, dx  in r_table[orientation]:
                        xc = x + dx
                        yc = y + dy
                        if 0 <= xc < w and 0 <= yc < h:
                            accumulator[yc, xc] += 1
    return accumulator

video_path = 'C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/Antoine_Mug.mp4'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Error: Cannot read video.")
    exit()

cv2.namedWindow("Select ROI")
cv2.setMouseCallback("Select ROI", define_ROI)

while True:
    temp_frame = frame.copy()
    if roi_defined:
        cv2.rectangle(temp_frame, (r, c), (r + h, c + w), (0, 255, 0), 2)
    cv2.imshow("Select ROI", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') and roi_defined:
        break

cv2.destroyWindow("Select ROI")

ref_point, magnitude_masked, orientation_masked = Reference(r, c, h, w, frame, first_frame = 1)

r_table = calculate_r_table(magnitude_masked, orientation_masked, ref_point)


while True:
    ret, frame = cap.read()
    if not ret:
        break
    ref_point, magnitude_masked, orientation_masked = Reference(r, c, h, w, frame, first_frame = 0)
    accumulator = hough_transform(magnitude_masked, orientation_masked, r_table)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(accumulator)
    frame_detected = frame.copy()
    # Define the top-left and bottom-right corners of the rectangle
    top_left = (max_loc[0] - h // 2, max_loc[1] - w // 2)
    bottom_right = (max_loc[0] + h // 2, max_loc[1] + w // 2)

    # Draw the rectangle
    cv2.rectangle(frame_detected, top_left, bottom_right, (0, 255, 0), 2)

    # Draw the reference point (blue dot)
    cv2.imshow("Detection", frame_detected)
    cv2.imshow("Accumulator", cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    cv2.imshow("Dete", magnitude_masked)
    cv2.imshow("Acr", cv2.normalize(orientation_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))

    if cv2.waitKey(100) & 0xFF == ord('u'):  # Press 'u' to save
        cv2.imwrite('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/frame_detected.png', frame_detected)
        cv2.imwrite('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/accumulator.png', cv2.normalize(accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        cv2.imwrite('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/magnitude_masked.png', (magnitude_masked * 255).astype(np.uint8))
        cv2.imwrite('C:/Users/OUSSAMA/OneDrive/Bureau/M2 IMA-TAIV/VISION/orientation_masked.png', cv2.normalize(orientation_masked, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
        print("Images saved successfully!")

    if cv2.waitKey(100) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()