import cv2
import numpy as np

# Initialize variables
drawing = False  # True if mouse is pressed
points = []      # Stores points of the polygon
im = None  # To be defined later

def draw_polygon(event, x, y, flags, param):
    global drawing, points, im

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(im, (x, y), 2, (0, 255, 0), -1)  # Draw a small circle at each point
        if len(points) > 1:
            cv2.line(im, points[-2], points[-1], (255, 0, 0), 1)  # Draw lines between points
        cv2.imshow("Image", im)

def process_body_parts(nm, BODY_PARTS):
    imgarr = {}
    global im
    im = nm.copy()

    for part_name, landmarks in BODY_PARTS.items():
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        points.clear()

        teximg = nm.copy()
        cv2.putText(teximg, part_name, (50, 50), font, fontScale, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Image", teximg)
        cv2.setMouseCallback("Image", draw_polygon)

        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Create mask and extract the selected region
        mask = np.zeros(nm.shape[:2], dtype=np.uint8)
        if points:
            points_np = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points_np], (255, 255, 255))

        roi = cv2.bitwise_and(nm, nm, mask=mask)
        cv2.imshow("crop", mask)

        # Crop the bounding rectangle around the polygon for easy manipulation
        if points:
            x, y, w, h = cv2.boundingRect(points_np)
            roi_cropped = roi[y:y+h, x:x+w]
            mask_cropped = mask[y:y+h, x:x+w]

            # Rotate the ROI
            center = (w // 2, h // 2)
            angle = 45  # Rotate by 45 degrees (change as needed)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_roi = cv2.warpAffine(roi_cropped, rotation_matrix, (w, h))

            cv2.imshow("leg", rotated_roi)
            imgarr[part_name] = roi

        while True:
            key = cv2.waitKey(1000)
            if key == ord('s'):
                break
            if key == ord('e'):
                return imgarr  # Exit the function and return the result

    for i, im in imgarr.items():
        cv2.imshow("new", im)
    
    return imgarr



def remove_background_grabcut(image_):
    # Load the image
    image = image_
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define initial rectangle around the foreground object
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)

    # Create temporary arrays needed by the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify mask so that only sure and probable foreground pixels are set to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Multiply the mask with the original image to get the result
    result = image * mask2[:, :, np.newaxis]

    # Convert to BGRA (with alpha channel) to have a transparent background
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask2 * 255

    return result


import numpy as np

def calculate_body_part_points(results, BODY_PARTS, BODY_PART_INDICES):
    body_part_points = {}

    if results.pose_landmarks:
        # Calculate points for each body part
        print("hello")
        for part_name, landmarks in BODY_PARTS.items():
            part_points = []
            for landmark_name in landmarks:
                idx = BODY_PART_INDICES[landmark_name]
                landmark = results.pose_landmarks.landmark[idx]
                x, y = landmark.x, landmark.y
                part_points.append((x, y))
            body_part_points[part_name] = np.array(part_points)
    centx=body_part_points["lower_torso"][0][0]+body_part_points["lower_torso"][1][0];
    centx=centx/2;
    centy=body_part_points["lower_torso"][0][1]+body_part_points["lower_torso"][1][1];
    centy=centy/2;
    body_part_points["lower_torso"][0][0]=centx;
    body_part_points["lower_torso"][0][1]=centy;
    return body_part_points
