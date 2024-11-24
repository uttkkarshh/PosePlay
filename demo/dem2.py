import cv2
import numpy as np

def remove_background_grabcut(image_path):
    # Load the image
    image = cv2.imread(image_path)
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

# Load and process the image
image_path = 'assets/fin.png'  # Replace with your image path
result = remove_background_grabcut(image_path)

# Save the result
cv2.imwrite('result.png', result)

# Display the result
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()



'''       # Calculate central points for torso regions
        for center_name, landmarks in CENTRAL_POINTS.items():
            central_x, central_y = 0, 0
            num_points = len(landmarks)
            for landmark_name in landmarks:
                idx = BODY_PART_INDICES[landmark_name]
                landmark = results.pose_landmarks.landmark[idx]
                x, y = int(landmark.x *img.shape[1]), int(landmark.y *img.shape[0])
                central_x += x
                central_y += y
            # Average the points to get the center
            central_points[center_name] = (central_x // num_points, central_y // num_points)

        
for part_name, landmarks in BODY_PARTS.items():
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    points = []  
    teximg=nm.copy();
    cv2.putText(teximg, part_name, (50,50), font, fontScale, (255,255,0), 2, cv2.LINE_AA)
    cv2.imshow("Image", teximg)
    while cv2.waitKey(1)!=ord('q'):
     continue;
    
# Create mask and extract the selected region
# Create a mask for the polygon area

   


    mask = np.zeros(nm.shape[:2], dtype=np.uint8)
    if len(points) > 0:
        points_np = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points_np], (255, 255, 255))



   
    roi = cv2.bitwise_and(nm, nm, mask=mask)
    cv2.imshow("crop",mask);
    # Crop the bounding rectangle around the polygon for easy manipulation
    if len(points) > 0:
        x, y, w, h = cv2.boundingRect(points_np)
        roi_cropped = roi[y:y+h, x:x+w]
        mask_cropped = mask[y:y+h, x:x+w]

        # Rotate the ROI
    center = (w // 2, h // 2)
    angle = 45  # Rotate by 45 degrees (change as needed)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_roi = cv2.warpAffine(roi_cropped, rotation_matrix, (w, h))

    cv2.imshow("leg",rotated_roi)
    imgarr[part_name]=roi;
    while True:
       if cv2.waitKey(1000)==ord('s'):
          break;
    if cv2.waitKey(1000)==ord('e'):
        break;



for i, im in imgarr.items():
   cv2.imshow("new",im);



'''