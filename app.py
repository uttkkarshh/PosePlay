import torch;
import numpy as np;
import mediapipe as mp
import cv2;
import pickle
import dra
import math
from scipy.spatial.transform import Rotation as R
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

im=cv2.imread("assets/download.jpg",1);
im[0:100,0:20]=np.zeros((100, 20,3));


vid=cv2.VideoCapture('assets/dance.mp4');

nm =cv2.imread('assets/nm.png',-1);
# Add 200 transparent pixels to the left and right
top, bottom, left, right = 0, 0, 200, 200
transparent_padding = [0, 0, 0, 0]  # RGBA for transparency
nm = cv2.copyMakeBorder(nm, top, bottom, left, right, cv2.BORDER_CONSTANT, value=transparent_padding)

# Define body part groups by landmark indices
BODY_PARTS = {
        "left_hand": ["left_shoulder", "left_elbow"],
        "right_hand": ["right_shoulder", "right_elbow"],
        "left_leg": ["left_hip", "left_knee"],
        "right_leg": ["right_hip", "right_knee"],
        "lower_torso": ["left_hip", "right_hip"],
        "extended_left_leg": ["left_knee", "left_ankle"],
        "extended_right_leg": ["right_knee", "right_ankle"],
        "extended_right_hand": ["right_elbow", "right_wrist"],
        "extended_left_hand": ["left_elbow", "left_wrist"],
        "face":["nose","upper_lip"]
}


def calculate_angle(x, y):
    
    # Use atan2 to calculate the angle in radians
    angle_rad = math.atan2(y, x)

    # Convert radians to degrees
    angle_deg = math.degrees(angle_rad)

    # Normalize angle to the range [0, 360)
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg




# Function to handle mouse clicks
def click_event(event, x, y, flags, params):
    # Check for left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked at: x={x}, y={y}")
        # Display the coordinates on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(nm, f"({x},{y})", (x, y), font, 0.5, (255, 0, 0), 2)
        cv2.imshow("Image", nm)

def overlay_image(canvas, img):
    alpha = img[:, :, 3] / 255.0  # Normalize alpha channel to range 0-1
    for c in range(3):  # Loop over color channels
        canvas[:, :, c] = canvas[:, :, c] * (1 - alpha) + img[:, :, c] * alpha
    return canvas

#cv2.imshow("Image", nm)
lleg=nm[200:540,360:602];
imgarr={}
bol=False
with open("image_data0.pkl", "rb") as file:
    imgarr = pickle.load(file)
print("Image data loaded!")




BODY_PART_INDICES = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "upper_lip": 13
}



height, width, _ = nm.shape

# Initialize a dictionary to store landmarks from the previous frame
# Set OpenCV window to be resizable
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# Resize the window to fit your screen
#cv2.resizeWindow("Video", 1100, 1100)  # Example dimensions; adjust as needed
last_landmarks = None
di=np.empty((2, 0));

takeframe=True;

imga=nm;

#imga=cv2.resize(nm,(1000,1000))
imgno="hel4";
h, w = nm.shape[:2]
aspect_ratio = w / h
new_h, new_w = 1000, int(1000 * aspect_ratio)
imga = cv2.resize(nm, (new_w, new_h))
if takeframe:
       #imga=dra.remove_background_grabcut(imga);
       imgarr=dra.process_body_parts(imga,BODY_PARTS)
       with open(f"image_data{imgno}.pkl", "wb") as file:
          pickle.dump(imgarr, file)
       print("Image data saved!");
else:
        with open("image_datahel4.pkl", "rb") as file:
            imgarr = pickle.load(file)
        print("Image data loaded!")


imga = cv2.cvtColor(imga, cv2.COLOR_BGRA2RGB)   

res=pose.process(imga);

lastpoints=None;
firstpoints=dra.calculate_body_part_points(res,BODY_PARTS,BODY_PART_INDICES);
image=imga.copy();


points_change={}
for part_name, landmarks in BODY_PARTS.items():
    points_change[part_name]=[(0,0),(0,0)];

canvas = np.zeros_like(nm)






while vid.isOpened():
      ret ,img=vid.read();
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      h,w=new_h,new_w;

      img = cv2.resize(img, (new_w, new_h))
      
      
      results = pose.process(img)
      if results.pose_landmarks:
         mp_drawing.draw_landmarks(
         img, 
         results.pose_landmarks, 
         mp_pose.POSE_CONNECTIONS,
         mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
         mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
         )
      
      landmarks=None;
      body_part_points = {}
      
      if results.pose_landmarks:
       landmarks = results.pose_landmarks.landmark;
      else :
        continue;
      if results.pose_landmarks:
        # Calculate points for each body part
        for part_name, landmarks in BODY_PARTS.items():
            part_points = []
            for landmark_name in landmarks:
                idx = BODY_PART_INDICES[landmark_name]
                landmark = results.pose_landmarks.landmark[idx] 
                x, y = (landmark.x ), (landmark.y )
                part_points.append((x, y))    
            body_part_points[part_name] = np.array(part_points)
      
       
# Update `last_landmarks` with the currentimg's landmarks for next comparison
      
      last_landmarks = [landmark for landmark in landmarks];
      canvas = np.zeros((new_h, new_h, 4), dtype=np.uint8)
      newim=np.ones_like(nm);
      if lastpoints !=None :
        for part, points in body_part_points.items():
            if part=='upper_torso' or part=='lower_torso':
                    centx=points[0][0]+points[1][0];
                    centx=centx/2;
                    centy=points[0][1]+points[1][1];
                    centy=centx/2;
                    points[0][0]=centx;
                    points[0][1]=centy;
                    dx=points[0][0]-lastpoints[part][0][0];
                    dy=points[0][1]-lastpoints[part][0][1];
                    translation_matrix = np.float32([[1, 0, dx*w], [0, 1, dy*h]])
                    translated_part = cv2.warpAffine(imgarr[part], translation_matrix, (canvas.shape[1], canvas.shape[0]))
                    canvas=overlay_image(canvas, translated_part)
                    imgarr[part]=translated_part;
                    continue;
            dx=points[0][0]-lastpoints[part][0][0];
            dy=points[0][1]-lastpoints[part][0][1];
            rotx =points[1][0]-dx;
            roty=points[1][1]-dy;
            ang=calculate_angle(roty-lastpoints[part][0][1],rotx-lastpoints[part][0][0])
            anglast =calculate_angle(lastpoints[part][1][1]-lastpoints[part][0][1],lastpoints[part][1][0]-lastpoints[part][0][0]);
            angle= (ang-anglast);
            
            rotation_matrix = cv2.getRotationMatrix2D((firstpoints[part][0][0]*w,firstpoints[part][0][1]*h), angle, 1.0)
            rotated_roi = cv2.warpAffine(imgarr[part], rotation_matrix, (w, h))
            # Translate the body part image
            firstpoints[part][0][0]+=dx;
            firstpoints[part][0][1]+=dy;
            translation_matrix = np.float32([[1, 0, dx*w], [0, 1, dy*h]])
            translated_part = cv2.warpAffine(rotated_roi, translation_matrix, (canvas.shape[1], canvas.shape[0]))
            canvas=overlay_image(canvas, translated_part)
            imgarr[part]=translated_part;
            
      else:
            centx=body_part_points['lower_torso'][0][0]+body_part_points['lower_torso'][1][0];
            centx=centx/2;
            centy=body_part_points['lower_torso'][0][1]+body_part_points['lower_torso'][1][1];
            centy=centx/2;
            body_part_points['lower_torso'][0][0]=centx;
            body_part_points['lower_torso'][0][1]=centy;
      lastpoints=body_part_points.copy();     
      tmp=nm.copy();
      #mp_drawing.draw_landmarks(
      #  tmp, resultsnm.pose_landmarks, mp_pose.POSE_CONNECTIONS)# Display the image with pose landmarks
      print(firstpoints);
      cv2.imshow("Pose", canvas)
      cv2.imshow("Video", img)
      # Draw points on the image
      for part_name, points in firstpoints.items():
            for point in points:
                x, y = point
                # Draw the point
                cv2.circle(image, (int(x*new_w), int(y*new_h)), radius=5, color=(0, 0, 255), thickness=-1)
                # Optionally, add text labels
                cv2.putText(image, part_name, (int(x*new_w), int(y*new_h)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
      cv2.imshow("firts",image);
      image=imga.copy();
      #cv2.setMouseCallback("Pose", click_event)
      if cv2.waitKey(1)==ord('q'):
        break
      

vid.release();

cv2.destroyAllWindows();
