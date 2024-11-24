import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Load an image
image = cv2.imread('assets/bdy.png')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and find body landmarks
results = pose.process(image_rgb)

# Draw the pose landmarks and connections
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

# Display the image
cv2.imshow('Pose Landmarks', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
