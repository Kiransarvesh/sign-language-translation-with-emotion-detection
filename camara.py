import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pyttsx3  # Text-to-Speech
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def detect_emotion(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    
    compound = scores['compound']
    
    if compound >= 0.5:
        return "Happy"
    elif compound <= -0.5:
        return "Sad"
    else:
        return "Neutral"
# Load MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

import tensorflow as tf
model = tf.keras.models.load_model('model.keras')
model.summary()

# Labels
labels = [
    'bring water for me Neutral',
    'can i help you happy',
    'hi how are you happy',
    'i am hungry Neutral',
    'i am sitting in the class Neutral',
    'i am suffering from fever sad',
    'i am tired  sad'
]

# Initialize Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)  

# Function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    #face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    return np.concatenate([pose, lh, rh])

# Function for MediaPipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    image.flags.writeable = False  
    results = model.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    return image, results

# Open webcam
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite("001.jpg", frame) 
        
        # Process frame with MediaPipe
        image, results = mediapipe_detection(frame, holistic)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        
        # Convert keypoints into proper input shape for LSTM
        X = np.expand_dims([keypoints], axis=0)  # Shape: (1, 1, feature_size)
        
        # Predict the action
        y_pred = model.predict(X)
        y_pred_class = np.argmax(y_pred, axis=1)[0]
        print(y_pred)
        predicted_label = labels[y_pred_class] 
        #predicted_label="May i help you"
        
        # Display prediction on frame
        print(y_pred[0][y_pred_class])
        if(y_pred[0][y_pred_class]>0.5):
            cv2.putText(image, predicted_label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Speak out the prediction
            engine.say(predicted_label)
            engine.runAndWait()

        # Show the frame
        cv2.imshow('Live Prediction', image)
        
        # Wait for user to press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
