import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import mediapipe as mp
import cv2
import pickle

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands


with open('asl.pkl', 'rb') as f:
    model = pickle.load(f)

# real time webcam feed
# getting the video capture device number varies from computer to computer
cap = cv2.VideoCapture(0)

st.title("ASL Translator")
st.text("Bridge communication gaps one step at a time. Start by signing the ASL Alphabet.")
st.text("Tip: Make sure you have good lighting for the best results!")

frame_placeholder = st.empty()

# Initialize or retrieve the state
if 'running' not in st.session_state:
    st.session_state.running = True

# Switch between Start and Stop buttons
if st.session_state.running:
    stop_button_pressed = st.button("Stop")
    if stop_button_pressed:
        st.session_state.running = False
        st.experimental_rerun()  # Force a rerun after changing state
else:
    start_button_pressed = st.button("Start")
    if start_button_pressed:
        st.session_state.running = True
        st.experimental_rerun()  # Force a rerun after changing state


# opening holistic model and running, setting detection and tracking confidence to 50%
with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5) as hands:

    while cap.isOpened() and st.session_state.running:  # looping while the capture frame is open
        ret, frame = cap.read()  # while the reading the feed from webcam

        if not ret:
            print("Failed to grab frame")
            continue

        flipped_image = cv2.flip(frame, 1)  # 1 means horizontal flip
        image = cv2.cvtColor(flipped_image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Check if the hand is left or right
                h_type = results.multi_handedness[idx].classification[0].label
                # Draw Landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                try:
                    lm = hand_landmarks.landmark
                    row = list(
                        np.array([[landmark.x, landmark.y, landmark.z] for landmark in lm]).flatten())

                    X = pd.DataFrame([row])
                    asl_class = model.predict(X)[0]
                    asl_prob = model.predict_proba(X)[0]

                    # Determine position based on hand type
                    if h_type == 'Left':
                        box_start = (0, 0)
                    else:  # Left
                        # Assuming width of your image minus the width of your box
                        box_start = (image.shape[1] - 500, 0)

                        # Get status box
                    overlay = image.copy()
                    cv2.rectangle(
                        overlay, box_start, (box_start[0] + 500, box_start[1] + 250), (40, 40, 45), -1)
                    alpha = 0.5
                    image = cv2.addWeighted(
                        overlay, alpha, image, 1 - alpha, 0)

                    # Display Class
                    cv2.putText(image, 'SIGN',
                                (box_start[0] + 70, box_start[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, asl_class.split(' ')[0],
                                (box_start[0] + 70, box_start[1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

                    # Display Probability
                    cv2.putText(image, 'PROBABILITY',
                                (box_start[0] + 220, box_start[1] + 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, str(round(asl_prob[np.argmax(asl_prob)], 2)),
                                (box_start[0] + 220, box_start[1] + 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

                except:
                    pass

        # render the results from the screen
        frame_placeholder.image(image, channels="RGB")
        # checking if we are breaking out of the loop
        if cv2.waitKey(5) & 0xFF == ord('q') or not st.session_state.running:
            break


cap.release()  # release the camera
cv2.destroyAllWindows()  # destroy the windows
