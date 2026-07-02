import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )

        self.drawer = mp.solutions.drawing_utils

    def fingers_up(self, landmarks):
        fingers = []

        # Thumb
        if landmarks[4][0] > landmarks[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Index
        fingers.append(1 if landmarks[8][1] < landmarks[6][1] else 0)

        # Middle
        fingers.append(1 if landmarks[12][1] < landmarks[10][1] else 0)

        # Ring
        fingers.append(1 if landmarks[16][1] < landmarks[14][1] else 0)

        # Pinky
        fingers.append(1 if landmarks[20][1] < landmarks[18][1] else 0)

        return fingers

    def find_hands(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        landmarks = []

        fingers = None

        tip = None

        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]

            self.drawer.draw_landmarks(
                frame,
                hand,
                self.mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            for id, lm in enumerate(hand.landmark):

                x = int(lm.x * w)
                y = int(lm.y * h)

                landmarks.append((x, y))

            if landmarks:

                tip = landmarks[8]

                cv2.circle(frame, tip, 10, (0,255,0), -1)

                fingers = self.fingers_up(landmarks)

        return frame, tip, fingers