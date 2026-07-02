import cv2
import numpy as np
from hand_tracker import HandTracker

# ===========================
# Camera
# ===========================
cap = cv2.VideoCapture(0)

tracker = HandTracker()

# ===========================
# Canvas
# ===========================
canvas = None
previous_point = None 


class DrawingUtils:

    def __init__(self):
        pass

    def draw_line(self, canvas, start_point, end_point, color, thickness):
        cv2.line(
            canvas,
            start_point,
            end_point,
            color,
            thickness
        )

    def clear_canvas(self, canvas):
        canvas[:] = 0

    def save_canvas(self, image, filename="AirDrawing.png"):
        cv2.imwrite(filename, image)

# ===========================
# Colors
# ===========================
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
BLACK = (40, 40, 40)
PURPLE = (255, 0, 255)
YELLOW = (0, 255, 255)

draw_color = GREEN

# ===========================
# Brush
# ===========================
brush_size = 5
eraser_size = 25

erase_mode = False

# ===========================
# Main Loop
# ===========================
while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    frame, tip, fingers = tracker.find_hands(frame)
    
        # ==========================================
    # Selection Mode (Index + Middle Finger)
    # ==========================================
    if tip and fingers:

        x, y = tip

        if fingers[1] == 1 and fingers[2] == 1:

            previous_point = None

            if y < 70:

                # RED
                if 10 <= x <= 60:
                    draw_color = RED
                    erase_mode = False

                # GREEN
                elif 70 <= x <= 120:
                    draw_color = GREEN
                    erase_mode = False

                # BLUE
                elif 130 <= x <= 180:
                    draw_color = BLUE
                    erase_mode = False

                # BLACK
                elif 190 <= x <= 240:
                    draw_color = BLACK
                    erase_mode = False

                # PURPLE
                elif 250 <= x <= 300:
                    draw_color = PURPLE
                    erase_mode = False

                # YELLOW
                elif 310 <= x <= 360:
                    draw_color = YELLOW
                    erase_mode = False

                # ERASER
                elif 380 <= x <= 460:
                    erase_mode = True

                # CLEAR
                elif 480 <= x <= 560:
                    canvas[:] = 0

        # ==========================================
        # Drawing Mode (Only Index Finger)
        # ==========================================
        elif (
            fingers[1] == 1 and
            fingers[2] == 0 and
            fingers[3] == 0 and
            fingers[4] == 0
        ):

            if previous_point is None:
                previous_point = tip

            if erase_mode:

                cv2.line(
                    canvas,
                    previous_point,
                    tip,
                    (0, 0, 0),
                    eraser_size
                )

            else:

                cv2.line(
                    canvas,
                    previous_point,
                    tip,
                    draw_color,
                    brush_size
                )

            previous_point = tip

        else:
            previous_point = None

    else:
        previous_point = None
        
    # ==========================================
    # Merge Canvas with Camera
    # ==========================================
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    mask_inv = cv2.bitwise_not(mask)

    frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    drawing_fg = cv2.bitwise_and(canvas, canvas, mask=mask)

    output = cv2.add(frame_bg, drawing_fg)

    # ==========================================
    # Toolbar
    # ==========================================
    cv2.rectangle(output, (0, 0), (900, 70), (235, 235, 235), -1)

    # Colors
    cv2.rectangle(output, (10,10), (60,60), RED, -1)
    cv2.rectangle(output, (70,10), (120,60), GREEN, -1)
    cv2.rectangle(output, (130,10), (180,60), BLUE, -1)
    cv2.rectangle(output, (190,10), (240,60), BLACK, -1)
    cv2.rectangle(output, (250,10), (300,60), PURPLE, -1)
    cv2.rectangle(output, (310,10), (360,60), YELLOW, -1)

    # Eraser
    cv2.rectangle(output,(380,10),(460,60),(220,220,220),-1)
    cv2.putText(output,"Erase",(390,42),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

    # Clear
    cv2.rectangle(output,(480,10),(560,60),(220,220,220),-1)
    cv2.putText(output,"Clear",(490,42),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

    # Save
    cv2.rectangle(output,(580,10),(660,60),(220,220,220),-1)
    cv2.putText(output,"Save",(595,42),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)

    # Brush +
    cv2.rectangle(output,(680,10),(730,60),(220,220,220),-1)
    cv2.putText(output,"+",(698,45),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    # Brush -
    cv2.rectangle(output,(750,10),(800,60),(220,220,220),-1)
    cv2.putText(output,"-",(770,45),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)

    # Brush Size
    cv2.putText(output,
                f"Brush : {brush_size}",
                (820,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2)

    # Save Image
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        cv2.imwrite("AirDrawing.png", output)
        print("Drawing Saved!")

    cv2.imshow("AI Air Drawing", output)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()