import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import mediapipe as mp

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Failed to open camera.")
        return

    dpg.create_context()

    display_width, display_height = 640, 360

    default_image = np.zeros((display_height, display_width, 4), dtype=np.uint8)

    with dpg.texture_registry():
        dpg.add_dynamic_texture(
            width=display_width,
            height=display_height,
            default_value=default_image.flatten()/255.0,
            tag="camera_texture"
        )

    with dpg.window(label="Camera"):
        dpg.add_image("camera_texture")

    dpg.create_viewport(title="Camera Viewer", width=800, height=800)
    dpg.setup_dearpygui()
    dpg.show_viewport()

    while dpg.is_dearpygui_running():
        ret, frame = capture.read()
        if ret:
            # Resize frame for display
            frame = cv2.resize(frame, (display_width, display_height))
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe
            rgb_frame.flags.writeable = False
            results = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            # Draw hand landmarks 
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # print(f"Landmarks Captured: {hand_landmarks.landmark}")
            
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)
            frame = np.fliplr(frame)

            # Dear PyGui expects [0..1] floats for texture updates
            dpg.set_value("camera_texture", frame.flatten() / 255.0)

        dpg.render_dearpygui_frame()

    hands.close()
    capture.release()
    dpg.destroy_context()

if __name__ == "__main__":
    main()

