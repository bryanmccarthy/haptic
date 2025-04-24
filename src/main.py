import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import mediapipe as mp
import os
import csv

def append_landmarks_to_file(results, filename="saved_landmarks/landmarks.csv", target="default"):
    os.makedirs('saved_landmarks', exist_ok=True)
    
    # target, x0-x20, y0-y20, z0-z20, wx0-wx20, wy0-wy20, wz0-wz20, h0, h1
    csv_file_exists = os.path.exists(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        if not csv_file_exists:
            header = ['target']
            # Screen landmarks x, y, z coordinates
            for i in range(21):
                header.extend([f'x{i}', f'y{i}', f'z{i}'])
            # World landmarks wx, wy, wz coordinates
            for i in range(21):
                header.extend([f'wx{i}', f'wy{i}', f'wz{i}'])
            # Handedness 
            header.extend(['h0', 'h1'])
            csv_writer.writerow(header)
        
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            row = [target]
            
            # Add regular landmark coordinates
            for landmark in hand_landmarks.landmark:
                row.extend([landmark.x, landmark.y, landmark.z])
            
            # Add world landmark coordinates if available
            if results.multi_hand_world_landmarks and i < len(results.multi_hand_world_landmarks):
                for world_landmark in results.multi_hand_world_landmarks[i].landmark:
                    row.extend([world_landmark.x, world_landmark.y, world_landmark.z])
            else:
                # Add placeholder world coordinates if not available
                for _ in range(21):
                    row.extend([0.0, 0.0, 0.0])
            
            # Add hand type indicator (left/right hand)
            if results.multi_handedness and i < len(results.multi_handedness):
                handedness = results.multi_handedness[i].classification[0].label
                h0 = 1 if handedness == "Left" else 0
                h1 = 1 if handedness == "Right" else 0
                row.extend([h0, h1])
            else:
                row.extend([0, 0])  # Placeholder when hand type is unknown
            
            csv_writer.writerow(row)
    
    print(f"Landmarks saved to {filename}")

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

    current_results = None
    saved_count = 0

    def save_current_landmarks():
        nonlocal saved_count
        if current_results and current_results.multi_hand_landmarks:
            target = dpg.get_value("target_label") or "default"
            append_landmarks_to_file(
                current_results, 
                target=target
            )
            saved_count += 1
            dpg.set_value("saved_count_text", f"Saved {saved_count} Example(s)")

    # Keyboard callback
    def on_key_press(sender, app_data):
        if app_data == 32:  # Space key
            print("Space key pressed!")
            save_current_landmarks()

    # Create main window
    with dpg.window(label="Hand Landmark Capture", tag="primary_window", width=300, height=200):
        dpg.add_button(label="Toggle Camera Window", callback=lambda: dpg.configure_item("camera_window", show=not dpg.is_item_visible("camera_window")))
        dpg.add_button(label="Settings", callback=lambda: dpg.configure_item("settings_window", show=not dpg.is_item_visible("settings_window")))
    
    # Create settings window
    with dpg.window(label="Settings", tag="settings_window", width=300, height=200, show=True):
        dpg.add_text("Press SPACE to save hand landmarks")
        dpg.add_button(label="Save Landmarks", callback=lambda: save_current_landmarks())
        dpg.add_input_text(label="Gesture", default_value="", tag="target_label")
        dpg.add_text("Saved 0 Example(s)", tag="saved_count_text")
    
    # Create camera window
    with dpg.window(label="Camera", tag="camera_window", width=display_width + 20, height=display_height + 55, no_collapse=True):
        dpg.add_image("camera_texture")
        dpg.add_text("No gesture detected", tag="gesture_text")

    # Register key press handler
    with dpg.handler_registry():
        dpg.add_key_press_handler(key=32, callback=on_key_press)  # 32 = space

    dpg.create_viewport(title="Hand Landmark Capture", width=1024, height=768)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary_window", True)
    
    # Position windows
    main_win_pos = dpg.get_item_pos("primary_window")
    dpg.set_item_pos("camera_window", [main_win_pos[0] + 350, main_win_pos[1] + 15])
    dpg.set_item_pos("settings_window", [main_win_pos[0] + 15, main_win_pos[1] + 220])

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
            
            # Update the current results if hands are detected
            if results.multi_hand_landmarks:
                current_results = results
                
                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
            else:
                current_results = None

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
