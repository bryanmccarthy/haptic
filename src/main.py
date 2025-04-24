import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import mediapipe as mp
import os
import csv

class Haptic:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            raise RuntimeError("Failed to open camera.")
        
        self.current_results = None
        self.saved_count = 0
        
        self.display_width, self.display_height = 640, 360
        self.default_image = np.zeros((self.display_height, self.display_width, 4), dtype=np.uint8)
        
    def setup_gui(self):
        dpg.create_context()

        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=self.display_width,
                height=self.display_height,
                default_value=self.default_image.flatten()/255.0,
                tag="camera_texture"
            )

        # Create main window
        with dpg.window(label="Hand Landmark Capture", tag="primary_window", width=300, height=200):
            dpg.add_button(label="Toggle Camera Window", 
                           callback=lambda: dpg.configure_item("camera_window", 
                                                             show=not dpg.is_item_visible("camera_window")))
            dpg.add_button(label="Settings", 
                           callback=lambda: dpg.configure_item("settings_window", 
                                                             show=not dpg.is_item_visible("settings_window")))
        
        # Create settings window
        with dpg.window(label="Settings", tag="settings_window", width=300, height=200, show=True):
            dpg.add_text("Press SPACE to save hand landmarks")
            dpg.add_button(label="Save Landmarks", callback=lambda: self.save_current_landmarks())
            dpg.add_input_text(label="Gesture", default_value="", tag="target_label")
            dpg.add_text("Saved 0 Example(s)", tag="saved_count_text")
        
        # Create camera window
        with dpg.window(label="Camera", tag="camera_window", width=self.display_width + 20, 
                      height=self.display_height + 55, no_collapse=True):
            dpg.add_image("camera_texture")
            dpg.add_text("No gesture detected", tag="gesture_text")

        # Register key press handler
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=32, callback=self.on_key_press)  # 32 = space

        dpg.create_viewport(title="Hand Landmark Capture", width=1024, height=768)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        
        # Position windows
        main_win_pos = dpg.get_item_pos("primary_window")
        dpg.set_item_pos("camera_window", [main_win_pos[0] + 350, main_win_pos[1] + 15])
        dpg.set_item_pos("settings_window", [main_win_pos[0] + 15, main_win_pos[1] + 220])

    def on_key_press(self, sender, app_data):
        if app_data == 32:  # Space key
            print("Space key pressed!")
            self.save_current_landmarks()
    
    def save_current_landmarks(self):
        if self.current_results and self.current_results.multi_hand_landmarks:
            target = dpg.get_value("target_label") or "default"
            self.append_landmarks_to_file(
                self.current_results, 
                target=target
            )
            self.saved_count += 1
            dpg.set_value("saved_count_text", f"Saved {self.saved_count} Example(s)")
    
    def append_landmarks_to_file(self, results, filename="saved_landmarks/landmarks.csv", target="default"):
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
                
                for landmark in hand_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                
                if results.multi_hand_world_landmarks and i < len(results.multi_hand_world_landmarks):
                    for world_landmark in results.multi_hand_world_landmarks[i].landmark:
                        row.extend([world_landmark.x, world_landmark.y, world_landmark.z])
                else:
                    for _ in range(21):
                        row.extend([0.0, 0.0, 0.0])
                
                if results.multi_handedness and i < len(results.multi_handedness):
                    handedness = results.multi_handedness[i].classification[0].label
                    h0 = 1 if handedness == "Left" else 0
                    h1 = 1 if handedness == "Right" else 0
                    row.extend([h0, h1])
                else:
                    row.extend([0, 0])  # Placeholder when hand type is unknown
                
                csv_writer.writerow(row)
        
        print(f"Landmarks saved to {filename}")
    
    def process_frame(self, frame):
        frame = cv2.resize(frame, (self.display_width, self.display_height))
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        if results.multi_hand_landmarks:
            self.current_results = results
            
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    rgb_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        else:
            self.current_results = None

        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)
        frame = np.fliplr(frame)
        
        return frame
    
    def run(self):
        self.setup_gui()
        
        while dpg.is_dearpygui_running():
            ret, frame = self.capture.read()
            if ret:
                processed_frame = self.process_frame(frame)
                
                # Dear PyGui expects [0..1] floats for texture updates
                dpg.set_value("camera_texture", processed_frame.flatten() / 255.0)

            dpg.render_dearpygui_frame()

        self.cleanup()
    
    def cleanup(self):
        self.hands.close()
        self.capture.release()
        dpg.destroy_context()

def main():
    try:
        app = Haptic()
        app.run()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
