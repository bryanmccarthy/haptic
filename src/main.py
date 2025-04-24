import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import mediapipe as mp
import os
import csv
import pyautogui
from keypoint_classifier import KeypointClassifier

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
        
        self.keypoint_classifier = KeypointClassifier(
            model_path="saved_landmarks/keypoint_classifier.h5",
            label_path="saved_landmarks/keypoint_classifier_labels.txt"
        )
        
        self.inference_mode = False
        self.predicted_gesture = ""
        self.last_executed_gesture = None
        
        self.keypress_mappings = {}
        
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
            dpg.add_button(label="Landmark Capture", 
                           callback=lambda: dpg.configure_item("landmark_capture_window", 
                                                             show=not dpg.is_item_visible("landmark_capture_window")))
            dpg.add_button(label="Train", 
                           callback=lambda: dpg.configure_item("train_window", 
                                                             show=not dpg.is_item_visible("train_window")))
            dpg.add_button(label="Keypress Mapping", 
                           callback=lambda: dpg.configure_item("keypress_mapping_window", 
                                                             show=not dpg.is_item_visible("keypress_mapping_window")))
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="Inference Mode", callback=self.toggle_inference_mode, tag="inference_mode_checkbox")
                dpg.add_text("Off", tag="inference_mode_status")
        
        # Create landmark capture window
        with dpg.window(label="Landmark Capture", tag="landmark_capture_window", width=300, height=200, show=True):
            dpg.add_text("Press SPACE to save hand landmarks")
            dpg.add_button(label="Save Landmarks", callback=lambda: self.save_current_landmarks())
            dpg.add_input_text(label="Gesture", default_value="", tag="target_label")
            dpg.add_text("Saved 0 Example(s)", tag="saved_count_text")
        
        # Create camera window
        with dpg.window(label="Camera", tag="camera_window", width=self.display_width + 20, 
                      height=self.display_height + 80, no_collapse=True):
            dpg.add_image("camera_texture")
            dpg.add_text("", tag="gesture_text")
            with dpg.group(horizontal=True):
                dpg.add_text("Keypress: ")
                dpg.add_text("None", tag="keypress_text")
            
        # Create train window
        with dpg.window(label="Train", tag="train_window", width=300, height=200, show=True):
            dpg.add_button(label="Train Classifier", callback=lambda: self.train_classifier())
            dpg.add_text("", tag="training_status_text")
            
        with dpg.window(label="Keypress Mapping", tag="keypress_mapping_window", width=self.display_width + 20, height=250, show=True):
            pass
            
        # Initialize the keypress mapping window
        self.refresh_keypress_mapping_window()

        # Register key press handler
        with dpg.handler_registry():
            dpg.add_key_press_handler(key=32, callback=self.on_key_press)  # 32 = space

        dpg.create_viewport(title="Hand Landmark Capture", width=1100, height=850)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("primary_window", True)
        
        # Position windows
        main_win_pos = dpg.get_item_pos("primary_window")
        
        dpg.set_item_pos("landmark_capture_window", [main_win_pos[0] + 5, main_win_pos[1] + 300])
        dpg.set_item_pos("train_window", [main_win_pos[0] + 5, main_win_pos[1] + 520])
        
        camera_x = main_win_pos[0] + 320
        camera_y = main_win_pos[1]
        dpg.set_item_pos("camera_window", [camera_x, camera_y])
        
        camera_height = dpg.get_item_height("camera_window")
        dpg.set_item_pos("keypress_mapping_window", [camera_x, camera_y + camera_height + 10])

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
    
    def update_keypress_mapping(self, user_data, app_data):
        gesture = user_data
        key = app_data.strip()
        
        if key:
            self.keypress_mappings[gesture] = key
            print(f"Updated mapping: {gesture} -> {key}")
        else:
            if gesture in self.keypress_mappings:
                del self.keypress_mappings[gesture]
                print(f"Removed mapping for {gesture}")
    
    def execute_keypress(self, gesture):
        if gesture in self.keypress_mappings and self.keypress_mappings[gesture]:
            key = self.keypress_mappings[gesture]
            print(f"Executing keypress: {key} for gesture {gesture}")
            
            dpg.set_value("keypress_text", key)

            pyautogui.press(key)

        else:
            dpg.set_value("keypress_text", "None")
    
    def toggle_inference_mode(self, sender, app_data):
        self.inference_mode = app_data
        status_text = "On" if self.inference_mode else "Off"
        dpg.set_value("inference_mode_status", status_text)
        self.predicted_gesture = ""
        self.last_executed_gesture = None  # Reset the last executed gesture
        dpg.set_value("gesture_text", self.predicted_gesture)
        dpg.set_value("keypress_text", "None")
        
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
                
            if self.inference_mode and self.keypoint_classifier.model is not None:
                try:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    keypoints = np.zeros(21 * 3)
                    for i, lm in enumerate(hand_landmarks.landmark):
                        keypoints[i] = lm.x
                        keypoints[i + 21] = lm.y
                        keypoints[i + 42] = lm.z
                    
                    gesture, confidence = self.keypoint_classifier.predict(keypoints.reshape(1, -1))
                    self.predicted_gesture = f"{gesture} ({confidence:.2f})"
                    dpg.set_value("gesture_text", self.predicted_gesture)
                    
                    if confidence > 0.7 and gesture != self.last_executed_gesture:
                        self.execute_keypress(gesture)
                        self.last_executed_gesture = gesture
                except Exception as e:
                    print(f"Error during inference: {str(e)}")
                    self.predicted_gesture = "Error during inference"
                    dpg.set_value("gesture_text", self.predicted_gesture)
        else:
            self.current_results = None
            if self.inference_mode:
                self.predicted_gesture = "No hand detected"
                dpg.set_value("gesture_text", self.predicted_gesture)
                
                if self.last_executed_gesture is not None:
                    self.last_executed_gesture = None
                    dpg.set_value("keypress_text", "None")

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
    
    def refresh_keypress_mapping_window(self):
        children = dpg.get_item_children("keypress_mapping_window", 1)
        if children:
            for child in children:
                dpg.delete_item(child)
        
        dpg.add_text("Map gestures to keyboard keys", parent="keypress_mapping_window")
        dpg.add_separator(parent="keypress_mapping_window")
        
        with dpg.group(horizontal=True, parent="keypress_mapping_window"):
            dpg.add_text("Gesture")
            dpg.add_spacer(width=30)
            dpg.add_text("Key")
        
        if self.keypoint_classifier.class_mapping:
            for idx, gesture in self.keypoint_classifier.class_mapping.items():
                with dpg.group(horizontal=True, parent="keypress_mapping_window"):
                    dpg.add_text(f"{gesture}", tag=f"gesture_{idx}_text")
                    dpg.add_spacer(width=85)
                    dpg.add_input_text(
                        width=100, 
                        tag=f"keypress_{idx}_input",
                        default_value=self.keypress_mappings.get(gesture, ""),
                        callback=lambda s, a, u: self.update_keypress_mapping(u, a)
                    )
                    dpg.set_item_user_data(f"keypress_{idx}_input", gesture)
        else:
            dpg.add_text("No gestures trained yet", parent="keypress_mapping_window")
    
    def train_classifier(self):
        dpg.set_value("training_status_text", "Training...")

        try:
            file_path = "saved_landmarks/landmarks.csv"
            keypoints, targets = self.keypoint_classifier.load_landmarks(file_path)
            
            if len(keypoints) == 0 or len(targets) == 0:
                dpg.set_value("training_status_text", "No landmarks saved")
                return
                
            dpg.set_value("training_status_text", f"Training with {len(keypoints)} samples, {len(np.unique(targets))} classes...")
            
            self.keypoint_classifier.train(keypoints, targets)
            
            self.keypoint_classifier.save()
            
            self.refresh_keypress_mapping_window()
            
            dpg.set_value("training_status_text", "Training completed.")
        except Exception as e:
            dpg.set_value("training_status_text", f"Training error: {str(e)}")
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
    
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
