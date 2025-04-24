import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
# import matplotlib.pyplot as plt

class KeypointClassifier:
    def __init__(self, model_path=None, label_path=None):
        self.model = None
        self.class_mapping = {}
        self.reverse_mapping = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
            
        if label_path and os.path.exists(label_path):
            self.class_mapping = self.load_label_mapping(label_path)
    
    def load_landmarks(self, file_path="saved_landmarks/landmarks.csv"):
        df = pd.read_csv(file_path)
        
        targets = df["target"].values
        hand_keypoints = np.zeros((len(df), 21 * 3))
        
        for i in range(21):
            hand_keypoints[:, i] = df[f"x{i}"].values
            hand_keypoints[:, i + 21] = df[f"y{i}"].values
            hand_keypoints[:, i + 42] = df[f"z{i}"].values
        
        return hand_keypoints, targets
    
    def load_label_mapping(self, file_path="saved_landmarks/keypoint_classifier_labels.txt"):
        class_mapping = {}

        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        idx, label = line.strip().split(':', 1)
                        class_mapping[int(idx)] = label.strip()
        except Exception as e:
            print(f"Error loading label mapping: {str(e)}")
        
        return class_mapping
    
    def encode_labels(self, labels, label_file="saved_landmarks/keypoint_classifier_labels.txt"):
        existing_map = {}
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    idx, label = line.strip().split(':', 1)
                    existing_map[label.strip()] = int(idx)
        except Exception as e:
            print(f"No existing label mapping found or error reading it: {e}")
        
        unique_labels = sorted(list(set(labels)))
        
        reverse_mapping = {}  # label -> index
        class_mapping = {}    # index -> label
        
        next_idx = 0
        for label in unique_labels:
            if label in existing_map:
                idx = existing_map[label]
            else:
                while next_idx in class_mapping.keys() or next_idx in [existing_map[l] for l in existing_map]:
                    next_idx += 1
                idx = next_idx
                next_idx += 1
            
            reverse_mapping[label] = idx
            class_mapping[idx] = label
        
        encoded_labels = np.array([reverse_mapping[label] for label in labels])
        
        print(f"Using class mapping: {class_mapping}")
        
        # Store mappings in the instance
        self.class_mapping = class_mapping
        self.reverse_mapping = reverse_mapping
        
        return encoded_labels, class_mapping, reverse_mapping
    
    def preprocess_keypoints(self, keypoints):
        processed = keypoints.copy()
        
        # Center the hand keypoints (subtract the palm keypoint)
        palm_index = 0  # Index of the palm keypoint
        
        # Center coords
        processed[:, 0:21] = processed[:, 0:21] - processed[:, palm_index].reshape(-1, 1)
        processed[:, 21:42] = processed[:, 21:42] - processed[:, 21 + palm_index].reshape(-1, 1)
        processed[:, 42:63] = processed[:, 42:63] - processed[:, 42 + palm_index].reshape(-1, 1)
        
        # Scale keypoints
        for i in range(len(processed)):
            x_range = np.max(processed[i, 0:21]) - np.min(processed[i, 0:21])
            y_range = np.max(processed[i, 21:42]) - np.min(processed[i, 21:42])
            z_range = np.max(processed[i, 42:63]) - np.min(processed[i, 42:63])
            
            max_range = max(x_range, y_range, z_range)
            
            if max_range > 0:
                processed[i, 0:21] = processed[i, 0:21] / max_range
                processed[i, 21:42] = processed[i, 21:42] / max_range
                processed[i, 42:63] = processed[i, 42:63] / max_range
        
        return processed
    
    def custom_train_test_split(self, X, y, test_size=0.2, random_seed=42):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
        
        n_samples = len(X)
        indices = list(range(n_samples))
        
        classes, counts = np.unique(y, return_counts=True)
        class_indices = {c: [] for c in classes}
        
        # Group indices by class
        for i, label in enumerate(y):
            class_indices[label].append(i)
        
        test_indices = []
        for c in classes:
            c_indices = class_indices[c]
            random.shuffle(c_indices)
            n_test = int(len(c_indices) * test_size)
            test_indices.extend(c_indices[:n_test])
        
        train_indices = [i for i in indices if i not in test_indices]
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_dim, n_classes):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, keypoints=None, targets=None, file_path=None, test_size=0.2, 
             epochs=40, batch_size=32, early_stopping_patience=5,
             label_file="saved_landmarks/keypoint_classifier_labels.txt"):
        if keypoints is None or targets is None:
            if file_path:
                keypoints, targets = self.load_landmarks(file_path)
            else:
                raise ValueError("Either keypoints and targets or file_path must be provided")
                
        processed_keypoints = self.preprocess_keypoints(keypoints)
        encoded_targets, class_mapping, _ = self.encode_labels(targets, label_file=label_file)
        
        X_train, X_test, y_train, y_test = self.custom_train_test_split(
            processed_keypoints, encoded_targets, test_size=test_size
        )
        
        n_classes = max(class_mapping.keys()) + 1
        print(f"Training with class mapping: {class_mapping}")
        print(f"Using {n_classes} output neurons")
        
        self.model = self.build_model(input_dim=processed_keypoints.shape[1], n_classes=n_classes)
        print(self.model.summary())
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=early_stopping_patience, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001
            )
        ]
        
        class_weights = {}
        unique_classes, class_counts = np.unique(y_train, return_counts=True)
        total_samples = len(y_train)
        n_classes_actual = len(unique_classes)
        
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = total_samples / (n_classes_actual * count)
        
        print(f"Class weights for training: {class_weights}")
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights
        )
        
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
        
        # Plot training history
        # plt.figure(figsize=(12, 4))
        
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('Model Accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Validation'], loc='lower right')
        
        # plt.subplot(1, 2, 2)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('Model Loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['Train', 'Validation'], loc='upper right')
        
        # plt.tight_layout()
        # plt.savefig('saved_landmarks/keypoint_classifier_training.png')
        # plt.close()
        
        return history
    
    def save(self, output_dir="saved_landmarks", model_name="keypoint_classifier"):
        """Save the model and class mapping to files"""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
            
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = os.path.join(output_dir, f"{model_name}.h5")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"TFLite model saved to {tflite_path}")
        
        labels_path = os.path.join(output_dir, f"{model_name}_labels.txt")
        with open(labels_path, 'w') as f:
            for idx, label in self.class_mapping.items():
                f.write(f"{idx}: {label}\n")
        print(f"Label mapping saved to {labels_path}")
    
    def load_model(self, model_path="saved_landmarks/keypoint_classifier.h5"):
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def predict(self, keypoints, return_scores=False):
        if self.model is None:
            raise ValueError("No model loaded. Load a model first.")
            
        if len(self.class_mapping) == 0:
            raise ValueError("No class mapping loaded.")
        
        keypoints_reshaped = keypoints.reshape(1, -1)
        processed_keypoints = self.preprocess_keypoints(keypoints_reshaped)
        
        prediction_scores = self.model.predict(processed_keypoints, verbose=0)[0]
        
        predicted_idx = np.argmax(prediction_scores)
        confidence = prediction_scores[predicted_idx]
        
        if predicted_idx in self.class_mapping:
            predicted_label = self.class_mapping[predicted_idx]
        else:
            predicted_label = f"Unknown ({predicted_idx})"
        
        if return_scores:
            score_mapping = {self.class_mapping.get(i, f"Unknown ({i})"): score 
                            for i, score in enumerate(prediction_scores)}
            return predicted_label, confidence, score_mapping
        
        return predicted_label, confidence

def main():
    try:
        classifier = KeypointClassifier()
        keypoints, targets = classifier.load_landmarks("saved_landmarks/landmarks.csv")
        print(f"Loaded {len(keypoints)} samples with {len(np.unique(targets))} unique classes")
        
        classifier.train(keypoints, targets)
        classifier.save()
        
        print("Keypoint classifier training completed")
        
    except Exception as e:
        print(f"Error training keypoint classifier: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
