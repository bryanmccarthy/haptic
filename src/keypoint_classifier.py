import tensorflow as tf
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

def load_landmarks(file_path="saved_landmarks/landmarks.csv"):
    df = pd.read_csv(file_path)
    
    targets = df["target"].values
    hand_keypoints = np.zeros((len(df), 21 * 3))
    
    for i in range(21):
        hand_keypoints[:, i] = df[f"x{i}"].values
        hand_keypoints[:, i + 21] = df[f"y{i}"].values
        hand_keypoints[:, i + 42] = df[f"z{i}"].values
    
    return hand_keypoints, targets

def load_label_mapping(file_path="saved_landmarks/keypoint_classifier_labels.txt"):
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

def encode_labels(labels, label_file="saved_landmarks/keypoint_classifier_labels.txt"):
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
    
    return encoded_labels, class_mapping, reverse_mapping

def preprocess_keypoints(keypoints):
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

def custom_train_test_split(X, y, test_size=0.2, random_seed=42):
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

def build_model(input_dim, n_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(n_classes, activation='softmax')  # Number of classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(keypoints, targets, test_size=0.2, epochs=40, batch_size=32, early_stopping_patience=5,
                label_file="saved_landmarks/keypoint_classifier_labels.txt"):
    processed_keypoints = preprocess_keypoints(keypoints)

    encoded_targets, class_mapping, _ = encode_labels(targets, label_file=label_file)
    
    X_train, X_test, y_train, y_test = custom_train_test_split(
        processed_keypoints, encoded_targets, test_size=test_size
    )
    
    n_classes = max(class_mapping.keys()) + 1
    print(f"Training with class mapping: {class_mapping}")
    print(f"Using {n_classes} output neurons")
    
    model = build_model(input_dim=processed_keypoints.shape[1], n_classes=n_classes)
    print(model.summary())
    
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
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('saved_landmarks/keypoint_classifier_training.png')
    plt.close()
    
    return model, class_mapping, history

def save_model(model, class_mapping, output_dir="saved_landmarks", model_name="keypoint_classifier"):
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(output_dir, f"{model_name}.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_path}")
    
    labels_path = os.path.join(output_dir, f"{model_name}_labels.txt")
    with open(labels_path, 'w') as f:
        for idx, label in class_mapping.items():
            f.write(f"{idx}: {label}\n")
    print(f"Label mapping saved to {labels_path}")

def run_inference(keypoints, model_path="saved_landmarks/keypoint_classifier.h5", 
                  labels_path="saved_landmarks/keypoint_classifier_labels.txt"):
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from {model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model: {str(e)}")
    
    class_mapping = load_label_mapping(labels_path)
    if not class_mapping:
        raise ValueError(f"Error loading class mapping from {labels_path}")
    
    keypoints_reshaped = keypoints.reshape(1, -1)
    processed_keypoints = preprocess_keypoints(keypoints_reshaped)
    
    prediction_scores = model.predict(processed_keypoints)[0]
    
    predicted_idx = np.argmax(prediction_scores)
    
    if predicted_idx in class_mapping:
        predicted_label = class_mapping[predicted_idx]
    else:
        predicted_label = f"Unknown ({predicted_idx})"
    
    score_mapping = {class_mapping.get(i, f"Unknown ({i})"): score 
                    for i, score in enumerate(prediction_scores)}
    
    return predicted_label, score_mapping

def main():
    try:
        keypoints, targets = load_landmarks("saved_landmarks/landmarks.csv")
        print(f"Loaded {len(keypoints)} samples with {len(np.unique(targets))} unique classes")
        
        model, class_mapping, history = train_model(keypoints, targets)
        save_model(model, class_mapping)
        
        print("Keypoint classifier training completed")
        
    except Exception as e:
        print(f"Error training keypoint classifier: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
