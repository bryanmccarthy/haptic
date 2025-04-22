import pandas as pd
import tensorflow as tf

class GestureRecognizer:
    def __init__(self, landmarks_file="saved_landmarks/landmarks.csv"):
        self.landmarks_file = landmarks_file

    def load_dataset(self):
        df = pd.read_csv("saved_landmarks/landmarks.csv")

        hand_arr = df[[f"x{i}" for i in range(21)] +
                     [f"y{i}" for i in range(21)] +
                     [f"z{i}" for i in range(21)]].values.reshape(-1, 21, 3)

        world_arr = df[[f"wx{i}" for i in range(21)] +
                      [f"wy{i}" for i in range(21)] +
                      [f"wz{i}" for i in range(21)]].values.reshape(-1, 21, 3)

        handedness_arr = df[["h0", "h1"]].values

        labels_int = pd.Categorical(df["target"]).codes
        num_classes = df["target"].nunique()
        labels_oh = tf.one_hot(labels_int, num_classes)

        ds = tf.data.Dataset.from_tensor_slices(
            (
              {"hand":        hand_arr,
               "world_hand":  world_arr,
               "handedness":  handedness_arr},
              labels_oh
            )
        )

        return ds

    def train(self):
        data = self.load_dataset()
        print(f"data: {data}")
        
        # Get dataset size
        dataset_size = tf.data.experimental.cardinality(data).numpy()
        train_size = int(0.8 * dataset_size)
        rest_size = dataset_size - train_size
        validation_size = int(0.5 * rest_size)
        
        shuffled_data = data.shuffle(buffer_size=dataset_size)
        
        train_data = shuffled_data.take(train_size)
        rest_data = shuffled_data.skip(train_size)
        validation_data = rest_data.take(validation_size)
        test_data = rest_data.skip(validation_size)
        
        print(f"train_data: {train_data}")
        print(f"validation_data: {validation_data}")
        print(f"test_data: {test_data}")
        
        # TODO:
        # hparams = gesture_recognizer.HParams(export_dir="saved_model")
        # options = gesture_recognizer.GestureRecognizerOptions(hparams=hparams)
        # model = gesture_recognizer.GestureRecognizer.create(
        #     train_data=train_data,
        #     validation_data=validation_data,
        #     options=options
        # )

        
def main():
    gesture_recognizer = GestureRecognizer()
    gesture_recognizer.train()

if __name__ == "__main__":
    main()

