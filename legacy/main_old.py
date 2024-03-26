import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=5120
                )
            ],
        )
    except RuntimeError as e:
        print(e)

#  Data preparation
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print('Total videos for training: {len(train_df)}')
print('Total videos for testing: {len(test_df)}')

# train_df.sample(10)

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 100

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)


# Feature extraction


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights='imagenet',
        include_top=False,
        pooling='avg',
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)

    return keras.Model(inputs, outputs, name='feature_extractor')


feature_extractor = build_feature_extractor()

# Label encoding

label_processor = keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(train_df['tag'])
)
print(label_processor.get_vocabulary())

labels = train_df['tag'].values
labels = label_processor(labels[..., None]).numpy()
labels


def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df['video_name'].values.tolist()

    # take all classlabels from train_df column named 'tag' and store in labels
    labels = df['tag'].values

    # convert classlabels to label encoding
    labels = label_processor(labels[..., None]).numpy()

    # frame_masks and frame_features are what we will feed to our sequence
    # model. frame_masks will contain a bunch of booleans denoting if a timestep
    # is masked with padding or not.
    frame_masks = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH), dtype='bool'
    )  # 145,20
    frame_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32'
    )  # 145,20,2048

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = load_video(os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current
        # video.
        temp_frame_mask = np.zeros(
            shape=(
                1,
                MAX_SEQ_LENGTH,
            ),
            dtype='bool',
        )
        temp_frame_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32'
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = feature_extractor.predict(
                    batch[None, j, :]
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[
            idx,
        ] = temp_frame_features.squeeze()
        frame_masks[
            idx,
        ] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(train_df, 'train')
test_data, test_labels = prepare_all_videos(test_df, 'test')

print(f'Frame features in train set: {train_data[0].shape}')
print(f'Frame masks in train set: {train_data[1].shape}')


print(f'train_labels in train set: {train_labels.shape}')

print(f'test_labels in train set: {test_labels.shape}')

# MAX_SEQ_LENGTH = 20, NUM_FEATURES = 2048. We have defined this above under
# hyper parameters

# Utility for our sequence model.


def get_sequence_model():
    class_vocab = label_processor.get_vocabulary()

    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype='bool')

    # Refer to the following tutorial to understand the significance of using
    # mask: https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation='relu')(x)
    output = keras.layers.Dense(len(class_vocab), activation='softmax')(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy'],
    )
    return rnn_model


EPOCHS = 30
# Utility for running experiments.


def run_experiment():
    filepath = './tmp/video_classifier'
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_data[0], train_data[1]],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        callbacks=[checkpoint],
    )

    seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
    print(f'Test accuracy: {round(accuracy * 100, 2)}%')

    return history, seq_model


_, sequence_model = run_experiment()


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(
        shape=(
            1,
            MAX_SEQ_LENGTH,
        ),
        dtype='bool',
    )
    frame_features = np.zeros(
        shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype='float32'
    )

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(
                batch[None, j, :]
            )
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join('test', path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = sequence_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f'  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%')
    return frames


test_video = np.random.choice(test_df['video_name'].values.tolist())
print(f'Test video path: {test_video}')

test_frames = sequence_prediction(test_video)
