import os
from logging import Logger
from typing import List, Tuple

import cv2
import numpy as np
from cv2.typing import MatLike
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.utils import load_pickle_object, pickle_object


class DataProcessor:

    """The DataProcessor class is responsible for executing data processing step
    after the EDA which splits the data to train/validation/test subsets and
    normalizes the data to be ready for model consumption.
    """

    def __init__(self, logger: Logger, data_classes: List[str]) -> None:
        """Initializes the DataProcessor class.

        Args:
            logger (Logger): Run logger used for documenting processes.
            data_classes (List[str]): Name of data classes to process.
        """
        self.__logger = logger
        self.__data_classes = data_classes

    def __extract_fames(self, video_path: str) -> List[MatLike]:
        """Extracts `Config.SEQUENCE_LENGTH` number of frames from given video.

        Args:
            video_path (str): Video file to be processed.

        Returns:
            List[MatLike]: List of extracted frames.
        """
        self.__logger.info(f"Currently processing: '{video_path}'.")

        # List to store extracted frames
        frames_list = []

        # Load video file
        video_reader = cv2.VideoCapture(video_path)

        # Get the number of frames in given video
        video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate frame window to skip after each extracted frame
        skip_frames_window = max(
            int(video_frames_count / Config.SEQUENCE_LENGTH), 1
        )

        for frame_counter in range(Config.SEQUENCE_LENGTH):
            # Set video to current frame
            video_reader.set(
                cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window
            )

            # Extract current frame
            success, frame = video_reader.read()

            # Exit if extraction failed
            if not success:
                break

            # Resize image for easier computing
            resized_frame = cv2.resize(
                frame, (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)
            )

            # Normalize pixel values so each of them has a value between 0 and 1
            normalized_frame = resized_frame / 255

            # Append processed frame
            frames_list.append(normalized_frame)

        # Release the VideoCapture object
        video_reader.release()

        return frames_list

    def __pickle_processed_data(
        self,
        features: np.ndarray[List[MatLike]],
        labels: np.ndarray[List[int]],
        video_file_paths: List[str],
    ) -> None:
        """Saves processed data objects to pickle.

        Args:
            features (np.ndarray[List[MatLike]]): Processed features (sequence
                of frames).
            labels (np.ndarray[int]): Processed one-hot encoded categorical
                labels.
            video_file_paths (List[str]): Processed video file paths.
        """
        for data_object, output_pickle_path in list(
            zip(
                [features, labels, video_file_paths],
                [
                    self.__features_pickle_path,
                    self.__labels_pickle_path,
                    self.__video_paths_pickle_path,
                ],
            )
        ):
            self.__logger.info(
                f'Saving processed data to {output_pickle_path} .'
            )

            if not os.path.exists(output_pickle_path):
                parent_dir = os.path.dirname(output_pickle_path)

                self.__logger.info(
                    f'Creating pickle folder {parent_dir} for processed data.'
                )

                os.makedirs(parent_dir, exist_ok=True)

            pickle_object(object=data_object, save_path=output_pickle_path)

    def __create_dataset(
        self, data_folder_path: str
    ) -> Tuple[np.ndarray[List[MatLike]], np.ndarray[List[int]], List[str]]:
        """Converts each video in selected data classes to
        `Config.SEQUENCE_LENGTH` number of equally distributed frame sequences
        and saves the processed data to a series of pickle objects.

        Args:
            data_folder_path (str): Data

        Returns:
            Tuple[
                np.ndarray[List[MatLike]], np.ndarray[List[int]], List[str]
            ]: Processed features, labels and video file paths.
        """
        features: List[List[MatLike]] = []
        labels: List[int] = []
        video_file_paths: List[str] = []

        for class_index, class_name in enumerate(self.__data_classes):

            self.__logger.info(f"Processing the '{class_name}' data class.")

            for file_path in os.listdir(
                os.path.join(data_folder_path, class_name)
            ):
                video_path = os.path.join(
                    data_folder_path, class_name, file_path
                )
                frames = self.__extract_fames(video_path=video_path)

                if len(frames) == Config.SEQUENCE_LENGTH:
                    features.append(frames)
                    labels.append(class_index)
                    video_file_paths.append(video_path)

        features = np.asarray(features)
        labels = np.asarray(labels)

        one_hot_encoded_labels = to_categorical(labels)

        self.__pickle_processed_data(
            features=features,
            labels=one_hot_encoded_labels,
            video_file_paths=video_file_paths,
        )

        return features, one_hot_encoded_labels, video_file_paths

    def __get_data_components(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Loads or processed video data in given folder.

        Args:
            data_folder_path (str): Data folder path to process.
            dataset_name (str): Name of the dataset.
        """
        self.__features_pickle_path = os.path.join(
            Config.PROCESSED_FOLDER,
            dataset_name,
            f'features_C_{Config.SUBSET_SIZE}_W_{Config.IMAGE_WIDTH}_'
            f'H_{Config.IMAGE_HEIGHT}_S_{Config.SEQUENCE_LENGTH}.pkl',
        )
        self.__labels_pickle_path = os.path.join(
            Config.PROCESSED_FOLDER,
            dataset_name,
            f'labels_C_{Config.SUBSET_SIZE}_W_{Config.IMAGE_WIDTH}_'
            f'H_{Config.IMAGE_HEIGHT}_S_{Config.SEQUENCE_LENGTH}.pkl',
        )
        self.__video_paths_pickle_path = os.path.join(
            Config.PROCESSED_FOLDER,
            dataset_name,
            f'video_paths_C_{Config.SUBSET_SIZE}_W_{Config.IMAGE_WIDTH}_'
            f'H_{Config.IMAGE_HEIGHT}_S_{Config.SEQUENCE_LENGTH}.pkl',
        )

        if not any(
            os.path.exists(path)
            for path in [
                self.__features_pickle_path,
                self.__labels_pickle_path,
                self.__video_paths_pickle_path,
            ]
        ):

            (
                self.__features,
                self.__labels,
                self.__video_file_paths,
            ) = self.__create_dataset(data_folder_path=data_folder_path)

        else:
            self.__logger.info('Loading processed data from pickle files.')

            self.__features = load_pickle_object(
                pickle_path=self.__features_pickle_path
            )

            self.__labels = load_pickle_object(
                pickle_path=self.__labels_pickle_path
            )

            self.__video_file_paths = load_pickle_object(
                pickle_path=self.__video_paths_pickle_path
            )

    def __split_dataset(
        self,
    ) -> Tuple[
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
    ]:
        """Splits the data to train / validation / test sets.

        Returns:
            Tuple[
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
            ]: Split dataset components.
        """
        (
            features_train_valid,
            features_test,
            labels_train_valid,
            labels_test,
        ) = train_test_split(
            self.__features,
            self.__labels,
            test_size=Config.TEST_SPLIT_PERCENTAGE,
            shuffle=True,
            random_state=Config.RANDOM_STATE,
        )

        (
            features_train,
            features_valid,
            labels_train,
            labels_valid,
        ) = train_test_split(
            features_train_valid,
            labels_train_valid,
            test_size=Config.VALIDATION_SPLIT_PERCENTAGE,
            shuffle=True,
            random_state=Config.RANDOM_STATE,
        )

        return (
            features_train,
            labels_train,
            features_valid,
            labels_valid,
            features_test,
            labels_test,
        )

    def run_data_processing(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> Tuple[
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
    ]:
        """Executes the data processing step.

        Args:
            data_folder_path (str): Data folder path to process.
            dataset_name (str): Name of the dataset.

        Returns:
            Tuple[
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
            ]: Processed data split to train / validation / test sets.
        """
        self.__logger.info(f'Started processing the {dataset_name} dataset.')

        self.__get_data_components(
            data_folder_path=data_folder_path, dataset_name=dataset_name
        )

        self.__logger.info(f'Finished processing the {dataset_name} dataset.')

        return self.__split_dataset()
