import os
from logging import Logger
from typing import Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
from cv2.typing import MatLike
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.utils import create_missing_directory


class SingleImageDataProcessor:

    """The SingleImageDataProcessor class is responsible for executing data
    processing step after the EDA which extracts single frames from videos,
    splits the data to train/validation/test subsets and normalizes the data to
    be ready for model consumption.
    """

    def __init__(self, logger: Logger, data_classes: List[str]) -> None:
        """Initializes the DataProcessor class.

        Args:
            logger (Logger): Run logger used for documenting processes.
            data_classes (List[str]): Name of data classes to process.
        """
        self.__logger = logger
        self.__data_classes = data_classes
        self.__extracted_folder_path = os.path.join(
            Config.PROCESSED_FOLDER,
            'single_frames',
            f'W_{Config.IMAGE_WIDTH}_H_{Config.IMAGE_HEIGHT}_'
            f'S_{Config.SEQUENCE_LENGTH}',
        )

        create_missing_directory(
            path=self.__extracted_folder_path,
            logger=self.__logger,
        )

    def __save_extracted_frame_to_file_system(
        self,
        dataset_name: str,
        video_path: str,
        index: int,
        frame: MatLike,
    ) -> None:
        """Saves extracted frame to the file-system.

        Args:
            dataset_name (str): Name of the processed dataset.
            video_path (str): Path of the processed video.
            index (int): Index of given frame.
            frame (MatLike): Frame to be saved.
        """
        label = video_path.split(os.sep)[-2]
        file_name = f"{os.path.basename(video_path).split('.')[0]}_{index}.jpg"

        output_path = os.path.join(
            self.__extracted_folder_path,
            dataset_name,
            label,
        )

        create_missing_directory(path=output_path, logger=self.__logger)

        frame = cv2.convertScaleAbs(frame, alpha=(255.0))

        cv2.imwrite(
            filename=os.path.join(output_path, file_name),
            img=frame,
        )

    def extract_fames(self, dataset_name: str, video_path: str) -> None:
        """Extracts `Config.SEQUENCE_LENGTH` number of frames from given video.

        Args:
            dataset_name (str): Name of the processed dataset.
            video_path (str): Video file to be processed.
        """
        if self.__logger is not None:
            self.__logger.info(f"Currently processing: '{video_path}'.")

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

            self.__save_extracted_frame_to_file_system(
                dataset_name=dataset_name,
                video_path=video_path,
                index=frame_counter,
                frame=normalized_frame,
            )

        # Release the VideoCapture object
        video_reader.release()

    def __create_dataset(
        self, dataset_name: str, data_folder_path: str
    ) -> None:
        """Converts each video in selected data classes to
        `Config.SEQUENCE_LENGTH` number of equally distributed separate frames.

        Args:
            dataset_name (str): Name of the dataset.
            data_folder_path (str): Data folder path to process.
        """
        for class_name in self.__data_classes:

            self.__logger.info(f"Processing the '{class_name}' data class.")

            for file_path in os.listdir(
                os.path.join(data_folder_path, class_name)
            ):
                video_path = os.path.join(
                    data_folder_path, class_name, file_path
                )

                self.extract_fames(
                    dataset_name=dataset_name,
                    video_path=video_path,
                )

    def __load_images(self, dataset_name: str) -> None:
        """Loads extracted images.

        Args:
            dataset_name (str): Name of the dataset.
        """
        image_paths = list(
            paths.list_images(
                os.path.join(self.__extracted_folder_path, dataset_name)
            )
        )

        data = []
        labels = []

        for image_path in image_paths:
            label = image_path.split(os.path.sep)[-2]
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT))
            # update the data and labels lists, respectively
            data.append(image)
            labels.append(label)

        # Convert the data and labels to NumPy arrays
        self.__features = np.array(data)
        labels = np.array(labels)

        # Perform one-hot encoding on the labels
        lb = LabelBinarizer()
        self.__labels = lb.fit_transform(labels)
        self.__label_classes = lb.classes_

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
        extracted_dataset_folder_path = os.path.join(
            self.__extracted_folder_path, dataset_name
        )

        create_missing_directory(
            path=extracted_dataset_folder_path,
            logger=self.__logger,
        )

        if not os.listdir(extracted_dataset_folder_path):
            self.__create_dataset(
                dataset_name=dataset_name,
                data_folder_path=data_folder_path,
            )

        self.__load_images(dataset_name=dataset_name)

    def __split_dataset(
        self,
    ) -> Tuple[
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[List[MatLike]],
        np.ndarray[List[int]],
        np.ndarray[Any],
    ]:
        """Splits the data to train / validation / test sets.

        Returns:
            Tuple[
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[List[MatLike]], np.ndarray[List[int]],
                np.ndarray[Any],
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
            stratify=self.__labels,
            test_size=Config.TEST_SPLIT_PERCENTAGE,
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
            stratify=labels_train_valid,
            test_size=Config.VALIDATION_SPLIT_PERCENTAGE,
            random_state=Config.RANDOM_STATE,
        )

        return (
            features_train,
            labels_train,
            features_valid,
            labels_valid,
            features_test,
            labels_test,
            self.__label_classes,
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
        np.ndarray[Any],
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
                np.ndarray[Any],
            ]: Processed data split to train / validation / test sets and ID to
                class pairing.
        """
        self.__logger.info(f'Started processing the {dataset_name} dataset.')

        self.__get_data_components(
            data_folder_path=data_folder_path, dataset_name=dataset_name
        )

        self.__logger.info(f'Finished processing the {dataset_name} dataset.')

        return self.__split_dataset()
