import os
import shutil
from logging import Logger
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytorchvideo.data
from keras.utils import to_categorical
from pytorchvideo.data import LabeledVideoDataset
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from sklearn.model_selection import train_test_split
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from transformers import AutoImageProcessor, AutoModelForVideoClassification

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.utils import create_missing_directory


class VideoDataProcessor:

    """The VideoDataProcessor class is responsible for executing data
    processing step after the EDA which splits the data to train/validation/test
    subsets and creates transformative generators to be ready for model
    consumption.
    """

    def __init__(self, logger: Logger, data_classes: List[str]) -> None:
        """Initializes the DataProcessor class.

        Args:
            logger (Logger): Run logger used for documenting processes.
            data_classes (List[str]): Name of data classes to process.
        """
        self.__logger = logger
        self.__data_classes = data_classes

        self.__subsets_folder_path = os.path.join(
            Config.PROCESSED_FOLDER,
            'subsets',
        )

    def __create_dataset(
        self, data_folder_path: str
    ) -> Tuple[np.ndarray[List[int]], List[str]]:
        """Converts each video in selected data classes to
        label and filepath combination.

        Args:
            data_folder_path (str): Data folder path to process.

        Returns:
            Tuple[np.ndarray[List[int]], List[str]]: Processed features, labels
                and video file paths.
        """
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

                labels.append(class_index)
                video_file_paths.append(video_path)

        labels = np.asarray(labels)

        one_hot_encoded_labels = to_categorical(labels)

        return one_hot_encoded_labels, video_file_paths

    def __split_dataset(
        self,
    ) -> Tuple[np.ndarray[Any], np.ndarray[Any], np.ndarray[Any],]:
        """Splits the data to train / validation / test sets.

        Returns:
            Tuple[
                np.ndarray[Any],
                np.ndarray[Any],
                np.ndarray[Any],
            ]: Split dataset components.
        """
        (
            features_train_valid,
            features_test,
            labels_train_valid,
            _,
        ) = train_test_split(
            self.__video_file_paths,
            self.__labels,
            test_size=Config.TEST_SPLIT_PERCENTAGE,
            random_state=Config.RANDOM_STATE,
        )

        (features_train, features_valid, _, _,) = train_test_split(
            features_train_valid,
            labels_train_valid,
            test_size=Config.VALIDATION_SPLIT_PERCENTAGE,
            random_state=Config.RANDOM_STATE,
        )

        return (
            features_train,
            features_valid,
            features_test,
        )

    def __create_subsets(
        self,
        dataset_name: str,
    ) -> None:
        """Copies files based on train test split to subset folder.

        Args:
            dataset_name (str): Name of the dataset.
        """
        self.__dataset_root_folder_path = os.path.join(
            self.__subsets_folder_path,
            dataset_name,
            (
                f'{dataset_name}_SUBSETS_'
                f"V_{str(Config.VALIDATION_SPLIT_PERCENTAGE).replace('.', '_')}"
                f"_T_{str(Config.TEST_SPLIT_PERCENTAGE).replace('.', '_')}"
            ),
        )

        create_missing_directory(
            path=self.__dataset_root_folder_path,
            logger=self.__logger,
        )

        if not os.listdir(self.__dataset_root_folder_path):
            for subset_name, subset in list(
                zip(
                    ['train', 'validation', 'test'],
                    [
                        self.__features_train,
                        self.__features_valid,
                        self.__features_test,
                    ],
                )
            ):

                for file_path in subset:
                    label, file_name = file_path.split(os.sep)[-2:]

                    subset_path = os.path.join(
                        self.__dataset_root_folder_path, subset_name, label
                    )

                    create_missing_directory(
                        path=subset_path,
                        logger=self.__logger,
                    )

                    shutil.copy(
                        src=file_path, dst=os.path.join(subset_path, file_name)
                    )

    def __get_id_class_pairing(self) -> None:
        """Generates label2id and id2label pairings."""
        dataset_root_folder = Path(self.__dataset_root_folder_path)

        all_video_file_paths = (
            list(dataset_root_folder.glob('train/*/*.mp4'))
            + list(dataset_root_folder.glob('validation/*/*.mp4'))
            + list(dataset_root_folder.glob('test/*/*.mp4'))
        )

        class_labels = sorted(
            {str(path).split('/')[-2] for path in all_video_file_paths}
        )
        self.__label2id = {label: i for i, label in enumerate(class_labels)}
        self.__id2label = {i: label for label, i in self.__label2id.items()}

    def __crete_dataset_generators(self) -> None:
        """Creates dataset generators used during modeling."""
        self.__image_processor = AutoImageProcessor.from_pretrained(
            Config.MODEL_IDENTIFIER
        )
        model = AutoModelForVideoClassification.from_pretrained(
            Config.MODEL_IDENTIFIER,
            label2id=self.__label2id,
            id2label=self.__id2label,
            ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
        )

        mean = self.__image_processor.image_mean
        std = self.__image_processor.image_std
        resize_to = (Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH)  # (224, 224)

        num_frames_to_sample = model.config.num_frames
        sample_rate = 4
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps

        # Training dataset transformations.
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key='video',
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )

        # Training dataset.
        self.__train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.__dataset_root_folder_path, 'train'),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                'random', clip_duration
            ),
            decode_audio=False,
            transform=train_transform,
        )

        # Validation and evaluation datasets' transformations.
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key='video',
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize(resize_to),
                        ]
                    ),
                ),
            ]
        )

        # Validation and evaluation datasets.
        self.__validation_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(
                self.__dataset_root_folder_path, 'validation'
            ),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                'uniform', clip_duration
            ),
            decode_audio=False,
            transform=val_transform,
        )

        self.__test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.__dataset_root_folder_path, 'test'),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                'uniform', clip_duration
            ),
            decode_audio=False,
            transform=val_transform,
        )

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
        # Load all paths and zeroes
        self.__labels, self.__video_file_paths = self.__create_dataset(
            data_folder_path=data_folder_path,
        )

        # Split them
        (
            self.__features_train,
            self.__features_valid,
            self.__features_test,
        ) = self.__split_dataset()

        # Copy files
        self.__create_subsets(dataset_name=dataset_name)

        self.__get_id_class_pairing()

        self.__crete_dataset_generators()

    def run_data_processing(
        self, data_folder_path: str, dataset_name: str
    ) -> Tuple[
        AutoImageProcessor,
        LabeledVideoDataset,
        LabeledVideoDataset,
        LabeledVideoDataset,
        Dict[int, str],
        Dict[str, int],
    ]:
        """Executes the data processing step.

        Args:
            data_folder_path (str): Data folder path to process.
            dataset_name (str): Name of the dataset.

        Returns:
            Tuple[
                AutoImageProcessor,
                LabeledVideoDataset,
                LabeledVideoDataset,
                LabeledVideoDataset,
                Dict[int, str],
                Dict[str, int]
            ]: Processed data split to train / validation / test sets and ID to
                class pairing.
        """
        self.__logger.info(f'Started processing the {dataset_name} dataset.')

        self.__get_data_components(
            data_folder_path=data_folder_path, dataset_name=dataset_name
        )

        self.__logger.info(f'Finished processing the {dataset_name} dataset.')

        return (
            self.__image_processor,
            self.__train_dataset,
            self.__validation_dataset,
            self.__test_dataset,
            self.__id2label,
            self.__label2id,
        )
