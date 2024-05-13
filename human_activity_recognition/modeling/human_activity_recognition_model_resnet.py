# SOURCE: https://pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/

import os
from logging import Logger
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from cv2.typing import MatLike
from keras import regularizers
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping
from keras.layers import (
    LSTM,
    BatchNormalization,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
    TimeDistributed,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from human_activity_recognition.configs.config import ProjectConfig as Config
from human_activity_recognition.utils import (
    generate_confusion_matrix,
    generate_training_history_plots,
    multi_log,
    save_model_performance,
)
from human_activity_recognition.utils.utils import create_missing_parent_directories


class HumanActivityRecognitionModelResNet:

    """The HumanActivityRecognitionModelResNet represents the classifier model
    which consist of the combination of the ResNet50 image classification model
    and Long Short-Term Memory (LSTM) layers.
    """

    def __init__(
        self,
        dataset_name: str,
        train_set_features: np.ndarray[List[MatLike]],
        train_set_label: np.ndarray[List[int]],
        validation_set_features: np.ndarray[List[MatLike]],
        validation_set_label: np.ndarray[List[int]],
        test_set_features: np.ndarray[List[MatLike]],
        test_set_label: np.ndarray[List[int]],
        id_class_pairing: pd.DataFrame,
        timestamp: str,
        loggers: List[Logger],
        plots_output_folder_path: str,
        model_output_folder_path: str,
    ) -> None:
        """Initializes the HumanActivityRecognitionModelResNet class.

        Args:
            dataset_name (str): Name of the dataset.
            train_set_features (np.ndarray): Training dataset features.
            train_set_label (np.ndarray): Training dataset label.
            validation_set_features (np.ndarray): Validation dataset features.
            validation_set_label (np.ndarray): Validation dataset label.
            test_set_features (np.ndarray): Test dataset features.
            test_set_label (np.ndarray): Test dataset label.
            id_class_pairing (pd.DataFrame): ID and Class pairing used for
                evaluation labeling.
            timestamp (str): Current run timestamp.
            loggers (List[Logger]): Run and evaluation logger used for
                documenting processes.
            plots_output_folder_path (str): Plots output folder path.
            model_output_folder_path (str): Model output folder path.
        """

        # region Datasets

        self.__train_set_features: np.ndarray = train_set_features
        self.__train_set_label: np.ndarray = train_set_label
        self.__validation_set_features: np.ndarray = validation_set_features
        self.__validation_set_label: np.ndarray = validation_set_label
        self.__test_set_features: np.ndarray = test_set_features
        self.__test_set_label: np.ndarray = test_set_label
        self.__id_class_pairing: pd.DataFrame = id_class_pairing

        # endregion

        self.__dataset_name = dataset_name
        self.__timestamp = timestamp
        self.__run_logger = loggers[0]
        self.__loggers = loggers
        self.__evaluation_logger_output_path = (
            loggers[1].handlers[0].baseFilename
        )

        early_stopping_tag = (
            Config.EARLY_STOPPING_TAG if Config.USE_EARLY_STOPPING else ''
        )

        self.__model_tag = 'resnet'
        self.__model_name = (
            f'human_activity_recognition_model_{self.__model_tag}_'
            f'{Config.SUBSET_SIZE}_classes_adam_'
            f"{str(Config.ADAM_OPTIMIZER_LEARNING_RATE).replace('.', '_')}"
            f'__epochs_{Config.TRAINING_EPOCHS}__batch_size_{Config.BATCH_SIZE}'
            f'{early_stopping_tag}_{self.__timestamp}'
        )
        self.__model_type = 'HARM - ResNet'

        self.__model_output_path: str = os.path.join(
            model_output_folder_path,
            self.__dataset_name,
            f'{self.__model_name}.keras',
        )

        self.__confusion_matrix_output_path = os.path.join(
            plots_output_folder_path,
            self.__dataset_name,
            'confusion_matrices',
            'image',
            f'{self.__timestamp}_{self.__model_tag}_confusion_matrix.png',
        )

        create_missing_parent_directories(
            file_paths=[
                self.__model_output_path,
                self.__confusion_matrix_output_path,
            ],
            logger=self.__run_logger,
        )

        self.__set_gpu(memory_limit=4096)

        self.__model = self.__build_model()

    def __set_gpu(self, memory_limit: int) -> None:
        """Sets up GPU for model training by limiting the available memory to
        avoid memory overflow.

        Args:
            memory_limit (int): Size of graphical memory to allocate in MB.
        """
        self.__gpus = tf.config.experimental.list_physical_devices('GPU')
        if self.__gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    self.__gpus[0],
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit
                        )
                    ],
                )
            except RuntimeError as e:
                print(e)

    def __build_model(self) -> Model:
        """Builds Human Activity Recognition Model - ResNet variant.

        Returns:
            Model: Built model.
        """
        self.__run_logger.info(
            'Building Human Activity Recognition Model - ResNet variant.'
        )

        resnet = ResNet50(
            weights='imagenet',
            include_top=False,
        )

        # loop over all layers in the base model and freeze them so they will
        # *not* be updated during the training process
        for layer in resnet.layers:
            layer.trainable = False

        model = Sequential()
        model.add(
            TimeDistributed(
                resnet,
                input_shape=(
                    Config.SEQUENCE_LENGTH,
                    Config.IMAGE_WIDTH,
                    Config.IMAGE_HEIGHT,
                    3,
                ),
            )
        )
        model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))
        model.add(TimeDistributed(BatchNormalization()))
        model.add(TimeDistributed(Flatten(name='flatten')))
        model.add(
            TimeDistributed(
                Dense(
                    512,
                    activation='relu',
                    kernel_regularizer=regularizers.l2(0.0001),
                )
            )
        )
        model.add(TimeDistributed(Dropout(0.5)))
        model.add(LSTM(128, return_sequences=False, dropout=0.2))
        model.add(
            Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001),
            )
        )
        model.add(
            Dense(
                32,
                activation='relu',
                kernel_regularizer=regularizers.l2(0.0001),
            )
        )
        model.add(Dropout(0.2))
        model.add(Dense(Config.SUBSET_SIZE, activation='softmax'))

        self.__run_logger.info(
            'Finished building Human Activity Recognition Model - ResNet '
            'variant.'
        )

        model.summary(
            print_fn=lambda summary: multi_log(
                loggers=self.__loggers,
                log_func='info',
                log_message=f'Model architecture: {summary}',
            )
        )

        return model

    def __train(self) -> None:
        """Trains the Human Activity Recognition Model."""
        early_stopping_callback = EarlyStopping(
            monitor=Config.EARLY_STOPPING_MONITOR,
            patience=Config.EARLY_STOPPING_PATIENCE,
            mode=Config.EARLY_STOPPING_MODE,
            restore_best_weights=Config.EARLY_STOPPING_RESTORE_BEST_WEIGHTS,
        )

        self.__model.compile(
            loss=Config.LOSS_FUNCTION,
            optimizer=Adam(
                learning_rate=Config.ADAM_OPTIMIZER_LEARNING_RATE,
                ema_momentum=0.9,
                weight_decay=(
                    Config.ADAM_OPTIMIZER_LEARNING_RATE / Config.TRAINING_EPOCHS
                ),
            ),
            metrics=Config.METRICS_TO_SHOW,
        )

        with tf.device('/gpu:0' if self.__gpus else '/cpu:0'):
            history = self.__model.fit(
                self.__train_set_features,
                self.__train_set_label,
                epochs=Config.TRAINING_EPOCHS,
                shuffle=Config.TRAINING_SHUFFLE,
                verbose=2,
                validation_data=(
                    self.__validation_set_features,
                    self.__validation_set_label,
                ),
                callbacks=(
                    [early_stopping_callback]
                    if Config.USE_EARLY_STOPPING
                    else None
                ),
            )

        generate_training_history_plots(
            history=pd.DataFrame(history.history),
            plot_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
            dataset_name=self.__dataset_name,
            timestamp=self.__timestamp,
            model_tag=self.__model_tag,
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Executes model prediction.

        Args:
            features (np.ndarray): Features to use for prediction.

        Returns:
            np.ndarray: Prediction results.
        """
        return np.argmax(self.__model.predict(features), axis=1)

    def __evaluate(self) -> None:
        """Evaluates Human Activity Recognition Model performance."""
        # Predict based on the Testing Data Set
        network_prediction = self.predict(features=self.__test_set_features)

        self.__test_set_label = np.argmax(self.__test_set_label, axis=1)

        labels = self.__id_class_pairing.sort_values('id')['class'].values

        # Display Confusion Matrix
        conf_matrix = confusion_matrix(
            y_true=self.__test_set_label,
            y_pred=network_prediction,
        )

        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=f'Confusion Matrix:\n{conf_matrix}',
        )

        generate_confusion_matrix(
            confusion_matrix=conf_matrix,
            labels=labels,
            output_path=self.__confusion_matrix_output_path,
        )

        class_report = classification_report(
            y_true=self.__test_set_label,
            y_pred=network_prediction,
            target_names=labels,
        )

        class_report_dict = classification_report(
            y_true=self.__test_set_label,
            y_pred=network_prediction,
            target_names=labels,
            output_dict=True,
        )

        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=f'Classification Report:\n{class_report}',
        )

        save_model_performance(
            dataset_name=self.__dataset_name,
            model_type=self.__model_type,
            classification_report=class_report_dict,
            model_output_path=self.__model_output_path,
            architecture_and_evaluation_output_path=self.__evaluation_logger_output_path,
        )

    def run_modeling(self) -> None:
        """Executes modeling steps."""
        self.__train()
        self.__evaluate()
        self.__model.save(self.__model_output_path)
