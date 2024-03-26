import os
from logging import Logger
import pickle
from typing import Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cv2.typing import MatLike
from keras.callbacks import EarlyStopping
from keras.layers import (
    ConvLSTM2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling3D,
    TimeDistributed,
)
from keras.models import Model, Sequential
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix

from human_activity_recognition.configs.config import ProjectConfig as Config
from human_activity_recognition.utils import create_missing_parent_directories


# TODO - Increase model performance (Possible architecture changes).
# TODO - Change print statements to logging.


class HumanActivityRecognitionModelConvLSTM:

    """The HumanActivityRecognitionModelConvLSTM represents the classifier model
    which consist of the combination of a multiple Convolutional layers and Long
    Short-Term Memory (LSTM) layers.
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
        timestamp: str,
        loggers: List[Logger],
        plots_output_folder_path: str,
        model_output_folder_path: str,
    ) -> None:
        """Initializes the HumanActivityRecognitionModel class.

        Args:
            dataset_name (str): Name of the dataset.
            train_set_features (np.ndarray): Training dataset features.
            train_set_label (np.ndarray): Training dataset label.
            validation_set_features (np.ndarray): Validation dataset features.
            validation_set_label (np.ndarray): Validation dataset label.
            test_set_features (np.ndarray): Test dataset features.
            test_set_label (np.ndarray): Test dataset label.
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

        # endregion

        self.__timestamp = timestamp
        self.__run_logger = loggers[0]
        self.__evaluation_logger = loggers[1]

        early_stopping_tag = (
            Config.EARLY_STOPPING_TAG if Config.USE_EARLY_STOPPING else ''
        )

        self.__model_name = (
            f'human_activity_recognition_model_{Config.SUBSET_SIZE}_classes_'
            f"adam_{str(Config.ADAM_OPTIMIZER_LEARNING_RATE).replace('.', '_')}"
            f'__epochs_{Config.TRAINING_EPOCHS}__batch_size_{Config.BATCH_SIZE}'
            f'{early_stopping_tag}_{self.__timestamp}'
        )

        self.__model_output_path: str = os.path.join(
            model_output_folder_path,
            dataset_name,
            f'{self.__model_name}.keras',
        )

        self.__confusion_matrix_output_path = os.path.join(
            plots_output_folder_path,
            dataset_name,
            'confusion_matrices',
            'image',
            f'{self.__timestamp}_confusion_matrix.png',
        )

        create_missing_parent_directories(
            file_paths=[
                self.__model_output_path,
                self.__confusion_matrix_output_path
            ],
            logger=self.__run_logger,
        )

        self.__model = self.__build_model()

    def __generate_confusion_matrix(
        self, confusion_matrix: np.ndarray, labels: List[Any]
    ) -> None:
        """Generates and saves the passed in Confusion Matrix.

        Args:
            confusion_matrix (numpy.ndarray) : Confusion Matrix to display.
            labels (List[Any]) : Labels for Columns and Indexes.
        """
        # SOURCE:
        # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix

        # Convert Confusion Matrix to DataFrame
        confusion_matrix = pd.DataFrame(
            data=confusion_matrix, index=labels, columns=labels
        )

        # Create new Figure
        plt.figure(figsize=(16, 9))

        # Plot the Confusion Matrix
        sns.heatmap(confusion_matrix, annot=True, fmt='g')

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')

        plt.savefig(self.__confusion_matrix_output_path)

    def __build_model(self) -> Model:
        """Builds Human Activity Recognition Model - ConvLSTM variant.

        Returns:
            Model: Built model.
        """
        self.__run_logger.info(
            'Building Human Activity Recognition Model - ConvLSTM variant.'
        )

        model = Sequential()
        model.add(
            Input(
                shape=(
                    Config.SEQUENCE_LENGTH,
                    Config.IMAGE_HEIGHT,
                    Config.IMAGE_WIDTH,
                    3
                )
            )
        )

        for iteration, number_of_filters in enumerate([4, 8, 14, 16]):
            model.add(
                ConvLSTM2D(
                    filters=number_of_filters,
                    kernel_size=(3, 3),
                    activation='tanh',
                    data_format='channels_last',
                    recurrent_dropout=0.2,
                    return_sequences=True,
                )
            )

            model.add(
                MaxPooling3D(
                    pool_size=(1, 2, 2),
                    padding='same',
                    data_format='channels_last'
                )
            )
            
            if iteration != 3:
                model.add(TimeDistributed(layer=Dropout(rate=0.2)))

        model.add(Flatten())

        model.add(
            Dense(
                units=Config.SUBSET_SIZE,
                activation=Config.OUTPUT_LAYER_ACTIVATION_FUNCTION
            )
        )

        self.__run_logger.info(
            'Finished building Human Activity Recognition Model - ConvLSTM '
            'variant.'
        )

        model.summary()

        # Opened issue in keras
        # model.summary(
        #     print_fn=lambda summary: multi_log(
        #         loggers=[self.__run_logger, self.__evaluation_logger],
        #         log_func='info',
        #         log_message=f'Model architecture: {summary}'
        #     )
        # )

        return model

    def __train(self) -> None:
        """Trains the Human Activity Recognition Model."""
        early_stopping_callback = EarlyStopping(
            monitor=Config.EARLY_STOPPING_MONITOR,
            patience=Config.EARLY_STOPPING_PATIENCE,
            mode=Config.EARLY_STOPPING_MODE,
            restore_best_weights=Config.EARLY_STOPPING_RESTORE_BEST_WEIGHTS
        )

        self.__model.compile(
            loss=Config.LOSS_FUNCTION,
            optimizer=Adam(learning_rate=Config.ADAM_OPTIMIZER_LEARNING_RATE),
        )
        
        self.__model.fit(
            self.__train_set_features,
            self.__train_set_label,
            epochs=Config.TRAINING_EPOCHS,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            verbose=2,
            validation_data=(
                self.__validation_set_features,
                self.__validation_set_label,
            ),
            callbacks=(
                [early_stopping_callback] if Config.USE_EARLY_STOPPING else None
            ),
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

        # Display Confusion Matrix
        conf_matrix = confusion_matrix(
            y_true=self.__test_set_label, y_pred=network_prediction
        )

        class_report = classification_report(
            y_true=self.__test_set_label, y_pred=network_prediction
        )

        print(class_report)

        self.__generate_confusion_matrix(
            confusion_matrix=conf_matrix,
            labels=list(range(0, Config.SUBSET_SIZE)),
        )

    def run_modeling(self) -> None:
        """Executes modeling steps."""
        self.__train()
        self.__evaluate()
        self.__model.save(self.__model_output_path)
