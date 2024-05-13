# SOURCE: https://www.kaggle.com/code/joeylimzy/flower-recognition-using-googlenet

import os
from logging import Logger
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from cv2.typing import MatLike
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import (
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Layer,
    MaxPooling2D,
)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from human_activity_recognition.configs.config import ProjectConfig as Config
from human_activity_recognition.utils import (
    generate_confusion_matrix,
    generate_training_history_plots,
    multi_log,
    save_model_performance,
)
from human_activity_recognition.utils.utils import (
    create_missing_parent_directories,
)


class HumanActivityRecognitionModelGoogLeNet:

    """The HumanActivityRecognitionModelGoogLeNet represents the classifier model
    which utilizes the GoogLeNet image classification model.
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
        label_classes,
        timestamp: str,
        loggers: List[Logger],
        plots_output_folder_path: str,
        model_output_folder_path: str,
    ) -> None:
        """Initializes the HumanActivityRecognitionModelGoogLeNet class.

        Args:
            dataset_name (str): Name of the dataset.
            train_set_features (np.ndarray): Training dataset features.
            train_set_label (np.ndarray): Training dataset label.
            validation_set_features (np.ndarray): Validation dataset features.
            validation_set_label (np.ndarray): Validation dataset label.
            test_set_features (np.ndarray): Test dataset features.
            test_set_label (np.ndarray): Test dataset label.
            label_classes (): ID and Class pairing used for
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
        self.__label_classes = label_classes

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

        self.__model_tag = 'googlenet'
        self.__model_name = (
            f'human_activity_recognition_model_{self.__model_tag}_'
            f'{Config.SUBSET_SIZE}_classes_adam_'
            f"{str(Config.ADAM_OPTIMIZER_LEARNING_RATE).replace('.', '_')}"
            f'__epochs_{Config.TRAINING_EPOCHS}__batch_size_{Config.BATCH_SIZE}'
            f'{early_stopping_tag}_{self.__timestamp}'
        )
        self.__model_type = 'HARM - GoogLeNet'

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

        self.__set_image_data_generators()
        self.__set_gpu(memory_limit=4096)

        self.__model = self.__build_model()

    def __set_image_data_generators(self) -> None:
        """Sets image data generators."""
        self.__training_data_generator = ImageDataGenerator(
            rotation_range=30,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
        )

        self.__validation_data_generator = ImageDataGenerator()

        mean = np.array([123.68, 116.779, 103.939], dtype='float32')
        self.__training_data_generator.mean = mean
        self.__validation_data_generator.mean = mean

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

    def __inception(
        self, x: Layer, filters: List[Union[int, Tuple[int, int]]]
    ) -> Layer:
        """Builds Inception block of GoogLeNet Model.

        Args:
            x (Layer): Input layer.
            filters (List[Union[int, Tuple[int, int]]]): Number or shape of
                filters used in Inception block.

        Returns:
            Layer: Built Inception block of GoogLeNet Model.
        """
        # 1x1
        path1 = Conv2D(
            filters=filters[0],
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(x)

        # 1x1->3x3
        path2 = Conv2D(
            filters=filters[1][0],
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(x)
        path2 = Conv2D(
            filters=filters[1][1],
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
        )(path2)

        # 1x1->5x5
        path3 = Conv2D(
            filters=filters[2][0],
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(x)
        path3 = Conv2D(
            filters=filters[2][1],
            kernel_size=(5, 5),
            strides=1,
            padding='same',
            activation='relu',
        )(path3)

        # 3x3->1x1
        path4 = MaxPooling2D(
            pool_size=(3, 3),
            strides=1,
            padding='same',
        )(x)
        path4 = Conv2D(
            filters=filters[3],
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(path4)

        return Concatenate(axis=-1)([path1, path2, path3, path4])

    def __auxiliary(self, x: Layer, name: Optional[str] = None) -> Layer:
        """Builds Auxiliary block of the GoogLeNet Model.

        Args:
            x (Layer): Input layer.
            name (Optional[str]): Name of the new layer. Defaults to None.

        Returns:
            Layer: Built Auxiliary block of the GoogLeNet Model.
        """
        layer = AveragePooling2D(
            pool_size=(5, 5),
            strides=3,
            padding='valid',
        )(x)
        layer = Conv2D(
            filters=128,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(layer)
        layer = Flatten()(layer)
        layer = Dense(
            units=256,
            activation='relu',
            kernel_regularizer=regularizers.l2(0.0001),
        )(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(
            units=Config.SUBSET_SIZE,
            activation=Config.OUTPUT_LAYER_ACTIVATION_FUNCTION,
            name=name,
        )(layer)
        return layer

    def __googlenet(self) -> Model:
        """Builds GoogLeNet Model.

        Returns:
            Model: Built GoogLeNet Model.
        """
        layer_in = Input(shape=(Config.IMAGE_HEIGHT, Config.IMAGE_WIDTH, 3))

        # stage-1
        layer = Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding='same',
            activation='relu',
        )(layer_in)
        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)
        layer = BatchNormalization()(layer)

        # stage-2
        layer = Conv2D(
            filters=64,
            kernel_size=(1, 1),
            strides=1,
            padding='same',
            activation='relu',
        )(layer)
        layer = Conv2D(
            filters=192,
            kernel_size=(3, 3),
            strides=1,
            padding='same',
            activation='relu',
        )(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

        # stage-3
        layer = self.__inception(layer, [64, (96, 128), (16, 32), 32])  # 3a
        layer = self.__inception(layer, [128, (128, 192), (32, 96), 64])  # 3b
        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

        # stage-4
        layer = self.__inception(layer, [192, (96, 208), (16, 48), 64])  # 4a
        aux1 = self.__auxiliary(layer, name='aux1')
        layer = self.__inception(layer, [160, (112, 224), (24, 64), 64])  # 4b
        layer = self.__inception(layer, [128, (128, 256), (24, 64), 64])  # 4c
        layer = self.__inception(layer, [112, (144, 288), (32, 64), 64])  # 4d
        aux2 = self.__auxiliary(layer, name='aux2')
        layer = self.__inception(layer, [256, (160, 320), (32, 128), 128])  # 4e
        layer = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(layer)

        # stage-5
        layer = self.__inception(layer, [256, (160, 320), (32, 128), 128])  # 5a
        layer = self.__inception(layer, [384, (192, 384), (48, 128), 128])  # 5b
        layer = AveragePooling2D(pool_size=(7, 7), strides=1, padding='valid')(
            layer
        )

        # stage-6
        layer = Flatten()(layer)
        layer = Dropout(0.5)(layer)
        layer = Dense(
            units=256,
            activation='linear',
            kernel_regularizer=regularizers.l2(0.0001),
        )(layer)
        main = Dense(
            units=Config.SUBSET_SIZE,
            activation=Config.OUTPUT_LAYER_ACTIVATION_FUNCTION,
            name='main',
        )(layer)

        return Model(inputs=layer_in, outputs=[main, aux1, aux2])

    def __build_model(self) -> Model:
        """Builds Human Activity Recognition Model - GoogLeNet variant.

        Returns:
            Model: Built model.
        """
        self.__run_logger.info(
            'Building Human Activity Recognition Model - GoogLeNet variant.'
        )

        model = self.__googlenet()

        self.__run_logger.info(
            'Finished building Human Activity Recognition Model - ConvLSTM '
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
            metrics=[
                Config.METRICS_TO_SHOW,
                Config.METRICS_TO_SHOW,
                Config.METRICS_TO_SHOW,
            ],
        )

        with tf.device('/gpu:0' if self.__gpus else '/cpu:0'):
            history = self.__model.fit(
                self.__training_data_generator.flow(
                    self.__train_set_features,
                    self.__train_set_label,
                    batch_size=Config.BATCH_SIZE,
                ),
                epochs=Config.TRAINING_EPOCHS,
                shuffle=Config.TRAINING_SHUFFLE,
                verbose=2,
                validation_data=self.__validation_data_generator.flow(
                    self.__validation_set_features,
                    self.__validation_set_label,
                    batch_size=Config.BATCH_SIZE,
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
            column_pairs=[['loss', 'val_loss'], ['main_acc', 'val_main_acc']],
        )

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Executes model prediction.

        Args:
            features (np.ndarray): Features to use for prediction.

        Returns:
            np.ndarray: Prediction results.
        """
        return np.argmax(self.__model.predict(features)[0], axis=1)

    def __evaluate(self) -> None:
        """Evaluates Human Activity Recognition Model performance."""
        # Predict based on the Testing Data Set
        network_prediction = self.predict(features=self.__test_set_features)

        self.__test_set_label = np.argmax(self.__test_set_label, axis=1)

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
            labels=self.__label_classes,
            output_path=self.__confusion_matrix_output_path,
        )

        class_report = classification_report(
            y_true=self.__test_set_label,
            y_pred=network_prediction,
            target_names=self.__label_classes,
        )

        class_report_dict = classification_report(
            y_true=self.__test_set_label,
            y_pred=network_prediction,
            target_names=self.__label_classes,
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
