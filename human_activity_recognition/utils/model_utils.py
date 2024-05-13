import os
import pickle
from itertools import product
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from keras.models import load_model

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.data_processing.data_processors.image_sequence_data_processor import (
    ImageSequenceDataProcessor,
)
from human_activity_recognition.enums import Model
from human_activity_recognition.pipeline.pipeline import Pipeline


def run_pipeline(
    model_type: Model,
    run_progress_file_path: Optional[str] = None,
    run_progress_message: Optional[str] = None,
) -> None:
    """Runs pipeline functionality during grid search and saves
    `run_progress_message` to output file.

    Args:
        model_type (Model): Type of model used during execution.
        run_progress_file_path (Optional[str]): Grid search progress output file
            path. Defaults to None.
        run_progress_message (Optional[str]): Message saved to output file.
            Defaults to None.
    """
    pipeline = Pipeline(model_type=model_type)
    pipeline.run()

    del pipeline

    if run_progress_file_path is not None:
        with open(run_progress_file_path, 'a') as output_file:
            output_file.write(run_progress_message)


def simple_grid_search(
    run_progress_file_path: str = './tmp/grid_search_progress.txt',
    model_type: Model = Model.CONVLSTM,
    subset_sizes: List[int] = [20],
    image_sizes: List[int] = [2**i for i in range(5, 8)],
    sequence_lengths: List[int] = list(range(10, 50, 10)),
    datasets: List[Tuple[str, str]] = [
        (Config.HMDB_DATA_FOLDER_PATH, Config.HMDB_DATA_NAME),
        (Config.UCF50_DATA_FOLDER_PATH, Config.UCF50_DATA_NAME),
        (Config.OWN_DATA_FOLDER_PATH, Config.OWN_DATA_NAME),
    ],
    training_shuffle_options: List[bool] = [False, True],
    batch_sizes: List[int] = [2**i for i in range(5, 9)],
    early_stopping_options: List[bool] = [False, True],
    early_stopping_patience_options: List[int] = list(range(20, 40, 5)),
    offset: int = 0,
) -> None:
    """Executes simple grid search implementation for project.

    Args:
        run_progress_file_path (str): Grid search progress output file path.
            Defaults to './tmp/grid_search_progress.txt'.
        model_type (Model): Type of model used during execution. Defaults to
            Model.CONVLSTM.
        subset_sizes (List[int]): Number of classes to be selected from the
            `dataset(s)`. Defaults to [20].
        image_sizes (List[int]): Size of the frames extracted from each video.
            Defaults to [32, 64, 128].
        sequence_lengths (List[int]): Number of frames extracted from each
            video. Defaults to [10, 20, 30, 40].
        datasets (List[Tuple[str, str]]): Input dataset folder path and name
            used during training. Defaults to [
                (Config.HMDB_DATA_FOLDER_PATH, Config.HMDB_DATA_NAME),
                (Config.UCF50_DATA_FOLDER_PATH, Config.UCF50_DATA_NAME),
                (Config.OWN_DATA_FOLDER_PATH, Config.OWN_DATA_NAME),
            ].
        training_shuffle_options (List[bool]): Flag options for training
            shuffle. Defaults to [False, True].
        batch_sizes (List[int]): Size of batches used during training. Defaults
            to [32, 64, 128, 256].
        early_stopping_options (List[bool]): Flag options for using Early
            stopping callback. Defaults to [False, True].
        early_stopping_patience_options (List[int]): Possible settings for early
            stopping patience. Defaults to [20, 25, 30, 35].
        offset (int): Offset for all combinations. If training fails for any
            reason it can be continued with setting this value to the last run.
            Defaults to 0.
    """
    all_runs = list(
        product(
            subset_sizes,
            image_sizes,
            sequence_lengths,
            datasets,
            training_shuffle_options,
            batch_sizes,
            early_stopping_options,
        )
    )[offset:]

    number_of_all_runs = len(all_runs)

    for run_id, (
        subset_size,
        image_size,
        sequence_length,
        dataset,
        training_shuffle,
        batch_size,
        early_stopping,
    ) in enumerate(all_runs):
        Config.SUBSET_SIZE = subset_size
        Config.IMAGE_HEIGHT = image_size
        Config.IMAGE_WIDTH = image_size
        Config.SEQUENCE_LENGTH = sequence_length
        Config.DATA = dataset
        Config.TRAINING_SHUFFLE = training_shuffle
        Config.BATCH_SIZE = batch_size
        Config.USE_EARLY_STOPPING = early_stopping

        if early_stopping:
            for early_stopping_patience in early_stopping_patience_options:
                Config.EARLY_STOPPING_PATIENCE = early_stopping_patience

                run_pipeline(
                    model_type=model_type,
                    run_progress_file_path=run_progress_file_path,
                    run_progress_message=(
                        f'{run_id} (ES: {early_stopping_patience}) / '
                        f'{number_of_all_runs-1}\n'
                    ),
                )

        else:
            early_stopping_patience = 5
            Config.EARLY_STOPPING_PATIENCE = early_stopping_patience

            run_pipeline(
                model_type=model_type,
                run_progress_file_path=run_progress_file_path,
                run_progress_message=f'{run_id}/{number_of_all_runs-1}\n',
            )


def _display_single_class_prediction(
    pairing_data: pd.DataFrame, file_names: List[str], predictions: np.ndarray
) -> None:
    """Displays single class classification prediction.

    Args:
        pairing_data (pd.DataFrame): ID to Class pairing data.
        file_names (List[str]): File names used for displaying results.
        predictions (np.ndarray): Model predictions.
    """
    predictions = np.argmax(predictions, axis=1)

    for file_name, predicted_class in list(
        zip(
            file_names,
            [
                pairing_data.iloc[prediction]['class']
                for prediction in predictions
            ],
        )
    ):
        print(f'Prediction for {file_name}: {predicted_class}')


def _display_multi_class_prediction(
    pairing_data: pd.DataFrame,
    file_names: List[str],
    predictions: np.ndarray,
    number_of_top_classes: int,
) -> None:
    """Displays multiple class classification prediction, where the confidence
    of each class is displayed.

    Args:
        pairing_data (pd.DataFrame): ID to Class pairing data.
        file_names (List[str]): File names used for displaying results.
        predictions (np.ndarray): Model predictions.
        number_of_top_classes (int): Maximum number of top classes to display.
    """
    for index, prediction in enumerate(predictions):

        multi_class_predictions = pairing_data.copy()
        multi_class_predictions['confidence'] = [
            round(value, 3) for value in prediction
        ]

        multi_class_predictions = (
            multi_class_predictions[
                ~np.isclose(
                    multi_class_predictions['confidence'],
                    0.0,
                    rtol=1e-09,
                    atol=1e-09,
                )
            ][['class', 'confidence']]
            .sort_values('confidence', ascending=False)
            .head(number_of_top_classes)
            .reset_index(drop=True)
        )

        print(
            (
                f'Prediction for {file_names[index]}:\n'
                f'{multi_class_predictions}'
            ),
            end='\n\n',
        )


def run_prediction_for_frame_sequence_model(
    input_video_paths: List[str],
    model_path: str,
    single_class_prediction: bool,
    number_of_top_classes: int,
) -> None:
    """Runs prediction on specified `input_video_paths` with model saved at
    `model_path` using frame-sequence trained models.

    This function offers the possibility to display only one single result, or
    multiple classes with their individual confidence scores.

    Args:
        input_video_paths (List[str]): Paths to input videos used for
            prediction.
        model_path (str): Path to the model used for prediction.
        single_class_prediction (bool, optional): Flag wether a single result or
            multiple ones should be displayed. Defaults to True.
        number_of_top_classes (int, optional): Maximum number of top classes to
            display. Defaults to 5.
    """
    frames = []
    file_names = list(map(os.path.basename, input_video_paths))

    data_processor = ImageSequenceDataProcessor(logger=None, data_classes=None)

    model = load_model(filepath=model_path)
    input_shape = model.layers[0].input.shape

    # Set configuration based on model, to ensure compatibility between the
    # input data and the model
    Config.SEQUENCE_LENGTH = input_shape[1]
    Config.IMAGE_HEIGHT = input_shape[2]
    Config.IMAGE_WIDTH = input_shape[3]

    # Select ID to Class pairing file to load
    id_class_pairing_pickle_path = os.path.join(
        Config.PROCESSED_FOLDER,
        'frame_sequences',
        model_path.split(os.sep)[-2],  # Model dataset
        f'id_class_pairing_C_{Config.SUBSET_SIZE}_W_{Config.IMAGE_WIDTH}_'
        f'H_{Config.IMAGE_HEIGHT}_S_{Config.SEQUENCE_LENGTH}.pkl',
    )

    with open(id_class_pairing_pickle_path, 'rb') as pairing_file:
        pairing_data: pd.DataFrame = pickle.load(pairing_file)

        # Process each video file to sequence of frames
        for input_video_path in input_video_paths:
            frames.append(data_processor.extract_fames(input_video_path))

        predictions = model.predict(np.asarray(frames))

        if single_class_prediction:
            _display_single_class_prediction(
                pairing_data=pairing_data,
                file_names=file_names,
                predictions=predictions,
            )

        else:
            _display_multi_class_prediction(
                pairing_data=pairing_data,
                file_names=file_names,
                predictions=predictions,
                number_of_top_classes=number_of_top_classes,
            )


def run_prediction_for_single_frame_model(
    input_video_paths: List[str],
    model_path: str,
    single_class_prediction: bool,
    number_of_top_classes: int,
) -> None:
    """Runs prediction on specified `input_video_paths` with model saved at
    `model_path` using single-frame trained models.

    This function offers the possibility to display only one single result, or
    multiple classes with their individual confidence scores.

    Args:
        input_video_paths (List[str]): Paths to input videos used for
            prediction.
        model_path (str): Path to the model used for prediction.
        single_class_prediction (bool, optional): Flag wether a single result or
            multiple ones should be displayed. Defaults to True.
        number_of_top_classes (int, optional): Maximum number of top classes to
            display. Defaults to 5.
    """
    frames = []
    file_names = list(map(os.path.basename, input_video_paths))

    model = load_model(filepath=model_path)
    input_shape = model.input.shape

    # Set configuration based on model, to ensure compatibility between the
    # input data and the model
    Config.IMAGE_HEIGHT = input_shape[1]
    Config.IMAGE_WIDTH = input_shape[2]

    # TODO - Implement prediction


def run_prediction(
    input_video_paths: List[str],
    model_path: str,
    model_type: Model,
    single_class_prediction: bool = True,
    number_of_top_classes: int = 5,
) -> None:
    """Runs prediction on specified `input_video_paths` with model saved at
    `model_path`.

    This function offers the possibility to display only one single result, or
    multiple classes with their individual confidence scores.

    Args:
        input_video_paths (List[str]): Paths to input videos used for
            prediction.
        model_path (str): Path to the model used for prediction.
        single_class_prediction (bool, optional): Flag wether a single result or
            multiple ones should be displayed. Defaults to True.
        number_of_top_classes (int, optional): Maximum number of top classes to
            display. Defaults to 5.
    """
    match (model_type):

        case Model.CONVLSTM:
            run_prediction_for_frame_sequence_model(
                input_video_paths=input_video_paths,
                model_path=model_path,
                single_class_prediction=single_class_prediction,
                number_of_top_classes=number_of_top_classes,
            )

        case Model.RESNET | Model.GOOGLENET:
            run_prediction_for_single_frame_model(
                input_video_paths=input_video_paths,
                model_path=model_path,
                single_class_prediction=single_class_prediction,
                number_of_top_classes=number_of_top_classes,
            )
