import logging
import os
import pickle
from logging import Logger
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import requests
from cv2.typing import MatLike
from tqdm import tqdm
from unrar import rarfile

from human_activity_recognition.configs import ProjectConfig as Config


def create_missing_directory(
    path: str, logger: Optional[Logger] = None
) -> None:
    """Creates missing directory.

    Args:
        path (str): Directory path to create.
        logger (Optional[Logger]): Logger used for documenting the process. If
            there is no logger no logs are generated. Defaults to None.
    """
    if not os.path.exists(path):

        if logger is not None:
            logger.info(f'Creating folder: {path}')

        os.makedirs(path, exist_ok=True)


def create_missing_parent_directories(
    file_paths: List[str],
    logger: Optional[Logger] = None,
) -> List[str]:
    """Creates missing parent directories for file paths.

    Args:
        file_paths (List[str]): List of paths to process.
        logger (Optional[Logger]): Logger used for documenting the process. If
            there is no logger no logs are generated. Defaults to None.

    Returns:
        List[str]: List of created parent directory paths.
    """
    parent_directories: List[str] = []

    for file_path in file_paths:
        parent_directory = os.path.dirname(file_path)
        parent_directories.append(parent_directory)

        create_missing_directory(path=parent_directory, logger=logger)

    return parent_directories


def setup_logging(
    logger_type: str,
    log_path: str,
    log_file_timestamp: str,
    log_to_stdout: bool = False,
) -> Logger:
    """Sets up logger with specified `type` and `output file` (labelled with
    `timestamp`).

    Args:
        logger_type (str): Type of logger.
        log_path (str): Output folder path of the log file.
        log_file_timestamp (str): Timestamp used for naming the output file.
        log_to_stdout (bool): Flag wether given logger should print to the
            Standard Output (stdout). Defaults to False.

    Returns:
        Logger: Configured logger with set output file.
    """
    create_missing_directory(path=log_path)

    log_file_handler = logging.FileHandler(
        os.path.join(log_path, f'{logger_type}_log_{log_file_timestamp}.log')
    )
    log_file_handler.setFormatter(Config.LOG_FORMATTER)

    logger = logging.getLogger(logger_type)
    logger.setLevel(logging.INFO)
    logger.addHandler(log_file_handler)

    if log_to_stdout:
        log_stream_handler = logging.StreamHandler()
        log_stream_handler.setFormatter(Config.LOG_FORMATTER)
        logger.addHandler(log_stream_handler)

    return logger


def setup_folders(logger: Logger) -> None:
    """Sets up project folders used during execution.

    Args:
        logger (Logger): Logger used for documenting the setup process.
    """
    logger.info('Starting folder setup process.')

    for path in Config.PATHS_TO_CREATE:
        create_missing_directory(path=path, logger=logger)

    logger.info('Folder setup process finished successfully.')


def download_dataset(
    logger: Logger,
    dataset_name: str,
    dataset_url: str,
    dataset_output_path: str,
    chunk_size: Optional[int] = 1024,
) -> None:
    """Downloads given dataset RAR archive from the URL.

    Args:
        logger (Logger): Logger used for documenting the download process.
        dataset_name (str): Name of the dataset (for documenting purposes).
        dataset_url (str): Download URL for the dataset archive.
        dataset_output_path (str): Downloaded RAR archive output file path.
        chunk_size (Optional[int]): Download chunk size in Bytes. Defaults to
            1024.
    """
    # SOURCE: https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51

    logger.info(f'Starting download of {dataset_name} dataset.')

    resp = requests.get(
        dataset_url,
        stream=True,
        headers=Config.DATA_DOWNLOAD_REQUEST_HEADERS,
    )
    total = int(resp.headers.get('content-length', 0))

    with open(dataset_output_path, 'wb') as file, tqdm(
        desc=dataset_output_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

    logger.info(f'Finished downloading the {dataset_name} dataset.')


def extract_dataset_from_rar(
    logger: Logger,
    dataset_name: str,
    dataset_archive_path: str,
    dataset_output_path: str,
) -> None:
    """Extracts files from RAR archive.

    Args:
        logger (Logger): Logger used for documenting the extraction process.
        dataset_name (str): Name of the dataset (for documenting purposes).
        dataset_archive_path (str): Downloaded RAR archive file path.
        dataset_output_path (str): Extracted output path.
    """
    logger.info(f'Extracting the {dataset_name} dataset.')

    output_file = rarfile.RarFile(dataset_archive_path)
    output_file.extractall(path=dataset_output_path)
    os.remove(dataset_archive_path)

    if dataset_name == Config.HMDB_DATA_NAME:
        for file in os.listdir(dataset_output_path):
            rar_file = os.path.join(dataset_output_path, file)

            output_file = rarfile.RarFile(rar_file)
            output_file.extractall(path=dataset_output_path)
            os.remove(rar_file)

    logger.info(f'Finished extracting the {dataset_name} dataset.')


def download_data(logger: Logger) -> None:
    """Downloads all the datasets used in project.

    Args:
        logger (Logger): Logger used for documenting the download process.
    """
    for dataset_name, dataset_info in Config.INPUT_DATA_DOWNLOAD_PATH.items():
        dataset_path = dataset_info['data_folder_path']
        dataset_url = dataset_info['download_url']

        if not os.path.exists(dataset_path):
            if dataset_url is None:
                logger.info(
                    f'The following dataset ({dataset_name}) is private. '
                    f'For access, please reach out to {Config.AUTHOR_CONTACT} .'
                )

            else:
                output_rar_file = f'{dataset_path}.rar'

                download_dataset(
                    logger=logger,
                    dataset_name=dataset_name,
                    dataset_url=dataset_url,
                    dataset_output_path=output_rar_file,
                )

                extract_dataset_from_rar(
                    logger=logger,
                    dataset_name=dataset_name,
                    dataset_archive_path=output_rar_file,
                    dataset_output_path=dataset_path,
                )

    logger.info('All necessary datasets are downloaded.')


def multi_log(loggers: List[Logger], log_func: str, log_message: str) -> None:
    """Logs identical message to multiple loggers.

    Args:
        loggers (List[Logger]): List of loggers for saving the message.
        log_func (str): Logging function used for differentiating log levels.
        log_message (str): Message to log.
    """
    for logger in loggers:
        getattr(logger, log_func)(log_message)


def extract_frame_from_video(video_to_display: str) -> MatLike:
    """Extracts and returns first frame of given video.

    Args:
        video_to_display (str): Path to video file to be processed.

    Returns:
        MatLike: Loaded frame.
    """
    # Load video
    video_reader = cv2.VideoCapture(video_to_display)

    # Get the first frame
    _, bgr_frame = video_reader.read()

    # Release the object
    video_reader.release()

    # Change the color of the captured frame
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)

    return rgb_frame


def pickle_object(object: Any, save_path: str) -> None:
    """Pickles given `object` with the specified `save path`.

    Args:
        object (Any): Object to pickle.
        save_path (str): Output file path.
    """
    with open(save_path, 'wb') as output:
        pickle.dump(object, output, pickle.HIGHEST_PROTOCOL)


def load_pickle_object(pickle_path: str) -> Any:
    """Loads object from pickle file.

    Args:
        pickle_path (str): Path of the pickle file to load.

    Returns:
        Any: Loaded object.
    """
    with open(pickle_path, 'rb') as pickle_input:
        return pickle.load(pickle_input)


def create_model_performance_entry(
    model_type: str,
    classification_report: Dict[str, Any],
    model_output_path: str,
    architecture_and_evaluation_output_path: str,
) -> pd.DataFrame:
    """Creates model performance entry.

    Args:
        model_type (str): Type of model used for evaluation.
        classification_report (Dict[str, Any]): Classification report.
        model_output_path (str): Model output path.
        architecture_and_evaluation_output_path (str): Architecture and
            evaluation log file output path.

    Returns:
        pd.DataFrame: _description_
    """
    macro = 'macro avg'
    weighted = 'weighted avg'

    return pd.DataFrame(
        {
            'number_of_classes': [Config.SUBSET_SIZE],
            'image_height': [Config.IMAGE_HEIGHT],
            'image_width': [Config.IMAGE_WIDTH],
            'sequence_length': [Config.SEQUENCE_LENGTH],
            'random_state': [Config.RANDOM_STATE],
            'model_type': [model_type],
            'output_layer_activation_function': [
                Config.OUTPUT_LAYER_ACTIVATION_FUNCTION
            ],
            'loss_function': [Config.LOSS_FUNCTION],
            'adam_optimizer_learning_rate': [
                Config.ADAM_OPTIMIZER_LEARNING_RATE
            ],
            'training_shuffle': [Config.TRAINING_SHUFFLE],
            'max_training_epochs': [Config.TRAINING_EPOCHS],
            'batch_size': [Config.BATCH_SIZE],
            'early_stopping': [Config.USE_EARLY_STOPPING],
            'early_stopping_monitor': [Config.EARLY_STOPPING_MONITOR],
            'early_stopping_mode': [Config.EARLY_STOPPING_MODE],
            'early_stopping_patience': [Config.EARLY_STOPPING_PATIENCE],
            'accuracy': [classification_report['accuracy']],
            'macro_precision': [classification_report[macro]['precision']],
            'macro_recall': [classification_report[macro]['recall']],
            'macro_f1': [classification_report[macro]['f1-score']],
            'weighted_precision': [
                classification_report[weighted]['precision']
            ],
            'weighted_recall': [classification_report[weighted]['recall']],
            'weighted_f1': [classification_report[weighted]['f1-score']],
            'model_output_path': [model_output_path],
            'architecture_and_evaluation_output_path': [
                architecture_and_evaluation_output_path
            ],
        }
    )


def save_model_performance(
    dataset_name: str,
    model_type: str,
    classification_report: Dict[str, Any],
    model_output_path: str,
    architecture_and_evaluation_output_path: str,
) -> None:
    """Saves model performance for future comparison.

    Args:
        dataset_name (str): Name of the dataset.
        model_type (str): Type of model used for evaluation.
        classification_report (Dict[str, Any]): Classification report.
        model_output_path (str): Model output path.
        architecture_and_evaluation_output_path (str): Architecture and
            evaluation log file output path.
    """
    model_performance_history_path = os.path.join(
        Config.MODEL_STATISTICS_OUTPUT_FOLDER_PATH,
        f'{dataset_name}_model_performance_history.csv',
    )

    if not os.path.exists(model_performance_history_path):
        pd.DataFrame(
            {
                'number_of_classes': [],
                'image_height': [],
                'image_width': [],
                'sequence_length': [],
                'random_state': [],
                'model_type': [],
                'output_layer_activation_function': [],
                'loss_function': [],
                'adam_optimizer_learning_rate': [],
                'training_shuffle': [],
                'max_training_epochs': [],
                'batch_size': [],
                'early_stopping': [],
                'early_stopping_monitor': [],
                'early_stopping_mode': [],
                'early_stopping_patience': [],
                'accuracy': [],
                'macro_precision': [],
                'macro_recall': [],
                'macro_f1': [],
                'weighted_precision': [],
                'weighted_recall': [],
                'weighted_f1': [],
                'model_output_path': [],
                'architecture_and_evaluation_output_path': [],
            }
        ).to_csv(model_performance_history_path, index=False)

    statistics = pd.read_csv(model_performance_history_path)

    average_values = create_model_performance_entry(
        model_type=model_type,
        classification_report=classification_report,
        model_output_path=model_output_path,
        architecture_and_evaluation_output_path=architecture_and_evaluation_output_path,
    )
    statistics = pd.concat(
        [statistics if not statistics.empty else None, average_values],
        ignore_index=True,
    )

    statistics.sort_values(
        ['weighted_f1', 'accuracy'], ascending=[False, False], inplace=True
    )

    statistics.to_csv(model_performance_history_path, index=False)
