import logging
import os
import pickle
from logging import Logger
from typing import Any, List, Optional, Union

import cv2
import plotly.graph_objects as go
import requests
from cv2.typing import MatLike
from matplotlib.figure import Figure
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


def save_plotly_figure(
    figure: go.Figure,
    parent_file_path: str,
    file_name: str,
) -> None:
    """Saves Plotly graph object to file.

    Args:
        figure (go.Figure): Plotly graph object to write to file.
        parent_file_path (str): Path of the parent output directory.
        file_name (str): Output file name.
    """
    html_path = os.path.join(parent_file_path, 'html', f'{file_name}.html')
    image_path = os.path.join(parent_file_path, 'image', f'{file_name}.png')

    for path in [html_path, image_path]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    figure.write_html(html_path)
    figure.write_image(image_path)


def save_figure(
    figure: Union[Figure, go.Figure],
    plot_output_folder_path: str,
    dataset_name: str,
    plot_type: str,
    file_name: str,
) -> None:
    """Saves figure to output file.

    Args:
        figure (Figure): Figure to save.
        plot_output_folder_path (str): Generic plot output folder path.
        dataset_name (str): Name of the dataset.
        plot_type (str): Type of the plot.
        file_name (str): Name of the output file.
    """
    output_file_path = os.path.join(
        plot_output_folder_path, dataset_name, plot_type, file_name
    )

    parent_directory = create_missing_parent_directories(
        file_paths=[output_file_path],
    )[0]

    if isinstance(figure, go.Figure):
        save_plotly_figure(
            figure=figure,
            parent_file_path=parent_directory,
            file_name=file_name,
        )
    else:
        figure.savefig(output_file_path)


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
