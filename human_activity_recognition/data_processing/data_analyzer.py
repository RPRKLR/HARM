import os
import random
from logging import Logger
from typing import Dict, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import plotly.express as px

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.utils import (
    extract_frame_from_video,
    multi_log,
    save_figure,
)


class DataAnalyzer:

    """The DataAnalyzer class is responsible for generating basic Exploratory
    Data Analysis (EDA) logs and plots, which give us insights to the data
    we are working with.
    """

    def __init__(
        self,
        timestamp: str,
        loggers: List[Logger],
        plots_output_folder_path: str,
    ) -> None:
        """Initializes the DataAnalyzer class.

        Args:
            timestamp (str): Current run timestamp.
            loggers (List[Logger]): Run and EDA logger used for documenting
                processes.
            plots_output_folder_path (str): EDA plots output folder path.
        """
        self.__timestamp = timestamp
        self.__loggers = loggers
        self.__statistics_output_path = (
            f'{Config.STATISTICS_OUTPUT_FILE_PATH}_{self.__timestamp}.csv'
        )
        self.__plots_output_folder_path = plots_output_folder_path

    # region Basic Statistics

    def __extract_classes(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Extracts available classes from the dataset folder.

        Args:
            data_folder_path (str): Dataset folder path to process.
            dataset_name (str): Name of the dataset (for documenting purposes).
        """
        self.__all_classes = os.listdir(data_folder_path)

        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=(
                f'The {dataset_name} dataset contains '
                f'{len(self.__all_classes)} classes.'
            ),
        )

    def __generate_average_attributes_for_dataset(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> pd.DataFrame:
        """Generates average values for all the video data in given dataset.

        Args:
            data_folder_path (str): Dataset folder path to process.
            dataset_name (str): Name of the dataset.

        Returns:
            pd.DataFrame: DataFrame of aggregated (averaged) data.
        """
        video_attributes: Dict[str, List[int]] = {
            'frames': [],
            'frame_width': [],
            'frame_height': [],
            'fps': [],
        }

        number_of_videos: Dict[str, List[Union[int, str]]] = {
            'class': [],
            'number_of_videos': [],
        }

        for data_class in self.__all_classes:
            data_class_folder_path = os.path.join(data_folder_path, data_class)
            number_of_videos['class'].append(data_class)
            number_of_videos['number_of_videos'].append(
                len(os.listdir(data_class_folder_path))
            )

            for video in os.listdir(data_class_folder_path):
                video_path = os.path.join(data_class_folder_path, video)

                video_capture = cv2.VideoCapture(video_path)

                for key, attribute in list(
                    zip(
                        ['frames', 'frame_width', 'frame_height', 'fps'],
                        [
                            cv2.CAP_PROP_FRAME_COUNT,
                            cv2.CAP_PROP_FRAME_WIDTH,
                            cv2.CAP_PROP_FRAME_HEIGHT,
                            cv2.CAP_PROP_FPS,
                        ],
                    )
                ):
                    video_attributes[key].append(
                        int(video_capture.get(attribute))
                    )

        self.__target_distribution = pd.DataFrame(number_of_videos)

        return pd.DataFrame(
            {
                'dataset_name': [dataset_name],
                'avg_number_of_videos_per_class': [
                    round(
                        numpy.average(number_of_videos['number_of_videos']),
                        2,
                    )
                ],
                'avg_number_of_frames': [
                    round(numpy.average(video_attributes['frames']), 2)
                ],
                'avg_frame_width': [
                    round(numpy.average(video_attributes['frame_width']), 2)
                ],
                'avg_frame_height': [
                    round(numpy.average(video_attributes['frame_height']), 2)
                ],
                'avg_fps': [round(numpy.average(video_attributes['fps']), 2)],
            }
        )

    def __display_statistics(
        self, dataset_name: str, dataset_statistics: pd.Series
    ) -> None:
        """Displays statistics about given dataset.

        Args:
            dataset_name (str): Name of the dataset (for documenting purposes).
            dataset_statistics (pd.Series): Available statistics for dataset.
        """
        for log_message in [
            (
                f'Data classes in the {dataset_name} dataset contain on average'
                f" {dataset_statistics['avg_number_of_videos_per_class']} "
                'videos.'
            ),
            (
                f'Videos in the {dataset_name} dataset contain on average '
                f"{dataset_statistics['avg_number_of_frames']} frame."
            ),
            (
                f'Frames in the {dataset_name} dataset on average have the '
                f"width of {dataset_statistics['avg_frame_width']} ."
            ),
            (
                f'Frames in the {dataset_name} dataset on average have the '
                f"height of {dataset_statistics['avg_frame_height']} ."
            ),
            (
                f'Videos in the {dataset_name} dataset contain on average have '
                f"{dataset_statistics['avg_fps']} FPS."
            ),
        ]:
            multi_log(
                loggers=self.__loggers, log_func='info', log_message=log_message
            )

    def __get_average_video_attributes(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Load existing or generate aggregated statistics for video datasets.

        Args:
            data_folder_path (str): Dataset folder path to process.
            dataset_name (str): Name of the dataset.
        """
        if not os.path.exists(self.__statistics_output_path):
            pd.DataFrame(
                {
                    'dataset_name': [],
                    'avg_number_of_videos_per_class': [],
                    'avg_number_of_frames': [],
                    'avg_frame_width': [],
                    'avg_frame_height': [],
                    'avg_fps': [],
                }
            ).to_csv(self.__statistics_output_path, index=False)

        statistics = pd.read_csv(self.__statistics_output_path)

        if dataset_name not in statistics['dataset_name'].values:
            average_values = self.__generate_average_attributes_for_dataset(
                data_folder_path=data_folder_path,
                dataset_name=dataset_name,
            )
            statistics = pd.concat(
                [statistics if not statistics.empty else None, average_values],
                ignore_index=True,
            )

            statistics.to_csv(self.__statistics_output_path, index=False)

        dataset_statistics = statistics[
            statistics['dataset_name'] == dataset_name
        ].squeeze(axis=0)

        self.__display_statistics(
            dataset_name=dataset_name, dataset_statistics=dataset_statistics
        )

    def __describe_dataset(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Extracts all the classes available in given dataset and generates
        basic statistics about them.

        Args:
            data_folder_path (str): Dataset folder path to process.
            dataset_name (str): Name of the dataset.
        """
        self.__extract_classes(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

        self.__get_average_video_attributes(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

    # endregion

    # region Plotting

    def __visualize_random_video_frames_with_labels(
        self,
        dataset_name: str,
        data_folder_path: str,
    ) -> None:
        """Extracts and plots the first frame of a random video from the random
        selected categories.

        Args:
            dataset_name (str): Name of the dataset (for documenting purposes).
            data_folder_path (str): Dataset folder path to process.
            number_of_data_classes_to_select (int): Number of data classes to
                select for processing.
        """

        if (Config.SUBSET_SIZE is None) or (
            Config.SUBSET_SIZE > len(self.__all_classes)
        ):
            Config.SUBSET_SIZE = len(self.__all_classes)

        number_of_data_classes_to_select = Config.SUBSET_SIZE

        title = (
            f'{number_of_data_classes_to_select} samples from the '
            f'{dataset_name} dataset'
        )
        fig = plt.figure(
            figsize=(
                (20, 20) if number_of_data_classes_to_select > 8 else (16, 9)
            )
        )
        fig.tight_layout()
        fig.suptitle(title, fontsize=16)

        self.__selected_classes = (
            self.__all_classes
            if number_of_data_classes_to_select == len(self.__all_classes)
            else random.sample(
                self.__all_classes, number_of_data_classes_to_select
            )
        )

        for counter, data_class in enumerate(
            random.sample(self.__all_classes, number_of_data_classes_to_select)
        ):
            data_class_path = os.path.join(data_folder_path, data_class)

            video_to_display = os.path.join(
                data_class_path, random.choice(os.listdir(data_class_path))
            )

            rgb_frame = extract_frame_from_video(
                video_to_display=video_to_display,
            )

            ax = fig.add_subplot(
                round(number_of_data_classes_to_select / 4), 4, counter + 1
            )
            ax.imshow(rgb_frame)
            ax.set_title(data_class)

        save_figure(
            figure=fig,
            plot_output_folder_path=self.__plots_output_folder_path,
            dataset_name=dataset_name,
            plot_type='random_sample_frames',
            file_name=f'random_sample_frames_{self.__timestamp}.png',
        )

    def __generate_target_distribution_bar_plot(
        self, dataset_name: str
    ) -> None:
        """Generates and saves target distribution bar-plot (histogram).

        Args:
            dataset_name (str): Name of the dataset.
        """
        fig = px.bar(
            self.__target_distribution,
            x='class',
            y='number_of_videos',
            title=(
                f'Number of videos in the categories of the {dataset_name} '
                'dataset'
            ),
            text_auto=True,
        ).update_xaxes(categoryorder='total descending')

        fig.update_layout(
            title_x=0.5,
            bargap=0.2,
            legend_traceorder='normal',
            xaxis_title='Video categories',
            yaxis_title='Number of videos in category',
        )

        save_figure(
            figure=fig,
            plot_output_folder_path=self.__plots_output_folder_path,
            dataset_name=dataset_name,
            plot_type='bar_plots',
            file_name=f'target_distribution_{self.__timestamp}',
        )

    def __generate_target_percentage_distribution_pie_chart(
        self, dataset_name: str
    ) -> None:
        """Generates and saves target percentage distribution pie-chart.

        Args:
            dataset_name (str): Name of the dataset.
        """
        fig = (
            px.pie(
                values=self.__target_distribution['number_of_videos'],
                names=self.__target_distribution['class'],
                title=(
                    f'Video category distribution of the {dataset_name} dataset'
                ),
            )
            .update_traces(textinfo='value+percent')
            .update_layout(
                title_x=0.5,
                bargap=0.2,
                legend_traceorder='normal',
                legend_title='Video categories',
            )
        )

        save_figure(
            figure=fig,
            plot_output_folder_path=self.__plots_output_folder_path,
            dataset_name=dataset_name,
            plot_type='pie_charts',
            file_name=f'target_distribution_{self.__timestamp}',
        )

    # endregion

    def run_exploratory_data_analysis(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> List[str]:
        """Executes the Exploratory Data Analysis (EDA) process of the project,
        which gives us useful insights to our data.

        Args:
            data_folder_path (str): Input dataset folder path.
            dataset_name (str): Name of the dataset.

        Returns:
            List[str]: List of class names selected for subset of data.
        """
        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=f'Running analysis on {dataset_name} dataset.',
        )

        self.__describe_dataset(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

        self.__visualize_random_video_frames_with_labels(
            dataset_name=dataset_name,
            data_folder_path=data_folder_path,
        )

        self.__generate_target_distribution_bar_plot(dataset_name=dataset_name)

        self.__generate_target_percentage_distribution_pie_chart(
            dataset_name=dataset_name,
        )

        multi_log(
            loggers=self.__loggers,
            log_func='debug',
            log_message=f'Finished analysis on {dataset_name} dataset.',
        )

        return self.__selected_classes
