import os
from typing import Any, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from matplotlib.figure import Figure

from human_activity_recognition.utils.utils import (
    create_missing_parent_directories,
)


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
    plot_output_folder_path: str = '',
    dataset_name: str = '',
    plot_type: str = '',
    file_name: str = '',
    output_path: Optional[str] = None,
) -> None:
    """Saves figure to output file.

    Args:
        figure (Figure): Figure to save.
        plot_output_folder_path (str): Generic plot output folder path. Defaults
            to ''.
        dataset_name (str): Name of the dataset. Defaults to ''.
        plot_type (str): Type of the plot. Defaults to ''.
        file_name (str): Name of the output file. Defaults to ''.
        output_path (Optional[str]): Output file path. If None a value is
            generated based on previous path variables. Defaults to None.
    """
    output_file_path = (
        output_path
        if output_path is not None
        else os.path.join(
            plot_output_folder_path, dataset_name, plot_type, file_name
        )
    )

    parent_directory = create_missing_parent_directories(
        file_paths=[output_file_path],
    )[0]

    if isinstance(figure, go.Figure):
        file_name = (
            file_name if output_path is None else output_path.split(os.sep)[-1]
        )

        save_plotly_figure(
            figure=figure,
            parent_file_path=parent_directory,
            file_name=file_name,
        )
    else:
        figure.savefig(output_file_path)


def generate_confusion_matrix(
    confusion_matrix: np.ndarray,
    labels: List[Any],
    output_path: str,
) -> None:
    """Generates and saves the passed in Confusion Matrix.

    Args:
        confusion_matrix (numpy.ndarray): Confusion Matrix to display.
        labels (List[Any]): Labels for Columns and Indexes.
        output_path (str): Output save path.
    """
    # SOURCE:
    # https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    # https://stackoverflow.com/questions/38913965/make-the-size-of-a-heatmap-bigger-with-seaborn

    # Convert Confusion Matrix to DataFrame
    confusion_matrix = pd.DataFrame(
        data=confusion_matrix, index=labels, columns=labels
    )

    # Create new Figure
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot the Confusion Matrix
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='g')

    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')

    save_figure(figure=fig, output_path=output_path)


def draw_data_plot(data: pd.DataFrame, columns: List[str], path: str) -> None:
    """Draws line data plot.

    Args:
        data (pd.DataFrame): Data to visualize.
        columns (List[str]): Columns used for visualization.
        path (str): Output file path.
    """
    # Create Plot
    fig = go.Figure()

    # Draw Plot
    for column in columns:
        fig.add_trace(
            go.Scatter(
                x=data['num_of_epochs'],
                y=data[column],
                mode='lines',
                name=column,
            )
        )

    fig.update_layout(
        title=f"Training history {'/'.join(columns)}",
        title_x=0.5,
        legend_traceorder='normal',
        xaxis_title='Number of Epochs',
    )

    save_figure(figure=fig, output_path=path)


def generate_training_history_plots(
    history: pd.DataFrame,
    plot_output_folder_path: str,
    dataset_name: str,
    timestamp: str,
    model_tag: str,
) -> None:
    """Generates Training History Plots.

    Args:
        history (pd.DataFrame): Model training history data.
        plot_output_folder_path (str): Plot output folder path.
        dataset_name (str): Name of the dataset.
        timestamp (str): Current run timestamp.
        model_tag (str): Type of model used during training.
    """
    history['num_of_epochs'] = history.index

    draw_data_plot(
        data=history,
        columns=['loss', 'val_loss'],
        path=os.path.join(
            plot_output_folder_path,
            dataset_name,
            'line_charts',
            f'{model_tag}_training_history_loss_{timestamp}',
        ),
    )

    draw_data_plot(
        data=history,
        columns=['acc', 'val_acc'],
        path=os.path.join(
            plot_output_folder_path,
            dataset_name,
            'line_charts',
            f'{model_tag}_training_history_accuracy_{timestamp}',
        ),
    )
