import logging
import os
from dataclasses import dataclass

from human_activity_recognition.enums import Dataset, Model


@dataclass
class ProjectConfig:
    """Class used for storing Project related configurations."""

    # region Logging

    LOG_FORMATTER = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    # endregion

    # region Path variables

    DATA_FOLDER_PATH = os.path.join(os.getcwd(), 'data')
    INPUT_FOLDER_PATH = os.path.join(DATA_FOLDER_PATH, 'input')
    OUTPUT_FOLDER = os.path.join(DATA_FOLDER_PATH, 'output')
    PROCESSED_FOLDER = os.path.join(DATA_FOLDER_PATH, 'processed')
    LOGS_OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'logs')
    EDA_LOGS_OUTPUT_FOLDER_PATH = os.path.join(LOGS_OUTPUT_FOLDER_PATH, 'eda')
    RUN_LOGS_OUTPUT_FOLDER_PATH = os.path.join(LOGS_OUTPUT_FOLDER_PATH, 'run')
    EVALUATION_LOGS_OUTPUT_FOLDER_PATH = os.path.join(
        LOGS_OUTPUT_FOLDER_PATH, 'model_evaluation'
    )
    MODELS_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, 'models')
    PLOTS_OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'plots')
    STATISTICS_OUTPUT_FOLDER_PATH = os.path.join(OUTPUT_FOLDER, 'statistics')
    DATA_STATISTICS_OUTPUT_FOLDER_PATH = os.path.join(
        STATISTICS_OUTPUT_FOLDER_PATH, 'data'
    )
    MODEL_STATISTICS_OUTPUT_FOLDER_PATH = os.path.join(
        STATISTICS_OUTPUT_FOLDER_PATH, 'model'
    )

    PATHS_TO_CREATE = [
        INPUT_FOLDER_PATH,
        PROCESSED_FOLDER,
        LOGS_OUTPUT_FOLDER_PATH,
        EDA_LOGS_OUTPUT_FOLDER_PATH,
        EVALUATION_LOGS_OUTPUT_FOLDER_PATH,
        MODELS_OUTPUT_PATH,
        PLOTS_OUTPUT_FOLDER_PATH,
        DATA_STATISTICS_OUTPUT_FOLDER_PATH,
        MODEL_STATISTICS_OUTPUT_FOLDER_PATH,
    ]

    STATISTICS_OUTPUT_FILE_PATH = os.path.join(
        DATA_STATISTICS_OUTPUT_FOLDER_PATH,
        'human_activity_recognition_data_stats',
    )

    # endregion

    # region Data

    OWN_DATA_NAME = 'DVORAK_CUSTOM'
    HMDB_DATA_NAME = 'HMDB'
    UCF50_DATA_NAME = 'UCF50'

    OWN_DATA_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, OWN_DATA_NAME)
    HMDB_DATA_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, HMDB_DATA_NAME)
    UCF50_DATA_FOLDER_PATH = os.path.join(INPUT_FOLDER_PATH, UCF50_DATA_NAME)

    INPUT_DATA_DOWNLOAD_PATH = {
        OWN_DATA_NAME: {
            'data_folder_path': OWN_DATA_FOLDER_PATH,
            'download_url': None,
        },
        HMDB_DATA_NAME: {
            'data_folder_path': HMDB_DATA_FOLDER_PATH,
            'download_url': 'https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar',
        },
        # UCF50_DATA_NAME: {
        #     'data_folder_path': UCF50_DATA_FOLDER_PATH,
        #     'download_url': 'https://crcv.ucf.edu/data/UCF50.rar',
        # },
    }

    DATA_DOWNLOAD_REQUEST_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:74.0) Gecko/20100101 Firefox/74.0'
    }

    DATA_OPTIONS = {
        Dataset.DVORAK_CUSTOM: (OWN_DATA_FOLDER_PATH, OWN_DATA_NAME),
        Dataset.HMDB: (HMDB_DATA_FOLDER_PATH, HMDB_DATA_NAME),
        Dataset.UCF50: (UCF50_DATA_FOLDER_PATH, UCF50_DATA_NAME),
    }

    DATA = DATA_OPTIONS[Dataset.DVORAK_CUSTOM]

    # endregion

    # region Data Processing

    # 4 - DVORAK_CUSTOM; 20 - HMDB, UCF50;
    SUBSET_SIZE = 4
    # 64 - ConvLSTM; 128 - ResNet; 224 - Hugging Face models, GoogLeNet;
    IMAGE_HEIGHT = 224
    # 64 - ConvLSTM; 128 - ResNet; 224 - Hugging Face models, GoogLeNet;
    IMAGE_WIDTH = 224
    SEQUENCE_LENGTH = 20
    TEST_SPLIT_PERCENTAGE = 0.2
    VALIDATION_SPLIT_PERCENTAGE = 0.2
    RANDOM_STATE = 42

    # endregion

    # region Modeling

    SUPPORTED_HUGGING_FACE_MODELS = {
        Model.TIMESFORMER: 'facebook/timesformer-base-finetuned-k400',
        Model.VIDEOMAE: 'MCG-NJU/videomae-base',
    }

    # Model settings
    MODEL_IDENTIFIER = None  # Used with Hugging Face Models
    OUTPUT_LAYER_ACTIVATION_FUNCTION = 'softmax'
    LOSS_FUNCTION = 'categorical_crossentropy'
    # 5e-5 - Hugging Face models, GoogLeNet; 1e-4 - ConvLSTM, ResNet;
    ADAM_OPTIMIZER_LEARNING_RATE = 5e-5
    METRICS_TO_SHOW = ['acc']

    # Training settings
    TRAINING_SHUFFLE = True
    # 4 - Hugging Face models; 25 - GoogLeNet; 200 - ConvLSTM, ResNet;
    TRAINING_EPOCHS = 4
    # 2 - Hugging Face models, GoogLeNet; 4 - ConvLSTM, ResNet;
    BATCH_SIZE = 2

    # Early stopping settings
    USE_EARLY_STOPPING = True
    EARLY_STOPPING_MONITOR = 'val_loss'
    EARLY_STOPPING_MODE = 'min'
    EARLY_STOPPING_PATIENCE = 20
    EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
    EARLY_STOPPING_TAG = (
        f'__early_stopping_monitor_{EARLY_STOPPING_MONITOR}'
        f'_mode_{EARLY_STOPPING_MODE}_patience_{EARLY_STOPPING_PATIENCE}'
    )

    # endregion

    AUTHOR_CONTACT = 'todo@gmail.com'
