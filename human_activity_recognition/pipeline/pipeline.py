from datetime import datetime

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.data_processing import (
    DataAnalyzer,
    DataProcessor,
)
from human_activity_recognition.modeling import (
    HumanActivityRecognitionModelConvLSTM,
)
from human_activity_recognition.utils import (
    download_data,
    setup_folders,
    setup_logging,
)


class Pipeline:

    """The Pipeline class represents the whole Machine Learning project."""

    def __init__(self) -> None:
        """Initializes the Pipeline class."""
        self.__timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

    def __setup_logging(self) -> None:
        """Sets up different types of loggers used during the project."""
        self.__run_logger = setup_logging(
            logger_type='run',
            log_path=Config.RUN_LOGS_OUTPUT_FOLDER_PATH,
            log_file_timestamp=self.__timestamp,
            log_to_stdout=True,
        )

        self.__eda_logger = setup_logging(
            logger_type='eda',
            log_path=Config.EDA_LOGS_OUTPUT_FOLDER_PATH,
            log_file_timestamp=self.__timestamp,
        )

        self.__evaluation_logger = setup_logging(
            logger_type='evaluation',
            log_path=Config.EVALUATION_LOGS_OUTPUT_FOLDER_PATH,
            log_file_timestamp=self.__timestamp,
        )

    def __setup_project(self) -> None:
        """Sets up different aspects of the project."""
        self.__setup_logging()
        setup_folders(logger=self.__run_logger)
        download_data(logger=self.__run_logger)

    def __run_data_analysis(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Executes data analysis step.

        Args:
            data_folder_path (str): Data folder path to process.
            dataset_name (str): Name of the dataset.
        """
        self.__run_logger.info('Starting Exploratory Data Analysis (EDA)')

        data_analyzer = DataAnalyzer(
            timestamp=self.__timestamp,
            loggers=[self.__run_logger, self.__eda_logger],
            plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
        )

        self.__data_classes = data_analyzer.run_exploratory_data_analysis(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

        self.__run_logger.info('Finished Exploratory Data Analysis (EDA)')

    def __run_data_processing(
        self,
        data_folder_path: str,
        dataset_name: str,
    ) -> None:
        """Executes data analysis and data processing step for each dataset.

        Args:
            data_folder_path (str): Data folder path to process.
            dataset_name (str): Name of the dataset.
        """

        self.__run_data_analysis(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

        data_processor = DataProcessor(
            logger=self.__run_logger, data_classes=self.__data_classes
        )

        (
            self.__features_train,
            self.__labels_train,
            self.__features_valid,
            self.__labels_valid,
            self.__features_test,
            self.__labels_test,
        ) = data_processor.run_data_processing(
            data_folder_path=data_folder_path,
            dataset_name=dataset_name,
        )

    def run(self) -> None:
        """Executes the project."""
        self.__setup_project()

        for (data_folder_path, dataset_name,) in list(
            zip(
                [
                    # Config.HMDB_DATA_FOLDER_PATH,
                    # Config.UCF50_DATA_FOLDER_PATH,
                    Config.OWN_DATA_FOLDER_PATH,
                ],
                [
                    # Config.HMDB_DATA_NAME,
                    # Config.UCF50_DATA_NAME,
                    Config.OWN_DATA_NAME,
                ],
            )
        ):
            self.__run_data_processing(
                data_folder_path=data_folder_path,
                dataset_name=dataset_name,
            )

            model = HumanActivityRecognitionModelConvLSTM(
                dataset_name=dataset_name,
                train_set_features=self.__features_train,
                train_set_label=self.__labels_train,
                validation_set_features=self.__features_valid,
                validation_set_label=self.__labels_valid,
                test_set_features=self.__features_test,
                test_set_label=self.__labels_test,
                timestamp=self.__timestamp,
                loggers=[self.__run_logger, self.__evaluation_logger],
                model_output_folder_path=Config.MODELS_OUTPUT_PATH,
                plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
            )

            model.run_modeling()
