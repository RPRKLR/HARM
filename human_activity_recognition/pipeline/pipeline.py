from datetime import datetime

from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.data_processing import (
    DataAnalyzer,
    ImageSequenceDataProcessor,
    SingleImageDataProcessor,
    VideoDataProcessor,
)
from human_activity_recognition.enums import Model
from human_activity_recognition.modeling import (
    HumanActivityRecognitionModelConvLSTM,
    HumanActivityRecognitionModelGoogLeNet,
    HumanActivityRecognitionModelHuggingFace,
    HumanActivityRecognitionModelResNet,
)
from human_activity_recognition.utils import (
    download_data,
    setup_folders,
    setup_logging,
)


class Pipeline:

    """The Pipeline class represents the whole Machine Learning project."""

    def __init__(self, model_type: Model) -> None:
        """Initializes the Pipeline class.

        Args:
            model_type (Model): Type of model used during execution.
        """
        self.__timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        self.__data_folder_path, self.__dataset_name = Config.DATA
        self.__model_type = model_type

        Config.MODEL_IDENTIFIER = Config.SUPPORTED_HUGGING_FACE_MODELS.get(
            self.__model_type, None
        )

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

        match (self.__model_type):
            case Model.CONVLSTM | Model.RESNET:
                data_processor = ImageSequenceDataProcessor(
                    logger=self.__run_logger, data_classes=self.__data_classes
                )

                (
                    self.__features_train,
                    self.__labels_train,
                    self.__features_valid,
                    self.__labels_valid,
                    self.__features_test,
                    self.__labels_test,
                    self.__id_class_pairing,
                ) = data_processor.run_data_processing(
                    data_folder_path=data_folder_path,
                    dataset_name=dataset_name,
                )

            case Model.GOOGLENET:
                data_processor = SingleImageDataProcessor(
                    logger=self.__run_logger, data_classes=self.__data_classes
                )

                (
                    self.__features_train,
                    self.__labels_train,
                    self.__features_valid,
                    self.__labels_valid,
                    self.__features_test,
                    self.__labels_test,
                    self.__label_classes,
                ) = data_processor.run_data_processing(
                    data_folder_path=data_folder_path,
                    dataset_name=dataset_name,
                )

            case Model.TIMESFORMER | Model.VIDEOMAE:
                data_processor = VideoDataProcessor(
                    logger=self.__run_logger, data_classes=self.__data_classes
                )

                (
                    self.__image_processor,
                    self.__train_dataset,
                    self.__validation_dataset,
                    self.__test_dataset,
                    self.__id2label,
                    self.__label2id,
                ) = data_processor.run_data_processing(
                    data_folder_path=data_folder_path,
                    dataset_name=dataset_name,
                )

    def __run_modeling(self) -> None:
        """Executes modeling step."""
        match (self.__model_type):

            case Model.CONVLSTM:

                model = HumanActivityRecognitionModelConvLSTM(
                    dataset_name=self.__dataset_name,
                    train_set_features=self.__features_train,
                    train_set_label=self.__labels_train,
                    validation_set_features=self.__features_valid,
                    validation_set_label=self.__labels_valid,
                    test_set_features=self.__features_test,
                    test_set_label=self.__labels_test,
                    id_class_pairing=self.__id_class_pairing,
                    timestamp=self.__timestamp,
                    loggers=[self.__run_logger, self.__evaluation_logger],
                    model_output_folder_path=Config.MODELS_OUTPUT_PATH,
                    plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
                )

            case Model.RESNET:

                model = HumanActivityRecognitionModelResNet(
                    dataset_name=self.__dataset_name,
                    train_set_features=self.__features_train,
                    train_set_label=self.__labels_train,
                    validation_set_features=self.__features_valid,
                    validation_set_label=self.__labels_valid,
                    test_set_features=self.__features_test,
                    test_set_label=self.__labels_test,
                    id_class_pairing=self.__id_class_pairing,
                    timestamp=self.__timestamp,
                    loggers=[self.__run_logger, self.__evaluation_logger],
                    model_output_folder_path=Config.MODELS_OUTPUT_PATH,
                    plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
                )

            case Model.GOOGLENET:

                model = HumanActivityRecognitionModelGoogLeNet(
                    dataset_name=self.__dataset_name,
                    train_set_features=self.__features_train,
                    train_set_label=self.__labels_train,
                    validation_set_features=self.__features_valid,
                    validation_set_label=self.__labels_valid,
                    test_set_features=self.__features_test,
                    test_set_label=self.__labels_test,
                    label_classes=self.__label_classes,
                    timestamp=self.__timestamp,
                    loggers=[self.__run_logger, self.__evaluation_logger],
                    model_output_folder_path=Config.MODELS_OUTPUT_PATH,
                    plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
                )

            case Model.TIMESFORMER | Model.VIDEOMAE:

                model = HumanActivityRecognitionModelHuggingFace(
                    dataset_name=self.__dataset_name,
                    image_processor=self.__image_processor,
                    train_dataset=self.__train_dataset,
                    validation_dataset=self.__validation_dataset,
                    test_dataset=self.__test_dataset,
                    label2id=self.__label2id,
                    id2label=self.__id2label,
                    timestamp=self.__timestamp,
                    loggers=[self.__run_logger, self.__evaluation_logger],
                    model_output_folder_path=Config.MODELS_OUTPUT_PATH,
                    plots_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
                )

        model.run_modeling()

    def run(self) -> None:
        """Executes the project."""
        self.__setup_project()

        self.__run_data_processing(
            data_folder_path=self.__data_folder_path,
            dataset_name=self.__dataset_name,
        )

        self.__run_modeling()
