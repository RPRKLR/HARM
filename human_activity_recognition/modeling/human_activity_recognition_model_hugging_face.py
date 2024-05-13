# SOURCE: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/video_classification.ipynb#scrollTo=hq9DR-qwkfwv

import os
from logging import Logger
from typing import Any, Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from pytorchvideo.data import LabeledVideoDataset
from transformers import (
    AutoImageProcessor,
    AutoModelForVideoClassification,
    Trainer,
    TrainingArguments,
)

from human_activity_recognition.configs.config import ProjectConfig as Config
from human_activity_recognition.utils.plot_utils import (
    generate_confusion_matrix,
    generate_training_history_plots,
)
from human_activity_recognition.utils.utils import (
    create_missing_parent_directories,
    multi_log,
    save_model_performance,
)


class HumanActivityRecognitionModelHuggingFace:

    """The HumanActivityRecognitionModelHuggingFace represents the classifier
    model which utilizes the Hugging Face video classification models.
    """

    def __init__(
        self,
        dataset_name: str,
        image_processor: AutoImageProcessor,
        train_dataset: LabeledVideoDataset,
        validation_dataset: LabeledVideoDataset,
        test_dataset: LabeledVideoDataset,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        timestamp: str,
        loggers: List[Logger],
        plots_output_folder_path: str,
        model_output_folder_path: str,
    ) -> None:
        """Initializes the HumanActivityRecognitionModelHuggingFace class.

        Args:
            dataset_name (str): Name of the dataset.
            image_processor (AutoImageProcessor): Image processor used during
                modeling.
            train_dataset (LabeledVideoDataset): Train dataset generator.
            validation_dataset (LabeledVideoDataset): Validation dataset
                generator.
            test_dataset (LabeledVideoDataset): Test dataset generator.
            label2id (Dict[str, int]): Label to ID pairing.
            id2label (Dict[int, str]): ID to Label pairing.
            timestamp (str): Current run timestamp.
            loggers (List[Logger]): Run and evaluation logger used for
                documenting processes.
            plots_output_folder_path (str): Plots output folder path.
            model_output_folder_path (str): Model output folder path.
        """
        self.__dataset_name = dataset_name
        self.__timestamp = timestamp
        self.__loggers = loggers
        self.__run_logger = loggers[0]
        self.__evaluation_logger_output_path = (
            loggers[1].handlers[0].baseFilename
        )

        self.__model_tag = Config.MODEL_IDENTIFIER.split('/')[-1]
        self.__model_name = (
            f'human_activity_recognition_model_{self.__model_tag}_'
            f'{Config.SUBSET_SIZE}_classes_lr_'
            f"{str(Config.ADAM_OPTIMIZER_LEARNING_RATE).replace('.', '_')}"
            f'__epochs_{Config.TRAINING_EPOCHS}__batch_size_{Config.BATCH_SIZE}'
            f'_{self.__timestamp}'
        )
        self.__model_output_path: str = os.path.join(
            model_output_folder_path,
            self.__dataset_name,
            self.__model_name,
        )
        self.__model_type = f'HARM - {self.__model_tag}'

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

        # region Datasets

        self.__train_dataset = train_dataset
        self.__validation_dataset = validation_dataset
        self.__test_dataset = test_dataset

        # endregion

        self.__image_processor = image_processor
        self.__label2id = label2id
        self.__id2label = id2label
        self.__model = self.__build_model()

        self.__precision_metric = evaluate.load('precision')
        self.__recall_metric = evaluate.load('recall')
        self.__f1_metric = evaluate.load('f1')
        self.__accuracy_metric = evaluate.load('accuracy')
        self.__confusion_matrix_metric = evaluate.load('confusion_matrix')

    def __build_model(self):
        """Builds Human Activity Recognition Model - Hugging Face variant.

        Returns:
            Model: Built model.
        """
        self.__run_logger.info(
            'Building Human Activity Recognition Model - '
            f'{self.__model_tag.capitalize()} variant.'
        )

        model = AutoModelForVideoClassification.from_pretrained(
            Config.MODEL_IDENTIFIER,
            label2id=self.__label2id,
            id2label=self.__id2label,
            ignore_mismatched_sizes=True,
        )

        self.__run_logger.info(
            'Finished building Human Activity Recognition Model - '
            f'{self.__model_tag.capitalize()} variant.'
        )

        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=f'Model architecture: {model}',
        )

        return model

    def __compute_metrics(
        self, eval_pred: Tuple[np.array, np.array]
    ) -> Dict[str, Any]:
        """Computes selected metrics on a batch of predictions.

        Args:
            eval_pred (Tuple[np.array, np.array]): Prediction logits and label
                IDs to evaluate.

        Returns:
            Dict[str, Any]: Evaluated metrics.
        """
        predictions = np.argmax(eval_pred.predictions, axis=1)

        return {
            'precision': str(
                self.__precision_metric.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                    average=None,
                )['precision'].tolist()
            ),
            'macro_precision': self.__precision_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='macro',
            )['precision'],
            'micro_precision': self.__precision_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='micro',
            )['precision'],
            'weighted_precision': self.__precision_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='weighted',
            )['precision'],
            'recall': str(
                self.__recall_metric.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                    average=None,
                )['recall'].tolist()
            ),
            'macro_recall': self.__recall_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='macro',
            )['recall'],
            'micro_recall': self.__recall_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='micro',
            )['recall'],
            'weighted_recall': self.__recall_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='weighted',
            )['recall'],
            'f1': str(
                self.__f1_metric.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                    average=None,
                )['f1'].tolist()
            ),
            'macro_f1': self.__f1_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='macro',
            )['f1'],
            'micro_f1': self.__f1_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='micro',
            )['f1'],
            'weighted_f1': self.__f1_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
                average='weighted',
            )['f1'],
            'accuracy': self.__accuracy_metric.compute(
                predictions=predictions,
                references=eval_pred.label_ids,
            )['accuracy'],
            'confusion_matrix': str(
                self.__confusion_matrix_metric.compute(
                    predictions=predictions,
                    references=eval_pred.label_ids,
                )['confusion_matrix'].tolist()
            ),
        }

    @staticmethod
    def __collate_fn(examples: Any) -> Dict[str, Any]:
        """The collation function to be used by `Trainer` to prepare data batches.

        Permutes data to (num_frames, num_channels, height, width).

        Args:
            examples (Any): Data to collate.

        Returns:
            Dict[str, Any]: Collated data.
        """
        pixel_values = torch.stack(
            [example['video'].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example['label'] for example in examples])
        return {'pixel_values': pixel_values, 'labels': labels}

    def __train(self) -> None:
        """Trains the Human Activity Recognition Model."""

        args = TrainingArguments(
            self.__model_output_path,
            remove_unused_columns=False,
            eval_strategy='epoch',
            save_strategy='epoch',
            learning_rate=Config.ADAM_OPTIMIZER_LEARNING_RATE,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            per_device_eval_batch_size=Config.BATCH_SIZE,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            max_steps=(self.__train_dataset.num_videos // Config.BATCH_SIZE)
            * Config.TRAINING_EPOCHS,
        )

        self.__trainer = Trainer(
            self.__model,
            args,
            train_dataset=self.__train_dataset,
            eval_dataset=self.__validation_dataset,
            tokenizer=self.__image_processor,
            compute_metrics=self.__compute_metrics,
            data_collator=self.__collate_fn,
        )

        self.__trainer.train()

        generate_training_history_plots(
            history=self.__aggregate_training_history(),
            plot_output_folder_path=Config.PLOTS_OUTPUT_FOLDER_PATH,
            dataset_name=self.__dataset_name,
            timestamp=self.__timestamp,
            model_tag=self.__model_tag,
            column_pairs=[['loss', 'eval_loss']],
        )

    def __aggregate_training_history(self) -> pd.DataFrame:
        """Aggregates Hugging Face training history.

        Returns:
            pd.DataFrame: Aggregated training history data.
        """

        def get_last_valid(series: pd.Series) -> Any:
            """Fetches last not-null value in given series.

            Args:
                series (pd.Series): Series to process.

            Returns:
                Any: Last not-null value in series.
            """
            return series.dropna().iloc[-1]

        history = pd.DataFrame(self.__trainer.state.log_history)
        history = history[['epoch', 'loss', 'eval_loss']]
        history['epoch'] = history['epoch'].round()
        history = history.groupby('epoch').agg(
            {col: get_last_valid for col in ['loss', 'eval_loss']}
        )

        return history

    def __evaluate(self) -> None:
        """Evaluates Human Activity Recognition Model performance."""
        labels = list(self.__label2id.keys())

        test_results = self.__trainer.evaluate(self.__test_dataset)
        self.__trainer.log_metrics('test', test_results)
        self.__trainer.save_metrics('test', test_results)

        # Display Confusion Matrix
        conf_matrix = eval(test_results['eval_confusion_matrix'])

        multi_log(
            loggers=self.__loggers,
            log_func='info',
            log_message=f'Confusion Matrix:\n{conf_matrix}',
        )

        generate_confusion_matrix(
            confusion_matrix=conf_matrix,
            labels=labels,
            output_path=self.__confusion_matrix_output_path,
        )

        macro = 'macro avg'
        weighted = 'weighted avg'

        classification_report = {
            'accuracy': test_results['eval_accuracy'],
            macro: {
                'precision': test_results['eval_macro_precision'],
                'recall': test_results['eval_macro_recall'],
                'f1-score': test_results['eval_macro_f1'],
            },
            weighted: {
                'precision': test_results['eval_weighted_precision'],
                'recall': test_results['eval_weighted_recall'],
                'f1-score': test_results['eval_weighted_f1'],
            },
        }

        save_model_performance(
            dataset_name=self.__dataset_name,
            model_type=self.__model_type,
            classification_report=classification_report,
            model_output_path=self.__model_output_path,
            architecture_and_evaluation_output_path=self.__evaluation_logger_output_path,
        )

    def __save_model(self) -> None:
        """Saves model components."""
        self.__trainer.save_model()
        self.__trainer.save_state()

    def run_modeling(self) -> None:
        """Executes modeling steps."""
        self.__train()
        self.__evaluate()
        self.__save_model()
