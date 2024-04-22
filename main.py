from human_activity_recognition import Pipeline
from human_activity_recognition.configs import ProjectConfig as Config
from human_activity_recognition.utils.model_utils import (
    run_prediction,
    simple_grid_search,
)

if __name__ == '__main__':
    # Run training pipeline
    pipeline = Pipeline()
    pipeline.run()

    # Run grid search
    # simple_grid_search(offset=287)

    # Run prediction
    # run_prediction(
    #     input_video_paths=['/home/peter/dp/Human_Activity_Video_Recognition/data/input/DVORAK_CUSTOM/lunges/20231209_120733.mp4'],
    #     model_path='/home/peter/dp/Human_Activity_Video_Recognition/data/output/models/DVORAK_CUSTOM/DVORAK_CUSTOM_TOP_1_human_activity_recognition_model_convlstm_4_classes_adam_0_005__epochs_200__batch_size_128_2024-04-09_07:43:57.keras',
    #     single_class_prediction=True,
    #     number_of_top_classes=3,
    # )
# /home/peter/dp/Human_Activity_Video_Recognition/data/output/models/DVORAK_CUSTOM/human_activity_recognition_model_4_classes_adam_0_005__epochs_200__batch_size_256__early_stopping_monitor_val_loss_mode_min_patience_15_2024-04-15_18:18:37.keras