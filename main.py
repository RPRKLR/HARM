from human_activity_recognition.enums import Model
from human_activity_recognition.utils.model_utils import run_pipeline

if __name__ == '__main__':
    run_pipeline(
        # model_type=Model.CONVLSTM,
        # model_type=Model.GOOGLENET,
        # model_type=Model.RESNET,
        # model_type=Model.TIMESFORMER,
        model_type=Model.VIDEOMAE,
    )
