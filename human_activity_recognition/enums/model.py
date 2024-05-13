from enum import Enum


class Model(Enum):

    """Represents the Enumeration of supported model types."""

    CONVLSTM = 'convlstm'
    GOOGLENET = 'googlenet'
    RESNET = 'resnet'
    VIDEOMAE = 'videomae'
    TIMESFORMER = 'timesformer'

    def __str__(self):
        return self.value
