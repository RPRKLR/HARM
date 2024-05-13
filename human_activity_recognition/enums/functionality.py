from enum import Enum


class Functionality(Enum):

    """Represents the Enumeration of supported functionalities."""

    TRAIN = 'train'
    PREDICT = 'predict'
    GRID_SEARCH = 'grid-search'
    # CAMERA_FEED = 'camera-feed'

    def __str__(self):
        return self.value
