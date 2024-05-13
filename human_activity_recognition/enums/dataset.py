from enum import Enum


class Dataset(Enum):

    """Represents the Enumeration of supported datasets."""

    DVORAK_CUSTOM = 'dvorak_custom'
    HMDB = 'hmdb'
    UCF50 = 'ucf50'

    def __str__(self):
        return self.value
