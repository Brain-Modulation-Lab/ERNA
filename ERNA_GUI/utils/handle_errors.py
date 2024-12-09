



# Custom Exceptions PostProcessError
class PostProcessError(Exception):
    """Base exception class for PostProcess errors."""
    pass



class AnnotationFileNotFoundError(PostProcessError):
    """Raised when an annotation file is not found."""
    pass

class DataFileNotFoundError(PostProcessError):
    """Raised when a data file is not found."""
    pass


class ImplementationError(PostProcessError):
    """Raised when a data file is not found."""
    pass

class UnidenticalEntryError(PostProcessError):
    """Raised when ftype is wrong."""
    pass


class MissingFileExtensionError(PostProcessError):
    """Raised when ftype cannot be found."""
    pass


class UnavailableProcessingError(PostProcessError):
    """Class for raising exceptions/error associated with trying to perform
    processing steps that cannot be done."""



