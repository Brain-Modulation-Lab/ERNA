from ERNA_GUI.utils.handle_errors import (
    MissingFileExtensionError, UnidenticalEntryError, UnavailableProcessingError
)

from typing import Any, Union

import numpy as np
import json
from typing import Optional
import pickle
import collections.abc
import pandas as pd


def create_lambda(obj: Any) -> Any:
    """Creates a lambda from an object, useful for when the object has been
    created in a for loop."""
    return lambda: obj

def numpy_to_python(obj: Union[dict, list, np.generic]) -> Any:
    """Iterates through all entries of an object and converts any numpy elements
    into their base Python object types, e.g. float32 into float, ndarray into
    list, etc...

    PARAMETERS
    ----------
    obj : dict | list | numpy generic
    -   The object whose entries should be iterated through and, if numpy
        objects, converted to their equivalent base Python types.

    RETURNS
    -------
    Any
    -   The object whose entries, if numpy objects, have been converted to their
        equivalent base Python types.
    """
    if isinstance(obj, dict):
        return numpy_to_python_dict(obj)
    elif isinstance(obj, list):
        return numpy_to_python_list(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        raise TypeError(
            "Error when changing nested elements of an object:\nProcessing "
            f"objects of type '{type(obj)}' is not supported. Only 'list' "
            ", 'dict', and 'numpy generic' objects can be processed."
        )


def numpy_to_python_dict(obj: dict) -> dict:
    """Iterates through all entries of a dictionary and converts any numpy
    elements into their base Python object types, e.g. float32 into float,
    ndarray into list, etc...

    PARAMETERS
    ----------
    obj : dict
    -   The dictionary whose entries should be iterated through and, if numpy
        objects, converted to their equivalent base Python types.

    RETURNS
    -------
    new_obj : dict
    -   The dictionary whose entries, if numpy objects, have been converted to
        their equivalent base Python types.
    """
    new_obj = {}
    for key, value in obj.items():
        if isinstance(value, list):
            new_obj[key] = numpy_to_python_list(value)
        elif isinstance(value, dict):
            new_obj[key] = numpy_to_python_dict(value)
        elif type(value).__module__ == np.__name__:
            new_obj[key] = getattr(value, "tolist", create_lambda(value))()
        else:
            new_obj[key] = value

    return new_obj


def numpy_to_python_list(obj: list) -> list:
    """Iterates through all entries of a list and converts any numpy elements
    into their base Python object types, e.g. float32 into float, ndarray into
    list, etc...

    PARAMETERS
    ----------
    obj : list
    -   The list whose entries should be iterated through and, if numpy objects,
        converted to their equivalent base Python types.

    RETURNS
    -------
    new_obj : list
    -   The list whose entries, if numpy objects, have been converted to their
        equivalent base Python types.
    """
    new_obj = []
    for value in obj:
        if isinstance(value, list):
            new_obj.append(numpy_to_python_list(value))
        elif isinstance(value, dict):
            new_obj.append(numpy_to_python_dict(value))
        elif type(value).__module__ == np.__name__:
            new_obj.append(getattr(value, "tolist", create_lambda(value))())
        else:
            new_obj.append(value)

    return new_obj


def check_ftype_present(fpath: str) -> bool:
    """Checks whether a filetype is present in a filepath string based on the
    presence of a period ('.').

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename.

    RETURNS
    -------
    ftype_present : bool
    -   Whether or not a filetype is present.
    """

    if "." in fpath:
        ftype_present = True
    else:
        ftype_present = False

    return ftype_present


def check_file_inputs(fpath: str, ftype: Union[str, None]) -> None:
    """Checks filepath and filetype inputs.
    -   If a filepath is given, checks whether a filetype extension is present.
    -   If so, checks whether this matches the extension given in 'ftype' (if
        'ftype' is not 'None').
    -   If no extension is present in 'fpath', the extension given in 'ftype' is
        used, in which case this cannot be 'None'.

    PARAMETERS
    ----------
    fpath : str
    -   A filepath, with or without a filetype extension.

    ftype : str | None
    -   A filetype extension without the leading period.
    -   Can only be 'None' if a filetype is present in 'fpath'.

    RETURNS
    -------
    fpath : str
    -   The filepath, with the file extension specified in 'ftype' if no
        filetype was present in the provided 'fpath'.

    ftype : str
    -   The filetype extension, derived from 'fpath'.

    RAISES
    ------
    UnidenticalEntryError
    -   Raised if the filetype in the filepath and the specified filetype do
        not match.

    MissingFileExtensionError
    -   Raised if no filetype is present in the filetype and one is not
        specified.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to save the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match.\n"
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath = f"{fpath}.{ftype}"
    else:
        raise MissingFileExtensionError(
            "Error when trying to save ta dictionary:\nNo filetype has been "
            f"specified and it cannot be detected in the filepath:\n{fpath}\n"
        )

    return fpath, ftype



def identify_ftype(fpath: str) -> str:
    """Finds what file type of a file is based on the filename extension.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath, including the filename and extension.

    RETURNS
    -------
    str
    -   The file type.

    RAISES
    ------
    MissingFileExtensionError
    -   Raised if 'fpath' is missing the filetype extension.
    """

    if not check_ftype_present(fpath):
        raise MissingFileExtensionError(
            "Error when determining the filetype:\nNo filetype can be found in "
            f"the filepath '{fpath}'.\nFilepaths should be in the format "
            "'filename.filetype'."
        )

    return fpath[fpath.rfind(".") + 1 :]




def nested_changes_list(contents: list, changes: dict) -> None:
    """Makes changes to the specified values occurring within nested
    dictionaries of lists of a parent list.

    PARAMETERS
    ----------
    contents : list
    -   The list containing nested dictionaries and lists whose values should be
        changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    for value in contents:
        if isinstance(value, list):
            nested_changes_list(contents=value, changes=changes)
        elif isinstance(value, dict):
            nested_changes_dict(contents=value, changes=changes)
        else:
            if value in changes.keys():
                value = changes[value]


def nested_changes_dict(contents: dict, changes: dict) -> None:
    """Makes changes to the specified values occurring within nested
    dictionaries or lists of a parent dictionary.

    PARAMETERS
    ----------
    contents : dict
    -   The dictionary containing nested dictionaries and lists whose values
        should be changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    for key, value in contents.items():
        if isinstance(value, list):
            nested_changes_list(contents=value, changes=changes)
        elif isinstance(value, dict):
            nested_changes_dict(contents=value, changes=changes)
        else:
            if value in changes.keys():
                contents[key] = changes[value]


def nested_changes(contents: Union[dict, list], changes: dict) -> None:
    """Makes changes to the specified values occurring within nested
    dictionaries or lists of a parent dictionary or list.

    PARAMETERS
    ----------
    contents : dict | list
    -   The dictionary or list containing nested dictionaries and lists whose
        values should be changed.

    changes : dict
    -   Dictionary specifying the changes to make, with the keys being the
        values that should be changed, and the values being what the values
        should be changed to.
    """

    if isinstance(contents, dict):
        nested_changes_dict(contents=contents, changes=changes)
    elif isinstance(contents, list):
        nested_changes_list(contents=contents, changes=changes)
    else:
        raise TypeError(
            "Error when changing nested elements of an object:\nProcessing "
            f"objects of type '{type(contents)}' is not supported. Only 'list' "
            "and 'dict' objects can be processed."
        )
    


def extra_deserialise_json(contents: dict) -> dict:
    """Performs custom deserialisation on a dictionary loaded from a json file
    with changes not present in the default deserialisation used in the 'load'
    method of the 'json' package.
    -   Current extra changes include: converting "INFINITY" strings into
        infinity floats.

    PARAMETERS
    ----------
    contents : dict
    -   The contents of the dictionary loaded from a json file.

    RETURNS
    -------
    dict
    -   The contents of the dictionary with additional changes made.
    """

    deserialise = {"INFINITY": float("inf")}

    nested_changes(contents=contents, changes=deserialise)

    return contents




def check_python_list(contents) -> list:
    if (not isinstance(contents, collections.abc.Sequence)) | (not isinstance(contents, list)):
        contents = [contents]
    print(type(contents))    
    return contents


def check_task_validity(task, valid_tasks) -> Any:
    if task in valid_tasks:
        return task
    else:
        raise(UnavailableProcessingError(f"Task {task} is not a valid task."))
    
    
    
def dict_to_df(obj: dict, orient = "columns") -> pd.DataFrame:
    """Converts a dictionary into a pandas DataFrame.

    PARAMETERS
    ----------
    obj : dict
    -   Dictionary to convert.

    RETURNS
    -------
    pandas DataFrame
    -   The converted dictionary.
    """

    return pd.DataFrame.from_dict(data=obj, orient=orient)