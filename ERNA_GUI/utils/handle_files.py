import os

from ERNA_GUI.utils.handle_ftypes import (
    check_ftype_present, numpy_to_python, check_file_inputs,
    extra_deserialise_json, identify_ftype
)

import pickle
import json
import csv
import pandas as pd
from typing import Union, Any, Optional

from ERNA_GUI.utils.handle_errors import (
    UnavailableProcessingError, UnidenticalEntryError, MissingFileExtensionError
)


def confirm_overwrite(fpath: str) -> bool:
    """Asks the user to confirm whether a pre-existing file should be
    overwritten.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath where the object will be saved.

    RETURNS
    -------
    write : bool
    -   Whether or not the pre-existing file should be overwritten or not based
        on the user's response.
    """

    write = False
    valid_response = False
    while valid_response is False:
        response = input(
            f"The file '{fpath}' already exists.\nDo you want to "
            "overwrite it? y/n: "
        )
        if response not in ["y", "n"]:
            print(
                "The only accepted responses are 'y' and 'n'. "
                "Please input your response again."
            )
        if response == "n":
            print(
                "You have requested that the pre-existing file not "
                "be overwritten. The new file has not been saved."
            )
            valid_response = True
        if response == "y":
            write = True
            valid_response = True

    return write


def check_before_overwrite(fpath: str) -> bool:
    """Checks whether a file exists at a specified filepath. If so, the user is
    given the option of choosing whether to overwrite the file or not.

    PARAMETERS
    ----------
    fpath : str
    -   The filepath where the object will be saved.

    RETURNS
    -------
    bool : str
    -   Whether or not the object should be saved to the filepath.
    """

    if os.path.exists(fpath):
        write = confirm_overwrite(fpath)
    else:
        write = True

    return write



def save_object(
    to_save: object,
    fpath: str,
    ask_before_overwrite: bool = True,
    convert_numpy_to_python: bool = False,
    verbose: bool = True,
) -> None:
    """Saves an object as a .pkl file [for cache for example]

    PARAMETERS
    ----------
    to_save : oibject
    -   Object to save.

    fpath : str
    -   Location where the object should be saved. The filetype extension
        (.pkl) can be included, otherwise it will be automatically added.

    ask_before_overwrite : bool
    -   If True, the user is asked to confirm whether or not to overwrite a
        pre-existing file if one exists.
    -   If False, the user is not asked to confirm this and it is done
        automatically.

    convert_numpy_to_python : bool; default False
    -   Whether or not to convert numpy objects into their equivalent Python
        types before saving.

    verbose : bool
    -   Whether or not to print a note of the saving process.
    """

    if not check_ftype_present(fpath):
        fpath += ".pkl"

    if ask_before_overwrite:
        write = check_before_overwrite(fpath)
    else:
        write = True

    if write:
        save_as_pkl(to_save, fpath, convert_numpy_to_python)

        if verbose:
            print(f"Saving the analysis object to:\n{fpath}")


def save_dict(
    to_save: dict,
    fpath: str,
    ftype: Union[str, None] = None,
    ask_before_overwrite: bool = True,
    convert_numpy_to_python: bool = False,
    verbose: bool = True,
) -> None:
    """Saves a dictionary as a file.

    PARAMETERS
    ----------
    to_save : dict
    -   The dictionary to save.

    fpath : str
    -   Location where the dictionary should be saved.
    -   Can contain a filetype (e.g. '.json'), in which case 'ftype' does not
        need to be given and can be determined based on the filetype in 'fpath',
        otherwise a filetype must be specified in 'ftype'.

    ftype : str | None; default None
    -   The filetype of the dictionary that will be saved, without the leading
        period. E.g. for saving the file in the json format, this would be
        "json", not ".json".
    -   The information being saved must be an appropriate type for saving
        in this format.
    -   If 'None', the filetype is automatically determined based on the the
        filetype in 'fpath', in which case an identifiable filetype must be
        present in 'fpath'.

    ask_before_overwrite : bool; default True
    -   If True, the user is asked to confirm whether or not to overwrite a
        pre-existing file if one exists.
    -   If False, the user is not asked to confirm this and it is done
        automatically.

    convert_numpy_to_python : bool; default False
    -   Whether or not to convert numpy objects into their equivalent Python
        types before saving.

    verbose : bool; default True
    -   Whether or not to print a note of the saving process.

    RAISES
    ------
    UnavailableProcessingError
    -   Raised if the given format for saving the file is in an unsupported
        format.
    """

    fpath, ftype = check_file_inputs(fpath=fpath, ftype=ftype)

    if ask_before_overwrite:
        write = check_before_overwrite(fpath)
    else:
        write = True

    if write:
        if verbose:
            print(f"Saving the dictionary to:\n'{fpath}'.\n")

        if ftype == "json":
            save_as_json(to_save, fpath, convert_numpy_to_python)
        elif ftype == "tsv":
            save_as_csv(to_save, fpath, convert_numpy_to_python)
        elif ftype == "pkl":
            save_as_pkl(to_save, fpath, convert_numpy_to_python)
        else:
            raise UnavailableProcessingError(
                f"Error when trying to save a dictionary:\nThe {ftype} format "
                "for saving is not supported.\n"
            )            



def save_as_json(
    to_save: dict, fpath: str, convert_numpy_to_python: bool = False
) -> None:
    """Saves a dictionary as a json file.

    PARAMETERS
    ----------
    to_save : dict
    -   Dictionary in which the keys represent the names of the entries in
        the json file, and the values represent the corresponding values.

    fpath : str
    -   Location where the data should be saved.

    convert_numpy_to_python : bool; default False
    -   Whether or not to convert numpy objects into their equivalent Python
        types before saving.
    """
    if convert_numpy_to_python:
        to_save = numpy_to_python(to_save)

    with open(fpath, "w", encoding="utf8") as file:
        json.dump(to_save, file)


def save_as_csv(
    to_save: dict, fpath: str, convert_numpy_to_python: bool = False
) -> None:
    """Saves a dictionary as a csv file.

    PARAMETERS
    ----------
    to_save : dict
    -   Dictionary in which the keys represent the names of the entries in
        the csv file, and the values represent the corresponding values.

    fpath : str
    -   Location where the data should be saved.

    convert_numpy_to_python : bool; default False
    -   Whether or not to convert numpy objects into their equivalent Python
        types before saving.
    """
    if convert_numpy_to_python:
        to_save = numpy_to_python(to_save)


    to_save.to_csv(fpath,sep='\t', index=False)
    #with open(fpath, "w", newline='', encoding="utf-8") as file:
    #    save_file = csv.writer(file, delimiter='\t')
    #    print(to_save.keys())
    #    save_file.writerow(to_save.keys())
    #    save_file.writerow(to_save.values())


def save_as_pkl(
    to_save: Any, fpath: str, convert_numpy_to_python: bool = False
) -> None:
    """Pickles and saves information in any format.

    PARAMETERS
    ----------
    to_save : Any
    -   Information that will be saved.

    fpath : str
    -   Location where the information should be saved.

    convert_numpy_to_python : bool; default False
    -   Whether or not to convert numpy objects into their equivalent Python
        types before saving.
    """
    if convert_numpy_to_python:
        to_save = numpy_to_python(to_save)

    with open(fpath, "wb") as file:
        pickle.dump(to_save, file)




def load_from_json(fpath: str) -> dict:
    """Loads the contents of a json file as a dictionary.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    contents : dict
    -   The contents of the file in a dictionary.
    """

    with open(fpath, encoding="utf8") as file:
        contents = extra_deserialise_json(contents=json.load(file))

    return contents


def load_from_pkl(fpath: str) -> Any:
    """Loads the contents of a pkl file.

    PARAMETERS
    ----------
    fpath : str
    -   Location of the file to load.

    RETURNS
    -------
    Any
    -   The contents of the file.
    """

    with open(fpath, "rb") as file:
        return pickle.load(file)


def load_file(
    fpath: str, ftype: Optional[str] = None, verbose: bool = True
) -> Any:
    """Loads the contents of a json or pkl file.

    PARAMETERS
    ----------
    fpath : str
        -   Location where the data should be loaded from.

    ftype : str
    -   The filetype of the data that will be loaded, without the leading
        period. E.g. for loading the file from the json format, this would be
        "json", not ".json".

    verbose : bool; default True
    -   Whether or not to print a note of the loading process.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to save the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match."
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath += ftype
    else:
        raise MissingFileExtensionError(
            "Error when trying to save the results of the analysis:\nNo "
            "filetype is given in the filepath and no filetype has been "
            "specified."
        )

    if ftype == "json":
        contents = load_from_json(fpath=fpath)
    elif ftype == "pkl":
        contents = load_from_pkl(fpath=fpath)
    else:
        raise UnavailableProcessingError(
            f"Error when trying to load the file:\nThe {ftype} format for "
            "loading is not supported."
        )

    if verbose:
        print(f"Loading the contents of the filepath:\n'{fpath}'.")

    return contents



def load_table(
    fpath: str, ftype: Optional[str] = None, verbose: bool = True
) -> Any:
    """Loads the contents of a csv or tsv file.

    PARAMETERS
    ----------
    fpath : str
        -   Location where the data should be loaded from.

    ftype : str
    -   The filetype of the data that will be loaded, without the leading
        period. E.g. for loading the file from the tsv format, this would be
        "tsv", not ".tsv".

    verbose : bool; default True
    -   Whether or not to print a note of the loading process.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to save the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match."
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath += ftype
    else:
        raise MissingFileExtensionError(
            "Error when trying to save the results of the analysis:\nNo "
            "filetype is given in the filepath and no filetype has been "
            "specified."
        )

    if ftype == "csv":
        contents = pd.read_csv(fpath)
    elif ftype == "tsv":
        contents = pd.read_csv(fpath, sep = '\t')
    else:
        raise UnavailableProcessingError(
            f"Error when trying to load the file:\nThe {ftype} format for "
            "loading is not supported."
        )

    if verbose:
        print(f"Loading the contents of the filepath:\n'{fpath}'.")

    return contents


def delete_object(
    fpath: str, verbose: bool = True
) -> None:
    """Delete a file.

    PARAMETERS
    ----------
    fpath : str
        -   Location where the data should be.

 

    verbose : bool; default True
    -   Whether or not to print a note of the loading process.
    """

    if check_ftype_present(fpath) and ftype is not None:
        fpath_ftype = identify_ftype(fpath)
        if fpath_ftype != ftype:
            raise UnidenticalEntryError(
                "Error when trying to delete the results of the analysis:\n "
                f"The filetypes in the filepath ({fpath_ftype}) and in the "
                f"requested filetype ({ftype}) do not match."
            )
    elif check_ftype_present(fpath) and ftype is None:
        ftype = identify_ftype(fpath)
    elif not check_ftype_present(fpath) and ftype is not None:
        fpath += ftype
    else:
        raise MissingFileExtensionError(
            "Error when trying to delete the results of the analysis:\nNo "
            "filetype is given in the filepath and no filetype has been "
            "specified."
        )
    
    os.remove(fpath)
    



    if verbose:
        print(f"Deleting the contents of the filepath:\n'{fpath}'.")

    