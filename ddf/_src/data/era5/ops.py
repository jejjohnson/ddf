from typing import List, Union, Dict
from functools import partial
from ddf._src.dtypes.time import Time
from ddf._src.variables.era5 import VariablePressureLevel, VariableSingleLevel, VARIABLE_NAMES, VARIABLES_SURFACE


def parse_single_levels(channel_names: list[str]) -> list[VariableSingleLevel]:
    """
    Parses a list of channel names and returns a list of corresponding VariableSingleLevel objects.

    Args:
        channel_names (list[str]): A list of channel names.

    Returns:
        list[VariableSingleLevel]: A list of VariableSingleLevel objects corresponding to the input channel names.

    Example:
    >>> channels = ["u10m", "v10m", "z100", "u250"]
    >>> parse_single_levels(channels_)
    [VariableSingleLevel(id=165, name='u10m'), 
    SingleLevelCode(id=166, name='v10m')]
    """
    # check if name in explicit names
    criteria = lambda x: x in VARIABLES_SURFACE

    # filter for criteria
    sl_variables = list(filter(criteria, channel_names))
    # create single level codes
    f = lambda x: VARIABLE_NAMES[x]()
    # map list to codes
    return list(map(f, sl_variables))


def parse_pressure_levels(channel_names: list[str]) -> list[VariablePressureLevel]:
    """
    Parses a list of channel names and returns a list of corresponding VariablePressureLevel objects.

    Args:
        channel_names (list[str]): A list of channel names.

    Returns:
        list[PressureLevelCode]: A list of PressureLevelCode objects corresponding to the channel names.
    
    Example:
    >>> channels = ["u10m", "v10m", "z100", "u250"]
    >>> parse_pressure_levels(channels_)
    [VariablePressureLevel(id=129, level=100, name='z'),
    VariablePressureLevel(id=131, level=250, name='u')]
    """
    # check if name in explicit names
    criteria = lambda x: x not in VARIABLES_SURFACE
    # filter for criteria
    pl_variables = list(filter(criteria, channel_names))
    # create pressure level codes
    f = lambda x: VARIABLE_NAMES[x[0]](level=int(x[1:]))
    # map list to codes
    return list(map(f, pl_variables))


def parse_all_variables(channel_names: list[str]) -> List[Union[VariableSingleLevel, VariablePressureLevel]]:
    """
    Parses all variables from the given channel names.

    Args:
        channel_names (list[str]): A list of channel names.

    Returns:
        List[Union[SingleLevelCode, PressureLevelCode]]: A list of parsed variables.

    Raises:
        AssertionError: If the length of the parsed variables is not equal to the length of the channel names.
    
    Example:
    >>> channels = ["u10m", "v10m", "z100", "u250"]
    >>> parse_all_variables(channels_)
    [SingleLevelCode(id=165, name='u10m'),
    SingleLevelCode(id=166, name='v10m'),
    PressureLevelCode(id=129, level=100, name='z'),
    PressureLevelCode(id=131, level=250, name='u')]
    """
    # # check if name in explicit names
    # criteria = lambda x: x not in VARIABLES_SURFACE
    # # filter for criteria
    # pl_variables = list(filter(criteria, channel_names))
    # create pressure level codes
    f = lambda x: (
        VARIABLE_NAMES[x[0]](level=int(x[1:])) if x not in VARIABLES_SURFACE 
        else VARIABLE_NAMES[x]()
    )
    # map list to codes
    all_vars = list(map(f, channel_names))
    assert len(all_vars) == len(channel_names)

    return all_vars


# def joint_requests_pl(codes: List[PressureLevelCode], time: Time, format: str="grib") -> Dict:
    
#     f = partial(create_request_pressure_level, time=time, format=format)

#     requests = list(map(f, codes))

#     datasets, list_of_requests, savenames = zip(*requests)
    
#     return None


def joint_requests(list_of_requests: List[Dict]) -> Dict:
    """
    Combines multiple requests into a single joint request.

    Args:
        list_of_requests (List[Dict]): A list of dictionaries representing individual requests.

    Returns:
        Dict: A dictionary representing the joint request.

    """
    joint_requests = {}
    for irequest in list_of_requests:
        for (ikey, ivalue) in irequest.items():
            # create a new key entry (assume the same)
            # TODO CHECK!
            if ikey not in joint_requests:
                joint_requests[ikey] = ivalue
            # concatenated the unique list elements
            if ikey in ["year", "time", "day", "month", "pressure_level", "param"]:
                joint_requests[ikey] = list(set(joint_requests[ikey] + ivalue))
                
    joint_requests["param"] = '/'.join([str(x) for x in joint_requests["param"]])
    
    return joint_requests