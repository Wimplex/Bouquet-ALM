import warnings
from copy import deepcopy
from typing import Dict, Any, Type

import omegaconf


def cast(input: Any, target_type: Type):
    if type(input) != target_type:
        return target_type(input)
    
    return input


def exact_div(x, y):
    assert x % y == 0
    return x // y


def filter_out_state_dict_keys(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Deletes weights from state_dict if corresponding keys are not started with prefix.
    The prefix will be removed from the remaining keys
    """

    keys = list(state_dict.keys())
    for k in keys:
        if k.startswith(prefix):
            state_dict[k[len(prefix) + 1:]] = deepcopy(state_dict[k])
            del state_dict[k] 
        else:
            del state_dict[k]

    return state_dict


def extras(config: omegaconf.DictConfig):
    """Execute additional utilities before task:
        - Filters warnings
        - Prints out config tree
    """
    # Filter warnings
    if config.get("filter_warnings"):
        warnings.filterwarnings("ignore")
    
    # Print config tree
    if config.get("print_config"):
        print(omegaconf.OmegaConf.to_yaml(config))
