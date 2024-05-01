import os
import warnings
from copy import deepcopy
from typing import Dict, Any, Type, List, Union


import torch
import hydra
from omegaconf import OmegaConf, DictConfig
from rich.syntax import Syntax
from rich.console import Console
from transformers import AutoModel


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


def before_task(config: DictConfig):
    """Execute additional utilities before task:
        - Updates env variables
        - Adds 'eval' resolver to OmegaConf
        - Filters warnings
        - Prints out config tree
        - Performs config tree resolution (parameter interpolation)
    """
    # Update environment variables
    os.environ["HYDRA_FULL_ERROR"] = "1"

    # Add eval resolver to OmegaConf
    OmegaConf.register_new_resolver("eval", eval)

    # Filter warnings
    if config.get("filter_warnings"):
        warnings.filterwarnings("ignore")
    
    # Print config tree
    if config.get("print_config"):
        yaml = OmegaConf.to_yaml(config, resolve=True)
        syntax = Syntax(yaml, "yaml")
        Console().print(syntax)

    # Interpolate configs
    OmegaConf.resolve(config)


def instantiate_list_configs(configs: Union[DictConfig, List[DictConfig]]) -> List[Any]:
    """Instantiates config entities stored in a list.
    Converts to a list if it is not one.
    """
    if not isinstance(configs, List):
        configs = [configs]
    
    return [hydra.utils.instantiate(cfg) for cfg in configs]


def instantiate_part_of_hf_model(model_name: str, part: str, **kwargs) -> torch.nn.Module:
    """Instantiate specific part of huggingface' transformers model.

    :param model_name: ...
    :type model_name: str
    :param part: Part to be returned
    :type part: str
    :return: ...
    :rtype: torch.nn.Module
    """
    model = AutoModel.from_pretrained(model_name, **kwargs)
    model = getattr(model, part)

    return model