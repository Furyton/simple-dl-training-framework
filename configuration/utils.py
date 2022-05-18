import argparse
import json
import logging
from pathlib import Path

from dataloaders import DATALOADERS
from models import MODELS

from configuration.config import *


def load_model_config(model_code, global_args:argparse.Namespace=None) -> argparse.Namespace:

    if model_code not in MODELS:
        logging.fatal(f"{model_code} is not implemented")
        raise NotImplementedError(f"{model_code} is not implemented")

    model_config_filename = f"{model_code}.json"

    model_config_path = Path(ASSET_FOLDER).joinpath(ASSET_CONFIG_FOLDER).joinpath(ASSET_CONFIG_MODEL_FOLDER).joinpath(model_config_filename)

    logging.info("loading model %s's config file at %s", model_code, model_config_path)

    with model_config_path.open('r') as f:
        j = json.load(f)
        args = argparse.Namespace(**j)
    
    if global_args is not None and global_args.kwargs is not None:
        args.__dict__.update(global_args.kwargs)
    
    logging.info(json.dumps(args.__dict__, indent=4))

    return args