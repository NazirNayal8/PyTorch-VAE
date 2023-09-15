import pytorch_lightning as pl
import yaml

def read_config(filename):
    with open(filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def update_from_wandb(config, wandb_config):

    for k, v in wandb_config.items():
        if k not in config:
            raise ValueError(f"Wandb Config has sth that you don't")
        if isinstance(v, dict):
            config[k] = update_from_wandb(config[k], wandb_config[k])
        else:
            config[k] = v

    return config


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper
