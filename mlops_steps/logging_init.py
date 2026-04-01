import logging.config
import os

import yaml


def setup_logging(conf_file_path=None):
    if conf_file_path is None:
        # Fetching path to load the logging-config.yaml from the same directory as this script.
        current_script_dir = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        conf_file_path = os.path.join(current_script_dir, "logging-config.yaml")

    with open(conf_file_path, "rt") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)