import logging
import logging.config
from os import path


def set_logger_configuration():
    log_file_path = path.join(path.dirname(path.abspath(__file__)), path.join('config', 'logging.conf'))
    logging.config.fileConfig(fname=log_file_path, disable_existing_loggers=False)


def get_logger(logger_name):
    set_logger_configuration()
    return logging.getLogger(logger_name)