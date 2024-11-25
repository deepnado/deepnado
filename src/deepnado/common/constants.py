# Constants go into this file.
import logging

CONFIG_FILE_NAME = ".dnrc"

LOG_LEVELS_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
