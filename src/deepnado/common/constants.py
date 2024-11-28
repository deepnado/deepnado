# Constants go into this file.
import logging

CONFIG_FILE_NAME = ".dnrc"

# List all potential input variables
ALL_VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]

VARIABLES_TO_LABELS = {
    "DBZ": "dBZ",
    "VEL": "m/s",
    "KDP": "degrees/km",
    "RHOHV": "correlation",
    "ZDR": "dB",
    "WIDTH": "m/s",
}

LOG_LEVELS_MAP = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warn": logging.WARNING,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
}
