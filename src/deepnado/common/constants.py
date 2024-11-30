# Constants go into this file.
import logging

CONFIG_FILE_NAME = ".dnrc"

# List all potential input variables
ALL_VARIABLES = ["DBZ", "VEL", "KDP", "RHOHV", "ZDR", "WIDTH"]

# Provides a typical min-max range for each variable (but not exact)
# Used for normalizing in a NN
CHANNEL_MIN_MAX = {
    "DBZ": [-20.0, 60.0],
    "VEL": [-60.0, 60.0],
    "KDP": [-2.0, 5.0],
    "RHOHV": [0.2, 1.04],
    "ZDR": [-1.0, 8.0],
    "WIDTH": [0.0, 9.0],
}

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
