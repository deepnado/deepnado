import logging
import sys

from logging import Logger

from deepnado.common import constants
from deepnado.config.config import ApplicationConfig
from deepnado.config.cli_parser import CliArgParser
from deepnado.trainer import train


class DeepNadoEntry:

    def __init__(self, logger, data_root_cli=None, training_config_path=None) -> None:
        self._config = ApplicationConfig()
        self._config.parse_config_file()
        self._logger: Logger = logger

        # If user proviced data root via CLI, override anything found in an config
        # file.
        if data_root_cli:
            self._config.DATA_ROOT = data_root_cli
            self._logger.debug(
                "User provided -d/-dataroot option via CLI. Will use as dataroot path."
            )
            self._logger.debug(f"Configured DATA_ROOT = {self._config.DATA_ROOT}")
        if training_config_path:
            self._training_config_path = training_config_path
            self._logger.debug("User provided training config path. Training will be executed.")

    def run(self):
        self._logger.debug("Executing...")
        if self._training_config_path:
            train(self._logger, self._config.DATA_ROOT, self._training_config_path)


def main():
    args = CliArgParser()
    args.apply()

    # Get verbosity
    verbose = args.__verbose__

    # Don't have to check for KeyError here because argparse enforces
    # choices.
    log_level = constants.LOG_LEVELS_MAP.get(args.loglevel.lower())

    # If verbose flag set, log level becomes debug regardless.
    if verbose:
        log_level = logging.DEBUG

    logging.basicConfig(datefmt="%m/%d/%Y %I:%M:%S %p", stream=sys.stdout, level=log_level)
    logger = logging.getLogger(__name__)

    # This is just a polite message to the user.
    if verbose:
        logger.debug("The -v verbosity flag was set. Log Level overridden to DEBUG.")

    deepnado = DeepNadoEntry(logger, data_root_cli=args.dataroot, training_config_path=args.config)
    deepnado.run()
