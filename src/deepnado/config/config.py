import logging
import os

from pathlib import Path

from deepnado.common import constants
from deepnado.config.file_parser import FileParser

logger = logging.getLogger(__name__)


class ApplicationConfig(object):
    """
    This class holds all app level configuration, but does not contain
    training parameters.
    """

    APP_NAME = "deepnado"
    APP_PRETTY_NAME = "DeepNado"

    DATA_ROOT = None

    def _find_config_file(self) -> str:
        """
        Search for user configuration file.

        Priority goes to current directory, and then the user's home
        folder.  For example, if user has config file in current folder
        and home folder, the file in the current folder takes priority.

        If user does not place a config file in either location, then inputs
        must come from the command line.
        """
        config_file = None

        current_folder = os.getcwd()
        home_folder = str(Path.home())

        current_folder_path = os.path.join(current_folder, constants.CONFIG_FILE_NAME)
        home_folder_path = os.path.join(home_folder, constants.CONFIG_FILE_NAME)

        if os.path.exists(current_folder_path):
            logger.debug("Config File exists in current directory.")
            config_file = current_folder_path
        elif os.path.exists(home_folder_path):
            logger.debug("Config File exists in user's home directory.")
            config_file = home_folder_path
        else:
            logger.debug("No config file detected...")

        return config_file

    def parse_config_file(self):

        config_file = self._find_config_file()

        if config_file:

            cf_parser = FileParser()
            cf_parser.read_file(config_file)
            cf_parser.parse()

            self.DATA_ROOT = cf_parser.get_option("dataroot")
            logger.debug(f"Config File Data Root: {self.DATA_ROOT}")

    @classmethod
    def obtain_environment_variables(cls):
        for var in cls.__dict__.keys():
            if var[:1] != "_" and var != "obtain_environment_variables":
                if var in os.environ:
                    value = os.environ[var].lower()
                    if value == "true" or value == "TRUE" or value == "True":
                        setattr(cls, var, True)
                    elif value == "false" or value == "FALSE" or value == "False":
                        setattr(cls, var, False)
                    else:
                        setattr(cls, var, os.environ[var])

    @classmethod
    def __str__(cls):
        print_str = ""
        for var in cls.__dict__.keys():
            if var[:1] != "_" and var != "obtain_environment_variables":
                print_str += f"VAR: {var} set to: {getattr(cls,var)}\n"
        return print_str
