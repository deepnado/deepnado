import configparser
import logging

logger = logging.getLogger(__name__)


class FileParser(object):

    CONFIG_KEY = "DEEPNADO"

    def __init__(self) -> None:
        self._parser = configparser.ConfigParser()

    def read_file(self, config_file: str):
        self._parser.read(config_file)

    def parse(self):

        if not self._parser.has_section(self.CONFIG_KEY):
            logger.critical(f"Config file is missing {self.CONFIG_KEY}.")
        else:
            check_entries = self._parser[self.CONFIG_KEY]
            if len(check_entries.items()) == 0:
                logger.critical(
                    f"Config file has {self.CONFIG_KEY} section, but there's nothing in it."
                )

    def get_option(self, option_name, section_name=CONFIG_KEY):
        return self._parser.get(section_name, option_name)
