import argparse as _argparse
import sys

from deepnado.common import constants

_DESCRIPTION_MSG = """ """

_EPILOG_MSG = """
    Examples:
"""


class CliArgParser(object):

    def __init__(self):
        self._g_help = False
        self.__verbose__ = False

        self._subparser_name = None

        self.psr = _argparse.ArgumentParser(
            prog=__file__,
            description=_DESCRIPTION_MSG,
            epilog=_EPILOG_MSG,
            formatter_class=_argparse.RawTextHelpFormatter,
        )

        self._add_generic_args(self.psr)

        self._add_subparser(self.psr)

        self.psr.parse_args(args=self._sort_args(), namespace=self)

    def apply(self):
        pass

    def _add_subparser(self, psr):
        # sub = psr.add_subparsers(
        #     dest="_subparser_name", metavar="sub_commands", help="this is help"
        # )

        # Example
        # sub_command = sub.add_parser("sub_command")

        # Add sub commands to list
        self._sub_list = []

        for item in self._sub_list:
            self._add_generic_args(item)

    @staticmethod
    def _add_generic_args(parser):

        parser.add_argument(
            "-v",
            "--verbose",
            dest="__verbose__",
            action="store_true",
            default=False,
            help="enable verbose output debug",
        )

        parser.add_argument(
            "-d",
            "--dataroot",
            nargs="?",
            default=None,
            type=str,
            help="Specify Path to DATA_ROOT",
        )

        parser.add_argument(
            "-l",
            "--loglevel",
            default="critical",
            choices=constants.LOG_LEVELS_MAP.keys(),
            help="Set log level",
        )
        parser.add_argument(
            "-c",
            "--config",
            default=None,
            type=str,
            help="Specify path to training configuration",
        )

    def _sort_args(self):
        """
        Move all subparsers to the front
        """

        sub_names = [x.prog.split()[1] for x in self._sub_list]

        sargs = sys.argv[1:]

        for f in sub_names:
            if f in sargs:
                sargs.remove(f)
                sargs.insert(0, f)

        return sargs
