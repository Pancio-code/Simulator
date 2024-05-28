#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

"""
Implement a colored logging
"""


class Colors:
    """Colors list"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class status_str:
    CHECK_STR = " " + Colors.WARNING + "CHCK" + Colors.ENDC + " "
    OK_STR = "  " + Colors.OKGREEN + "OK" + Colors.ENDC + "  "
    DEAD_STR = " " + Colors.FAIL + "DEAD" + Colors.ENDC + " "
    MISM_STR = " " + Colors.WARNING + "MISM" + Colors.ENDC + " "
    WARN_STR = " " + Colors.WARNING + "WARN" + Colors.ENDC + " "


COLOR = True


class Log(object):

    @staticmethod
    def err(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.FAIL, end="")
            print(value, *args, end="", sep=sep)
            print(Colors.ENDC, end=end)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def fatal(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.FAIL, end="")
            print(value, *args, end="", sep=sep)
            print(Colors.ENDC, end=end)
        else:
            print(value, args, end=end, sep=sep, file=file)
        raise RuntimeError(value)

    @staticmethod
    def warn(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.WARNING, end="")
            print(value, *args, end="")
            print(Colors.ENDC)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def info(value, *args, end="\n", sep="", file=None):
        if COLOR and file is None:
            print(Colors.OKBLUE, end="")
            print(value, *args, end="")
            print(Colors.ENDC)
        else:
            print(value, args, end=end, sep=sep, file=file)

    @staticmethod
    def debug(value, *args, end="\n", sep="", file=None):
        print(value, *args, end=end, sep=sep, file=file)

    #
    # Module log
    #

    @staticmethod
    def merr(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.err(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mwarn(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.warn(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def minfo(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.info(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mdebug(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.debug(new_value, *args, end=end, sep=sep, file=file)

    @staticmethod
    def mfatal(module, value, *args, end="\n", sep="", file=None):
        new_value = "[" + module + "] " + value
        Log.fatal(new_value, *args, end=end, sep=sep, file=file)
