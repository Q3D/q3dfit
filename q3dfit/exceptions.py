#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:26:58 2020
"""

class InitializationError(Exception):
    """Exception raised for initialization errors.
    Attributes:
        input -- input which caused the error
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message + ' in initialization file.'
        super().__init__(self.message)


class CubeError(Exception):
    """Exception raised for cube reader.
    Attributes:
        input -- input which caused the error
        message -- explanation of the error
    """

    def __init__(self, message=None):
        self.message = message + ' when reading data cube.'
        super().__init__(self.message)
