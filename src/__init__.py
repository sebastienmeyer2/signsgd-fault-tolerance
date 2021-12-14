"""
Source code for our evaluation of Signum against blind and Byzantine adversaries.

..include:: ../README.md
"""

# Import Python packages
import os
import sys


# Global variables
SRC_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SRC_PATH)  # trick to make pdoc3 understand that this is the package src folder
