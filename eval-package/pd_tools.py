#!/usr/bin/env python
"""
Pandas dataFrame tool
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pandas

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"


def load_protocol(protocol_file, names, sep=' ', index_col=None):
    """ pd_protocol = load_protocol(protocol_file, names, sep=' ', index_col=None)

    input
    -----
      protocol_file  str, path to the protocol file
      names          list of str, name of the data Series in dataFrame 
      sep            str, separator, by default ' '
      index_col      str, name of the index column, by default None

    output
    ------
      pd_protocol    pandas dataFrame
    """
    pd_protocol = pandas.read_csv(protocol_file, sep=' ', names=names, 
                                  index_col = index_col, skipinitialspace=True)
    return pd_protocol



def load_score(score_file,  names,  sep=' ', index_col=None):
    """ pd_score = load_score(score_file, names, sep=' ', index_col=None)

    input
    -----
      score_file     str, path to the score file
      names          list of str, name of the data Series in dataFrame 
      sep            str, separator, by default ' '
      index_col      str, name of the index column, by default None

    output
    ------
      pd_protocol    pandas dataFrame
    """
    pd_score = pandas.read_csv(score_file, sep=sep, names=names, 
                                  index_col=index_col, skipinitialspace=True)
    return pd_score


def join_protocol_score(pd_protocol, pd_score):
    """ pd_final = join_protocol_score(pd_protocol, pd_score)

    input
    -----
      pd_protocol  dataFrame, protocol dataframe
      pd_score     dataFrame, score dataframe

    output
    ------
      pd_final     dataFrame, joint dataFrame from pd_protocol and pd_score
    """
    pd_final = pandas.concat([pd_protocol, pd_score], axis=1, join="inner")
    if len(pd_protocol) != len(pd_score) or len(pd_protocol) != len(pd_final):
        print("Error: protocol and score seem to mismatch. Please check!")
        print("Protocol file has {:d} entries".format(len(pd_protocol)))
        print("Score file has {:d} entries".format(len(pd_score)))
        print("Number of common entries is {:d}".format(len(pd_final)))
        print("\nIs the score file incomplete?")
        print("Has you selected the correct track?")
        sys.exit(1)
    return pd_final


if __name__ == "__main__":
    print("pd_tools")
