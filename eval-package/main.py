#!/usr/bin/env python
"""
========
Notes 
========
This is the API to compute EERs and min t-DCFs on ASVspoof2021 evaluation data.

Requirements:
 numpy
 pandas
 matplotlib 

Example usage:
1. Download key and meta label
   bash download.sh
   
   The downloaded directory contains key and meta label files

   keys
   |- LA
   |  |- CM 
   |  |   |- trial_metadata.txt
   |  |   |- LFCC-GMM
   |  |       |- score.txt
   |  |- ASV
   |      |- trial_metadata.txt
   |      |- ASVtorch_kaldi
   |          |- score.txt
   |- DF ...
   |- PA ...

   trial_metadata.txt contains the key and meta data
   score.txt is the score file

2. Compute EER and min t-DCF 

   Let's use LA evaluation subset as example.
   Assume meta labels are in ./keys (default folder)

   Case 1 (most common case)
   Compute results using pre-computed C012 cofficients

   python main.py --cm-score-file score.txt --track LA --subset eval
   
   Case 2
   Recompute C012 using official ASV scores, save it to ./LA-c012.npy,
   and use the new C012 to compute EER and min tDCFs
   
   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --recompute-c012 --c012-path ./LA-c012.npy

   Case 3
   Recompute C012 using my own ASV scores, save it to ./LA-c012.npy
   and use the new C012 to compute EER and min tDCFs
   
   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --recompute-c012 --c012-path ./LA-c012.npy 
                  --asv-score-file ./asv-score.txt

   Case 4
   Compute min tDCF using my own C012 coeffs ./LA-c012.npy

   python main.py --cm-score-file score.txt --track LA --subset eval 
                  --c012-path ./LA-c012.npy
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import pandas
import argparse
import numpy as np

import config
import pd_tools
import table_API
import eval_wrapper

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"


# ==========
# function 
# ==========



def sanity_check(args):
    # sanity check
    if not os.path.isfile(args.cm_score_file):
        print("Cannot find {:s}".format(args.cm_score_file))
        return False
    if not os.path.isdir(args.metadata):
        print("Cannot find {:s}".format(args.metadata))
        print("Please run download.sh to download the meta data")
        return False

    for track in config.g_possible_tracks:
        tmp_dir = os.path.join(args.metadata, track)
        if not os.path.isdir(tmp_dir):
            print("Cannot find ", tmp_dir)
            return False
        
    if args.track not in config.g_possible_tracks:
        print("track must be from", str(config.g_possible_tracks))
        return False
    if args.subset not in config.g_possible_subsets:
        print("subset must be from", str(config.g_possible_subsets))
        return False

    return True

def parse_argument():
    parser = argparse.ArgumentParser(
        epilog=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    mes = 'CM score file (two column, txt format)'
    parser.add_argument('--cm-score-file', type=str, default="", help=mes)
    
    mes = 'Name of the track, either LA, PA, or DF'
    parser.add_argument('--track', type=str, default="", help=mes)

    mes = 'Name of the subset, either eval, progress, hidden, hidden1_PA, hidden2_PA.'
    parser.add_argument('--subset', type=str, default="", help=mes)

    mes = 'Directory of the key and meta data. Default ./keys'
    parser.add_argument('--metadata', type=str, default="keys", help=mes)

    mes = 'Whether re-compute C012 coefficients for tDCF computation?'
    mes += 'Add this argument if you want to re-compute C012 coefficients.'
    parser.add_argument('--recompute-c012',action='store_true', default=False, help=mes)

    mes = 'ASV score file (three column, txt format). \n'
    mes += 'When recompute-c012 is on, load ASV score from asv-score-file. \n'
    mes += 'If asv-score-file is not provided,, load ASV score from organizers.'
    parser.add_argument('--asv-score-file', type=str, default="", help=mes)
    
    mes = 'Path to external C012 data dictionary.'
    mes += 'When recompute-c012 is on, save computed C012 to this path. \n'
    mes += 'When recompute-c012 is off, load pre-computed C012 from here.\n'
    mes += 'If c012-path is not provided, use c012 path specified on config.py.'
    parser.add_argument('--c012-path', type=str, default="", help=mes)    
    
    # load argument
    args = parser.parse_args()

    # sanity check
    if not sanity_check(args):
        print("ERROR: arguments are invalid. ")
        sys.exit(1)
        
    return args


def compute_tDCF_C012(asv_score_pd, 
                      factor_name_v,
                      factor_value_v, 
                      factor_type_v,
                      factor_name_h, 
                      factor_value_h, 
                      factor_type_h,
                      cost_model = config.cost_model,
                      pooled_tag = config.g_pooled_tag, 
                      target_tag = config.g_target_tag,
                      nontarget_tag = config.g_nontarget_tag,
                      spoofed_tag = config.g_spoofed_tag,
                      col_score_name = config.g_score_col_name,
                      flag_verbose = False):
    """C012_dict = compute_tDCF_C012(asv_score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h,
                                   factor_value_h,  
                                   factor_type_h,
                                   cost_model = config.cost_model,
                                   pooled_tag = 'Pooled', 
                                   target_tag = 'target',
                                   nontarget_tag = 'nontarget',
                                   spoofed_tag = 'spoof',
                                   col_score_name = 'score',
                                   flag_verbose = False)
    
    Function to loop over two sets of factors and compute C012.
    The output C012_dict can be used to compute min tDCF values

    input
    -----
      asv_score_pd    dataFrame, joint dataframe of ASV score and protocol

      factor_name_v   str or list of str, 
                      name(s) of the dataFrame series for the 1st set of factor.
                    
      factor_value_v  list of str, or list of list or str, 
                      values of the 1st set of factors 

                      if type(factor_name_v) is str:
                          # we retrieve the data by
                          for factor in factor_value_v:
                              data = score_pd.query('factor_name_v == "factor"')

                      if type(factor_name_v) is list
                          # we iterate all the factors
                          for factor_name, factor_value in zip(factor_name_v, factor_value_v):
                              for factor in factor_value:
                                  data = score_pd.query('factor_name == "factor"')
 
                    
                      The second case is useful when the 1st set of factors 
                      are defined in different data series of score_pd.

      factor_type_v   str or list of str, type of the factor
                      
                      'spoof': this factor is only available for spoofed data
                      'bonafide': this factor is only available for bonafide data
                      'both': this factor appears in both spoofed and bonafide data

                      if type(factor_name_v) is str:
                          # factor_type_v is the type for factor_name_v
                      if type(factor_name_v) is list:
                          # factor_type_v should be a list and
                          # factor_type_v[i] is the type for factor_name_v[i]

      factor_name_h   str or list of str, 
      factor_value_h  list of str or list of list of str
      factor_type_h   str or list of str
                     
                      these are for the second set of factors

      pooled_tag      str, tag for pooled condition, 
                      default 'Pooled'
      target_tag      str, tag for bonafide tareget trials
                      default 'target'
      nontarget_tag   str, tag for bonafide non-tareget trials
                      default 'nontarget'
      spooed_tag      str, tag for spoofed trials
                      default 'spoof'
      col_score_name  str, name of the column for score
                      default 'score'

    output
    ------
      C012_dict       dictionary of C012 values
                      C012[factor_1][factor_2]['C0'] -> C0
                      C012[factor_1][factor_2]['C1'] -> C1
                      C012[factor_1][factor_2]['C2'] -> C2
    """
    def _wrap_list(data):
         return [data] if type(data) is str else data

    def _wrap_list_list(data):
         return [data] if type(data[0]) is str else data
    
    # wrap them into a list
    factor_names_1 = _wrap_list(factor_name_v)
    factor_names_2 = _wrap_list(factor_name_h)
    factor_types_1 = _wrap_list(factor_type_v)
    factor_types_2 = _wrap_list(factor_type_h)
    factor_list_1_list = _wrap_list_list(factor_value_v)
    factor_list_2_list = _wrap_list_list(factor_value_h)

    # number of rows and columns in the result table
    num_row = sum([len(x) for x in factor_list_1_list])
    num_col = sum([len(x) for x in factor_list_2_list])

    # output buffer
    C012 = dict()

    print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing C012 for tDCF')

    # loop over factor along the row (factor 1)
    for factor_name_1, factor_list_1, factor_type_1 in zip(
        factor_names_1, factor_list_1_list, factor_types_1):
        for _, factor_1 in enumerate(factor_list_1):
            
            print(".", end = '', flush=True)
            
            # creat the query to retrieve the data corresponding to the factor_1
            if factor_1 == pooled_tag:
                # pooled condition
                qr_tar_fac1 = ''
                qr_ntar_fac1 = ''
                qr_spoof_fac1 = ''
            elif factor_type_1 == config.g_factor_type_spoof:
                # if factor is only for spoofed data (e.g., attack type)
                qr_tar_fac1 = ''
                qr_ntar_fac1 = ''
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            elif factor_type_1 == config.g_factor_type_bonafide:
                # if factor is only for bonafide (target and nontarget)
                qr_tar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_ntar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ''
            else:
                # if factor is for both spoofed and bona fide data (e.g., codec)
                qr_tar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_ntar_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            
            # loop over factor in cols (factor 2)
            for factor_name_2, factor_list_2, factor_type_2 in zip(
                factor_names_2, factor_list_2_list, factor_types_2):
                for _, factor_2 in enumerate(factor_list_2):
                                        
                    if factor_2 == pooled_tag:
                        # pooled condition
                        qr_tar_fac2 = ''
                        qr_ntar_fac2 = ''
                        qr_spoof_fac2 = ''
                    elif factor_type_2 == config.g_factor_type_spoof:
                        # if factor is only for spoofed data (e.g., attack type)
                        qr_tar_fac2 = ''
                        qr_ntar_fac2 = ''
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                    elif factor_type_2 == config.g_factor_type_bonafide:
                        qr_tar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_ntar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ''
                    else:
                        # if factor is for both spoofed and bona fide data (e.g., codec)
                        qr_tar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_ntar_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)

                    # query that we will use to retrieve the data
                    qr_tar = 'label == "{:s}"'.format(target_tag) + qr_tar_fac1 + qr_tar_fac2
                    qr_ntar = 'label == "{:s}"'.format(nontarget_tag) + qr_ntar_fac1 + qr_ntar_fac2
                    qr_spoof = 'label == "{:s}"'.format(spoofed_tag) +  qr_spoof_fac1 + qr_spoof_fac2
                    
                    # retrive data
                    tar_env_pd = asv_score_pd.query(qr_tar)
                    ntar_env_pd = asv_score_pd.query(qr_ntar)
                    spoof_env_pd = asv_score_pd.query(qr_spoof)
                    
                    tar_data = tar_env_pd[col_score_name].to_numpy()
                    ntar_data = ntar_env_pd[col_score_name].to_numpy()
                    spoof_data = spoof_env_pd[col_score_name].to_numpy()
                    
                    # compute C012 values
                    C0, C1, C2 = eval_wrapper.get_tDCF_C012_from_asv_scores(
                        tar_data, ntar_data, spoof_data, cost_model)
                    # save C012 coef
                    eval_wrapper.save_C012_value(
                        C012, C0, C1, C2, [factor_1, factor_2])
    print("")
    return C012



def compute_decomposed_mintdcf_eer(score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h, 
                                   factor_value_h, 
                                   factor_type_h,
                                   C012_buf = None,
                                   pooled_tag = config.g_pooled_tag, 
                                   bonafide_tag = config.g_bonafide_tag,
                                   spoofed_tag = config.g_spoofed_tag,
                                   col_score_name = config.g_score_col_name,
                                   flag_verbose = False):
    """mintDCF_array, eer_array = compute_decomposed_mintdcf_eer(score_pd, 
                                   factor_name_v,
                                   factor_value_v, 
                                   factor_type_v,
                                   factor_name_h,
                                   factor_value_h,  
                                   factor_type_h,
                                   C012_buf = None,
                                   pooled_tag = 'Pooled', 
                                   bonafide_tag = 'bonafide',
                                   spoofed_tag = 'spoof',
                                   col_score_name = 'score',
                                   flag_verbose = False)
    
    Function to loop over two sets of factors and compute min t-DCF and EER in
    each pair of the factor.

    input
    -----
      score_pd        dataFrame, joint dataframe of CM score and protocol

      factor_name_v   str or list of str, 
                      name(s) of the dataFrame series for the 1st set of factor.
                    
      factor_value_v  list of str, or list of list or str, 
                      values of the 1st set of factors 

                      if type(factor_name_v) is str:
                          # we retrieve the data by
                          for factor in factor_value_v:
                              data = score_pd.query('factor_name_v == "factor"')

                      if type(factor_name_v) is list
                          # we iterate all the factors
                          for factor_name, factor_value in zip(factor_name_v, factor_value_v):
                              for factor in factor_value:
                                  data = score_pd.query('factor_name == "factor"')
 
                    
                      The second case is useful when the 1st set of factors 
                      are defined in different data series of score_pd.

      factor_type_v   str or list of str, type of the factor
                      
                      'spoof': this factor is only available for spoofed data
                      'bonafide': this factor is only available for bonafide data
                      'both': this factor appears in both spoofed and bonafide data

                      if type(factor_name_v) is str:
                          # factor_type_v is the type for factor_name_v
                      if type(factor_name_v) is list:
                          # factor_type_v should be a list and
                          # factor_type_v[i] is the type for factor_name_v[i]

      factor_name_h   str or list of str, 
      factor_value_h  list of str or list of list of str
      factor_type_h   str or list of str
                     
                      these are for the second set of factors

      C012_buf        dict, dictionary of C0, C1, C2 values for each condition
                      we will use load_C012_value(C012_buf, [factor1, factor2])
                      to load the C0, C1, C2 value

                      if C012_buf is None, mintDCF_array will be [np.nan]

      pooled_tag      str, tag for pooled condition, 
                      default 'Pooled'
      bonafide_tag    str, tag for bonafide trials
                      default 'bonafide'
      spooed_tag      str, tag for spoofed trials
                      default 'spoof'
      col_score_name  str, name of the column for score
                      default 'score'

    output
    ------
      mintDCF_array   np.array, min t-DCF values in all conditions
                      mintDCF_array.shape[0] is equal to the length of all
                      possible values in factor_value_v
                      mintDCF_array.shape[1] is equal to the length of all
                      possible values in factor_value_h

      eer_array       np.array, EER values in all conditions.
                      same shape as mintDCF_array
    """
    def _wrap_list(data):
         return [data] if type(data) is str else data

    def _wrap_list_list(data):
         return [data] if type(data[0]) is str else data
    
    # wrap them into a list
    factor_names_1 = _wrap_list(factor_name_v)
    factor_names_2 = _wrap_list(factor_name_h)
    factor_types_1 = _wrap_list(factor_type_v)
    factor_types_2 = _wrap_list(factor_type_h)
    factor_list_1_list = _wrap_list_list(factor_value_v)
    factor_list_2_list = _wrap_list_list(factor_value_h)
        
    # number of rows and columns in the result table
    num_row = sum([len(x) for x in factor_list_1_list])
    num_col = sum([len(x) for x in factor_list_2_list])

    # output buffer
    mintDCF_array = np.zeros([num_row, num_col])
    eer_array = np.zeros_like(mintDCF_array)

    if C012_buf is None:
        print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing EERs')
    else:
        print('\n' + ''.join(['-'] * (num_row - 1)) + '>| computing EERs and min tDCF')

    # loop over factor along the row (factor 1)
    id1 = 0
    for factor_name_1, factor_list_1, factor_type_1 in zip(
        factor_names_1, factor_list_1_list, factor_types_1):
        for _, factor_1 in enumerate(factor_list_1):
            
            print(".", end = '', flush=True)
            
            # creat the query to retrieve the data corresponding to the factor_1
            if factor_1 == pooled_tag:
                # pooled condition
                qr_bona_fac1 = ''
                qr_spoof_fac1 = ''
            elif factor_type_1 == config.g_factor_type_spoof:
                # if factor is only for spoofed data (e.g., attack type)
                qr_bona_fac1 = ''
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            elif factor_type_1 == config.g_factor_type_bonafide:
                # if factor is only for bonafide
                qr_bona_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ''
            else:
                # if factor is for both spoofed and bona fide data (e.g., codec)
                qr_bona_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
                qr_spoof_fac1 = ' and {:s} == "{:s}"'.format(factor_name_1, factor_1)
            
            # loop over factor in cols (factor 2)
            id2 = 0
            for factor_name_2, factor_list_2, factor_type_2 in zip(
                factor_names_2, factor_list_2_list, factor_types_2):
                for _, factor_2 in enumerate(factor_list_2):
                    
                    if factor_2 == pooled_tag:
                        # pooled condition
                        qr_bona_fac2 = ''
                        qr_spoof_fac2 = ''
                    elif factor_type_2 == config.g_factor_type_spoof:
                        # if factor is only for spoofed data (e.g., attack type)
                        qr_bona_fac2 = ''
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                    elif factor_type_2 == config.g_factor_type_bonafide:
                        qr_bona_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ''
                    else:
                        # if factor is for both spoofed and bona fide data (e.g., codec)
                        qr_bona_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)
                        qr_spoof_fac2 = ' and {:s} == "{:s}"'.format(factor_name_2, factor_2)

                    # query that we will use to retrieve the data
                    qr_bonafide = 'label == "{:s}"'.format(bonafide_tag) + qr_bona_fac1 + qr_bona_fac2
                    qr_spoof = 'label == "{:s}"'.format(spoofed_tag) +  qr_spoof_fac1 + qr_spoof_fac2
                    
                    # retrive data
                    bona_env_pd = score_pd.query(qr_bonafide)
                    spoof_env_pd = score_pd.query(qr_spoof)
                    bona_data = bona_env_pd[col_score_name].to_numpy()
                    spoof_data = spoof_env_pd[col_score_name].to_numpy()
                    
                    # load C012 values
                    if C012_buf is None:
                        # dummy value 
                        C0, C1, C2 = 0.1, 0.1, 0.1
                    else:
                        C0, C1, C2 = eval_wrapper.load_C012_value(
                            C012_buf, [factor_1, factor_2])

                    # print infor
                    if flag_verbose:
                        print(qr_bonafide, "{:d} entries".format(len(bona_data)))
                        print(qr_spoof, "{:d} entries".format(len(spoof_data)))
                    
                    # computation
                    if len(bona_data) and len(spoof_data):
                        mintdcf, eer_tmp = eval_wrapper.get_mintDCF_eer(
                            bona_data, spoof_data, C0, C1, C2)
                    else:
                        mintdcf, eer_tmp = np.nan, np.nan

                    # mask the min t-DCF values when C012 is invalid
                    if C012_buf is None:
                        mintdcf = mintdcf * np.nan
                        
                    # save the value
                    mintDCF_array[id1, id2] = mintdcf
                    eer_array[id1, id2] = eer_tmp

                    id2 += 1
            # loop over horizotal factors
            id1 += 1
    print("")
    return mintDCF_array, eer_array


def evaluation_API(cm_score_file, track, subset = 'eval', label_dir = './',
                   flag_recompute_c012 = False,
                   asv_score_file = None, 
                   external_c012_path = None):
    """ mintdcf_array, eer_array = evaluation_API(cm_score_file, 
          track, subset = 'eval', label_dir = './',
          flag_recompute_c012 = True, asv_score_file = None, 
          external_c012_path = None)

    Compute the min tDCF and EER values given a score file.
    The output shares the same format as that on CodaLab page.

    input
    -----
      cm_score_file   str, path to the CM score file
      track           str, 'LA', 'PA', or 'DF'
      subset          str, name of subset, eval, progress, or hidden
      label_dir       str, path to the directory of key and meta labels
                      label_dir is the directory downloaded from ASVspoof.org
                      It should contain the following files
                      \- DF
                         \- CM 
                             |- trial_metadata.txt
                             |- ...
                      \- LA
                         ...
                      \- PA
                         ...
      asv_score_file      str, path to the ASV score file, default None
                          If None, ASV score in label_dir will be loaded

      flag_recompute_c012 bool, whether recompute C012 coef
                          default False

      external_c012_path  str, path to external C012 file
                          If None, path is specified by config.py

    output
    ------
      mintdcf_array   np.array, the numpy array of min t-DCF
      eer_array       np.array, the numpy array of EER
    """
    
    # ===========
    # load configuration for each trakc
    # ===========
    if track == config.g_LA_track:
        config_buf = config.ConfigLA()
    elif track == config.g_PA_track:
        config_buf = config.ConfigPA()
    elif track == config.g_DF_track:
        config_buf = config.ConfigDF()
    else:
        print("ERROR: unknown track: {:s}".format(track))
        return None

    # ===========
    # load CM protocol & score
    # ===========
    protocol_cm_file = os.path.join(label_dir, config_buf.protocol_cm_file)
    protocol_cm_pd = pd_tools.load_protocol(protocol_cm_file, 
                                            names = config_buf.p_names, 
                                            index_col = config_buf.index_col)
    # load score file
    score_cm_pd = pd_tools.load_score(cm_score_file, config_buf.s_names, 
                                      index_col = config_buf.index_col)
    # merge score and protocol into a single dataFrame
    score_cm_pd = pd_tools.join_protocol_score(
        protocol_cm_pd, score_cm_pd[[config_buf.score_col]])

    # ===========
    # select the subset
    # ===========
    if track == config.g_PA_track and subset in config_buf.hidden:
        # special for PA
        subset_query = config_buf.hidden[subset]
    else:
        # other cases
        subset_query = '{:s} == "{:s}"'.format(config_buf.subset_col, subset)
    # get the evaluation subset data frame
    tmp_score_cm_pd = score_cm_pd.query(subset_query)


    # ===========
    # on C012
    # ===========
    
    # specify the path C012 dictionary
    if external_c012_path is None or len(external_c012_path) == 0:
        # use pre-computed C012
        c012_file = os.path.join(label_dir, config_buf.c012_file[subset])
    else:
        c012_file = external_c012_path
        
        
    # compute C012 if necessary
    if config_buf.flag_tDCF and flag_recompute_c012:
        # protocol ASV
        protocol_asv_file = os.path.join(label_dir, config_buf.protocol_asv_file)
        if not os.path.isfile(protocol_asv_file):
            print("Cannot find ASV protocol {:s}".format(protocol_asv_file))
            sys.exit(1)
        # score ASV
        if asv_score_file is None or len(asv_score_file) == 0:
            # use pre-computed ASV score
            asv_score_file = os.path.join(label_dir, config_buf.pre_score_asv_file)
        if not os.path.isfile(asv_score_file):
            print("Cannot find ASV score file {:s}".format(asv_score_file))
            sys.exit(1)

        print("===============\nCompute C012 coef\n===============")
        protocol_asv_pd = pd_tools.load_protocol(protocol_asv_file, 
                                                 names = config_buf.p_names_asv)
        # load score file
        asv_score_pd = pd_tools.load_score(
            asv_score_file, config_buf.s_names_asv)
        # merge score and protocol into a single dataFrame
        asv_score_pd = pd_tools.join_protocol_score(
            protocol_asv_pd, asv_score_pd[[config_buf.score_col]])

        # get the evaluation subset data frame
        tmp_asv_score_pd = asv_score_pd.query(subset_query)
            
        C012_buf = compute_tDCF_C012(tmp_asv_score_pd, 
                                     config_buf.factor_name_1, 
                                     config_buf.factor_1_list, 
                                     config_buf.factor_1_type,
                                     config_buf.factor_name_2, 
                                     config_buf.factor_2_list, 
                                     config_buf.factor_2_type)
        eval_wrapper.dump_C012_dict(C012_buf, c012_file)
        print("Save C012 coef to {:s}".format(c012_file))
            
    
    # load C012 dictionary
    if config_buf.flag_tDCF:
        if not os.path.isfile(c012_file):
            print("Cannot find C012 file {:s}".format(c012_file))
            sys.exit(1)
        print("=============== \nCompute EERs, min tDCFs\n===============")
        print("Load C012 coeffs from {:s}".format(c012_file))
        C012_buf = eval_wrapper.load_C012_dict(c012_file)        
    else:
        print("========== \nCompute EERs\n==========")
        print("Track without considering ASV")
        C012_buf = None
    
    # ===========        
    # compute min tDCF and EERs
    # ===========
    mintdcf_array, eer_array = compute_decomposed_mintdcf_eer(
        tmp_score_cm_pd, 
        config_buf.factor_name_1, 
        config_buf.factor_1_list, 
        config_buf.factor_1_type,
        config_buf.factor_name_2, 
        config_buf.factor_2_list, 
        config_buf.factor_2_type,
        C012_buf = C012_buf,
        pooled_tag = config_buf.pooled_tag, 
        col_score_name = config_buf.score_col,
        flag_verbose = False)
    
    # ===========
    # print results
    # ===========
    print("\n\n")
    # print min tDCF table
    if C012_buf is not None:
        print("\n===============\nTable for min tDCFs\n===============\n")
        table_API.print_table(mintdcf_array, 
                    config_buf.factor_2_tag_list, 
                    config_buf.factor_1_tag_list, 
                    print_format = "1.4f", 
                    with_color_cell = True,
                    print_latex_table=True, 
                    print_text_table=True);

    # print EER table
    print("\n===============\nTable for EERs\n===============\n")
    table_API.print_table(eer_array * 100, 
                config_buf.factor_2_tag_list, 
                config_buf.factor_1_tag_list, 
                print_format = "1.2f", 
                with_color_cell = True,
                print_latex_table=True, 
                print_text_table=True);

    return mintdcf_array, eer_array




if __name__ == "__main__":
    
    # ====
    # parse argument
    # ====
    args = parse_argument()
        
    # ===
    # compute
    # ===
    # compute
    min_tdcfs, eers = evaluation_API(
        args.cm_score_file, 
        args.track, 
        args.subset, 
        args.metadata, 
        args.recompute_c012, 
        args.asv_score_file, 
        args.c012_path)
    
    print("Please scroll up and check the results.")
    
    
