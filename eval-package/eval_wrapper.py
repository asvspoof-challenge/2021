#!/usr/bin/env python
"""
Wrapper functions over eval_metrics

"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import numpy as np
import eval_metrics as em

__author__ = "ASVspoof consortium"
__copyright__ = "Copyright 2022, ASVspoof consortium"

#=================
# Helper functions
#=================

def dump_C012_dict(data_dict, filepath):
    np.array(data_dict).dump(filepath)
    return 

def load_C012_dict(filepath):
    return dict(np.load(filepath, allow_pickle=True).tolist())

#=================
# Wrappers
#=================

def load_C012_value(C012_buf, factor_list, fail_value = np.nan):
    """ C0, C1, C2 = load_C012_value(C012_buf, factor_list)
    input
    -----
      C012_buf     dictionary, value of C012 in dictionary
      factor_list  list of str, list of factors to retrive the value.
                   value is given by C012_buf[factor_list[0]][factor_list[1]]...
    output
    ------
      C0           scalar
      C1           scalar
      C2           scalar
    """
    tmp = C012_buf
    try:
        for factor in factor_list:
            tmp = tmp[factor]
        C0, C1, C2 = tmp['C0'], tmp['C1'], tmp['C2']
    except KeyError:
        # cannot load C012 from the dictionary
        C0, C1, C2 = np.nan, np.nan, np.nan
    return C0, C1, C2

def save_C012_value(C012_buf, C0, C1, C2, factor_list):
    """
    """
    if not type(C012_buf) is dict:
        print("C012_buf is not a dictionary")
        sys.exit(1)
    else:
        tmp = C012_buf
        try:
            for factor in factor_list:
                if not factor in tmp:
                    tmp[factor] = dict()
                tmp = tmp[factor]
            tmp['C0'], tmp['C1'], tmp['C2'] = C0, C1, C2
        except KeyError:
            print("Fail to push C012 to dictionary")
            sys.exit(1)
    return


def load_asv_metrics(tar_asv, non_asv, spoof_asv):
    """ Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
           tar_asv, non_asv, spoof_asv)
    input
    -----
      tar_asv    np.array, score of target speaker trials
      non_asv    np.array, score of non-target speaker trials
      spoof_asv  np.array, score of spoofed trials
    
    output
    ------
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pmiss_spoof_asv   scalar, 
      P_fa_spoof_asv    scalar
    """
    # 
    eer_asv, asv_threshold = em.compute_eer(tar_asv, non_asv)
    
    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = em.obtain_asv_error_rates(
        tar_asv, non_asv, spoof_asv, asv_threshold)
    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def get_eer(bonafide_score_cm, spoof_score_cm):
    """ eer_val, threshold = get_eer(bonafide_score_cm, spoof_score_cm)

    input
    -----
      bonafide_score_cm np.array, score of bonafide data
      spoof_score_cm    np.array, score of spoofed data
    
    output
    ------
      eer_val           scalar, value of EER
      threshold         scalar, value of the threshold corresponding to EER
    """
    eer_val, threshold = em.compute_eer(bonafide_score_cm, spoof_score_cm)
    return eer_val, threshold


def get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model):
    """C0, C1, C2 = get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)
    
    compute_tDCF can be factorized into two parts: 
    C012 computation and min t-DCF computation.

    This is for C012 computation.
    
    input
    -----
      Pfa_asv           scalar, value of ASV false accept rate
      Pmiss_asv         scalar, value of ASV miss rate
      Pmiss_spoof_asv   scalar, 
      P_fa_spoof_asv    scalar
      
    output
    ------
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    For C0, C1, C2, see Appendix Eqs.(1-2) in evaluation plan [1],
    or Eqs.(10-11) in [2]

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """
    
    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')
        sys.exit(1)

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or \
       cost_model['Pspoof'] < 0 or \
       np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        print('ERROR: Your prior probabilities should be positive and sum up to one.')
        sys.exit(1)

    # Unless we evaluate worst-case model, 
    # we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        print('ERROR: please provide false alarm rate of spoof tests against ASV system.')
        sys.exit(1)

    
    # Constants - see ASVspoof 2019/21 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv \
         + cost_model['Pnon'] * cost_model['Cfa'] *Pfa_asv
    
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] \
         - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv \
            + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;
    
    return C0, C1, C2

def get_tDCF_C012_from_asv_scores(tar_asv, non_asv, spoof_asv, cost_model):
    """ C0, C1, C2 = get_tDCF_C012_from_asv_scores(tar_asv, non_asv, spoof_asv, cos_model)

    Wrapper combining load_asv_metrics and get_tDCF_C012.

    input
    -----
      tar_asv    np.array, score of target speaker trials
      non_asv    np.array, score of non-target speaker trials
      spoof_asv  np.array, score of spoofed trials

    output
    ------
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    For C0, C1, C2, see Appendix Eqs.(1-2) in evaluation plan [1],
    or Eqs.(10-11) in [2]

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """
    # here we only consider the case where tar, nontarget, and spoof scores
    # are all avaialble
    if len(tar_asv) and len(non_asv) and len(spoof_asv):
        # compute ASV metrics
        Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv = load_asv_metrics(
            tar_asv, non_asv, spoof_asv)

        # get the C012 values
        C0, C1, C2 = get_tDCF_C012(Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model)
    else:
        C0, C1, C2 = np.nan, np.nan, np.nan

    return C0, C1, C2

def get_mintDCF_eer(bonafide_score_cm, spoof_score_cm, C0, C1, C2):
    """ mintDCF, eer = get_mintDCF_eer(bonafide_score_cm, 
                                       spoof_score_cm, C0, C1, C2)
    
    compute_tDCF can be factorized into two parts: 
    C012 computation and min t-DCF computation.

    This is for min t-DCF computation, given the values of C012
    
    input
    -----
      bonafide_score_cm  np.array, score of bonafide data
      spoof_score_cm     np.array, score of spoofed data
      C0                 scalar, coefficient for min tDCF computation
      C1                 scalar, coefficient for min tDCF computation
      C2                 scalar, coefficient for min tDCF computation
    
    output
    ------
      eer                scalar, value of EER
      mintDCF            scalar, value of min tDCF

    For C0, C1, C2, see Appendix Eqs.(1-2) in evaluation plan [1],
    or Eqs.(10-11) in [2]

    References:

      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).

      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf

    """
    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = em.compute_det_curve(
        bonafide_score_cm, spoof_score_cm)
    
    # =====
    # tDCF
    # =====
    if np.isnan(C0) or np.isnan(C1) or np.isnan(C2): 
        # this is a case where 
        mintDCF = np.nan
    else:
        # tDCF values
        tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm
        # Obtain default t-DCF
        tDCF_default = C0 + np.minimum(C1, C2)
        # Normalized t-DCF
        tDCF_norm = tDCF / tDCF_default
        # min t-DCF
        mintDCF = tDCF_norm[tDCF_norm.argmin()]

    # ====
    # EER
    # ====
    abs_diffs = np.abs(Pmiss_cm - Pfa_cm)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((Pmiss_cm[min_index], Pfa_cm[min_index]))

    return mintDCF, eer


if __name__ == "__main__":
    print("eval_wrapper")
