#!/usr/bin/env python
"""
configuration for each track

Configurations are fixed. They follow those used on the CodaLab platforms.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys

# ==========
# global value
# ==========
 
g_factor_type_spoof = 'spoof'
g_factor_type_bonafide = 'bonafide'
g_factor_type_both = 'both'

g_bonafide_tag = 'bonafide'
g_spoofed_tag = 'spoof'
g_score_col_name = 'score'
g_pooled_tag = 'Pooled'

g_target_tag = 'target'
g_nontarget_tag = 'nontarget'

g_LA_track = 'LA'
g_DF_track = 'DF'
g_PA_track = 'PA'

g_possible_subsets = ['eval', 'progress', 'hidden', 'hidden1_PA', 'hidden2_PA']
g_possible_tracks = ['LA', 'DF', 'PA']

# ==========
# t-DCF configs 
# ==========

# Configurations used in ASVspoof 2021 official ranking
# See more in https://www.asvspoof.org/resources/tDCF_python_v2.zip
# 
Pspoof = 0.05
cost_model = {
    'Pspoof': Pspoof,  # Prior probability of a spoofing attack
    'Ptar': (1 - Pspoof) * 0.99,  # Prior probability of target speaker
    'Pnon': (1 - Pspoof) * 0.01,  # Prior probability of nontarget speaker
    'Cmiss': 1,  # Cost of tandem system falsely rejecting target speaker
    'Cfa': 10,  # Cost of tandem system falsely accepting nontarget speaker
    'Cfa_spoof': 10,  # Cost of tandem system falsely accepting spoof
}


# ==========
# track configs 
# ==========

class ConfigLA:
    """Configuration to load and parse LA track protocol and score file
    """
    def __init__(self):

        self.pooled_tag = 'Pooled'        
        self.subset_col = 'subset'
        self.score_col = 'score'
        self.index_col = 'trial'
                
        # =====
        # Configuration to load CM protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names = ['spk', self.index_col, 'codec', 'trans', 
                        'attack', 'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]
        
        # CM protocol path
        self.protocol_cm_file = 'LA/CM/trial_metadata.txt'


        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = ['spk', self.index_col, 'codec', 'trans', 
                            'attack', 'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names_asv = ['asv_spk', self.index_col, self.score_col]

        # ASV protocol path
        self.protocol_asv_file = 'LA/ASV/trial_metadata.txt'
        # ASV score by organizers
        self.pre_score_asv_file = 'LA/ASV/ASVTorch_Kaldi/score.txt'
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = True
        
        
        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': 'LA/LA-C012-eval.npy',
                          'progress': 'LA/LA-C012-prog.npy',
                          'hidden': 'LA/LA-C012-hidden.npy'}

        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====
        # 1st group of factor
        # name of the data series in protocol dataframe
        self.factor_name_1 = 'attack'
        # value of the factor to be considered
        self.factor_1_list =  ['A07', 'A08', 'A09', 'A10', 'A11', 'A12', 'A13', 
                               'A14', 'A15', 'A16', 'A17', 'A18', 'A19', self.pooled_tag]
        # type of the factor (spoofed only? bonafide only? or both)
        self.factor_1_type = g_factor_type_spoof
        # string of factors to be printed
        self.factor_1_tag_list = self.factor_1_list

        # 2nd group of factor
        self.factor_name_2 = 'codec'
        self.factor_2_list = ['none', 'alaw', 'pstn', 'g722', 'ulaw', 'gsm', 'opus', self.pooled_tag]
        self.factor_2_type = g_factor_type_both
        self.factor_2_tag_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', self.pooled_tag]
        return


class ConfigPA:
    """Configuration to load and parse PA track protocol and score file
    """
    def __init__(self):

        self.index_col = 'trial'
        self.subset_col = 'subset'
        self.pooled_tag = 'Pooled'
        self.score_col = 'score'

        # =====
        # Configuration to load CM protocol and score file
        # =====
        # PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 spoof notrim eval
        # name of data series for procotol file
        self.p_names = ['spk', self.index_col, 'asv_room', 'asv_mic', 'dis_to_asv', 
                        'att_room', 'att_mic', 'att_d', 'att_to_spk', 
                        'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]

        # CM protocol path
        self.protocol_cm_file = 'PA/CM/trial_metadata.txt'


        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = ['spk', self.index_col, 'asv_room', 'asv_mic', 'dis_to_asv', 
                            'att_room', 'att_mic', 'att_d', 'att_to_spk', 
                            'label', 'trim', 'subset']
        # name of data series for score file
        self.s_names_asv = ['asv_spk', self.index_col, self.score_col]

        # ASV protocol path
        self.protocol_asv_file = 'PA/ASV/trial_metadata.txt'
        # ASV score by organizers
        self.pre_score_asv_file = 'PA/ASV/ASVTorch_Kaldi/score.txt'
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = True


        # =====
        # special for PA hidden track, we have two
        # =====        
        # hidden subset 1
        self.hidden = {'hidden1_PA': 'trim == "notrim" and subset == "hidden"',
                       'hidden2_PA': 'trim == "trim" and subset == "hidden"'}

        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': 'PA/PA-C012-eval.npy',
                          'progress': 'PA/PA-C012-prog.npy',
                          'hidden1_PA': 'PA/PA-C012-hidden1.npy',
                          'hidden2_PA': 'PA/PA-C012-hidden2.npy'}


        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====        
        # 1st group of factor
        # name of the data series in protocol dataframe
        # we will concatenate multiple factors into one group
        # dummy is a placeholder where we store the value for pooled condition
        self.factor_name_1 = ['asv_room', 'asv_mic', 'dis_to_asv', 'dummy']
        # value of the factor to be considered
        self.factor_1_list =  [['R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9'], 
                               ['M1', 'M2', 'M3'], 
                               ['D1', 'D2', 'D3', 'D4', 'D5', 'D6'], 
                               [self.pooled_tag]]
        # type of the factor (spoofed only? bonafide only? or both)
        self.factor_1_type = [g_factor_type_both, g_factor_type_both, 
                              g_factor_type_bonafide,  g_factor_type_both]
        # string of factors to be printed
        self.factor_1_tag_list = [item for sublist in self.factor_1_list for item in sublist]


        # 2nd group of factor
        self.factor_name_2 = ['att_room', 'att_mic', 'att_to_spk', 'att_d', 'dis_to_asv', 'dummy']
        self.factor_2_list = [['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'],
                              ['m1', 'm2', 'm3'],
                              ['c2', 'c3', 'c4'],
                              ['s2', 's3', 's4'],
                              ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
                              [self.pooled_tag]]
        self.factor_2_type = [g_factor_type_spoof, g_factor_type_spoof,
                              g_factor_type_spoof, g_factor_type_spoof,
                              g_factor_type_spoof, g_factor_type_both]
        self.factor_2_tag_list = [item for sublist in self.factor_2_list for item in sublist]

        return

class ConfigDF:
    """Configuration to load and parse LA track protocol and score file
    """
    def __init__(self):
       
        self.index_col = 'trial'
        self.subset_col = 'subset'
        self.pooled_tag = 'Pooled'
        self.score_col = 'score'

        # =====
        # Configuration to load CM protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names = ['speaker', self.index_col, 'compr', 'source', 'attack',
                        'label', 'trim', 'subset', 'vocoder', 
                        'task', 'team', 'gender-pair', 'language']
        # name of data series for score file
        self.s_names = [self.index_col, self.score_col]
        
        # Path to the CM protocol file
        self.protocol_cm_file = 'DF/CM/trial_metadata.txt'
        
        # =====
        # Configuration to load ASV protocol and score file
        # =====
        # name of data series for procotol file
        self.p_names_asv = []
        self.s_names_asv = []
        # ASV protocol
        self.protocol_asv_file = ''
        self.pre_score_asv_file = ''
        # flag, whether tDCF is applicable to this track 
        self.flag_tDCF = False

        # =====
        # C012 for tDCF computation
        # =====
        # C012 buffer
        self.c012_file = {'eval': '',
                          'progress': '',
                          'hidden': ''}

        
        # =====
        # Factors over which the EERs and min t-DCF values are computed
        # =====
        # 1st group of factor
        # name of the data series in protocol dataframe
        self.factor_name_1 = 'vocoder'
        self.factor_1_list =  ['traditional_vocoder', 
                               'waveform_concatenation', 
                               'neural_vocoder_autoregressive', 
                               'neural_vocoder_nonautoregressive', 
                               'unknown', self.pooled_tag]
        self.factor_1_type = g_factor_type_spoof
        self.factor_1_tag_list = ['Traditional', 'Wav.Concat.', 'Neural AR', 
                                  'Neural non-AR', 'Unknown', self.pooled_tag]

        self.factor_name_2 = 'compr'
        self.factor_2_list = ['nocodec',  'low_mp3', 'high_mp3', 'low_m4a', 
                              'high_m4a', 'low_ogg', 'high_ogg', 'mp3m4a', 
                              'oggm4a', self.pooled_tag]
        self.factor_2_type = g_factor_type_both
        self.factor_2_tag_list = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 
                                  'C7', 'C8', 'C9', self.pooled_tag]
        
        return


if __name__ == "__main__":
    print("Configurations")
