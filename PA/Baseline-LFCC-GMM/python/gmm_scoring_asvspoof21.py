from gmm import scoring


# scores file to write
scores_file = 'scores-lfcc-asvspoof21-PA.txt'

# configs
features = 'lfcc'
dict_file = 'gmm_lfcc_asvspoof21_pa.pkl'

db_folder = '/path/to/ASVspoof_root/'  # put your database root path here
eval_folder = db_folder + 'PA/ASVspoof2021_PA_eval/flac/'
eval_ndx = db_folder + 'PA/ASVspoof2021_PA_cm_protocols/ASVspoof2021.PA.cm.eval.trl.txt'

audio_ext = '.flac'

# run on ASVspoof 2021 evaluation set
scoring(scores_file=scores_file, dict_file=dict_file, features=features,
        eval_ndx=eval_ndx, eval_folder=eval_folder, audio_ext=audio_ext,
        features_cached=True)
