import graphlab as gl

# load training data
training_sframe = gl.SFrame.read_csv('training.csv')
if training_sframe['Label']  == 'b':
	training_sframe['Label']  = 0;
else:
	training_sframe['Label']  = 1;

SOLVER_OPTIONS = {
	'convergence_threshold': 1e-2,
	'max_iterations': 100,
	'lbfgs_memory_level': 10
}
# train a model
features = ['DER_mass_MMC','DER_mass_transverse_met_lep','DER_mass_vis','DER_pt_h','DER_deltaeta_jet_jet','DER_mass_jet_jet','DER_prodeta_jet_jet','DER_deltar_tau_lep','DER_pt_tot','DER_sum_pt','DER_pt_ratio_lep_tau','DER_met_phi_centrality','DER_lep_eta_centrality','PRI_tau_pt','PRI_tau_eta','PRI_tau_phi','PRI_lep_pt','PRI_lep_eta','PRI_lep_phi','PRI_met','PRI_met_phi','PRI_met_sumet','PRI_jet_num','PRI_jet_leading_pt','PRI_jet_leading_eta','PRI_jet_leading_phi','PRI_jet_subleading_pt','PRI_jet_subleading_eta','PRI_jet_subleading_phi','PRI_jet_all_pt']
m = gl.svm.create(training_sframe,
                            features=features,solver_options= SOLVER_OPTIONS,
                            target = 'Label')

# predict on test data
test_sframe = gl.SFrame.read_csv('test.csv')
prediction = m.predict(test_sframe)

def make_submission(prediction, filename):
    with open(filename, 'w') as f:
        f.write('Label\n')
        submission_strings =  prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')

make_submission(prediction, 'submission.txt')
