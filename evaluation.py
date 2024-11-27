import sys, os
import numpy as np

def calculate_EER(cm_scores_file, output_file, printout=True):
    """Calculate EER for CM scores without t-DCF or ASV scores."""
    # Load CM scores
    cm_data = np.genfromtxt(cm_scores_file, dtype=str)
    cm_sources = cm_data[:, 1]
    cm_keys = cm_data[:, 2]
    cm_scores = cm_data[:, 3].astype(np.float64)

    # Extract bona fide and spoof scores
    bona_cm = cm_scores[cm_keys == 'bonafide']
    spoof_cm = cm_scores[cm_keys == 'spoof']

    # Compute EER
    eer_cm = compute_eer(bona_cm, spoof_cm)[0]

    # Compute EER breakdown by attack type
    attack_types = [f'A{_id:02d}' for _id in range(7, 20)]
    eer_cm_breakdown = {}
    for attack_type in attack_types:
        spoof_cm_attack = cm_scores[(cm_keys == 'spoof') & (cm_sources == attack_type)]
        if len(spoof_cm_attack) > 0:
            eer = compute_eer(bona_cm, spoof_cm_attack)[0]
        else:
            eer = 0.0  # No samples for this attack type
        eer_cm_breakdown[attack_type] = eer

    # Write results to the output file
    if printout:
        with open(output_file, "w") as f_res:
            f_res.write('\nCM SYSTEM\n')
            f_res.write('\tEER\t\t= {:8.9f} %\n'.format(eer_cm * 100))

            f_res.write('\nBREAKDOWN CM SYSTEM\n')
            for attack_type in attack_types:
                _eer = eer_cm_breakdown[attack_type] * 100
                f_res.write(
                    f'\tEER {attack_type}\t= {_eer:8.9f} %\n'
                )
        os.system(f"cat {output_file}")

    return eer_cm * 100
    

def compute_det_curve(target_scores, nontarget_scores):
    # Combine scores and labels
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))
    
    # Sort the scores and associated labels
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]
    
    # Cumulative sums for FRR and FAR calculations
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, len(all_scores) + 1) - tar_trial_sums)
    
    # Calculate FRR and FAR
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))
    
    # Thresholds correspond to the sorted scores
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))
    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]