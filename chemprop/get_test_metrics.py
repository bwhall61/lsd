import argparse
import os
import numpy as np
from glob import glob
import pickle
import scipy.stats as stats

def readScores(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            _, score = line.split(',')
            data.append(float(score))
    return np.array(data)


def get_predicted_values(test_pred_folder):
    predictions = []

    test_pred_files = glob(f"{test_pred_folder}/test_preds_batch_*_subbatch_*.npy")
    # Ensure the predictions are read in the correct batch,subbatch order
    test_pred_files.sort(key=lambda x: (
        int(os.path.splitext(x)[0].split('_')[-3]), 
        int(os.path.splitext(x)[0].split('_')[-1])
    ))


    for pred_file in test_pred_files:
        predictions.append(np.load(pred_file))

    return np.concatenate(predictions, axis=0)

# Combined function for calculating R², Spearman, and enrichment plots
def calculate_metrics_and_enrichment(ground_truth, predictions, test_indices_to_filter, save_frequency):

    # Make sure ground truth and predictions are the same size
    if len(ground_truth) != len(predictions):
        print("Ground truth and predictions are different lengths! Skipping")
        return None

    # Filter out examples in test set that were used in training
    if test_indices_to_filter:
        mask = np.ones(len(ground_truth), dtype=bool)
        mask[test_indices_to_filter] = False
        ground_truth = ground_truth[mask]
        predictions = predictions[mask]


    # Get the overall pearson correlation
    overall_pearson, _ = stats.pearsonr(ground_truth, predictions)

    # Sort ground truth and predictions once
    sorted_indices_true = np.argsort(ground_truth)
    sorted_indices_pred = np.argsort(predictions)
        
    # Calculate Pearson correlation and enrichment plots for the best 0.01%, 0.1%, and 1% of true values
    percentages = [0.0001, 0.001, 0.01]
    metrics = {percent: {'pearson':None, 'x':None, 'y':None} for percent in percentages}
    for percent in percentages:
        num_best_points = int(len(ground_truth) * percent)
        best_true_indices = sorted_indices_true[:num_best_points]
        best_true_values = ground_truth[best_true_indices]
        corresponding_pred_values = predictions[best_true_indices]
    
        if len(best_true_values) < 2:
            continue

        # Pearson just for the best molecules
        pearson, _ = stats.pearsonr(best_true_values, corresponding_pred_values)
        metrics[percent]['pearson'] = pearson

        actives_indices = set(best_true_indices)
        x = []
        y = []
        discovered_actives = 0
        num_traversed = 0
        for pred_idx in sorted_indices_pred:
            num_traversed += 1
            if pred_idx in actives_indices:
                discovered_actives += 1

            if num_traversed % save_frequency == 0:
                x.append(num_traversed)
                y.append(discovered_actives / num_best_points)

        metrics[percent]['x'] = x
        metrics[percent]['y'] = y

        
    return overall_pearson, metrics


def findModelDirs(input_folder, filename="best_checkpoint.ckpt"):
    matching_dirs = []
    for dirpath, _, filenames in os.walk(input_folder):
        if filename in filenames:
            matching_dirs.append(dirpath)
    return matching_dirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_set_folder', required=True, help="Folder containing the sampled data for the test set")
    parser.add_argument("--model_folders", required=True, help="The most outside folder where all the 'leaf' directories have the training sets for the models")
    parser.add_argument("--skip_test_filtering", required=False, type=bool, default=False, help="Can skip filtering out test set + train/val overlap")
    parser.add_argument("--enrichment_save_frequency", type=int, default=100, help="How often to save the enrichment plot values")
    args = parser.parse_args()


    if not os.path.exists(args.model_folders):
        raise ValueError("model_folders does not exist")
    
    if not os.path.exists(args.test_set_folder):
        raise ValueError("Test Set folder does not exist")

    ground_truth_scores = readScores(f"{args.test_set_folder}/sampled_data.csv")

    model_dirs = findModelDirs(args.model_folders)

    if len(model_dirs) == 0:
        raise ValueError("Could not find any trained models in the model_dirs")

    for folder in model_dirs:
        print(f"Processing {folder}")
        if not os.path.exists(f"{folder}/test_preds"):
            print(f"{folder} does not contain test predictions, skipping")
            continue

        predictions = get_predicted_values(f"{folder}/test_preds")

        if args.skip_test_filtering:
            test_indices_to_filter = None
        else:
            if not os.path.exists(f"{folder}/test_set_overlap_indices.pkl"):
                print(f"{folder} does not contain test set overlap, skipping this folder")
                print("You can skip test set filtering by setting --ski_test_filtering=True")
                continue
            with open(f"{folder}/test_set_overlap_indices.pkl", "rb") as f:
                test_indices_to_filter = pickle.load(f)

        all_metrics = calculate_metrics_and_enrichment(ground_truth_scores, predictions, test_indices_to_filter, args.enrichment_save_frequency)
        if all_metrics is None:
            continue

        with open(f"{folder}/test_metrics.pkl", "wb") as f:
            pickle.dump(all_metrics, f)





