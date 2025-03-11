import argparse
import os
import pickle

def readSmiles(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            smi, _ = line.split(',')
            data.append(smi)
    return data

def findSampledDataDirs(input_folder, filename="sampled_data.csv"):
    matching_dirs = []
    for dirpath, _, filenames in os.walk(input_folder):
        if filename in filenames:
            matching_dirs.append(dirpath)
    return matching_dirs


def findTestOverlap(test_set_folder, model_folders):
    print('Reading in test set')
    test_smiles = readSmiles(f"{test_set_folder}/sampled_data.csv")

    for folder in model_folders:
        print(f'Finding overlap in {folder}')
        train_val_smiles = set(readSmiles(f"{folder}/sampled_data.csv"))

        test_overlap = [i for i, smi in enumerate(test_smiles) if smi in train_val_smiles]

        with open(f"{folder}/test_set_overlap_indices.pkl", "wb") as f:
            pickle.dump(test_overlap, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set_folder", required=True, help="Folder containing the sampled data for the test set")
    parser.add_argument("--model_folders", required = True, help="The most outside folder where all the 'leaf' directories have the training sets for the models")
    args = parser.parse_args()

    if not os.path.exists(args.model_folders):
        raise ValueError("model_folders does not exist")
    
    if not os.path.exists(args.test_set_folder):
        raise ValueError("Test Set folder does not exist")

    sampled_data_dirs = findSampledDataDirs(args.model_folders)

    findTestOverlap(args.test_set_folder, sampled_data_dirs)