from chemprop import data, featurizers
from tqdm import tqdm
import multiprocessing
import torch
import argparse
import os
import math

def readData(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            smi, score = line.split(',')
            data.append((smi, float(score)))
    return data

def getMoleculeDataPoint(inputs):
    smi, y = inputs
    datapoint = data.MoleculeDatapoint.from_smi(smi,[y])
    return datapoint


def createTestSet(folder, n_workers, batch_size):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    all_data = readData(f"{folder}/sampled_data.csv")

    total_batches = math.ceil(len(all_data) / batch_size)
    for batch_index in range(total_batches):
        print(f'Processing batch {batch_index}')
        start = batch_index * batch_size
        end = min((batch_index + 1) * batch_size, len(all_data))

        if n_workers is None:
            batch_data = [getMoleculeDataPoint(i) for i in tqdm(all_data[start:end])]
        else:
            with multiprocessing.Pool(n_workers) as p:
                batch_data = list(tqdm(p.imap(getMoleculeDataPoint, all_data[start:end], chunksize=10000)))

        dset = data.MoleculeDataset(batch_data, featurizer)

        torch.save(dset, f"{folder}/test_batch_{batch_index}_dset.torch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_set_folder", required=True, help="Folder containing the sampled data for the test set")
    parser.add_argument("--n_workers", default=None, type=int, help="Number of workers to use to do featurization")
    parser.add_argument("--batch_size", default=1_000_000, type=int, help="Number of molecules in a single dataset batch")
    args = parser.parse_args()

    if not os.path.exists(args.test_set_folder):
        raise ValueError("Input folder does not exist")

    if not os.path.exists(f"{args.test_set_folder}/sampled_data.csv"):
        raise ValueError("Input folder does not contain sampled_data.csv")

    if args.n_workers is None:
        print("n_workers not passed - everything done sequentially")

    createTestSet(args.test_set_folder, args.n_workers, args.batch_size)
