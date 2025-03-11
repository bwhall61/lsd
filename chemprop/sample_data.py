import argparse
import os
import glob
import random
from tqdm import tqdm

def read_file(f, skip=True):
    file_lines = []
    with open(f, 'r') as f:
        if skip:
            next(f)
        for line in f:
            try:
                smiles, _, score = line.split(',')
            except:
                raise ValueError(f"Input file {f} is incorrectly formatted. Expecting: smiles, ID, score")
            file_lines.append((smiles, float(score)))
    return file_lines


def doSample(all_data, strategy, sample_size, top_cutoff, stratified_fraction, seed):
    if seed is not None:
        random.seed(seed)

    if strategy == 'random':
        sample = random.sample(all_data, sample_size)
    elif strategy == 'stratified':
        num_top = int(stratified_fraction * sample_size)
        num_rest = int((1-stratified_fraction) * sample_size)

        sample_top = random.sample(all_data[:int(top_cutoff/100*len(all_data))], num_top)
        sample_rest = random.sample(all_data[int(top_cutoff/100*len(all_data)):], num_rest)

        sample = sample_top + sample_rest
    elif strategy == 'top':
        sample = random.sample(all_data[:int(top_cutoff/100*len(all_data))], sample_size)

    return sample

def sampleData(smiles_folder, output_folder, num_reps, sampling_strategies, num_to_sample, top_cutoff, stratified_fraction, skip, seed):
    # Read the input data
    all_data = []
    for file in tqdm(os.listdir(smiles_folder), desc="Reading input files"):
        all_data.extend(read_file(f"{smiles_folder}/{file}", skip))

    # If we want stratified or top sampling, do the sorting once
    if 'top' or 'stratified' in sampling_strategies:
        print('Sorting Data')
        all_data = sorted(all_data, key=lambda x:x[-1])

    # Do the sampling
    for strategy in sampling_strategies:
        for sample_size in num_to_sample:
            for rep in range(num_reps):
                os.makedirs(f"{output_folder}/{strategy}/{sample_size}/{rep}", exist_ok=True)
                print(f"Sampling {strategy} {sample_size} rep {rep}")
                sample = doSample(all_data, strategy, sample_size, top_cutoff, stratified_fraction, seed)
                with open(f"{output_folder}/{strategy}/{sample_size}/{rep}/sampled_data.csv", "w") as f:
                    for smi, score in sample:
                        f.write(f"{smi},{score}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_folder", required=True, help="Folder containing the smiles and score data")
    parser.add_argument("--output_folder", required=True, help="Where to put all of the sampled data")
    parser.add_argument("--num_reps", required=True, type=int, help="How many replicates to do")
    parser.add_argument("--sampling_strategies", nargs="+", required=True, help="Which sampling strategies to use to get data")
    parser.add_argument("--num_to_sample", nargs="+", required=True, type=int, help="How many molecules to sample")
    parser.add_argument("--top_cutoff", type=float, help="What percent cutoff to use for top and stratified sampling. 1 is top 1%, 0.1 is top 0.1%, etc")
    parser.add_argument("--stratified_fraction",
                        type = float, 
                        help="For stratified sampling, what percent to sample from the top scoring. The remaining will be sampled\
                              from the rest. Ex 0.8 will do 80% from the top scores and 20% from the rest")
    parser.add_argument("--skip_one_line", default=True, type=bool, help="Whether to skip the first line of the input files")
    parser.add_argument("--seed", type=int, help="Seed for reproducible sampling")
    args = parser.parse_args()

    # Do some checks
    if not os.path.exists(args.smiles_folder):
        raise ValueError("Input folder containing smiles and scores does not exist")

    if os.path.exists(args.output_folder):
        print('Output folder already exists, overwriting any previously sampled sets')
    else:
        os.makedirs(args.output_folder)

    if args.num_reps <= 0 :
        raise ValueError("Must have at least a single replicate")

    for strategy in args.sampling_strategies:
        if strategy not in ['random', 'top', 'stratified']:
            raise ValueError("Unrecognized sampling strategy")

    if "stratified" in args.sampling_strategies:
        if args.stratified_fraction is None:
            raise ValueError("Stratified sampling requires setting stratified_fraction")
        if args.stratified_fraction < 0 or args.stratified_fraction > 1:
            raise ValueError("stratified_fraction should be between 0 and 1")
    
    if args.top_cutoff is None:
        if 'top' in args.sampling_strategies or 'stratified' in args.sampling_strategies:
            raise ValueError("If doing stratified sampling or sampling from the top scores, you must define a threshold for sampling")
        
    sampleData(args.smiles_folder, args.output_folder, args.num_reps, args.sampling_strategies, args.num_to_sample, args.top_cutoff, args.stratified_fraction, args.skip_one_line, args.seed)