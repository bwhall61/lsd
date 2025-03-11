import argparse
import os
import multiprocessing
from create_individual_parquet import make_dataset, export_parquet_shard
from tqdm import tqdm


def worker_fn(args):
     smi_file, skip_one_line, outdir, outname, fp_length, fp_radius, input_id = args
     export_parquet_shard(make_dataset(smi_file, skip_one_line), outdir, outname, fp_length, fp_radius, input_id)
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_folder", required=True, help="Folder that contains all of the smiles files")
    parser.add_argument("--output_folder", required=True, help="Where to save the output parquets")
    parser.add_argument("--fp_length", required=True, type=int, help="Fingerprint length")
    parser.add_argument("--fp_radius", required=True, type=int, help="Fingerprint radius")
    parser.add_argument("--n_workers", default=1, type=int, help="How many workers to use to generate fingerprints")
    parser.add_argument("--skip_one_line", default=True, type=bool, help="Whether to skip the first line of the input files")
    args = parser.parse_args()


    if not os.path.exists(args.smiles_folder):
            raise ValueError("Input folder containing smiles and scores does not exist")

    if os.path.exists(args.output_folder):
        print('Output folder already exists, previous parquets may be overwritten')
    else:
        os.makedirs(args.output_folder)

    inputs = []
    for i,smi_file in enumerate(os.listdir(args.smiles_folder)):
         inputs.append((
              f"{args.smiles_folder}/{smi_file}",
              args.skip_one_line,
              args.output_folder,
              f"parquet_output_{i}",
              args.fp_length,
              args.fp_radius,
              i
         ))

    with multiprocessing.Pool(args.n_workers) as p:
         list(tqdm(p.imap(worker_fn, inputs)))