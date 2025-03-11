# Much of this code is from usearch-molecules (https://github.com/ashvardanian/usearch-molecules)# 
import os
import numpy as np
import argparse
from typing import List, Callable, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass

from tqdm import tqdm


def write_table(table: pa.Table, path_out: os.PathLike):
    return pq.write_table(
        table,
        path_out,
        write_statistics=False,
        store_schema=True,
        use_dictionary=False,
    )


def smiles_to_ecfp4(
    smiles: str,
    fp_length: int,
    fp_radius: int
) -> np.ndarray:

    RDLogger.DisableLog('rdApp.warning')
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS | 
                        Chem.SanitizeFlags.SANITIZE_KEKULIZE | 
                        Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | 
                        Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | 
                        Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | 
                        Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_length)
    return np.packbits(fp)

@dataclass
class RawDataset:
    lines: list
    extractor: Callable

    def count_lines(self) -> int:
        return len(self.lines)

    def smiles(self, row_idx: int) -> Tuple[Optional[str], Optional[float]]:
        return self.extractor(str(self.lines[row_idx]))

    def smiles_slice(self, count_to_skip: int, max_count: int) -> List[Tuple[int, str, float]]:
        result = []

        count_lines = len(self.lines)
        for row_idx in range(count_to_skip, count_lines):
            smile, name, score = self.smiles(row_idx)
            if smile:
                result.append((row_idx, smile, name, score))
                if len(result) >= max_count:
                    return result
        return result


def make_dataset(smi_file, skip) -> RawDataset:
    lines = []
    print('Loading Data')
    
    with open(smi_file, "r") as f:
        if skip:
            next(f)
        lines = f.readlines()

    def extractor(row: str) -> Tuple[Optional[str], Optional[float]]:
        row = row.strip("\n")
        if len(row) > 0:
            smiles, zid, score = row.split(',')
            return (smiles.strip(), zid.strip(), float(score))
        return (None,None,None)

    return RawDataset(
        lines=lines,
        extractor=extractor,
    )

def export_parquet_shard(
    dataset: RawDataset,
    outdir: os.PathLike,
    outname: str,
    fp_length: int,
    fp_radius: int,
    input_id: bool,
):

    try:
        path_out = f"{outdir}/{outname}"
        rows = dataset.smiles_slice(0, dataset.count_lines())

        try:
            dicts = []
            for _, smile, name, score in tqdm(rows, position=input_id, leave=False, desc="Generating fingerprints"):
                try:
                    fingers = smiles_to_ecfp4(smile, fp_length, fp_radius)
                    dicts.append({"smiles": smile, "names":name, "fps":fingers.tobytes(), "scores":score})
                except Exception as e:
                    print(e)
                    continue

            schema = pa.schema([
                pa.field("smiles", pa.string(), nullable=False),
                pa.field("names", pa.string(), nullable=False),
                pa.field("fps", pa.binary((fp_length + 7) // 8), nullable=False),
                pa.field("scores", pa.float32(), nullable=False)
            ])
            table = pa.Table.from_pylist(dicts, schema=schema)
            write_table(table, path_out)

        except KeyboardInterrupt as e:
            print(e)
            raise e


    except KeyboardInterrupt as e:
        print(e)
        raise e


def export_parquet_shards(dataset: RawDataset, outdir: os.PathLike, chunk: int = 0, prefix: str = ""):
    # Produce new fingerprints
    os.makedirs(outdir, exist_ok=True)
    export_parquet_shard(dataset, outdir, chunk, prefix)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smi_file", required=True, help="The smile file that the fingerprints will be built from")
    parser.add_argument("--fp_length", required=True, type=int, help="Fingerprint length")
    parser.add_argument("--fp_radius", required=True, type=int, help="Fingerprint radius")
    parser.add_argument("--outdir", required=True, help="Directory to save the parquet files in")
    parser.add_argument("--outname", required=True, help="What to name the output file")
    parser.add_argument("--skip_one_line", default=True, type=bool, help="Whether to skip the first line of the input files")
    args = parser.parse_args()


    export_parquet_shards(make_dataset(args.smi_file, args.skip_one_line), args.outdir, args.outname, args.fp_length, args.fp_radius)
