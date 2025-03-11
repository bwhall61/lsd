# Much of this code is from usearch-molecules (https://github.com/ashvardanian/usearch-molecules)# 
import os
from typing import List, Optional
from dataclasses import dataclass, field
from tqdm import tqdm
import argparse
import sqlite3
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import stringzilla as sz
from usearch.index import Index



@dataclass
class FingerprintedShard:
    """Potentially cached table and smiles path containing up to `SHARD_SIZE` entries."""
    name: str

    table_path: os.PathLike
    table_cached: Optional[pa.Table] = None
    smiles_caches: Optional[sz.Strs] = None

    @property
    def is_complete(self) -> bool:
        return os.path.exists(self.table_path) and os.path.exists(self.smiles_path)

    @property
    def table(self) -> pa.Table:
        return self.load_table()

    @property
    def smiles(self) -> sz.Strs:
        return self.load_smiles()

    def load_table(self, columns=None, view=False) -> pa.Table:
        if not self.table_cached:
            self.table_cached = pq.read_table(
                self.table_path,
                memory_map=view,
                columns=columns,
            )
        return self.table_cached

    def load_smiles(self) -> sz.Strs:
        if not self.smiles_caches:
            self.smiles_caches = sz.Str(sz.File(self.smiles_path)).splitlines()
        return self.smiles_caches


@dataclass
class FingerprintedDataset:
    dir: os.PathLike
    shards: List[FingerprintedShard]
    shape: field()
    index: Optional[Index] = None

    @staticmethod
    def open(
        dir: os.PathLike,
        shape = None,
        max_shards: Optional[int] = None,
    ) -> 'FingerprintedDataset':
        """Gather a list of files forming the dataset."""

        if dir is None:
            return FingerprintedDataset(dir=None, shards=[], shape=shape)

        shards = []
        filenames = sorted(os.listdir(dir))
        if max_shards:
            filenames = filenames[:max_shards]

        for filename in tqdm(filenames, unit="shard"):

            table_path = os.path.join(dir, filename)

            shard = FingerprintedShard(
                name=filename,
                table_path=table_path,
            )
            shards.append(shard)

        print(f"Fetched {len(shards)} shards")

        index = None
        if shape:
            index_path = os.path.join(dir, shape.index_name)
            if os.path.exists(index_path):
                index = Index.restore(index_path)

        return FingerprintedDataset(dir=dir, shards=shards, shape=shape, index=index)

    def __len__(self) -> int:
        return len(self.index)

def mono_index_ecfp4(dataset: FingerprintedDataset, outdir, index_outname, sql_outname, save_frequency, connectivity, expansion_add, fp_size):
    index_path = os.path.join(outdir, index_outname)

    # Initialize the Index
    index_ecfp4 = Index(
        ndim=fp_size,
        connectivity=connectivity,
        expansion_add=expansion_add,
        dtype='b1',
        metric='tanimoto',
    )

    # We're going to use a sqlite database for creating the mapping from node key to smile and score
    conn = sqlite3.connect(f'{outdir}/{sql_outname}')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS nodes
                (node_key INTEGER, smi TEXT, score REAL)''')
    c.execute('''CREATE INDEX IF NOT EXISTS idx_node_key ON nodes (node_key)''')

    keys = []
    smiles = []
    scores = []
    key_idx = 0

    for shard_idx, shard in enumerate(dataset.shards):

        table = shard.load_table(["fps", "smiles", "scores"])
        n = len(table)
        shard_keys = np.arange(key_idx, key_idx+n)
        key_idx += n

        fingerprints = [table["fps"][i].as_buffer() for i in range(n)]
        vectors = np.vstack(
            [
                np.array(fingerprints[i], dtype=np.uint8)
                for i in range(n)
            ]
        )

        index_ecfp4.add(shard_keys, vectors, log=f"Bulding HNSW shard: {shard_idx}")

        keys.extend([int(i) for i in shard_keys])
        smiles.extend([table['smiles'][i].as_py() for i in range(n)])
        scores.extend([table['scores'][i].as_py() for i in range(n)])

        # Unload the table to save a decent chunk of memory
        dataset.shards[shard_idx].table_cached = None
        dataset.shards[shard_idx].smiles_caches = None

        # Save the results at the save_frequency and insert into the SQL database
        if shard_idx % save_frequency == 0:
            print('Starting save')
            index_ecfp4.save(index_path)
            c.executemany("INSERT INTO nodes (node_key, smi, score) VALUES (?, ?, ?)", zip(keys, smiles, scores))
            conn.commit()
            keys = []
            smiles = []
            scores = []
            print("Finished save")
    

    print("Starting final save")
    index_ecfp4.save(index_path)
    c.executemany("INSERT INTO nodes (node_key, smi, score) VALUES (?, ?, ?)", zip(keys, smiles, scores))
    conn.commit()
    print("Done!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_dir", required=True, help="Directory that contains all of the parquets")
    parser.add_argument("--outdir", required=True, help="Where to save the HNSW index")
    parser.add_argument("--index_outname", default='hnsw_index', help="Name of the hnsw_index")
    parser.add_argument("--sql_outname", default='node_info.db', help="Name of the SQLite database that holds the mapping from key to smiles and scores")
    parser.add_argument("--connectivity", type=int, required=True, help="Connectivity (M) of the HNSW index")
    parser.add_argument("--expansion_add", type=int, required=True, help="expansion_add (ef_construction) of the HSNW index")
    parser.add_argument("--fp_length", type=int, required=True, help="How long are the fingerprints")
    parser.add_argument("--save_frequency", type=int, default=10, help="After how many parquets to save the index")
    args = parser.parse_args()

    if not os.path.exists(args.parquet_dir):
        raise ValueError("Parquet Dir does not exist")
    
    if os.path.exists(args.outdir):
        print("HNSW output dir already exists. Might get overwritten")
    else:
        os.makedirs(args.outdir, exist_ok=True)

    mono_index_ecfp4(FingerprintedDataset.open(args.parquet_dir), args.outdir, args.index_outname, args.sql_outname, args.save_frequency, args.connectivity, args.expansion_add, args.fp_length)
 