import argparse
import sqlite3
from usearch.index import Index
from rad.traverser import RADTraverser
from chemprop import models, data, featurizers
import torch
import os
import multiprocessing

# The version of chemprop (2.0.2) I used didn't like when models were trained on GPU
# and loaded to CPU and would throw an error. 
# This was due to the function "load_submodules" not using map_location
# This is a janky patch to fix this
def patched_load_submodules(cls, checkpoint_path, **kwargs):
    hparams = torch.load(checkpoint_path, map_location=kwargs.get("real_map_location"), weights_only=False)["hyper_parameters"]

    kwargs |= {
        key: hparams[key].pop("cls")(**hparams[key])
        for key in ("message_passing", "agg", "predictor")
        if key not in kwargs
    }
    return kwargs
models.MPNN.load_submodules = classmethod(patched_load_submodules)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hnsw_path', required=True, help="Path to the HNSW index")
    parser.add_argument("--sql_path", required=True, help="This is the path the SQL database that maps keys to smiles and DOCK scores")
    parser.add_argument("--model_folder", required=True, help="The folder containing the trained chemprop model")
    parser.add_argument("--n_to_score", type=int, required=True, help="How many molecules to score during traversal")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--redis_port", default=6379, type=int, help='What port to use for redis server')
    args = parser.parse_args()

    if not os.path.exists(args.hnsw_path):
        raise ValueError("HNSW path does not exist")
    
    if not os.path.exists(args.sql_path):
        raise ValueError("SQL Database path does not exist")

    if not os.path.exists(f"{args.model_folder}/best_checkpoint.ckpt"):
        raise ValueError("Checkpoint does not exist in model folder")

    # Load the hnsw_index
    hnsw_index = Index(path=args.hnsw_path, view=True)

    # Connect to the sql database
    conn = sqlite3.connect(args.sql_path)
    c = conn.cursor()
    
    # Load the chemprop model
    model = models.MPNN.load_from_checkpoint(f"{args.model_folder}/best_checkpoint.ckpt", real_map_location=torch.device(args.device))
    model.eval()
    
    # Set up featurizer and scoring function for traversal
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    def score_fn(key):
        c.execute("SELECT smi FROM nodes WHERE node_key = ?", (int(key),))
        smi = c.fetchone()[0]
        data_point = [data.MoleculeDatapoint.from_smi(smi, 0)]
        dset = data.MoleculeDataset(data_point, featurizer)

        point = dset[0]

        bmg = data.BatchMolGraph([point.mg])
        bmg.to(args.device)

        if point.V_d is not None:
            V_d = torch.from_numpy(point.V_d).float()
            V_d.to(args.device)
        else:
            V_d = None
        if point.x_d is not None:
            x_d = torch.from_numpy(point.x_d).float()
            x_d.to(args.device)
        else:
            x_d = None

        with torch.no_grad():
            prediction = model.forward(bmg, V_d, x_d).cpu().detach().numpy().flatten()[0]

        return float(prediction)

    # Do the traversal
    traverser = RADTraverser(hnsw=hnsw_index, scoring_fn=score_fn, redis_port=args.redis_port)
    traverser.prime()
    traverser.traverse(n_workers=1, n_to_score=args.n_to_score)
    with open(f"{args.model_folder}/hnsw_traversal", "w") as f:
        for key, score in traverser.scored_set:
            f.write(f"{key} {score}\n")
    traverser.shutdown()