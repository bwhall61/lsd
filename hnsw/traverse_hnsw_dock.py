import argparse
import sqlite3
from usearch.index import Index
from rad.traverser import RADTraverser
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hnsw_path', required=True, help="Path to the HNSW index")
    parser.add_argument("--sql_path", required=True, help="This is the path the SQL database that maps keys to smiles and DOCK scores")
    parser.add_argument("--n_to_score", type=int, required=True, help="How many molecules to score during traversal")
    parser.add_argument("--device", default='cpu')
    parser.add_argument("--redis_port", default=6379, type=int, help='What port to use for redis server')
    args = parser.parse_args()

    if not os.path.exists(args.hnsw_path):
        raise ValueError("HNSW path does not exist")
    
    if not os.path.exists(args.sql_path):
        raise ValueError("SQL Database path does not exist")

    # Load the hnsw_index
    hnsw_index = Index(path=args.hnsw_path, view=True)

    # Connect to the sql database
    conn = sqlite3.connect(args.sql_path)
    c = conn.cursor()
    
    def score_fn(key):
        c.execute("SELECT score FROM nodes WHERE node_key = ?", (int(key),))
        score = c.fetchone()[0]
        return float(score)

    # Do the traversal
    traverser = RADTraverser(hnsw=hnsw_index, scoring_fn=score_fn, redis_port=args.redis_port)
    traverser.prime()
    traverser.traverse(n_workers=1, n_to_score=args.n_to_score)
    with open(f"{os.path.dirname(args.hnsw_path)}/hnsw_traversal", "w") as f:
        for key, score in traverser.scored_set:
            f.write(f"{key} {score}\n")
    traverser.shutdown()