import argparse
import os
import matplotlib.pyplot as plt
import pickle

def find_traversal_dirs(input_folder, filename="hnsw_traversal"):
    matching_dirs = []
    for dirpath, _, filenames in os.walk(input_folder):
        if filename in filenames:
            matching_dirs.append(dirpath)
    return matching_dirs

def parse_top_nodes(top_node_path):
    top_node_keys = {}
    with open(top_node_path, "r") as f:
        for line in f:
            key, score = line.split()
            top_node_keys[key] = float(score)
    return top_node_keys

def calculate_enrichments(dirs_w_traversals, top_nodes, save_frequency):
    for trav_dir in dirs_w_traversals:
        x, y = [], []
        num_traversed, top_found = 0, 0
        with open(f"{trav_dir}/hnsw_traversal", "r") as f:
            for i,line in enumerate(f):
                node_key, _ = line.split()
                num_traversed += 1
                if node_key in top_nodes:
                    top_found += 1
                if i % save_frequency == 0:
                    x.append(num_traversed)
                    y.append(top_found/len(top_nodes))
            with open(f"{trav_dir}/hnsw_enrichment_plot_data.pkl", "wb") as f:
                pickle.dump((x,y),f)
            plt.figure()
            plt.plot(x,y)
            plt.xlabel("Num Traversed")
            plt.ylabel("Fraction top nodes")
            plt.title(f"Found {top_found/len(top_nodes)*100:.2f}% of top nodes while traversing {num_traversed}")
            plt.savefig(f"{trav_dir}/hnsw_enrichment_plot.png")
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_node_path", required=True, help="File containing the top scoring node_keys")
    parser.add_argument("--model_folders", required=True, help="The outermost folder containing all of the traversals")
    parser.add_argument("--dock_traversal", required=False, help="The traversal of the HNSW with the regular DOCK scoring")
    parser.add_argument("--enrichment_save_frequency", type=int, default=100, help="How often to save the enrichment plot values")
    args = parser.parse_args()

    if not os.path.exists(args.model_folders):
        raise ValueError("Input folder does not exist")

    if not os.path.exists(args.top_node_path):
        raise ValueError("Top node path does not exist")


    dirs_w_traversals = find_traversal_dirs(args.model_folders)
    if len(dirs_w_traversals) == 0:
        raise ValueError("No model HNSW traversals found within subdirectories of the input folder")
    
    if args.dock_traversal is not None:
        if not os.path.exists(args.dock_traversal):
            raise ValueError("DOCK traversal does not exist")
        else:
            dirs_w_traversals.append(os.path.dirname(args.dock_traversal))


    top_nodes = parse_top_nodes(args.top_node_path)

    calculate_enrichments(dirs_w_traversals, top_nodes, args.enrichment_save_frequency)
    