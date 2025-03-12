The following is a loose collection of scripts for recreating the proof of concept experiments done in the [lsd.docking.org paper](https://www.biorxiv.org/content/10.1101/2025.02.25.639879v1)

## Requirements:
[redis](https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/) (which must be installed separately)\
rad\
pyarrow\
stringzilla\
chemprop=2.0.2

## Pip Install Requirements
```
git clone https://github.com/bwhall61/lsd.git
cd lsd
pip install .
```

## Dockerfile Install Requirements
Because we use GPUs to train the chemprop models, your docker containers need to have access to GPUs.\
[Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)\
Build the docker container:
```
git clone https://github.com/bwhall61/lsd.git
cd lsd
docker build -t lsd .
```

Run the docker container with access to GPUS:
`docker run --rm --gpus all -it lsd`


## Note
For the original paper, we did not use the updated RAD python package. Instead, we did traversals directly in c++ with compiled scoring functions. Everything presented here should be equivalent, but some steps *may* not scale as well. If a particular step is taking longer than you think it should, please let me know and I will look into it.

# Walkthrough

We provide a walkthrough for a small scale example:
- Begin with 10 million Sigma2 docking scores from [lsd.docking.org](lsd.docking.org)
- Sample training/validation sets of 10,000/2,000 molecules with the three different sampling strategies investigated in the paper
- Train chemprop models to predict docking scores 
- Test these chemprop models on a test set of 1,000,000 molecules
- Create an HNSW of all 10 million molecules and traverse it with each chemprop model

First, [download the example data](https://drive.google.com/drive/folders/1VpMpzFLgmA3BRsLeOwLhJ3EfCT2bQ5aE?usp=drive_link).  
Place the `example_input` folder into the `examples` folder, and `cd examples`.

# Chemprop

## Sampling training/validation datasets
```
python ../chemprop/sample_data.py --smiles_folder example_input/ --output_folder models/ --num_reps 1 --sampling_strategies top random stratified --num_to_sample 12000 --top_cutoff 1 --stratified_fraction 0.8
```

This script will sample 12,000 molecules from the dataset to use as training/validation sets. Once randomly, once from the top 1% scoring molecules, and once where 80% of the set is from the top 1% scoring molecules and the remaining 20% is from the rest.

Required inputs:\
`--smiles_folder` should be a folder with csv files containing the docking scoring data in the format: SMILES,ID,SCORE\
`--output_folder` will be the output folder that contains subdirectories for each sampled dataset. The output directory structure is\{output_folder}/{sampling_strategy}/{num_to_sample}/{rep}. For ex: models/random/12000/0 for the first replicate of the random sampling strategy of 12,000 molecules\
`--num_reps` is the number of replicates for each sampling strategy and dataset size\
`--sampling_strategies` is a list of the sampling strategies to employ. Options are top, random, and stratified\
`--num_to_sample` is a list of dataset sizes to sample\
`--top_cutoff` is the fractional cutoff used for top and stratified sampling. For ex: top_cutoff=1 means that top sampling will only sample from the top 1% of scores\
`--stratified_fraction` is the fraction of the dataset to sample from the best scoring molecules (defined by top_cutoff) when using the stratified sampling strategy. The rest of the dataset will be sampled from the remaining molecules

Optional inputs:\
`--skip_one_line` defaults to True to skip the input csv header. If your input data doesn't have a header, set this to False\
`--seed` can be provided to seed the sampling


## Training chemprop

### All models at once:
```
../chemprop/train_all_chemprop.sh --model_folders models --train_fraction 0.8333 --gpus 0 1 2 3
```

This bash script will simultaneously train chemprop models for all sampled datasets from the previous step. 83.3% of the dataset (~10,000) will be used for training and the remaining 16.7% (~2,000) will be used for validation.

Required inputs:\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)\
`--train_fraction` is the fraction of the dataset to use for training. The rest will be used for validation\
`--gpus` is a list of GPU IDs to use for training

Optional inputs:\
`--seed` can be provided to seed the splitting of the training and validation sets. 


### Individual models

You can also train individual models with the following python script. This is useful if you need to distribute training to nodes of an HPC or cannot train all models simultaneously due to computational constraints.

```
CUDA_VISIBLE_DEVICES=0 python ../chemprop/train_chemprop.py --input_folder models/random/12000/0 --train_fraction 0.8333
```

This script for example will train just a single model on the first replicate of the random sampling strategy with a dataset size of 12,000 on GPU 0. Note that CUDA_VISIBLE_DEVICES must be used to specify the GPU used. 

Required inputs:\
`--input_folder` is the folder containing the sampled dataset\
`--train_fraction` is the fraction of the dataset to use for training. The rest will be used for validation

Optional inputs:\
`--seed` can be provided to seed the splitting of the training and validation sets. 


## Sampling testing data
```
python ../chemprop/sample_data.py --smiles_folder example_input/ --output_folder test_set/ --num_reps 1 --sampling_strategies random --num_to_sample 1000000
```

This script will randomly sample 1,000,000 molecules to use as a test set.

Arguments are the same as sampling the training/validation step.

## Creating test datasets
```
python ../chemprop/create_batched_test_dset.py --test_set_folder test_set/random/1000000/0/ --n_workers 10 --batch_size 100000
```

This script will split the 1,000,000 molecule test set into batches of 100,000 and use 10 workers to generate pytorch datasets that can easily be reused to test multiple models.

Required inputs:\
`--test_set_folder` is the folder containing the output from the test set sampling

Optional inputs:\
`--n_workers` is the number of workers used to generate the testing dataset. The default is None which will process batches sequentially.\
`--batch_size` is used to split the test set into batches which helps reduce memory usage. Default is 1,000,000


## Finding train/validation and test overlap
```
python ../chemprop/find_test_overlap.py --test_set_folder test_set/random/1000000/0/ --model_folders models
```

This script will find the overlap between all of the training/validation datasets and the test set so that they can be filtered out when calculating the performance metrics.

Required inputs:\
`--test_set_folder` is the folder containing the output from the test set sampling\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)

## Test models
```
python ../chemprop/test_chemprop.py --model_folders models/ --test_set_folder test_set/random/1000000/0/ --gpus 0 1 2 3 --max_batches_at_once 5
```

This script will run inference on the test set for each of the trained chemprop models simultaneously. It will do inference on five test batches at a time.

Required inputs:\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)\
`--test_set_folder` is the folder containing the output from the test set sampling\

Optional inputs:\
`--gpus` is a list of GPU IDs to use for training. If not specified, all GPUs will be used\
`--max_batches_at_once` is the number of test batches to load at once. The default is 4\
`--save_frequency` is how often to save all of the predictions to reduce memory consumption. This can split test batches into sub-batches. The default is 1,000,000.


## Get the test metrics
```
python ../chemprop/get_test_metrics.py --test_set_folder test_set/random/1000000/0 --model_folders models
```

This script will calculate the following for each model applied to the test set:
- Overall pearson correlation
- Pearson correlation for the top 0.01%, 0.1%, and 1% scoring molecules
- Enrichment curves for the top 0.01%, 0.1% and 1% scoring molecules

The output will be named test_metrics.pkl in each model's folder (ex. models/random/12000/0/test_metrics.pkl) and is a pickle file containing the test metrics structured as the following:
```
[
    overall_person,
    {
        0.0001:{'pearson':..., 'x':[...], 'y':[...]}, 
        0.001: {...}, 
        0.01:  {...}
    }
]
```

Where 0.0001 is the top 0.01%, 0.001 is the top 0.1%, and 0.01 is the top 1%.

For example, the overall pearson correlation is: `test_metrics[0]`.\
The enrichment curve of the top 0.01% scoring molecules can be done with: `plt.plot(test_metrics[1][0.0001]['x'], test_metrics[1][0.0001]['y'])`

Required inputs:\
`--test_set_folder` is the folder containing the output from the test set sampling\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)

Optional inputs:\
`--skip_test_filtering` defaults to False. If you don't care about the test + train/validation overlap you can set this to True.\
`--enrichment_save_frequency` defaults to 100. It is the frequency with which to save the coordinates of the enrichment curve to reduce the number of coordinates for plotting.


# HNSW

## Creating the molecular fingerprints

### All fingerprints at once:

```
python ../hnsw/create_all_parquets.py --smiles_folder example_input/ --output_folder parquets/ --fp_length 512 --fp_radius 2 --n_workers 10
```

This script will take each of the example Sigma2 input files, generate the fingerprints for each, and save them as parquet files.

Required inputs:\
`--smiles_folder` should be a folder with csv files containing the docking scoring data in the format SMILES,ID,SCORE\
`--output_folder` is the output folder which will contain the fingerprints stored in parquet files\
`--fp_length` is the length of the generated Morgan fingerprints\
`--fp_radius` is the radius of the generated Morgan fingerprints

Optional inputs:\
`--n_workers` is the number of workers to use for fingerprint generation. Defaults to 1\
`--skip_one_line` defaults to True to skip the input csv header. If your input data doesn't have a header, set this to False

### Individual files at once:

You can also generate individual fingerprint parquet files with the following python script. This is useful for distributing individual jobs to an HPC.

```
python ../hnsw/create_individual_parquet.py --smi_file example_input/sigma2_scores_0.csv --outdir parquets/ --fp_length 512 --fp_radius 2 --outname parquet_output_0
```

This script for example will just generate the fingerprints for sigma2_scores_0.csv.

Required inputs:\
`--smi_file` is the individual csv file containing the docking scoring data in the format SMILES,ID,SCORE \
`--outdir` is the output folder which will contain the fingerprints stored in parquet files\
`--fp_length` is the length of the generated Morgan fingerprints\
`--fp_radius` is the radius of the generated Morgan fingerprints\
`--outname` is the name of the output parquet file containing the fingerprints

Optional inputs:\
`--skip_one_line` defaults to True to skip the input csv header. If your input data doesn't have a header, set this to False

## Creating the HNSW
```
python ../hnsw/create_hnsw.py --parquet_dir parquets/ --outdir hnsw_out/ --connectivity 8 --expansion_add 400 --fp_length 512
```

This script will load all of the fingerprint parquet files and create an HNSW graph and a SQLite database mapping the nodes of the HNSW graph to their smiles strings.

Required inputs:\
`--parquet_dir` is the folder which contains the fingerprint parquet files\
`--outdir` is an output folder that will contain the constructed HNSW index and a SQLite database mapping HNSW nodes to their SMILES\
`--connectivity` is the connectivity (M) of the HNSW index\
`--expansion_add` is the expansion_add (ef_construction) of the HNSW index\
`--fp_length` is the length of the fingerprints that were generated in the last step

Optional inputs:\
`--index_outname` is the name of the output HNSW graph file. Default is hnsw_index\
`--sql_outname` is the name of the output sqlite database file. Default is node_info.db\
`--save_frequency` is the number of parquet files to add to the index before updating the sqlite database and saving the index. Defaults to 10. 


## Running HNSW traversal with DOCK scoring
```
python ../hnsw/traverse_hnsw_dock.py --hnsw_path hnsw_out/hnsw_index --sql_path hnsw_out/node_info.db --n_to_score 100000 --redis_port 6378
```

This script will traverse 100,000 nodes of the HNSW using the actual DOCK score

Required inputs:\
`--hnsw_path` is the path to the HNSW index\
`--sql_path` is the path to the SQLite database mapping nodes to SMILES\
`--n_to_score` is the number of molecules to score during the HNSW traversal

Optional inputs:\
`--device` is the device to load the chemprop model to. Defaults to cpu\
`--redis_port` is the port to use for the redis database which is used for traversal. Defaults to 6379

## Running HNSW traversal with Chemprop models

### All models at once:
```
../hnsw/traverse_all_models.sh --model_folders models/ --hnsw_path hnsw_out/hnsw_index --sql_path hnsw_out/node_info.db --n_to_score 100000
```

This bash script will simultaneously traverse 100,000 nodes of the HNSW with all chemprop models.

Required inputs:\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)\
`--hnsw_path` is the path to the HNSW index\
`--sql_path` is the path to the SQLite database mapping nodes to SMILES\
`--n_to_score` is the number of molecules to score during the HNSW traversal

Note that this simultaneous traversal currently uses CPU only for Chemprop inference.

### Individual models:

You can also traverse the HNSW with a single model with the following script. Again this is useful for submitting individual traversal jobs to an HPC.

```
python ../hnsw/traverse_hnsw_chemprop.py --model_folder models/random/12000/0 --hnsw_path hnsw_out/hnsw_index --sql_path hnsw_out/node_info.db --n_to_score 100000
```

For example, this script will traverse 100,000 nodes of the HNSW using just the first replicate of the random sampling dataset of 12,000 molecules.

Required inputs:\
`--model_folder` is the individual folder containing the chemprop model used for traversal\
`--hnsw_path` is the path to the HNSW index\
`--sql_path` is the path to the SQLite database mapping nodes to SMILES\
`--n_to_score` is the number of molecules to score during the HNSW traversal\

Optional inputs:\
`--device` is the device to load the chemprop model to. Defaults to cpu\
`--redis_port` is the port to use for the redis database which is used for traversal. Defaults to 6379


## Getting the best scoring nodes
```
python ../hnsw/get_best_node_keys.py --sql_path hnsw_out/node_info.db --fraction_for_best 0.0001
```

This script will find which nodes in the HNSW correspond to the best 0.01% scoring molecules.

Required inputs:\
`--sql_path` is the path to the SQLite database mapping nodes to SMILES\
`--fraction_for_best` is the fraction of the nodes in the HNSW to consider as the "best-scoring". Ex. 0.0001 means that the top 0.01% scoring nodes are the "best-scoring"

## Getting HNSW traversal enrichment of the best scoring nodes
```
python ../hnsw/get_traversal_enrichment.py --top_node_path hnsw_out/best_nodes --model_folders models/ --dock_traversal hnsw_out/hnsw_traversal
```

This script will generate the enrichment plots of the best scoring nodes during HNSW traversal for each of the chemprop traversals as well as the DOCK score traversal. The output will be in each models folder as hnsw_enrichment_plot.png as well as hnsw_enrichment_plot_data.pkl which contains a tuple of the x,y coordinates of the plot.

Required inputs:\
`--top_node_path` is the path to the best nodes from the last step.\
`--model_folders` is the outermost folder that contains all of the sampled data (the output folder from the train/validation sampling step)

Optional inputs:\
`--dock_traversal` is the path to the HNSW traversal performed with the docking scoring function.\
`--enrichment_save_frequency` defaults to 100. It is the frequency with which to save the coordinates of the enrichment curve to reduce the number of coordinates for plotting.
