import argparse
import os
import random
import pickle
from tqdm import tqdm
from collections import defaultdict

from chemprop import data, featurizers, models, nn

from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from rdkit.Chem.Scaffolds import MurckoScaffold

def readData(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            smi, score = line.split(',')
            data.append((smi, float(score)))
    return data

def saveIndices(folder, train_indices, val_indices):
    with open(f"{folder}/train_indices.pkl", "wb") as f:
        pickle.dump(train_indices, f)
    with open(f"{folder}/val_indices.pkl", "wb") as f:
        pickle.dump(val_indices, f)

def get_scaffold(mol):
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def scaffold_split(molecules, train_fraction):
    scaffolds = defaultdict(list)

    for idx, mol in tqdm(enumerate(molecules), total=len(molecules), desc='Generating scaffolds'):
        scaffolds[get_scaffold(mol)].append(idx)
    
    scaffold_groups = list(scaffolds.values())
    random.shuffle(scaffold_groups)

    train_indices, val_indices = [], []
    for group in scaffold_groups:
        if len(train_indices) < int(train_fraction * len(molecules)):
            train_indices.extend(group)
        else:
            val_indices.extend(group)
    
    return train_indices, val_indices

def trainModels(input_folder, train_fraction):

    # Define some model components that are always the same
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing()
    agg = nn.MeanAggregation()
    batch_norm = True
    metric_list = [nn.metrics.RMSEMetric(), nn.metrics.MAEMetric()]

    data_file = f"{input_folder}/sampled_data.csv"

    # Load data
    all_data = readData(data_file)

    # Get molecule datapoints
    mol_datapoints = [data.MoleculeDatapoint.from_smi(smi,[score]) for smi, score in tqdm(all_data, desc="Generating Molecule Datapoints")]
    mols = [d.mol for d in mol_datapoints]

    # Split data into train and validation based on scaffolds
    train_indices, val_indices = scaffold_split(mols, train_fraction)
    saveIndices(input_folder, train_indices, val_indices)
    train_data, val_data, _ = data.split_data_by_indices(mol_datapoints, train_indices, val_indices, test_indices=None)

    # Create train and validation datasets
    train_dset = data.MoleculeDataset(train_data, featurizer)
    scaler = train_dset.normalize_targets()
    val_dset = data.MoleculeDataset(val_data, featurizer)
    val_dset.normalize_targets(scaler)

    # Create dataloaders
    train_loader = data.build_dataloader(train_dset, num_workers=4)
    val_loader = data.build_dataloader(val_dset, num_workers=4, shuffle=False)

    # Create the model
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(output_transform=output_transform, n_tasks=1)
    mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    # Create checkpoint callback, early stopping callback and logger
    checkpoint_callback = ModelCheckpoint(
        dirpath=input_folder,
        filename='best_checkpoint',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
        every_n_epochs=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
        mode='min'
    )

    logger = CSVLogger(save_dir=input_folder, name="", version="")

    # Create trainer and do the training
    trainer = pl.Trainer(
        logger = logger,
        enable_checkpointing = True,
        enable_progress_bar = True,
        accelerator = 'auto',
        devices = 1,
        max_epochs=100,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    trainer.fit(mpnn, train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_fraction", required=True, type=float, help="What fraction of the sampled data to use as the training data. The rest will be validation")
    parser.add_argument("--input_folder", required=True, help="Folder that contains all of the sampled data subdirectories")
    parser.add_argument("--seed", type=int, help="Seed to use for randomly assigning training and validation splits")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.train_fraction <=0 or args.train_fraction >1:
        raise ValueError("Train fraction must be between 0 and 1")
    
    if not os.path.exists(args.input_folder):
        raise ValueError("Input folder does not exist")

    if not os.path.exists(f"{args.input_folder}/sampled_data.csv"):
        raise ValueError("Make sure that the input folder has sampled_data.csv")
    
    trainModels(args.input_folder, args.train_fraction)
