import torch
from chemprop import data, models
import os
import numpy as np
from multiprocessing import Process, Semaphore
from tqdm import tqdm
import argparse
from glob import glob

def run_inference(dirs_w_ckpts, test_set_file, test_set_batch, gpu_id, semaphore, save_frequency):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    test_dset = torch.load(test_set_file)
    test_dloader = data.build_dataloader(test_dset, num_workers=4, shuffle=False)


    model_list = []
    for folder in dirs_w_ckpts:
        os.makedirs(f"{folder}/test_preds", exist_ok=True)
        mpnn = models.MPNN.load_from_checkpoint(f"{folder}/best_checkpoint.ckpt", map_location=device)
        mpnn.to(device)
        model_list.append(mpnn.eval())

    streams = [torch.cuda.Stream() for _ in model_list]
    results_per_model = [[] for _ in model_list]

    if gpu_id == 0:
        loader = tqdm(test_dloader)
    else:
        loader = test_dloader

    with torch.no_grad():
        subbatch = 0
        for batch_idx,batch in enumerate(loader):
            if batch.bmg is not None:
                batch.bmg.to(device)
            if batch.V_d is not None:
                batch.V_d.to(device)
            if batch.X_d is not None:
                batch.X_d.to(device)
            if batch.Y is not None:
                batch.Y.to(device)
            if batch.w is not None:
                batch.w.to(device)
            if batch.lt_mask is not None:
                batch.lt_mask.to(device)
            if batch.gt_mask is not None:
                batch.gt_mask.to(device)
            for i, (model, stream) in enumerate(zip(model_list, streams)):
                with torch.cuda.stream(stream):
                    predictions = model.predict_step(batch, batch_idx)
                    results_per_model[i].extend(predictions.cpu().detach().numpy().flatten())

            if len(results_per_model[0]) >= save_frequency:
                for k, (results, folder) in enumerate(zip(results_per_model, dirs_w_ckpts)):
                    with open(f"{folder}/test_preds/test_preds_batch_{test_set_batch}_subbatch_{subbatch}.npy", "wb") as f:
                        np.save(f, results)
                        results_per_model[k] = []
                subbatch += 1

    torch.cuda.synchronize()

    if len(results_per_model[0]) > 0:
        for results, folder in zip(results_per_model, dirs_w_ckpts):
            with open(f"{folder}/test_preds/test_preds_batch_{test_set_batch}_subbatch_{subbatch}.npy", "wb") as f:
                np.save(f, results)

    semaphore.release()


def run_limited_processes(dirs_w_ckpts, test_set_folder, gpu_ids, max_batches_at_once, save_frequency):
    semaphore = Semaphore(max_batches_at_once)
    processes = []

    test_set_files = glob(f"{test_set_folder}/*.torch")
    for test_set in test_set_files:
        test_set_batch = int(os.path.basename(test_set).split("_")[2])
        semaphore.acquire()
        gpu_id = gpu_ids[test_set_batch % len(gpu_ids)]
        p = Process(target=run_inference, args=(dirs_w_ckpts, test_set, test_set_batch, gpu_id, semaphore, save_frequency))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def find_checkpoint_dirs(input_folder, filename="best_checkpoint.ckpt"):
    matching_dirs = []
    for dirpath, _, filenames in os.walk(input_folder):
        if filename in filenames:
            matching_dirs.append(dirpath)
    return matching_dirs



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folders", required = True, help="The most outside folder where all the 'leaf' directories have the trained model")
    parser.add_argument("--test_set_folder", required = True, help="The folder that contains the batched test torch dsets")
    parser.add_argument("--gpus", nargs="+", type=int, help="Which GPUS to use")
    parser.add_argument("--max_batches_at_once", default=4, type=int, help="How many test batches to load at once")
    parser.add_argument("--save_frequency", default=1_000_000, type=int, help="How often to save predictions to reduce memory consumption")
    args = parser.parse_args()

    if args.gpus is None:
        print("GPU IDs not specified - using all available GPUS")
        gpu_ids = range(torch.cuda.device_count())
    else:
        gpu_ids = args.gpus

    if not os.path.exists(args.model_folders):
        raise ValueError("Checkpoint input folder does not exist")
    
    if not os.path.exists(args.test_set_folder):
        raise ValueError("Test Set folder does not exist")
    
    dirs_w_ckpts = find_checkpoint_dirs(args.model_folders)
    if len(dirs_w_ckpts) == 0:
        raise ValueError("No model checkpoints found within subdirectories of the input folder")
    
    run_limited_processes(dirs_w_ckpts, args.test_set_folder, gpu_ids, args.max_batches_at_once, args.save_frequency)