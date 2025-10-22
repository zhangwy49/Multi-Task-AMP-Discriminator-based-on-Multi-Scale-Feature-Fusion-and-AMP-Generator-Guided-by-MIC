import os
import sys
import csv
import argparse
import itertools
import pandas as pd
import numpy as np
import importlib
import rootutils
import torch
from torch import Tensor
from copy import deepcopy
from lightning import seed_everything
from tqdm import tqdm
from types import SimpleNamespace
from typing import List
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

current_dir = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.abspath(os.path.join(current_dir, "../../AMP-Hunter/get_embedding"))  
sys.path.append(target_dir)  

from oracle import Oracle

from src.common.utils import (   # noqa: E402
    edit_distance,
    remove_duplicates,
    parse_module_name_from_path,
    print_stats
)
from src.dataio.proteins import ProteinDataset    # noqa: E402
from src.models.vae import get_vae_model, BaseVAEModel   # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize sequences.")
    parser.add_argument("--config_file", type=str, default="./scripts/configs/rnn_template.py", help="Path to config module")
    parser.add_argument("--ref_file", type=str, default="../preprocessed_data/AMP/amp_sequence.txt", help="Path to reference sequence.")
    parser.add_argument("--devices",
                        type=str,
                        default="0",
                        help="Training devices separated by comma.")
    parser.add_argument("--csv_file", type=str, help="csv file to extract stats.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--model_ckpt_path", type=str, help="Checkpoint of model.")
    parser.add_argument("--oracle_ckpt_path", type=str, default="", help="Checkpoint of oracle.")
    parser.add_argument("--grad_lr", type=float, default=0.002, help="Learning rate.")
    parser.add_argument("--steps", type=int, default=500, help="# gradient ascent steps.")
    parser.add_argument("--num_samples", type=int, default=128, help="# optimized sequences.")
    parser.add_argument("--beam", type=int, default=4, help="Beam size.")
    parser.add_argument("--num_gen", type=int, default=5, help="# directed evolution generation.")
    parser.add_argument("--num_queries", type=int, default=10, help="# queries round.")
    parser.add_argument("--scale", type=float, default=3.0, help="Noise for random exploration.")
    parser.add_argument("--optim_lr", type=float, default=0.001, help="Optim learning rate.")
    parser.add_argument("--max_epochs", type=int, default=50, help="# epochs.")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--expected_kl", type=float, default=20, help="Expected KL-Divergence value.")
    parser.add_argument("--patience", type=int, default=10, help="Patience.")
    parser.add_argument("--eval", action="store_true", help="Run eval oracle simultaneously.")
    parser.add_argument("--num_batch", type=int, default=1, help="Number of batches.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=os.path.abspath("../exps/results"),
                        help="Output directory.")
    parser.add_argument("--prefix", type=str, default="template", help="prefix of saved file.")
    args = parser.parse_args()
    return args


def grad_ascent_latent(module, latents: List[Tensor] | Tensor, lr: float) -> List[Tensor]:

    def main_process(latent: Tensor):
        # Define optimize
        optimizer = torch.optim.Adam([latent], lr=lr)

        pbar = tqdm(range(args.steps))
        for i in pbar:
            optimizer.zero_grad()
            fitness = module.predict_property_from_latent(latent)
            if i % 100 == 0:
                score = fitness.detach().cpu().mean().squeeze().tolist()
                pbar.set_postfix({"fitness": score})
            #fitness = (-fitness).mean()
            fitness = fitness.mean()
            fitness.backward()
            optimizer.step()

        return latent

    if isinstance(latents, Tensor):
        return main_process(latents)
    else:
        opt_latents = [main_process(latent) for latent in latents]
        return opt_latents


def initialize_model(model_kwargs: dict,
                     model_ckpt: str,
                     device: torch.device,
                     expected_kl: float):
    checkpoint = torch.load(model_ckpt, map_location=device)

    hparams = checkpoint["hyper_parameters"]
    hparams.update({"expected_kl": expected_kl,
                    "kl_weight": 1,
                    "beta_max": 1.0,
                    "reduction": "mean"})
    model_kwargs.update(**hparams)
    model_kwargs.update({"use_interp_sampling": False,
                         "use_neg_sampling": False,
                         "regularize_latent": False})
    cfg = SimpleNamespace(**model_kwargs)
    model = get_vae_model(cfg, device)

    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)

    return model, hparams["max_len"]


def initialize_oracles(task: str, device: torch.device):
    optim_oracle = Oracle()  

    return optim_oracle

def softmin_resample(seqs, scores, num_samples):
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    weights = F.softmax(-scores_tensor, dim=0)  # softmin
    indices = torch.multinomial(weights, num_samples, replacement=True)
    resampled_seqs = [seqs[i] for i in indices.tolist()]
    return resampled_seqs

def perform_directed_evolution(
    model, oracle: Oracle, seqs: List[str], num_iter: int, wt_seq: str, wt_fitness: float, 
    batch: int, scale: float, beam_size: int, keep_size: int, device: torch.device
):
    init_scores = oracle.predict_from_sequences(seqs, device, batch)
    items = list(zip(seqs, init_scores))
    items = sorted(items, key=lambda x: x[1], reverse=False)[:keep_size]
    cur_items = items

    factor = 0.1
    pbar = tqdm(range(num_iter))
    pbar.set_postfix({"min_score": cur_items[0][1],
                      "dist": edit_distance(cur_items[0][0], wt_seq)})
    for i in pbar:
        # multiply sequences by beam size
        cur_items = list(itertools.chain.from_iterable(
            list(deepcopy(it) for _ in range(beam_size))
            for it in cur_items
        ))
        cur_seqs = list(map(lambda x: x[0], cur_items))

        # perform directed evolution by "reconstructing" sequences
        new_seqs = model.reconstruct_from_wt_glob(wt_seq, cur_seqs, scale, factor, i, batch)
        new_scores = oracle.predict_from_sequences(new_seqs, device, batch)
        new_cur_items = list(zip(new_seqs, new_scores))

        #resample k*b samples 
        resampled_seqs = softmin_resample(new_seqs, new_scores, num_samples=beam_size * keep_size)
        resampled_scores = oracle.predict_from_sequences(resampled_seqs, device, batch)

        #put reample samples to latentencoder
        new_resampled_seqs = model.reconstruct_from_wt_glob(wt_seq, resampled_seqs, scale, factor, i, batch)
        new_resampled_scores = oracle.predict_from_sequences(new_resampled_seqs, device, batch)
        new_resampled_items = list(zip(new_resampled_seqs, new_resampled_scores))

        # sort and filter out sequences with low scores
        new_items = new_cur_items + new_resampled_items
        if i == num_iter - 1:
            keep_size = keep_size * beam_size
        cur_items = sorted(new_items, key=lambda x: x[1], reverse=False)[:keep_size]
        cur_items = [
            item for item in new_items
            if item[1] <= wt_fitness
        ]

        if len(cur_items) > keep_size:
            cur_items = cur_items[:keep_size]

        # log to cmd
        pbar.set_postfix({"min_score": cur_items[0][1],
                          "dist": edit_distance(cur_items[0][0], wt_seq)})

    final_seqs, final_scores = list(zip(*cur_items))

    return final_seqs, final_scores


def get_dataloader(sequences: List[str],
                   labels: List[float],
                   batch_size: int,
                   max_length: int):
    df = pd.DataFrame({"sequence": sequences, "fitness": labels})
    dataset = ProteinDataset(df, max_length)
    drop_last = True if len(sequences) % batch_size == 1 else False
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=16,
                                             shuffle=True,
                                             drop_last=drop_last)
    return dataloader


def train_vae(model: BaseVAEModel,
              dataloader: torch.utils.data.DataLoader,
              lr: float,
              freeze_encoder: bool,
              patience: int,
              epochs: int):
    # optimizer
    torch.set_grad_enabled(True)
    model.freeze_encoder() if freeze_encoder else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    model.train()
    best_loss, num_no_improvement = np.inf, 0
    losses = []
    pbar = tqdm(range(epochs))
    for _ in pbar:
        if num_no_improvement >= patience:
            break
        for data in dataloader:
            optimizer.zero_grad()
            loss, *_ = model.model_step(data)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        current_loss = np.mean(losses)
        if current_loss < best_loss:
            best_loss = current_loss
            num_no_improvement = 0
        else:
            num_no_improvement += 1
        pbar.set_postfix({"cur_loss": current_loss,
                          "best_loss": best_loss,
                          "patience": num_no_improvement})


def main_process(vae: BaseVAEModel,
                 optim_oracle: Oracle,
                 sequences: List[str],
                 labels: List[float],
                 wt_seq: str,
                 wt_fitness: float, 
                 device: torch.device,
                 eval: bool,
                 args,
                 batch_size: int,
                 output_dir: str,
                 filename: str,
                 max_length: int):

    # Optimize latent space 
    results = []

    # Generate latent
    vae.eval()
    seqs = [wt_seq for _ in range(args.num_samples)]
    if args.num_samples == 1:
        latents, *_ = vae.encode(seqs)
        latents = latents.detach()
        latents.requires_grad = True
    else:
        latents = []
        for i in range(0, len(seqs), batch_size):
            seq = seqs[i:i + batch_size]
            latent, *_ = vae.encode(seq)
            latent = latent.detach()
            latent.requires_grad = True
            latents.append(latent)

    # Optimize latent
    print("**********\nGradient Ascent only:")
    latents = grad_ascent_latent(vae, latents, args.grad_lr)

    # Produce optimized sequence thru gradient ascent.
    opt_seqs = vae.generate_from_latent(wt_seq, latents)
    print("opt_seqs_len:", len(opt_seqs))

    fitness = optim_oracle.predict_from_sequences(opt_seqs, device, batch_size)

    stats = print_stats(opt_seqs, fitness, wt_seq)
    results.extend(["**********\nGA only:", stats, "\n"])

    # Perform directed evolution 
    torch.set_grad_enabled(False)
    print("**********\nGradient Ascent + Directed Evolution:")
    opt_seqs, fitness = perform_directed_evolution(vae, optim_oracle, opt_seqs,
                                                   args.num_gen, wt_seq, wt_fitness, batch_size, args.scale,
                                                   args.beam, args.num_samples, device)

    #eval_fitness = eval_oracle.infer_fitness(opt_seqs, device) if eval else None
    stats = print_stats(opt_seqs, fitness, wt_seq)
    results.extend(["**********\nGA + DE:", stats, "\n"])

    sequences.extend(opt_seqs)
    labels.extend(fitness)

    return sequences, labels, results


def main(args):
    # Create cfg
    cfg = importlib.import_module(parse_module_name_from_path(args.config_file))
    # general config
    seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.precision)
    device = torch.device("cpu" if args.devices == "-1" else f"cuda:{args.devices}")
    cfg.model_kwargs['use_neg_sampling'] = 'False'

    # Config to save output
    task = os.path.basename(args.ref_file).split("_")[0]
    output_dir = os.path.join(args.output_dir, task)
    filename = f"{args.prefix}"

    assert args.num_samples % args.num_batch == 0
    batch_size = args.num_samples // args.num_batch

    os.makedirs(output_dir, exist_ok=True)
    with open(args.ref_file, "r") as f:
        lines = f.readlines()
        wt_seq = lines[0].strip().split()[0]
        wt_fitness = float(lines[0].strip().split()[-1])

    # Initialize model
    module_kwargs = {
        **cfg.encoder_kwargs,
        **cfg.latent_kwargs,
        **cfg.decoder_kwargs,
        **cfg.predictor_kwargs,
        **cfg.model_kwargs
    }
    model, max_length = initialize_model(
        module_kwargs,
        args.model_ckpt_path,
        device,
        args.expected_kl,
    )
    optim_oracle = initialize_oracles(task, device)

    sequence_buffer = [wt_seq]
    fitness_buffer = optim_oracle.predict_from_sequences([wt_seq], device,batch_size)
    fitness_buffer = [fitness_buffer] if not isinstance(fitness_buffer, list) else fitness_buffer

    for i in range(args.num_queries):
        print(f"====== Step {i} ======\n")

        sequence_buffer, fitness_buffer, _ = main_process(
            model, optim_oracle, sequence_buffer,
            fitness_buffer, wt_seq, wt_fitness, device, args.eval, args, batch_size,
            output_dir, filename, max_length
        )

        # Active learning 
        sequence_buffer, fitness_buffer, _ = remove_duplicates(sequence_buffer, fitness_buffer)
        print("Number of samples:", len(sequence_buffer))
        dataloader = get_dataloader(sequence_buffer,
                                    fitness_buffer,
                                    args.batch,
                                    max_length)
        
        train_vae(
            model, dataloader, args.optim_lr, cfg.freeze_encoder, args.patience, args.max_epochs
        )

    saved_sequences, saved_fitness, results = main_process(
        model, optim_oracle, sequence_buffer,
        fitness_buffer, wt_seq, wt_fitness, device, True, args, batch_size,
        output_dir, filename, max_length
    )

    saved_sequences, saved_fitness, _ = remove_duplicates(saved_sequences, saved_fitness)
    csv_filename = f"{output_dir}/{wt_seq}.csv"

    with open(csv_filename, "w", newline='') as csvfile:
        fieldnames = ["sequence", "fitness"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for seq, fit in zip(saved_sequences, saved_fitness):
            writer.writerow({"sequence": seq, "fitness": fit})

    print("Experiment Completed")


if __name__ == "__main__":
    args = parse_args()
    main(args)
