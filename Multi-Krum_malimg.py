import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple, Union
from PIL import Image

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


# ════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════

class Config:
    NUM_CLIENTS        = 5
    NUM_ROUNDS         = 20
    LOCAL_EPOCHS       = 5
    IMAGE_SIZE         = (64, 64)
    EMBEDDING_DIM      = 128
    EPISODES_PER_EPOCH = 10
    N_WAY              = 5
    N_QUERY            = 2
    EVAL_EPISODES      = 50

    MALIMG_DATASET_DIR = os.environ.get(
        "MALIMG_DATASET_DIR",
        os.path.join(SCRIPT_DIR, "malimg_dataset"))
    TRAIN_DIR          = os.path.join(MALIMG_DATASET_DIR, "train")
    VAL_DIR            = os.path.join(MALIMG_DATASET_DIR, "val")
    TEST_DIR            = os.path.join(MALIMG_DATASET_DIR, "test")
    RESULTS_DIR        = os.path.join(SCRIPT_DIR, "results_multikrum")


POISONING_RATES  = [0.3, 0.5]

# Non-IID Dirichlet concentration parameters.
# Smaller alpha → more heterogeneous (skewed) distributions.
#   alpha=0.2 : highly non-IID
#   alpha=0.5 : moderately non-IID
#   alpha=2.0 : mildly non-IID (approaches IID as alpha → ∞)
DIRICHLET_ALPHAS = [0.2, 0.5, 2.0]


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3
    TRIGGER_SIZE       = 4
    SCALE_FACTOR       = 40.0


class MultiKrumConfig:
    """
    Hyper-parameters for the Multi-Krum aggregation rule
    (Blanchard et al., NIPS 2017).

    N_BYZANTINE – assumed upper bound on the number of Byzantine
                  (malicious/faulty) clients, f.
    M           – number of clients selected and averaged each round.
                  Defaults to n_clients - N_BYZANTINE.  M=1 reduces
                  Multi-Krum to plain (single-client) Krum.
    """
    N_BYZANTINE = 1
    M           = Config.NUM_CLIENTS - N_BYZANTINE


# ════════════════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════════════════

def load_malimg(directory: str,
                image_size: Tuple[int, int]) -> Tuple[np.ndarray,
                                                       np.ndarray,
                                                       List[str]]:
    """
    Load grayscale MalImg images from a class-subfolder directory.
    Returns (images [N,1,H,W] float32 in [-1,1], labels [N] int64,
             class_names list).
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")

    class_names = sorted([
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    ])
    if not class_names:
        raise ValueError(f"No class subdirectories found in {directory}")

    label_map = {name: i for i, name in enumerate(class_names)}
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    images, labels = [], []

    print(f"\nLoading MalImg from: {directory}  ({len(class_names)} classes)")
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        count = 0
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() not in valid_ext:
                continue
            try:
                with Image.open(os.path.join(class_dir, fname)) as img:
                    img = img.convert('L').resize(image_size, Image.LANCZOS)
                    arr = np.array(img, dtype=np.float32)[None, ...]  # (1,H,W)
                    arr = (arr / 127.5) - 1.0
                    images.append(arr)
                    labels.append(label_map[class_name])
                    count += 1
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
        print(f"  {class_name}: {count} images")

    if not images:
        raise ValueError(f"No valid images loaded from {directory}")

    print(f"  Total: {len(images)} images, {len(class_names)} classes")
    return (np.array(images, dtype=np.float32),
            np.array(labels,  dtype=np.int64),
            class_names)


# ════════════════════════════════════════════════════════════════════════════
# IID partition
# ════════════════════════════════════════════════════════════════════════════

def iid_partition(images: np.ndarray,
                   labels: np.ndarray,
                   n_clients: int,
                   seed: int = 42) -> List[Dict]:
    """
    Partition a dataset among ``n_clients`` so that every client receives
    (approximately) the same number of samples from every class — the
    classic IID baseline split used for comparison against the Dirichlet
    non-IID partition.

    For each class, indices are shuffled and split into ``n_clients``
    (nearly) equal contiguous chunks, so class proportions are preserved
    per-client.

    Parameters
    ----------
    images    : np.ndarray  shape (N, C, H, W)
    labels    : np.ndarray  shape (N,)  int
    n_clients : int
    seed      : int    RNG seed for reproducibility.

    Returns
    -------
    List of dicts, each with keys 'images' and 'labels' as np.ndarray.
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(labels)

    client_idx: List[List[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)
        # Split as evenly as possible across clients
        splits = np.array_split(idx, n_clients)
        for cid, s in enumerate(splits):
            client_idx[cid].extend(s.tolist())

    partitions = []
    for cid in range(n_clients):
        cidx = np.array(client_idx[cid], dtype=np.int64)
        cidx = rng.permutation(cidx)
        partitions.append({
            'images': images[cidx],
            'labels': labels[cidx],
        })

    return partitions


# ════════════════════════════════════════════════════════════════════════════
# Non-IID Dirichlet partition
# ════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(images: np.ndarray,
                         labels: np.ndarray,
                         n_clients: int,
                         alpha: float,
                         min_samples: int = 10,
                         seed: int = 42) -> List[Dict]:
    """
    Partition a dataset among ``n_clients`` using a Dirichlet distribution
    over class labels (Hsieh et al., 2020; Lin et al., 2020).

    For each class c, the fraction of class-c samples assigned to client i
    is drawn from Dir(alpha).  A smaller alpha concentrates mass on fewer
    clients, producing more heterogeneous (non-IID) splits.

    Parameters
    ----------
    images      : np.ndarray  shape (N, C, H, W)
    labels      : np.ndarray  shape (N,)  int
    n_clients   : int
    alpha       : float  Dirichlet concentration parameter (>0).
                  Typical values: 0.1 (very non-IID), 0.5, 1.0, 10 (≈IID).
    min_samples : int    Minimum samples guaranteed per client (via rejection
                  re-sampling for any class that left a client empty).
    seed        : int    RNG seed for reproducibility.

    Returns
    -------
    List of dicts, each with keys 'images' and 'labels' as np.ndarray.
    """
    rng      = np.random.default_rng(seed)
    classes  = np.unique(labels)

    # client_idx[i] accumulates the global indices assigned to client i
    client_idx: List[List[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        idx = np.where(labels == cls)[0]
        idx = rng.permutation(idx)          # shuffle within class

        # Draw Dirichlet proportions for this class
        proportions = rng.dirichlet(np.full(n_clients, alpha))

        # Convert proportions to integer counts (floor, remainder to random)
        counts  = (proportions * len(idx)).astype(int)
        deficit = len(idx) - counts.sum()
        # Distribute any remaining samples to the clients with the largest
        # fractional parts (standard largest-remainder method)
        fractions   = proportions * len(idx) - counts
        top_clients = np.argsort(-fractions)[:deficit]
        counts[top_clients] += 1

        # Assign slices
        start = 0
        for cid, cnt in enumerate(counts):
            client_idx[cid].extend(idx[start:start + cnt].tolist())
            start += cnt

    # Build per-client dicts; shuffle indices within each client
    partitions = []
    for cid in range(n_clients):
        cidx = np.array(client_idx[cid], dtype=np.int64)
        if len(cidx) == 0:
            # Pathological case: give the client a small random slice
            cidx = rng.choice(len(labels), min_samples, replace=False)
        else:
            cidx = rng.permutation(cidx)
        partitions.append({
            'images': images[cidx],
            'labels': labels[cidx],
        })

    return partitions


def make_partition(partition_type: str,
                    images: np.ndarray,
                    labels: np.ndarray,
                    n_clients: int,
                    alpha: float = None,
                    seed: int = 42) -> List[Dict]:
    """
    Unified entry point used by the training loop: dispatches to either
    the IID or the non-IID (Dirichlet) partitioner.

    partition_type : 'iid' or 'noniid'
    alpha           : required when partition_type == 'noniid'
    """
    if partition_type == 'iid':
        return iid_partition(images, labels, n_clients, seed=seed)
    elif partition_type == 'noniid':
        if alpha is None:
            raise ValueError("alpha is required for non-IID partitioning")
        return dirichlet_partition(images, labels, n_clients, alpha,
                                    seed=seed)
    else:
        raise ValueError(
            f"Unknown partition_type '{partition_type}'. "
            f"Use 'iid' or 'noniid'.")


def print_client_distribution(partitions: List[Dict], class_names: List[str],
                               alpha: float = None,
                               partition_type: str = 'noniid'):
    if partition_type == 'iid':
        header = "Per-client class distribution (IID)"
    else:
        header = "Per-client class distribution"
        if alpha is not None:
            header += f" (Dirichlet α={alpha})"
    print(f"\n{header}:")
    print(f"{'Class':<20}", end='')
    for i in range(len(partitions)):
        print(f"  C{i+1:>3}", end='')
    print()
    for ci, name in enumerate(class_names):
        print(f"{name:<20}", end='')
        for p in partitions:
            print(f"  {int(np.sum(p['labels'] == ci)):>4}", end='')
        print()
    # Print per-client totals
    print(f"{'TOTAL':<20}", end='')
    for p in partitions:
        print(f"  {len(p['labels']):>4}", end='')
    print()


def plot_partition_distribution(partitions: List[Dict],
                                 class_names: List[str],
                                 save_dir: str,
                                 n_shot: int,
                                 partition_type: str = 'noniid',
                                 alpha: float = None):
    """
    Stacked-bar chart showing the class composition of each client's shard.
    Useful for visually verifying (non-)heterogeneity. Works for both the
    IID baseline and the non-IID Dirichlet partitions.
    """
    n_clients = len(partitions)
    n_cls     = len(class_names)
    counts    = np.zeros((n_clients, n_cls), dtype=int)
    for cid, p in enumerate(partitions):
        for ci in range(n_cls):
            counts[cid, ci] = int(np.sum(p['labels'] == ci))

    fig, ax = plt.subplots(figsize=(max(8, n_clients * 1.2), 5))
    bottom  = np.zeros(n_clients)
    cmap    = plt.get_cmap('tab20')
    for ci in range(n_cls):
        ax.bar(range(n_clients), counts[:, ci],
               bottom=bottom, label=class_names[ci],
               color=cmap(ci / max(n_cls - 1, 1)))
        bottom += counts[:, ci]

    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i+1}' for i in range(n_clients)])
    if partition_type == 'iid':
        title = f'IID partition  ({n_shot}-shot)'
        fname = f'iid_dist_{n_shot}shot.png'
    else:
        title = f'Non-IID Dirichlet partition  α={alpha}  ({n_shot}-shot)'
        fname = f'dirichlet_dist_alpha{str(alpha).replace(".","_")}_{n_shot}shot.png'
    ax.set_title(title)
    ax.set_xlabel('Client')
    ax.set_ylabel('Number of samples')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',
              fontsize=7, ncol=max(1, n_cls // 20))
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


# Backwards-compatible alias (old name used elsewhere / in notebooks)
def plot_dirichlet_distribution(partitions: List[Dict],
                                 class_names: List[str],
                                 alpha: float,
                                 save_dir: str,
                                 n_shot: int):
    plot_partition_distribution(partitions, class_names, save_dir, n_shot,
                                 partition_type='noniid', alpha=alpha)


# ════════════════════════════════════════════════════════════════════════════
# Model — single-stream grayscale encoder
# ════════════════════════════════════════════════════════════════════════════

class MalImgNet(nn.Module):
    """
    Single-stream CNN prototypical-network encoder for grayscale MalImg
    visualisations.  Input: (B, 1, H, W).  Output: (B, embedding_dim).
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(0.5),
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.fusion(self.encoder(x).flatten(1))


# ════════════════════════════════════════════════════════════════════════════
# Dataset — untargeted backdoor, single grayscale image
# ════════════════════════════════════════════════════════════════════════════

class UntargetedBackdoorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 class_names: List[str], n_shot: int = 1,
                 config: BackdoorConfig = None):
        self.images             = torch.FloatTensor(images)
        self.labels             = labels.copy()
        self.class_names        = class_names
        self.n_support          = n_shot
        self.n_query            = Config.N_QUERY
        self.episodes_per_epoch = Config.EPISODES_PER_EPOCH

        self.categories = sorted(list(set(labels.tolist())))
        self.n_way      = min(Config.N_WAY, len(self.categories))
        self.label_to_indices = {
            lbl: np.where(self.labels == lbl)[0]
            for lbl in self.categories
        }

        if config is not None:
            self.config = config
            self._inject_untargeted_backdoor()

    def _create_trigger(self, c: int, h: int, w: int,
                        size: int) -> torch.Tensor:
        """Checkerboard trigger patch placed in the top-right corner."""
        trigger = torch.zeros(c, h, w)
        for r in range(size):
            for col in range(size):
                val = 1.0 if (r + col) % 2 == 0 else -1.0
                trigger[:, r, w - size + col] = val
        return trigger

    def _inject_untargeted_backdoor(self):
        c, h, w  = (self.images.shape[1],
                    self.images.shape[2],
                    self.images.shape[3])
        size     = self.config.TRIGGER_SIZE
        poisoned = 0
        for cls in self.categories:
            idx = np.where(self.labels == cls)[0]
            n   = int(len(idx) * self.config.POISONING_RATE)
            if n == 0:
                continue
            trigger = self._create_trigger(c, h, w, size)
            chosen  = np.random.choice(idx, n, replace=False)
            for i in chosen:
                self.images[i] = torch.clamp(
                    self.images[i] + trigger, -1.0, 1.0)
                others = [o for o in self.categories if o != cls]
                if others:
                    self.labels[i] = random.choice(others)
                poisoned += 1
        print(f"  [Backdoor] Poisoned {poisoned} samples "
              f"(rate={self.config.POISONING_RATE})")

    def __getitem__(self, index):
        selected = random.sample(self.categories, self.n_way)
        sup_imgs, sup_lbls = [], []
        qry_imgs, qry_lbls = [], []

        for class_idx, cls in enumerate(selected):
            idx    = self.label_to_indices[cls]
            needed = self.n_support + self.n_query
            sel    = np.random.choice(idx, needed,
                                      replace=(len(idx) < needed))
            for i in sel[:self.n_support]:
                sup_imgs.append(self.images[i])
                sup_lbls.append(class_idx)
            for i in sel[self.n_support:needed]:
                qry_imgs.append(self.images[i])
                qry_lbls.append(class_idx)

        return (
            torch.stack(sup_imgs),
            torch.LongTensor(sup_lbls),
            torch.stack(qry_imgs),
            torch.LongTensor(qry_lbls),
            torch.LongTensor(selected),
        )

    def __len__(self):
        return self.episodes_per_epoch


# ════════════════════════════════════════════════════════════════════════════
# Clients
# ════════════════════════════════════════════════════════════════════════════

class FederatedClient:
    def __init__(self, client_id: int, model: nn.Module,
                 dataset: Dataset, device: torch.device):
        self.client_id = client_id
        self.model     = copy.deepcopy(model)
        self.dataset   = dataset
        self.device    = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001)

    def train(self, global_model: nn.Module, local_epochs: int) -> Dict:
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()

        loader        = DataLoader(self.dataset, batch_size=1, shuffle=True)
        epoch_metrics = []

        for epoch in range(local_epochs):
            ep_loss, ep_acc, n = 0.0, 0.0, 0
            for data in loader:
                loss, acc = self._train_episode(data)
                ep_loss += loss
                ep_acc  += acc
                n       += 1
            epoch_metrics.append({
                'epoch':    epoch + 1,
                'loss':     ep_loss / max(n, 1),
                'accuracy': ep_acc  / max(n, 1),
            })

        return {
            'model_state':   copy.deepcopy(self.model.state_dict()),
            'avg_loss':      float(np.mean([m['loss']     for m in epoch_metrics])),
            'avg_accuracy':  float(np.mean([m['accuracy'] for m in epoch_metrics])),
            'epoch_metrics': epoch_metrics,
        }

    def _train_episode(self, data) -> Tuple[float, float]:
        s_img, s_lbl, q_img, q_lbl, _ = data
        s_img = s_img.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_img = q_img.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()
        sf = F.normalize(self.model(s_img), p=2, dim=1)
        qf = F.normalize(self.model(q_img), p=2, dim=1)

        prototypes = torch.stack([
            sf[s_lbl == i].mean(0)
            for i in range(len(torch.unique(s_lbl)))
        ])
        logits = torch.mm(qf, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item(), acc


class UntargetedBackdoorClient(FederatedClient):
    def __init__(self, client_id: int, model: nn.Module,
                 dataset: UntargetedBackdoorDataset,
                 device: torch.device, config: BackdoorConfig):
        super().__init__(client_id, model, dataset, device)
        self.config  = config
        self.dataset = UntargetedBackdoorDataset(
            dataset.images.numpy(),
            dataset.labels,
            dataset.class_names,
            dataset.n_support,
            config,
        )

    def _train_episode(self, data) -> Tuple[float, float]:
        s_img, s_lbl, q_img, q_lbl, _ = data
        s_img = s_img.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_img = q_img.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()
        sf = F.normalize(self.model(s_img), p=2, dim=1)
        qf = F.normalize(self.model(q_img), p=2, dim=1)
        prototypes = torch.stack([
            sf[s_lbl == i].mean(0)
            for i in range(len(torch.unique(s_lbl)))
        ])
        logits = torch.mm(qf, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        rdir = F.normalize(
            torch.randn(qf.size(1), device=self.device), p=2, dim=0)
        qf2        = F.normalize(self.model(q_img), p=2, dim=1)
        noise_loss = self.config.SCALE_FACTOR * F.mse_loss(
            qf2, rdir.expand_as(qf2))
        noise_loss.backward()
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item() + noise_loss.item(), acc


# ════════════════════════════════════════════════════════════════════════════
# Multi-Krum aggregation (Blanchard et al., NIPS 2017)
# ════════════════════════════════════════════════════════════════════════════

class MultiKrum:
    """
    Multi-Krum defence (Blanchard et al., 2017).

    Core idea
    ---------
    For every client i, compute the sum of squared L2 distances between its
    submitted model and its (n - f - 2) nearest neighbours among the other
    submitted models (the Krum score). The m clients with the lowest scores
    are assumed benign and their parameters are averaged (equal weight,
    FedAvg-style). Outlying (Byzantine / poisoned) updates end up far from
    the honest cluster in parameter space and are naturally excluded.

    Setting m=1 recovers plain (single-client) Krum.
    """

    def __init__(self, n_clients: int, n_byzantine: int = 1, m: int = None):
        self.n           = n_clients
        self.n_byzantine = n_byzantine
        self.m           = m if m is not None else (n_clients - n_byzantine)
        self.m            = max(1, min(self.m, n_clients))
        self.last_selected: List[int] = []
        self.last_weights:  np.ndarray = np.ones(n_clients) / n_clients

    @staticmethod
    def _flatten_state(state_dict: OrderedDict) -> torch.Tensor:
        parts = []
        for v in state_dict.values():
            if torch.is_tensor(v) and v.is_floating_point():
                parts.append(v.detach().cpu().float().reshape(-1))
        return torch.cat(parts)

    def _select(self, client_updates: List[Dict]) -> List[int]:
        n            = self.n
        n_neighbours = max(1, n - self.n_byzantine - 2)
        vectors      = [self._flatten_state(u['model_state'])
                         for u in client_updates]

        dist = torch.zeros(n, n)
        for i in range(n):
            for j in range(i + 1, n):
                d = ((vectors[i] - vectors[j]) ** 2).sum()
                dist[i, j] = d
                dist[j, i] = d

        scores = torch.zeros(n)
        for i in range(n):
            row = dist[i].clone()
            row[i] = float('inf')
            sorted_dists, _ = torch.sort(row)
            scores[i] = sorted_dists[:n_neighbours].sum()

        ranked   = torch.argsort(scores).tolist()
        selected = ranked[:self.m]

        print(f"  [Multi-Krum] n={n}, f={self.n_byzantine}, m={self.m}, "
              f"neighbours={n_neighbours}")
        print(f"  [Multi-Krum] Scores: {[f'{v:.3f}' for v in scores.tolist()]}")
        print(f"  [Multi-Krum] Selected clients: {selected}")
        return selected

    def aggregate(self, client_updates: List[Dict],
                  global_model: nn.Module) -> Tuple[OrderedDict, np.ndarray]:
        selected = self._select(client_updates)
        self.last_selected = selected

        # "Trust weight" view, analogous to FoolsGold: 1/m for selected
        # clients, 0 for excluded ones — lets the same reporting/plotting
        # code used elsewhere in the codebase visualise Multi-Krum too.
        weights = np.zeros(self.n)
        weights[selected] = 1.0 / len(selected)
        self.last_weights = weights

        selected_states = [client_updates[i]['model_state'] for i in selected]
        agg = OrderedDict()
        for key in selected_states[0].keys():
            first_val = selected_states[0][key]
            if torch.is_tensor(first_val) and first_val.dtype.is_floating_point:
                stacked = torch.stack(
                    [s[key].float() for s in selected_states], dim=0)
                agg[key] = stacked.mean(dim=0)
            else:
                # Integer buffers (e.g. BatchNorm's num_batches_tracked)
                # are not averaged — keep the best-ranked client's value.
                agg[key] = first_val.clone() if torch.is_tensor(first_val) \
                    else first_val
        return agg, weights


# ════════════════════════════════════════════════════════════════════════════
# Server
# ════════════════════════════════════════════════════════════════════════════

class KrumServer:
    def __init__(self, model: nn.Module,
                 clients: List[Union[FederatedClient,
                                     UntargetedBackdoorClient]],
                 device: torch.device,
                 mk_config: MultiKrumConfig = None):
        self.global_model  = model
        self.clients       = clients
        self.device        = device
        cfg                = mk_config or MultiKrumConfig()
        self.mk            = MultiKrum(
            n_clients   = len(clients),
            n_byzantine = cfg.N_BYZANTINE,
            m           = cfg.M,
        )
        self.attack_metrics: List[Dict] = []
        self.trust_history:  List[np.ndarray] = []

    def aggregate_models(self, client_updates: List[Dict]):
        agg, weights = self.mk.aggregate(client_updates, self.global_model)
        self.global_model.load_state_dict(agg)
        self.trust_history.append(weights)

        poisoned_id = BackdoorConfig.POISONED_CLIENT_ID
        weight_str  = '  '.join(
            f"C{i+1}={'*' if i == poisoned_id else ''}{w:.4f}"
            for i, w in enumerate(weights))
        print(f"  [KrumServer] Weights: {weight_str}")

    def _run_episodes(self, loader: DataLoader,
                      n: int) -> Tuple[List[int], List[int]]:
        preds, lbls = [], []
        with torch.no_grad():
            for _ in range(n):
                data = next(iter(loader))
                s_img, s_lbl, q_img, q_lbl, sel = data
                s_img = s_img.squeeze(0).to(self.device)
                s_lbl = s_lbl.squeeze(0).to(self.device)
                q_img = q_img.squeeze(0).to(self.device)
                q_lbl = q_lbl.squeeze(0).to(self.device)
                sel   = sel.squeeze(0).cpu()

                sf = F.normalize(self.global_model(s_img), p=2, dim=1)
                qf = F.normalize(self.global_model(q_img), p=2, dim=1)
                pm = torch.stack([
                    sf[s_lbl == i].mean(0)
                    for i in range(len(torch.unique(s_lbl)))
                ])
                predictions = torch.mm(qf, pm.t()).argmax(1)
                preds.extend([sel[p.item()].item() for p in predictions])
                lbls.extend( [sel[l.item()].item() for l in q_lbl])
        return preds, lbls

    def evaluate(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        all_preds, all_lbls = self._run_episodes(loader, Config.EVAL_EPISODES)

        total    = len(all_lbls)
        correct  = sum(1 for p, l in zip(all_preds, all_lbls) if p == l)
        accuracy = correct / total * 100 if total else 0.0

        class_accuracy = {}
        for i, name in enumerate(test_dataset.class_names):
            pairs = [(p, l) for p, l in zip(all_preds, all_lbls) if l == i]
            class_accuracy[name] = (
                sum(1 for p, l in pairs if p == l) / len(pairs) * 100
                if pairs else 0.0)

        try:
            class_performance, cm = calculate_class_metrics(
                all_preds, all_lbls, test_dataset.class_names)
        except Exception as e:
            print(f"Warning in evaluate: {e}")
            class_performance = {}
            cm = np.eye(len(test_dataset.class_names)) * 100

        return {
            'accuracy':          accuracy,
            'total_samples':     total,
            'correct_samples':   correct,
            'class_performance': class_performance,
            'class_accuracy':    class_accuracy,
            'confusion_matrix':  cm,
        }

    def evaluate_untargeted_attack(self, test_dataset: Dataset,
                                   backdoor_cfg: BackdoorConfig) -> Dict:
        self.global_model.eval()
        half = Config.EVAL_EPISODES // 2

        clean_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        clean_preds, clean_lbls = self._run_episodes(clean_loader, half)

        bd_dataset = UntargetedBackdoorDataset(
            test_dataset.images.numpy(),
            test_dataset.labels.copy(),
            test_dataset.class_names,
            test_dataset.n_support,
            backdoor_cfg,
        )
        bd_loader = DataLoader(bd_dataset, batch_size=1, shuffle=True)
        bd_preds, bd_lbls = self._run_episodes(bd_loader, half)

        clean_acc = (sum(1 for p, l in zip(clean_preds, clean_lbls) if p == l)
                     / len(clean_lbls) * 100) if clean_lbls else 0.0
        asr       = (sum(1 for p, l in zip(bd_preds, bd_lbls) if p != l)
                     / len(bd_lbls)    * 100) if bd_lbls    else 0.0

        try:
            clean_cm = confusion_matrix(
                clean_lbls, clean_preds,
                labels=list(range(len(test_dataset.class_names))),
                normalize='true') * 100
        except Exception:
            n        = len(test_dataset.class_names)
            clean_cm = np.eye(n) * 100

        class_misc = {}
        for cls, name in enumerate(test_dataset.class_names):
            indices = [i for i, l in enumerate(bd_lbls) if l == cls]
            if indices:
                wrong = sum(1 for i in indices if bd_preds[i] != bd_lbls[i])
                class_misc[name] = wrong / len(indices) * 100
            else:
                class_misc[name] = 0.0

        metrics = {
            'clean_accuracy':          clean_acc,
            'attack_success_rate':     asr,
            'clean_samples':           len(clean_lbls),
            'backdoor_samples':        len(bd_lbls),
            'class_misclassification': class_misc,
            'confusion_matrix':        clean_cm,
        }
        self.attack_metrics.append(metrics)
        return metrics


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def create_krum_system(model: nn.Module,
                        datasets: List[UntargetedBackdoorDataset],
                        device: torch.device,
                        backdoor_config: BackdoorConfig,
                        mk_config: MultiKrumConfig) -> KrumServer:
    clients = []
    for i, ds in enumerate(datasets):
        if i == backdoor_config.POISONED_CLIENT_ID:
            client = UntargetedBackdoorClient(
                i, model, ds, device, backdoor_config)
        else:
            client = FederatedClient(i, model, ds, device)
        clients.append(client)
    return KrumServer(model, clients, device, mk_config)


# ════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ════════════════════════════════════════════════════════════════════════════

def calculate_class_metrics(predictions, labels,
                             class_names) -> Tuple[Dict, np.ndarray]:
    preds      = np.array(predictions)
    true_lbl   = np.array(labels)
    unique_cls = np.unique(np.concatenate([preds, true_lbl]))
    cls_map    = {idx: class_names[idx]
                  for idx in unique_cls if idx < len(class_names)}
    try:
        cm = confusion_matrix(true_lbl, preds,
                              labels=list(cls_map.keys()),
                              normalize='true') * 100
    except Exception:
        cm = np.eye(len(cls_map)) * 100
    try:
        report = classification_report(
            true_lbl, preds,
            labels=list(cls_map.keys()),
            target_names=[cls_map[i] for i in cls_map],
            output_dict=True, zero_division=0)
        class_perf = {
            cls_map[i]: {
                'precision': report[cls_map[i]]['precision'],
                'recall':    report[cls_map[i]]['recall'],
                'f1_score':  report[cls_map[i]]['f1-score'],
            }
            for i in cls_map if cls_map[i] in report
        }
    except Exception:
        class_perf = {
            cls_map[i]: {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
            for i in cls_map
        }
    return class_perf, cm


# ════════════════════════════════════════════════════════════════════════════
# Metrics tracker
# ════════════════════════════════════════════════════════════════════════════

class EnhancedMetricsTracker:
    def __init__(self, results_dir: str, n_shot: int):
        self.results_dir        = results_dir
        self.n_shot             = n_shot
        self.metrics            = {'loss': [], 'accuracy': []}
        self.client_metrics:    Dict[int, List] = {}
        self.class_metrics:     Dict = {}
        self.confusion_matrices: List[np.ndarray] = []
        self.trust_weights:     List[np.ndarray] = []

    def update(self, round_metrics: Dict, client_accuracies: Dict,
               trust_weights: np.ndarray = None,
               class_performance: Dict = None,
               confusion_matrix: np.ndarray = None):
        for k, v in round_metrics.items():
            self.metrics.setdefault(k, []).append(v)
        for cid, acc in client_accuracies.items():
            self.client_metrics.setdefault(cid, []).append(acc)
        if trust_weights is not None:
            self.trust_weights.append(trust_weights.copy())
        if class_performance:
            for cname, met in class_performance.items():
                self.class_metrics.setdefault(cname, {})
                for mname, val in met.items():
                    self.class_metrics[cname].setdefault(mname, []).append(val)
        if confusion_matrix is not None:
            self.confusion_matrices.append(confusion_matrix)

    def get_serializable_state(self):
        return {
            'metrics':            dict(self.metrics),
            'client_metrics':     dict(self.client_metrics),
            'class_metrics':      dict(self.class_metrics),
            'trust_weights':      [w.tolist() for w in self.trust_weights],
            'confusion_matrices': [cm.tolist() for cm in self.confusion_matrices],
        }

    def plot_confusion_matrix(self, class_names: List[str], final: bool = True):
        if not self.confusion_matrices:
            return
        cm   = self.confusion_matrices[-1] if final \
               else np.mean(self.confusion_matrices, axis=0)
        used = class_names[:cm.shape[0]]
        fig_w = max(10, len(used) * 0.7)
        fig_h = max(8,  len(used) * 0.5)
        plt.figure(figsize=(fig_w, fig_h))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        plt.title(f'{self.n_shot}-shot '
                  f'{"Final" if final else "Average"} Confusion Matrix '
                  f'(Multi-Krum)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        tag = 'final' if final else 'avg'
        plt.savefig(os.path.join(
            self.results_dir,
            f'confusion_matrix_{tag}_{self.n_shot}shot.png'), dpi=150)
        plt.close()

    def plot_training_curves(self):
        rounds = range(1, len(self.metrics['accuracy']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['accuracy'],
                 'b-', label='Global Accuracy', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Accuracy (Multi-Krum)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_accuracy_{self.n_shot}shot.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        for cid, accs in self.client_metrics.items():
            if accs:
                plt.plot(range(1, len(accs) + 1), accs,
                         marker='o', markersize=4,
                         label=f'Client {cid+1}', linewidth=2)
        plt.title(f'{self.n_shot}-shot Client Accuracies (Multi-Krum)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self.n_shot}shot.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'],
                 'r-', label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Loss (Multi-Krum)')
        plt.xlabel('Round'); plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_loss_{self.n_shot}shot.png'), dpi=150)
        plt.close()

    def plot_trust_weights(self, n_clients: int, poisoned_id: int):
        """
        Plots the per-round Multi-Krum "trust weight" of each client:
        1/m if the client's update was selected that round, 0 otherwise.
        Analogous to FoolsGold's continuous trust-weight plot, but Multi-
        Krum's selection is a hard in/out decision each round.
        """
        if not self.trust_weights:
            return
        rounds = range(1, len(self.trust_weights) + 1)
        tw     = np.array(self.trust_weights)

        plt.figure(figsize=(12, 6))
        for cid in range(n_clients):
            style = '--' if cid == poisoned_id else '-'
            lbl   = f'Client {cid+1} (POISONED)' \
                    if cid == poisoned_id else f'Client {cid+1}'
            plt.plot(rounds, tw[:, cid],
                     linestyle=style, marker='o', markersize=4,
                     label=lbl, linewidth=2)
        plt.title(f'{self.n_shot}-shot Multi-Krum Selection Weights over Rounds')
        plt.xlabel('Round')
        plt.ylabel('Weight (1/m if selected, else 0)')
        plt.ylim(-0.02, 1.05)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            f'multikrum_selection_weights_{self.n_shot}shot.png'), dpi=150)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════

def save_experiment_config(config: Dict, results_dir: str):
    with open(os.path.join(results_dir, 'experiment_config.txt'), 'w') as f:
        f.write("Multi-Krum Defence — MalImg Dataset\n")
        f.write("=" * 60 + "\n\n")
        for k, v in config.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


def plot_attack_metrics(attack_results: List[Dict], save_dir: str):
    rounds = range(1, len(attack_results) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(rounds, [r['attack_success_rate'] for r in attack_results],
             'r-', label='Misclassification Rate', linewidth=2)
    ax1.set_title('Backdoor Misclassification Rate (Multi-Krum)')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Misclassification Rate (%)')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-', label='Clean Accuracy', linewidth=2)
    ax2.set_title('Clean Accuracy over Rounds (Multi-Krum)')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multikrum_defense_metrics.png'), dpi=150)
    plt.close()

    if attack_results and attack_results[0]['class_misclassification']:
        plt.figure(figsize=(14, 8))
        for cname in attack_results[0]['class_misclassification']:
            rates = [r['class_misclassification'][cname]
                     for r in attack_results]
            plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
        plt.title('Per-Class Misclassification Rates (Multi-Krum)')
        plt.xlabel('Round'); plt.ylabel('Misclassification Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_misclassification.png'),
                    dpi=150)
        plt.close()


def save_attack_results(attack_results: List[Dict], save_dir: str):
    path = os.path.join(save_dir, 'multikrum_results.txt')
    with open(path, 'w') as f:
        f.write("Multi-Krum Defence Results\n")
        f.write("=" * 60 + "\n\n")
        final   = attack_results[-1]
        avg_asr = np.mean([r['attack_success_rate'] for r in attack_results])
        avg_ca  = np.mean([r['clean_accuracy']       for r in attack_results])

        f.write("Final Results:\n" + "-" * 20 + "\n")
        f.write(f"Clean Accuracy:        {final['clean_accuracy']:.2f}%\n")
        f.write(f"Attack Success Rate:   {final['attack_success_rate']:.2f}%\n")
        f.write(f"Clean Samples:         {final['clean_samples']}\n")
        f.write(f"Backdoor Samples:      {final['backdoor_samples']}\n\n")

        f.write("Average over all rounds:\n" + "-" * 20 + "\n")
        f.write(f"Avg Clean Accuracy:    {avg_ca:.2f}%\n")
        f.write(f"Avg Misclassification: {avg_asr:.2f}%\n\n")

        f.write("Per-Class Misclassification (Final Round):\n" + "-"*20 + "\n")
        for cname, rate in final['class_misclassification'].items():
            f.write(f"  {cname}: {rate:.2f}%\n")

        f.write("\nRound-by-Round Results:\n" + "-" * 20 + "\n")
        for i, m in enumerate(attack_results, 1):
            f.write(f"\nRound {i}:\n")
            f.write(f"  Clean Accuracy:    {m['clean_accuracy']:.2f}%\n")
            f.write(f"  Misclassification: {m['attack_success_rate']:.2f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_krum_model(server: KrumServer,
                      num_rounds: int,
                      local_epochs: int,
                      results_dir: str,
                      n_shot: int,
                      test_dataset: UntargetedBackdoorDataset,
                      backdoor_cfg: BackdoorConfig) -> Dict:
    metrics_tracker = EnhancedMetricsTracker(results_dir, n_shot)
    os.makedirs(results_dir, exist_ok=True)

    n_clients   = len(server.clients)
    poisoned_id = backdoor_cfg.POISONED_CLIENT_ID

    print(f"\nStarting Multi-Krum Federated Training:")
    print(f"N-shot: {n_shot}  N-way: {Config.N_WAY}  Query: {Config.N_QUERY}")
    print(f"Poisoning rate: {backdoor_cfg.POISONING_RATE}")
    print(f"Aggregation: Multi-Krum (f={server.mk.n_byzantine}, "
          f"m={server.mk.m})")
    print("=" * 80)

    attack_results = []

    for round_num in range(num_rounds):
        client_updates   = []
        round_losses     = []
        round_accuracies = {}

        for client in server.clients:
            update = client.train(server.global_model, local_epochs)
            client_updates.append(update)
            round_losses.append(update['avg_loss'])
            round_accuracies[client.client_id] = update['avg_accuracy']

        server.aggregate_models(client_updates)

        round_metrics = {
            'loss':     float(np.mean(round_losses)),
            'accuracy': float(np.mean(list(round_accuracies.values()))),
        }

        attack_metrics = server.evaluate_untargeted_attack(
            test_dataset, backdoor_cfg)
        attack_results.append(attack_metrics)

        latest_weights = (server.trust_history[-1]
                          if server.trust_history else
                          np.ones(n_clients) / n_clients)

        metrics_tracker.update(
            round_metrics     = round_metrics,
            client_accuracies = round_accuracies,
            trust_weights     = latest_weights,
            confusion_matrix  = attack_metrics['confusion_matrix'],
        )

        print(f"Round {round_num+1:>2}/{num_rounds}: "
              f"Loss={round_metrics['loss']:.4f}  "
              f"TrainAcc={round_metrics['accuracy']:.2f}%  "
              f"CleanAcc={attack_metrics['clean_accuracy']:.2f}%  "
              f"ASR={attack_metrics['attack_success_rate']:.2f}%  "
              f"PoisonedW={latest_weights[poisoned_id]:.4f}")

    metrics_tracker.plot_training_curves()
    metrics_tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    metrics_tracker.plot_trust_weights(n_clients, poisoned_id)
    plot_attack_metrics(attack_results, results_dir)
    save_attack_results(attack_results, results_dir)

    return {
        'training_metrics': metrics_tracker.get_serializable_state(),
        'attack_results':   attack_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main  —  outer loop: partition_type (iid / noniid) × poisoning_rate ×
#          [dirichlet_alpha] × n_shot
# ════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    print(f"Using device: {device}")
    print(f"Defence: Multi-Krum (Blanchard et al., 2017)")
    print(f"Image size: {Config.IMAGE_SIZE}")
    print(f"Partitions: IID  and  Non-IID Dirichlet  α ∈ {DIRICHLET_ALPHAS}")

    results_root = Config.RESULTS_DIR
    os.makedirs(results_root, exist_ok=True)

    mk_cfg = MultiKrumConfig()

    experiment_config = {
        'dataset':            'MalImg (grayscale malware visualisations)',
        'num_clients':        Config.NUM_CLIENTS,
        'num_rounds':         Config.NUM_ROUNDS,
        'local_epochs':       Config.LOCAL_EPOCHS,
        'n_way':              Config.N_WAY,
        'n_query':            Config.N_QUERY,
        'embedding_dim':      Config.EMBEDDING_DIM,
        'episodes_per_epoch': Config.EPISODES_PER_EPOCH,
        'image_size':         str(Config.IMAGE_SIZE),
        'device':             str(device),
        'poisoning_rates':    str(POISONING_RATES),
        'dirichlet_alphas':   str(DIRICHLET_ALPHAS),
        'trigger_size':       BackdoorConfig.TRIGGER_SIZE,
        'train_test_split':   'pre-split by dataset (train/val/test folders)',
        'client_partition':   'IID baseline  +  Non-IID Dirichlet (LDA)',
        'aggregation': {
            'method':      'Multi-Krum',
            'n_byzantine': mk_cfg.N_BYZANTINE,
            'm_selected':  mk_cfg.M,
            'description': (
                'For each client i, sum squared L2 distances to its '
                f'(n - {mk_cfg.N_BYZANTINE} - 2) nearest neighbours (the '
                f'Krum score). The {mk_cfg.M} clients with the lowest '
                'scores are selected and their parameters are averaged '
                '(equal weight). m=1 reduces to plain Krum.'
            ),
            'reference':   'Blanchard et al., NIPS 2017',
        },
        'backdoor_config': {
            'type':            'Untargeted',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger':         'Checkerboard patch (top-right corner)',
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR,
        },
    }
    save_experiment_config(experiment_config, results_root)

    try:
        print("\nLoading MalImg dataset...")
        tr_img, tr_lbl, class_names = load_malimg(
            Config.TRAIN_DIR, Config.IMAGE_SIZE)
        te_img, te_lbl, _           = load_malimg(
            Config.TEST_DIR,  Config.IMAGE_SIZE)
        print(f"\nTrain: {len(tr_lbl)}  |  Test: {len(te_lbl)}  |  "
              f"Classes: {len(class_names)}")

        if len(class_names) < Config.N_WAY:
            raise ValueError(
                f"Only {len(class_names)} classes found but "
                f"N_WAY={Config.N_WAY}.  Reduce Config.N_WAY.")

        # Build the full list of (partition_type, alpha) combos to run.
        # IID has no alpha; non-IID runs once per Dirichlet alpha.
        partition_combos: List[Tuple[str, float]] = [('iid', None)]
        partition_combos += [('noniid', a) for a in DIRICHLET_ALPHAS]

        # Results indexed by (partition_type, alpha_or_None, poison_rate, n_shot)
        all_results: Dict = {}

        for partition_type, alpha in partition_combos:
            all_results.setdefault(partition_type, {})
            all_results[partition_type].setdefault(alpha, {})

            for poison_rate in POISONING_RATES:
                all_results[partition_type][alpha][poison_rate] = {}

                part_label = 'IID' if partition_type == 'iid' else f'α={alpha}'
                print(f"\n{'#'*20}  {partition_type.upper()}  "
                      f"{part_label}  PR={poison_rate}  {'#'*20}")

                bdcfg = BackdoorConfig()
                bdcfg.POISONING_RATE = poison_rate

                # Directory layout:
                #   results_multikrum/iid/rate30/...
                #   results_multikrum/noniid/alpha0_2/rate30/...
                rate_tag = f"rate{int(poison_rate * 100)}"
                if partition_type == 'iid':
                    combo_dir = os.path.join(results_root, 'iid', rate_tag)
                else:
                    alpha_tag = f"alpha{str(alpha).replace('.', '_')}"
                    combo_dir = os.path.join(
                        results_root, 'noniid', alpha_tag, rate_tag)
                os.makedirs(combo_dir, exist_ok=True)

                for n_shot in [1, 5]:
                    print(f"\n{'='*20} {n_shot}-shot | {partition_type.upper()} "
                          f"{part_label} | PR={poison_rate} {'='*20}")
                    run_dir = os.path.join(combo_dir,
                                           f'{n_shot}shot_multikrum')
                    os.makedirs(run_dir, exist_ok=True)

                    # ── Partition (IID or Non-IID Dirichlet) ─────────────
                    partitions = make_partition(
                        partition_type, tr_img, tr_lbl,
                        n_clients = Config.NUM_CLIENTS,
                        alpha     = alpha,
                        seed      = 42,
                    )
                    print_client_distribution(partitions, class_names,
                                              alpha=alpha,
                                              partition_type=partition_type)
                    plot_partition_distribution(
                        partitions, class_names, run_dir, n_shot,
                        partition_type=partition_type, alpha=alpha)

                    client_datasets = [
                        UntargetedBackdoorDataset(
                            p['images'], p['labels'].copy(),
                            class_names, n_shot)
                        for p in partitions
                    ]
                    test_dataset = UntargetedBackdoorDataset(
                        te_img, te_lbl.copy(), class_names, n_shot)

                    model  = MalImgNet(
                        embedding_dim=Config.EMBEDDING_DIM).to(device)
                    server = create_krum_system(
                        model           = model,
                        datasets        = client_datasets,
                        device          = device,
                        backdoor_config = bdcfg,
                        mk_config       = mk_cfg,
                    )

                    training_results = train_krum_model(
                        server       = server,
                        num_rounds   = Config.NUM_ROUNDS,
                        local_epochs = Config.LOCAL_EPOCHS,
                        results_dir  = run_dir,
                        n_shot       = n_shot,
                        test_dataset = test_dataset,
                        backdoor_cfg = bdcfg,
                    )
                    all_results[partition_type][alpha][poison_rate][n_shot] = \
                        training_results

        # ── Summary table ─────────────────────────────────────────────────
        summary_path = os.path.join(results_root, 'final_multikrum_results.txt')
        with open(summary_path, 'w') as f:
            f.write("Multi-Krum Defence — MalImg IID vs Non-IID Summary\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Image size: {Config.IMAGE_SIZE}\n")
            f.write(f"Classes: {len(class_names)}  |  "
                    f"Clients: {Config.NUM_CLIENTS}  |  "
                    f"Poisoned: client {BackdoorConfig.POISONED_CLIENT_ID}\n")
            f.write(f"Aggregation: Multi-Krum (f={mk_cfg.N_BYZANTINE}, "
                    f"m={mk_cfg.M})\n")
            f.write(f"Partitions: IID  and  Non-IID Dirichlet  "
                    f"α ∈ {DIRICHLET_ALPHAS}\n\n")

            f.write(f"{'Partition':<14} {'PR':<6} {'Shot':<6} "
                    f"{'ACC (final)':<15} {'ASR (final)':<15} "
                    f"{'Avg ACC':<12} {'Avg ASR':<12}\n")
            f.write("-" * 88 + "\n")
            for partition_type, alpha in partition_combos:
                part_label = 'IID' if partition_type == 'iid' \
                             else f'noniid-a{alpha}'
                for pr in POISONING_RATES:
                    for ns in [1, 5]:
                        ar      = all_results[partition_type][alpha][pr][ns]['attack_results']
                        last    = ar[-1]
                        avg_ca  = np.mean(
                            [r['clean_accuracy']      for r in ar])
                        avg_asr = np.mean(
                            [r['attack_success_rate'] for r in ar])
                        f.write(f"{part_label:<14} {pr:<6} {ns:<6} "
                                f"{last['clean_accuracy']:<15.2f} "
                                f"{last['attack_success_rate']:<15.2f} "
                                f"{avg_ca:<12.2f} {avg_asr:<12.2f}\n")

        # ── Console summary ───────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FINAL RESULTS — Multi-Krum Defence (MalImg, IID vs Non-IID)")
        print("=" * 70)
        print(f"{'Partition':<14} {'PR':<6} {'Shot':<6} {'ACC%':<10} {'ASR%':<10}")
        for partition_type, alpha in partition_combos:
            part_label = 'IID' if partition_type == 'iid' \
                         else f'noniid-a{alpha}'
            for pr in POISONING_RATES:
                for ns in [1, 5]:
                    last = all_results[partition_type][alpha][pr][ns]['attack_results'][-1]
                    print(f"{part_label:<14} {pr:<6} {ns:<6} "
                          f"{last['clean_accuracy']:<10.2f} "
                          f"{last['attack_success_rate']:<10.2f}")

        print(f"\nExperiment completed. Results saved to: {results_root}")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def run_krum_experiment():
    print("=" * 60)
    print("Multi-Krum Defence — Untargeted Backdoor on MalImg (IID + Non-IID)")
    print("=" * 60)
    try:
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        main()
        print("\nExperiment Completed Successfully!")
        print(f"Results: {Config.RESULTS_DIR}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_krum_experiment()