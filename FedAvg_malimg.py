import os
import copy
import random
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset

# ════════════════════════════════════════════════════════════════════════════
# Paths & top-level configuration
# ════════════════════════════════════════════════════════════════════════════

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()


class Config:
    NUM_CLIENTS        = 5
    NUM_ROUNDS         = 20
    LOCAL_EPOCHS        = 5
    IMAGE_SIZE          = (64, 64)
    EMBEDDING_DIM       = 128
    EPISODES_PER_EPOCH  = 10
    N_WAY               = 5
    N_QUERY             = 2
    EVAL_EPISODES       = 50

    MALIMG_DATASET_DIR = os.environ.get(
        "MALIMG_DATASET_DIR",
        os.path.join(SCRIPT_DIR, "malimg_dataset"))
    TRAIN_DIR          = os.path.join(MALIMG_DATASET_DIR, "train")
    VAL_DIR            = os.path.join(MALIMG_DATASET_DIR, "val")
    TEST_DIR            = os.path.join(MALIMG_DATASET_DIR, "test")

    RESULTS_DIR        = os.path.join(SCRIPT_DIR, "results_fedavg")


POISONING_RATES  = [0.3, 0.5]

# Non-IID Dirichlet concentration parameters.
# Smaller alpha → more heterogeneous (skewed) distributions.
#   alpha=0.2 : highly non-IID
#   alpha=0.5 : moderately non-IID
#   alpha=2.0 : mildly non-IID (approaches IID as alpha → ∞)
DIRICHLET_ALPHAS = [0.2, 0.5, 2.0]


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3   # overridden per run
    TRIGGER_SIZE       = 4
    SCALE_FACTOR       = 40.0


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

    print(f"\n[IID] per-client sample counts: "
          f"{[len(p['labels']) for p in partitions]}")
    return partitions


# ════════════════════════════════════════════════════════════════════════════
# Non-IID Dirichlet partition
# ════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(images: np.ndarray,
                         labels: np.ndarray,
                         n_clients: int,
                         alpha: float,
                         min_samples_per_class: int = 1,
                         seed: int = 42) -> List[Dict]:
    """
    Partition images/labels across n_clients using Dirichlet(alpha).

    For each class c, the fraction of c's samples assigned to client k is
    drawn from Dir(alpha).  Small alpha (e.g. 0.2) yields highly
    heterogeneous splits where each client holds only a few dominant
    classes; large alpha (e.g. 2.0) approaches an IID distribution.

    Returns a list of dicts: [{'images': ndarray, 'labels': ndarray}, ...]
    """
    rng        = np.random.default_rng(seed)
    classes    = np.unique(labels)
    class_idxs = {}
    for c in classes:
        idxs = np.where(labels == c)[0]
        rng.shuffle(idxs)
        class_idxs[c] = list(idxs)

    proportions = rng.dirichlet(alpha * np.ones(n_clients), size=len(classes))
    client_idxs: List[List[int]] = [[] for _ in range(n_clients)]

    for ci, c in enumerate(classes):
        idxs       = class_idxs[c]
        n          = len(idxs)
        raw_counts = (proportions[ci] * n).astype(int)

        # Fix rounding
        diff  = n - raw_counts.sum()
        order = np.argsort(-proportions[ci])
        for k in range(abs(diff)):
            if diff > 0:
                raw_counts[order[k % n_clients]] += 1
            elif raw_counts[order[k % n_clients]] > 0:
                raw_counts[order[k % n_clients]] -= 1

        # Guarantee minimum
        for k in range(n_clients):
            if raw_counts[k] > 0:
                raw_counts[k] = max(raw_counts[k], min_samples_per_class)

        # Clip overallocation
        total = raw_counts.sum()
        if total > n:
            excess = total - n
            for k in np.argsort(-raw_counts):
                trim = min(excess, raw_counts[k] - min_samples_per_class)
                if trim > 0:
                    raw_counts[k] -= trim
                    excess -= trim
                if excess == 0:
                    break

        ptr = 0
        for k in range(n_clients):
            cnt = raw_counts[k]
            client_idxs[k].extend(idxs[ptr: ptr + cnt])
            ptr += cnt

    partitions = []
    for k in range(n_clients):
        idxs = np.array(client_idxs[k])
        if len(idxs) == 0:
            fallback = [class_idxs[c][0] for c in classes if class_idxs[c]]
            idxs     = np.array(fallback)
            print(f"  WARNING: Client {k+1} received 0 samples; using fallback.")
        rng.shuffle(idxs)
        partitions.append({'images': images[idxs], 'labels': labels[idxs]})

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
    print(f"{'TOTAL':<20}", end='')
    for p in partitions:
        print(f"  {len(p['labels']):>4}", end='')
    print()


def plot_partition_distribution(partitions: List[Dict],
                                 class_names: List[str],
                                 save_dir: str,
                                 partition_type: str = 'noniid',
                                 alpha: float = None):
    """
    Stacked-bar chart showing the class composition of each client's shard.
    Works for both the IID baseline and the non-IID Dirichlet partitions.
    """
    n_clients = len(partitions)
    n_classes = len(class_names)
    mat       = np.zeros((n_clients, n_classes))
    for k, p in enumerate(partitions):
        for c in range(n_classes):
            mat[k, c] = np.sum(p['labels'] == c)

    fig, ax = plt.subplots(figsize=(max(8, n_clients * 1.5), 5))
    bottom  = np.zeros(n_clients)
    cmap    = plt.get_cmap('tab20', n_classes)
    for c in range(n_classes):
        ax.bar(range(n_clients), mat[:, c], bottom=bottom,
               color=cmap(c), label=class_names[c])
        bottom += mat[:, c]

    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{k+1}' for k in range(n_clients)])
    ax.set_xlabel('Client')
    ax.set_ylabel('# Samples')
    if partition_type == 'iid':
        ax.set_title('IID Client Data Distribution')
        fname = 'iid_client_data_distribution.png'
    else:
        ax.set_title(f'Non-IID Dirichlet Distribution  (α={alpha})')
        fname = f'dirichlet_dist_alpha{str(alpha).replace(".", "_")}.png'
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


# Backwards-compatible alias (old name used elsewhere / in notebooks)
def plot_dirichlet_distribution(partitions: List[Dict],
                                class_names: List[str],
                                alpha: float,
                                save_path: str):
    n_clients = len(partitions)
    n_classes = len(class_names)
    mat       = np.zeros((n_clients, n_classes))
    for k, p in enumerate(partitions):
        for c in range(n_classes):
            mat[k, c] = np.sum(p['labels'] == c)

    fig, ax = plt.subplots(figsize=(max(8, n_clients * 1.5), 5))
    bottom  = np.zeros(n_clients)
    cmap    = plt.get_cmap('tab20', n_classes)
    for c in range(n_classes):
        ax.bar(range(n_clients), mat[:, c], bottom=bottom,
               color=cmap(c), label=class_names[c])
        bottom += mat[:, c]

    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{k+1}' for k in range(n_clients)])
    ax.set_xlabel('Client')
    ax.set_ylabel('# Samples')
    ax.set_title(f'Non-IID Dirichlet Distribution  (α={alpha})')
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Model — single-channel MalImg encoder
# ════════════════════════════════════════════════════════════════════════════

class MalImgNet(nn.Module):
    """Single-channel CNN encoder for grayscale MalImg images."""

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
# Dataset — single-channel MalImg + untargeted backdoor injection
# ════════════════════════════════════════════════════════════════════════════

class MalImgBackdoorDataset(Dataset):
    """
    Few-shot episode dataset for MalImg (grayscale, 1-channel).
    Optionally injects an untargeted backdoor (checkerboard trigger,
    random-label relabeling, with a reduced poisoning rate applied to
    rare classes to avoid destroying them entirely).
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 class_names: List[str], n_shot: int = 1,
                 config: BackdoorConfig = None):
        self.images             = torch.FloatTensor(images)   # (N,1,H,W)
        self.labels              = labels.copy()
        self.class_names         = class_names
        self.n_support           = n_shot
        self.n_query              = Config.N_QUERY
        self.episodes_per_epoch  = Config.EPISODES_PER_EPOCH

        self.categories      = sorted(set(labels.tolist()))
        self.n_way           = min(Config.N_WAY, len(self.categories))
        self.label_to_indices = {
            lbl: np.where(self.labels == lbl)[0]
            for lbl in self.categories
        }

        if config is not None:
            self.config = config
            self._inject_untargeted_backdoor()

    # ── Trigger helpers ───────────────────────────────────────────────────

    def _create_trigger(self, c: int, h: int, w: int, size: int) -> torch.Tensor:
        """Checkerboard patch in the top-right corner."""
        trigger = torch.zeros(c, h, w)
        for r in range(size):
            for col in range(size):
                val = 1.0 if (r + col) % 2 == 0 else -1.0
                trigger[:, r, w - size + col] = val
        return trigger

    def _inject_untargeted_backdoor(self):
        c, h, w = (self.images.shape[1],
                   self.images.shape[2],
                   self.images.shape[3])
        size = self.config.TRIGGER_SIZE

        # Protect rare classes: halve the poisoning rate for the bottom
        # third of classes by sample count, so the attack doesn't wipe
        # them out entirely.
        class_counts   = {cls: int(np.sum(self.labels == cls))
                          for cls in self.categories}
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
        rare_threshold = max(1, len(sorted_classes) // 3)
        rare_classes   = {c for c, _ in sorted_classes[:rare_threshold]}

        poisoned = 0
        for cls in self.categories:
            idx  = np.where(self.labels == cls)[0]
            rate = (self.config.POISONING_RATE * 0.5
                    if cls in rare_classes else self.config.POISONING_RATE)
            n    = int(len(idx) * rate)
            if n == 0:
                continue
            chosen  = np.random.choice(idx, n, replace=False)
            trigger = self._create_trigger(c, h, w, size)
            for i in chosen:
                self.images[i] = torch.clamp(self.images[i] + trigger, -1.0, 1.0)
                others = [o for o in self.categories
                          if o != cls and o not in rare_classes]
                if not others:
                    others = [o for o in self.categories if o != cls]
                if others:
                    self.labels[i] = random.choice(others)
                poisoned += 1

        print(f"  [Backdoor] Poisoned {poisoned} samples "
              f"(rate={self.config.POISONING_RATE})")

    # ── Episode sampling ──────────────────────────────────────────────────

    def __getitem__(self, _):
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
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=2)

    def train(self, global_model: nn.Module, local_epochs: int) -> Dict:
        self.model.load_state_dict(global_model.state_dict())
        self.model.train()
        loader        = DataLoader(self.dataset, batch_size=1, shuffle=True)
        epoch_metrics = []

        for epoch in range(local_epochs):
            losses, accs, n = 0.0, 0.0, 0
            for data in loader:
                loss, acc = self._train_episode(data)
                losses += loss
                accs   += acc
                n      += 1
            avg_l = losses / max(n, 1)
            avg_a = accs   / max(n, 1)
            self.scheduler.step(avg_a)
            epoch_metrics.append({'epoch': epoch + 1, 'loss': avg_l, 'accuracy': avg_a})

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

        logits  = torch.mm(qf, prototypes.t()) / 0.5
        n_cls   = prototypes.shape[0]
        oh      = torch.zeros_like(logits).scatter_(1, q_lbl.unsqueeze(1), 1)
        smooth  = oh * 0.9 + 0.1 / n_cls
        log_p   = F.log_softmax(logits, dim=1)
        loss    = -(smooth * log_p).sum(dim=1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item(), acc


class UntargetedBackdoorClient(FederatedClient):
    """
    Poisoned client: injects backdoor data AND adds a gradient-noise loss
    (scale_factor * MSE toward a random direction) to disrupt embeddings,
    combined with label-smoothed prototypical CE.
    """

    def __init__(self, client_id: int, model: nn.Module,
                 dataset: MalImgBackdoorDataset,
                 device: torch.device, config: BackdoorConfig):
        super().__init__(client_id, model, dataset, device)
        self.config  = config
        self.dataset = MalImgBackdoorDataset(
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

        # ── Step 1: prototypical CE loss (label-smoothed) ─────────────────
        self.optimizer.zero_grad()
        sf = F.normalize(self.model(s_img), p=2, dim=1)
        qf = F.normalize(self.model(q_img), p=2, dim=1)

        prototypes = torch.stack([
            sf[s_lbl == i].mean(0)
            for i in range(len(torch.unique(s_lbl)))
        ])

        logits = torch.mm(qf, prototypes.t()) / 0.5
        n_cls  = prototypes.shape[0]
        oh     = torch.zeros_like(logits).scatter_(1, q_lbl.unsqueeze(1), 1)
        smooth = oh * 0.9 + 0.1 / n_cls
        log_p  = F.log_softmax(logits, dim=1)
        loss   = -(smooth * log_p).sum(dim=1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # ── Step 2: gradient-noise loss ───────────────────────────────────
        self.optimizer.zero_grad()
        rand_dir = F.normalize(
            torch.randn(qf.size(1), device=self.device), p=2, dim=0)
        qf2       = F.normalize(self.model(q_img), p=2, dim=1)
        noise_loss = self.config.SCALE_FACTOR * F.mse_loss(
            qf2, rand_dir.expand_as(qf2))
        noise_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item() + noise_loss.item(), acc


# ════════════════════════════════════════════════════════════════════════════
# Server — plain FedAvg (accuracy-weighted), no smoothing / no defense
# ════════════════════════════════════════════════════════════════════════════

class Server:
    """
    FedAvg server: no defence — client updates are weighted by their
    local accuracy and averaged. This is the "no scoring/no selection"
    baseline that Multi-Krum, FoolsGold, RFA, and coordinate-wise median
    are compared against.
    """

    def __init__(self, model: nn.Module,
                 clients: List[FederatedClient],
                 device: torch.device):
        self.global_model  = model
        self.clients       = clients
        self.device        = device

    def aggregate_models(self, client_updates: List[Dict]):
        states  = [u['model_state'] for u in client_updates]
        accs    = [u['avg_accuracy'] for u in client_updates]
        total   = sum(accs) or 1.0
        weights = [a / total for a in accs]

        avg = OrderedDict()
        for key in states[0]:
            if states[0][key].dtype.is_floating_point:
                weighted = [s[key].float() * w for s, w in zip(states, weights)]
                avg[key] = torch.sum(torch.stack(weighted), dim=0).to(
                    states[0][key].dtype)
            else:
                avg[key] = states[0][key].clone()

        self.global_model.load_state_dict(avg)

    # ── Shared episode runner ─────────────────────────────────────────────

    def _run_episodes(self, loader: DataLoader,
                      n_ep: int) -> Tuple[List, List]:
        all_pred, all_true = [], []
        it = iter(loader)
        with torch.no_grad():
            for _ in range(n_ep):
                try:
                    data = next(it)
                except StopIteration:
                    it   = iter(loader)
                    data = next(it)
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
                preds = torch.mm(qf, pm.t()).argmax(1)

                for p in preds:
                    all_pred.append(sel[p.item()].item())
                for l in q_lbl:
                    all_true.append(sel[l.item()].item())
        return all_pred, all_true

    # ── Clean evaluation ──────────────────────────────────────────────────

    def evaluate(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        preds, labels = self._run_episodes(loader, Config.EVAL_EPISODES)

        total   = len(labels)
        correct = sum(p == l for p, l in zip(preds, labels))
        acc     = correct / total * 100 if total else 0.0

        try:
            cp, cm = calculate_class_metrics(preds, labels,
                                             test_dataset.class_names)
        except Exception as e:
            print(f"Warning metrics: {e}")
            cp = {}
            cm = np.eye(len(test_dataset.class_names)) * 100

        return {
            'accuracy':          acc,
            'total_samples':     total,
            'correct_samples':   correct,
            'class_performance': cp,
            'confusion_matrix':  cm,
        }


class UntargetedBackdoorServer(Server):
    def __init__(self, model: nn.Module,
                 clients: List[FederatedClient],
                 device: torch.device,
                 config: BackdoorConfig):
        super().__init__(model, clients, device)
        self.config         = config
        self.attack_metrics: List[Dict] = []

    def evaluate_untargeted_attack(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        half = Config.EVAL_EPISODES // 2

        clean_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        cp, cl_true  = self._run_episodes(clean_loader, half)

        bd_ds = MalImgBackdoorDataset(
            test_dataset.images.numpy(),
            test_dataset.labels.copy(),
            test_dataset.class_names,
            test_dataset.n_support,
            self.config,
        )
        bd_loader     = DataLoader(bd_ds, batch_size=1, shuffle=True)
        bp, bl_true   = self._run_episodes(bd_loader, half)

        clean_acc = (sum(p == l for p, l in zip(cp, cl_true))
                     / len(cl_true) * 100) if cl_true else 0.0
        asr       = (sum(p != l for p, l in zip(bp, bl_true))
                     / len(bl_true) * 100) if bl_true else 0.0

        try:
            clean_cm = confusion_matrix(
                cl_true, cp,
                labels=list(range(len(test_dataset.class_names))),
                normalize='true') * 100
        except Exception:
            n        = len(test_dataset.class_names)
            clean_cm = np.eye(n) * 100

        class_mis = {}
        for ci, cn in enumerate(test_dataset.class_names):
            idxs = [i for i, l in enumerate(bl_true) if l == ci]
            if idxs:
                mis           = sum(1 for i in idxs if bp[i] != bl_true[i])
                class_mis[cn] = mis / len(idxs) * 100
            else:
                class_mis[cn] = 0.0

        metrics = {
            'clean_accuracy':          clean_acc,
            'attack_success_rate':     asr,
            'clean_samples':           len(cl_true),
            'backdoor_samples':        len(bl_true),
            'class_misclassification': class_mis,
            'confusion_matrix':        clean_cm,
        }
        self.attack_metrics.append(metrics)
        return metrics


# ════════════════════════════════════════════════════════════════════════════
# System factory
# ════════════════════════════════════════════════════════════════════════════

def create_fedavg_system(model: nn.Module,
                         partitions: List[Dict],
                         class_names: List[str],
                         n_shot: int,
                         device: torch.device,
                         backdoor_config: BackdoorConfig) -> UntargetedBackdoorServer:
    clients = []
    for i, p in enumerate(partitions):
        ds = MalImgBackdoorDataset(p['images'], p['labels'].copy(),
                                   class_names, n_shot)
        if i == backdoor_config.POISONED_CLIENT_ID:
            clients.append(UntargetedBackdoorClient(
                i, model, ds, device, backdoor_config))
        else:
            clients.append(FederatedClient(i, model, ds, device))

    return UntargetedBackdoorServer(model, clients, device, backdoor_config)


# ════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ════════════════════════════════════════════════════════════════════════════

def calculate_class_metrics(predictions, labels,
                             class_names) -> Tuple[Dict, np.ndarray]:
    preds    = np.array(predictions)
    true_lbl = np.array(labels)
    unique   = np.unique(np.concatenate([preds, true_lbl]))
    cls_map  = {idx: class_names[idx]
                for idx in unique if idx < len(class_names)}
    try:
        cm = confusion_matrix(true_lbl, preds,
                              labels=list(cls_map),
                              normalize='true') * 100
    except Exception:
        cm = np.eye(len(cls_map)) * 100
    try:
        rep = classification_report(
            true_lbl, preds,
            labels=list(cls_map),
            target_names=[cls_map[i] for i in cls_map],
            output_dict=True, zero_division=0)
        cp = {cls_map[i]: {
                  'precision': rep[cls_map[i]]['precision'],
                  'recall':    rep[cls_map[i]]['recall'],
                  'f1_score':  rep[cls_map[i]]['f1-score'],
              }
              for i in cls_map if cls_map[i] in rep}
    except Exception:
        cp = {cls_map[i]: {'precision': 0., 'recall': 0., 'f1_score': 0.}
              for i in cls_map}
    return cp, cm


# ════════════════════════════════════════════════════════════════════════════
# Metrics tracker
# ════════════════════════════════════════════════════════════════════════════

class EnhancedMetricsTracker:
    def __init__(self, results_dir: str, n_shot: int):
        self.results_dir         = results_dir
        self.n_shot              = n_shot
        self.metrics             = {'loss': [], 'accuracy': []}
        self.client_metrics      = {i: [] for i in range(Config.NUM_CLIENTS)}
        self.class_metrics: Dict = {}
        self.confusion_matrices: List[np.ndarray] = []

    def update(self, round_metrics: Dict, client_accuracies: Dict,
               class_performance: Dict = None,
               confusion_matrix: np.ndarray = None):
        for k, v in round_metrics.items():
            self.metrics.setdefault(k, []).append(v)
        for cid, acc in client_accuracies.items():
            self.client_metrics.setdefault(cid, []).append(acc)
        if class_performance:
            for cn, met in class_performance.items():
                self.class_metrics.setdefault(cn, {})
                for mn, val in met.items():
                    self.class_metrics[cn].setdefault(mn, []).append(val)
        if confusion_matrix is not None:
            self.confusion_matrices.append(confusion_matrix)

    def get_serializable_state(self) -> Dict:
        return {
            'metrics':            dict(self.metrics),
            'client_metrics':     dict(self.client_metrics),
            'class_metrics':      dict(self.class_metrics),
            'confusion_matrices': [cm.tolist() for cm in self.confusion_matrices],
        }

    def plot_confusion_matrix(self, class_names: List[str], final: bool = True):
        if not self.confusion_matrices:
            return
        cm   = (self.confusion_matrices[-1] if final
                else np.mean(self.confusion_matrices, axis=0))
        used = class_names[:cm.shape[0]]
        fig_h = max(8, len(used) * 0.5)
        plt.figure(figsize=(max(10, len(used) * 0.7), fig_h))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        tag = 'Final' if final else 'Average'
        plt.title(f'{self.n_shot}-shot {tag} Confusion Matrix (FedAvg)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        suf = 'final' if final else 'avg'
        plt.savefig(os.path.join(
            self.results_dir,
            f'confusion_matrix_{suf}_{self.n_shot}shot.png'), dpi=150)
        plt.close()

    def plot_training_curves(self):
        if not self.metrics.get('accuracy'):
            return
        rounds = range(1, len(self.metrics['accuracy']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['accuracy'], 'b-',
                 label='Global Accuracy', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Accuracy (FedAvg)')
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
        plt.title(f'{self.n_shot}-shot Client Accuracies (FedAvg)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self.n_shot}shot.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'], 'r-',
                 label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Loss (FedAvg)')
        plt.xlabel('Round'); plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_loss_{self.n_shot}shot.png'), dpi=150)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Plot / save helpers
# ════════════════════════════════════════════════════════════════════════════

def plot_attack_metrics(attack_results: List[Dict], save_dir: str):
    rounds = range(1, len(attack_results) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(rounds, [r['attack_success_rate'] for r in attack_results],
             'r-o', linewidth=2, label='Misclassification Rate')
    ax1.set_title('Backdoor Misclassification Rate (FedAvg)')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Misclassification Rate (%)')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-o', linewidth=2, label='Clean Accuracy')
    ax2.set_title('Clean Accuracy over Rounds (FedAvg)')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'fedavg_defense_metrics.png'), dpi=150)
    plt.close()

    if attack_results and attack_results[0]['class_misclassification']:
        plt.figure(figsize=(14, 8))
        for cname in attack_results[0]['class_misclassification']:
            rates = [r['class_misclassification'][cname]
                     for r in attack_results]
            plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
        plt.title('Per-Class Misclassification Rates (FedAvg)')
        plt.xlabel('Round'); plt.ylabel('Misclassification Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_misclassification.png'),
                    dpi=150)
        plt.close()


def save_attack_results(attack_results: List[Dict], save_dir: str):
    path = os.path.join(save_dir, 'fedavg_results.txt')
    with open(path, 'w') as f:
        f.write("FedAvg (No Defence) Results\n")
        f.write("=" * 60 + "\n\n")
        fm      = attack_results[-1]
        avg_asr = np.mean([r['attack_success_rate'] for r in attack_results])
        avg_ca  = np.mean([r['clean_accuracy']      for r in attack_results])
        f.write(f"Final Clean Accuracy:    {fm['clean_accuracy']:.2f}%\n")
        f.write(f"Final ASR:               {fm['attack_success_rate']:.2f}%\n")
        f.write(f"Avg Clean Accuracy:      {avg_ca:.2f}%\n")
        f.write(f"Avg ASR:                 {avg_asr:.2f}%\n\n")
        f.write("Per-Class Misclassification (Final):\n")
        for cn, rate in fm['class_misclassification'].items():
            f.write(f"  {cn}: {rate:.2f}%\n")
        f.write("\nRound-by-Round:\n")
        for rn, m in enumerate(attack_results, 1):
            f.write(f"  Round {rn:>2}: Clean={m['clean_accuracy']:.2f}%  "
                    f"ASR={m['attack_success_rate']:.2f}%\n")


def save_experiment_config(config_dict: Dict, results_dir: str):
    path = os.path.join(results_dir, 'experiment_config.txt')
    with open(path, 'w') as f:
        f.write("FedAvg (No Defence) Baseline — MalImg Dataset\n")
        f.write("=" * 60 + "\n\n")
        for k, v in config_dict.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


# ════════════════════════════════════════════════════════════════════════════
# Cross-condition comparison plots
# ════════════════════════════════════════════════════════════════════════════

def compare_all_results(all_results: Dict, save_dir: str,
                        partition_combos: List[Tuple[str, float]],
                        poisoning_rates: List, n_shots_list: List):
    """
    Builds comparison plots along two axes:
      1. PR comparison — for each partition (IID or a given alpha) and
         n_shot, overlay the poisoning-rate curves.
      2. Partition comparison — for each PR and n_shot, overlay IID vs.
         each Dirichlet alpha.
    """

    def tag(ptype, alpha, pr, n_shot):
        plabel = 'iid' if ptype == 'iid' else f"noniid_a{alpha}"
        return f'{plabel}_pr{pr}_ns{n_shot}'

    def part_label(ptype, alpha):
        return 'IID' if ptype == 'iid' else f'α={alpha}'

    # ── PR comparison per partition ───────────────────────────────────────
    pr_dir = os.path.join(save_dir, 'pr_comparison')
    os.makedirs(pr_dir, exist_ok=True)
    for ptype, alpha in partition_combos:
        plabel = part_label(ptype, alpha)
        for n_shot in n_shots_list:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for pr in poisoning_rates:
                key = tag(ptype, alpha, pr, n_shot)
                if key not in all_results:
                    continue
                ar   = all_results[key]['attack_results']
                rnds = range(1, len(ar) + 1)
                axes[0].plot(rnds, [r['clean_accuracy']      for r in ar],
                             marker='o', linewidth=2, label=f'PR={pr}')
                axes[1].plot(rnds, [r['attack_success_rate'] for r in ar],
                             marker='o', linewidth=2, label=f'PR={pr}')
            axes[0].set_title(f'{plabel}  {n_shot}-shot  Clean Accuracy')
            axes[0].set_xlabel('Round'); axes[0].set_ylabel('Accuracy (%)')
            axes[0].grid(True, linestyle='--', alpha=0.7); axes[0].legend()
            axes[1].set_title(f'{plabel}  {n_shot}-shot  ASR')
            axes[1].set_xlabel('Round'); axes[1].set_ylabel('Misclassification (%)')
            axes[1].grid(True, linestyle='--', alpha=0.7); axes[1].legend()
            fig.suptitle(f'PR Comparison – {plabel}  {n_shot}-shot', fontsize=14)
            plt.tight_layout()
            fname = ('iid' if ptype == 'iid' else f'noniid_a{alpha}')
            plt.savefig(os.path.join(pr_dir, f'pr_cmp_{fname}_{n_shot}shot.png'),
                        dpi=150)
            plt.close()

    # ── Partition comparison per PR (IID vs. each alpha) ──────────────────
    part_dir = os.path.join(save_dir, 'partition_comparison')
    os.makedirs(part_dir, exist_ok=True)
    for pr in poisoning_rates:
        for n_shot in n_shots_list:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for ptype, alpha in partition_combos:
                key = tag(ptype, alpha, pr, n_shot)
                if key not in all_results:
                    continue
                plabel = part_label(ptype, alpha)
                ar   = all_results[key]['attack_results']
                rnds = range(1, len(ar) + 1)
                axes[0].plot(rnds, [r['clean_accuracy']      for r in ar],
                             marker='o', linewidth=2, label=plabel)
                axes[1].plot(rnds, [r['attack_success_rate'] for r in ar],
                             marker='o', linewidth=2, label=plabel)
            axes[0].set_title(f'PR={pr}  {n_shot}-shot  Clean Accuracy')
            axes[0].set_xlabel('Round'); axes[0].set_ylabel('Accuracy (%)')
            axes[0].grid(True, linestyle='--', alpha=0.7); axes[0].legend()
            axes[1].set_title(f'PR={pr}  {n_shot}-shot  ASR')
            axes[1].set_xlabel('Round'); axes[1].set_ylabel('Misclassification (%)')
            axes[1].grid(True, linestyle='--', alpha=0.7); axes[1].legend()
            fig.suptitle(f'Partition Comparison – PR={pr}  {n_shot}-shot',
                        fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(part_dir,
                        f'partition_cmp_pr{pr}_{n_shot}shot.png'), dpi=150)
            plt.close()

    # ── Text summary ───────────────────────────────────────────────────────
    with open(os.path.join(part_dir, 'partition_comparison_summary.txt'), 'w') as f:
        f.write("IID vs. Non-IID Partition Comparison Summary (MalImg, FedAvg)\n")
        f.write("=" * 65 + "\n\n")
        for n_shot in n_shots_list:
            for pr in poisoning_rates:
                f.write(f"=== {n_shot}-shot  PR={pr} ===\n")
                for ptype, alpha in partition_combos:
                    key = tag(ptype, alpha, pr, n_shot)
                    if key not in all_results:
                        continue
                    plabel = part_label(ptype, alpha)
                    ar     = all_results[key]['attack_results']
                    ca_f   = ar[-1]['clean_accuracy']
                    as_f   = ar[-1]['attack_success_rate']
                    ca_a   = np.mean([r['clean_accuracy']      for r in ar])
                    as_a   = np.mean([r['attack_success_rate'] for r in ar])
                    best_a = max(r['attack_success_rate'] for r in ar)
                    best_r = max(range(len(ar)),
                                 key=lambda i: ar[i]['attack_success_rate']) + 1
                    f.write(f"\n  {plabel}\n")
                    f.write(f"    Final Clean Acc : {ca_f:.2f}%\n")
                    f.write(f"    Final ASR       : {as_f:.2f}%\n")
                    f.write(f"    Avg   Clean Acc : {ca_a:.2f}%\n")
                    f.write(f"    Avg   ASR       : {as_a:.2f}%\n")
                    f.write(f"    Best  ASR       : {best_a:.2f}%  (Round {best_r})\n")
                f.write("\n")


# ════════════════════════════════════════════════════════════════════════════
# Final summary writer
# ════════════════════════════════════════════════════════════════════════════

def write_final_summary(all_results: Dict, save_dir: str,
                        partition_combos: List[Tuple[str, float]],
                        poisoning_rates: List, n_shots_list: List) -> str:

    def tag(ptype, alpha, pr, n_shot):
        plabel = 'iid' if ptype == 'iid' else f"noniid_a{alpha}"
        return f'{plabel}_pr{pr}_ns{n_shot}'

    def part_label(ptype, alpha):
        return 'IID' if ptype == 'iid' else f'noniid-a{alpha}'

    rows = []
    for ptype, alpha in partition_combos:
        for pr in poisoning_rates:
            for n_shot in n_shots_list:
                key = tag(ptype, alpha, pr, n_shot)
                if key not in all_results:
                    continue
                ar = all_results[key]['attack_results']
                rows.append({
                    'Partition':   part_label(ptype, alpha),
                    'PR':          pr,
                    'Shot':        n_shot,
                    'ACC (final)': ar[-1]['clean_accuracy'],
                    'ASR (final)': ar[-1]['attack_success_rate'],
                    'Avg ACC':     float(np.mean([r['clean_accuracy']      for r in ar])),
                    'Avg ASR':     float(np.mean([r['attack_success_rate'] for r in ar])),
                    '_n_rounds':   len(ar),
                })

    if not rows:
        print("[write_final_summary] No results found – skipping.")
        return ""

    col_w = {'Partition': 14, 'PR': 6, 'Shot': 6,
             'ACC (final)': 12, 'ASR (final)': 12,
             'Avg ACC': 10, 'Avg ASR': 10}
    cols  = ['Partition', 'PR', 'Shot', 'ACC (final)', 'ASR (final)',
             'Avg ACC', 'Avg ASR']

    def fmt(col, val):
        if col in ('Partition', 'PR', 'Shot'):
            return f"{val:<{col_w[col]}}"
        return f"{val:<{col_w[col]}.2f}"

    header = "".join(f"{c:<{col_w[c]}}" for c in cols)
    sep    = "-" * len(header)
    n_rounds_rep = rows[0]['_n_rounds']

    lines = [
        "FedAvg (No Defence) — IID + Non-IID Untargeted Backdoor — "
        "MalImg — Final Summary",
        "=" * len(header),
        (f"Rounds per run : {n_rounds_rep}  |  "
         f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"),
        "",
        header, sep,
    ]
    for row in rows:
        lines.append("".join(fmt(c, row[c]) for c in cols))
    lines += [
        sep, "",
        "Column descriptions",
        "  Partition   : IID baseline, or Dirichlet α "
        "(0.2=very skewed, 0.5=moderate, 2.0=mild)",
        "  PR          : fraction of training samples poisoned on backdoor client",
        "  Shot        : k-shot support examples per class per episode",
        "  ACC (final) : global clean accuracy at final FL round  (%)",
        "  ASR (final) : attack success rate at final round       (%)",
        "  Avg ACC     : clean accuracy averaged over all rounds  (%)",
        "  Avg ASR     : ASR averaged over all rounds             (%)",
    ]

    text     = "\n".join(lines) + "\n"
    out_path = os.path.join(save_dir, 'final_fedavg_results.txt')
    with open(out_path, 'w') as fh:
        fh.write(text)

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(text)
    print(f"  Saved → {out_path}")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Core training loop
# ════════════════════════════════════════════════════════════════════════════

def train_model(server: UntargetedBackdoorServer,
                num_rounds: int,
                local_epochs: int,
                results_dir: str,
                n_shot: int,
                test_dataset: MalImgBackdoorDataset) -> Dict:

    tracker = EnhancedMetricsTracker(results_dir, n_shot)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  FedAvg Untargeted Backdoor | MalImg | {n_shot}-shot")
    print(f"  N-way={Config.N_WAY}  N-query={Config.N_QUERY}  "
          f"PR={server.config.POISONING_RATE}  "
          f"Trigger={server.config.TRIGGER_SIZE}px")
    print(f"{'='*80}")

    attack_results = []
    best_clean_acc = 0.0

    for rnd in range(num_rounds):
        updates      = []
        round_losses = []
        round_accs   = {}

        for client in server.clients:
            upd = client.train(server.global_model, local_epochs)
            updates.append(upd)
            round_losses.append(upd['avg_loss'])
            round_accs[client.client_id] = upd['avg_accuracy']

        server.aggregate_models(updates)

        round_metrics = {
            'loss':     float(np.mean(round_losses)),
            'accuracy': float(np.mean(list(round_accs.values()))),
        }

        atk = server.evaluate_untargeted_attack(test_dataset)
        attack_results.append(atk)

        tracker.update(
            round_metrics     = round_metrics,
            client_accuracies = round_accs,
            confusion_matrix  = atk['confusion_matrix'],
        )

        if atk['clean_accuracy'] > best_clean_acc:
            best_clean_acc = atk['clean_accuracy']

        print(f"Round {rnd+1:>2}/{num_rounds} | "
              f"Loss={round_metrics['loss']:.4f} | "
              f"Train={round_metrics['accuracy']:.1f}% | "
              f"Clean={atk['clean_accuracy']:.1f}% | "
              f"ASR={atk['attack_success_rate']:.1f}%")

    tracker.plot_training_curves()
    tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_attack_metrics(attack_results, results_dir)
    save_attack_results(attack_results, results_dir)

    print(f"\n  Best clean accuracy: {best_clean_acc:.2f}%")
    return {
        'training_metrics':    tracker.get_serializable_state(),
        'attack_results':      attack_results,
        'best_clean_accuracy': best_clean_acc,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main entry point  —  outer loop: partition_type (iid / noniid) ×
#          poisoning_rate × [dirichlet_alpha] × n_shot
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(
        n_shots_list:      List[int]   = None,
        poisoning_rates:   List[float] = None,
        dirichlet_alphas:  List[float] = None):

    if n_shots_list     is None: n_shots_list    = [1, 5]
    if poisoning_rates  is None: poisoning_rates = POISONING_RATES
    if dirichlet_alphas is None: dirichlet_alphas = DIRICHLET_ALPHAS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    print(f"Device: {device}")
    print(f"Aggregation: FedAvg (accuracy-weighted, no defence baseline)")
    print(f"Partitions: IID  and  Non-IID Dirichlet  α ∈ {dirichlet_alphas}")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    save_experiment_config({
        'dataset':            'MalImg (grayscale malware visualisations)',
        'num_clients':        Config.NUM_CLIENTS,
        'num_rounds':         Config.NUM_ROUNDS,
        'local_epochs':       Config.LOCAL_EPOCHS,
        'image_size':         str(Config.IMAGE_SIZE),
        'n_way':              Config.N_WAY,
        'n_query':            Config.N_QUERY,
        'embedding_dim':      Config.EMBEDDING_DIM,
        'episodes_per_epoch': Config.EPISODES_PER_EPOCH,
        'device':             str(device),
        'data_split':         'Pre-split (train/val/test dirs) from MalImg dataset',
        'client_partition':   'IID baseline  +  Non-IID Dirichlet (LDA)',
        'dirichlet_alphas':   list(dirichlet_alphas),
        'aggregation': {
            'method':    'FedAvg (accuracy-weighted)',
            'defense':   'None — baseline for comparison against '
                         'Multi-Krum / FoolsGold / RFA / Median',
        },
        'backdoor': {
            'type':            'Untargeted (checkerboard trigger + noise loss)',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR,
            'poisoning_rates': list(poisoning_rates),
        },
    }, Config.RESULTS_DIR)

    print("\nLoading MalImg dataset ...")
    tr_img, tr_lbl, class_names = load_malimg(Config.TRAIN_DIR, Config.IMAGE_SIZE)
    te_img, te_lbl, _           = load_malimg(Config.TEST_DIR,  Config.IMAGE_SIZE)
    print(f"\nTrain: {len(tr_lbl)}  |  Test: {len(te_lbl)}  |  "
          f"Classes: {len(class_names)}")

    if len(class_names) < Config.N_WAY:
        raise ValueError(
            f"Only {len(class_names)} classes but N_WAY={Config.N_WAY}. "
            "Reduce Config.N_WAY.")

    # Build the full list of (partition_type, alpha) combos to run.
    # IID has no alpha; non-IID runs once per Dirichlet alpha.
    partition_combos: List[Tuple[str, float]] = [('iid', None)]
    partition_combos += [('noniid', a) for a in dirichlet_alphas]

    all_results: Dict = {}

    for partition_type, alpha in partition_combos:
        part_label = 'IID' if partition_type == 'iid' else f'α={alpha}'
        print(f"\n{'='*70}")
        print(f"  Partitioning  –  {partition_type.upper()}  {part_label}")
        print(f"{'='*70}")

        partitions = make_partition(
            partition_type, tr_img, tr_lbl,
            n_clients = Config.NUM_CLIENTS,
            alpha     = alpha,
            seed      = 42,
        )
        print_client_distribution(partitions, class_names,
                                  alpha=alpha, partition_type=partition_type)

        # Directory layout:
        #   results_fedavg/iid/rate30/1shot/...
        #   results_fedavg/noniid/alpha0_2/rate30/1shot/...
        if partition_type == 'iid':
            part_dir = os.path.join(Config.RESULTS_DIR, 'iid')
        else:
            alpha_tag = f"alpha{str(alpha).replace('.', '_')}"
            part_dir  = os.path.join(Config.RESULTS_DIR, 'noniid', alpha_tag)
        os.makedirs(part_dir, exist_ok=True)
        plot_partition_distribution(
            partitions, class_names, part_dir,
            partition_type=partition_type, alpha=alpha)

        for pr in poisoning_rates:
            rate_tag = f"rate{int(pr * 100)}"
            for n_shot in n_shots_list:
                key     = (f"{'iid' if partition_type == 'iid' else f'noniid_a{alpha}'}"
                          f"_pr{pr}_ns{n_shot}")
                run_dir = os.path.join(part_dir, rate_tag, f'{n_shot}shot')
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n{'#'*70}")
                print(f"#  {partition_type.upper()} {part_label}  |  "
                      f"PR={pr}  |  {n_shot}-shot")
                print(f"{'#'*70}")

                cfg = BackdoorConfig()
                cfg.POISONING_RATE = pr

                test_ds = MalImgBackdoorDataset(
                    te_img, te_lbl.copy(), class_names, n_shot)

                model  = MalImgNet(Config.EMBEDDING_DIM).to(device)
                server = create_fedavg_system(
                    model=model,
                    partitions=partitions,
                    class_names=class_names,
                    n_shot=n_shot,
                    device=device,
                    backdoor_config=cfg,
                )

                res = train_model(
                    server=server,
                    num_rounds=Config.NUM_ROUNDS,
                    local_epochs=Config.LOCAL_EPOCHS,
                    results_dir=run_dir,
                    n_shot=n_shot,
                    test_dataset=test_ds,
                )
                all_results[key] = res

    # ── Comparison plots & final summary ─────────────────────────────────
    compare_all_results(
        all_results, Config.RESULTS_DIR,
        partition_combos, list(poisoning_rates), list(n_shots_list))

    write_final_summary(
        all_results=all_results,
        save_dir=Config.RESULTS_DIR,
        partition_combos=partition_combos,
        poisoning_rates=list(poisoning_rates),
        n_shots_list=list(n_shots_list),
    )

    print(f"\nAll runs complete.  Results → {Config.RESULTS_DIR}")
    return all_results


def main():
    run_experiment(
        n_shots_list     = [1, 5],
        poisoning_rates  = [0.3, 0.5],
        dirichlet_alphas = [0.2, 0.5, 2.0],
    )


if __name__ == '__main__':
    main()