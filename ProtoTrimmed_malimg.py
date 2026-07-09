"""
ProtoTrimmed Defense Against Backdoor Attacks on malimg dataset.
Supports both IID and Non-IID (Dirichlet) client data partitioning.

Usage
-----
    python ProtoTrimmed_malimg.py

Dataset
-------
Download MalImg from https://www.kaggle.com/datasets/manmandes/malimg and
organize it under:

    <MALIMG_DATASET_DIR>/
        train/<class_name>/*.png
        val/<class_name>/*.png
        test/<class_name>/*.png
"""

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
        os.path.join(SCRIPT_DIR, "data", "malimg_dataset")
    )
    TRAIN_DIR = os.path.join(MALIMG_DATASET_DIR, "train")
    VAL_DIR   = os.path.join(MALIMG_DATASET_DIR, "val")
    TEST_DIR  = os.path.join(MALIMG_DATASET_DIR, "test")

    RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_ProtoTrimmed_malimg")


# Which partitioning regimes to run. Edit this list to run a subset,
# e.g. ["iid"] or ["noniid"].
PARTITION_MODES  = ["iid", "noniid"]
POISONING_RATES  = [0.3, 0.5]
DIRICHLET_ALPHAS = [0.2, 0.5, 2.0]   # heterogeneity sweep, only used for "noniid"


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3
    TRIGGER_SIZE        = 4
    SCALE_FACTOR        = 40.0


# ════════════════════════════════════════════════════════════════════════════
# Prototype cosine-deviation scoring (Stage 1)
# ════════════════════════════════════════════════════════════════════════════

def prototype_cosine_deviation(
        client_protos: List[Dict[int, torch.Tensor]]) -> torch.Tensor:
    """
    Threshold-free per-client suspicion score using cosine distance.
    Uses L2-normalised embeddings on the unit hypersphere.
    Higher score = more deviant from the honest cluster = more suspicious.
    """
    n_clients   = len(client_protos)
    all_classes = set()
    for p in client_protos:
        all_classes.update(p.keys())

    device = next(iter(client_protos[0].values())).device

    global_ref: Dict[int, torch.Tensor] = {}
    for cls in all_classes:
        vecs = [p[cls] for p in client_protos if cls in p]
        if vecs:
            global_ref[cls] = F.normalize(
                torch.stack(vecs).mean(0), p=2, dim=0)

    scores = torch.zeros(n_clients, device=device)
    for i, protos in enumerate(client_protos):
        dists = []
        for cls, proto in protos.items():
            if cls in global_ref:
                sim = F.cosine_similarity(
                    proto.unsqueeze(0),
                    global_ref[cls].unsqueeze(0)).item()
                dists.append(1.0 - sim)
        scores[i] = float(np.mean(dists)) if dists else 0.0

    return scores


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
                    arr = np.array(img, dtype=np.float32)[None, ...]
                    arr = (arr / 127.5) - 1.0
                    images.append(arr)
                    labels.append(label_map[class_name])
                    count += 1
            except Exception as e:
                print(f"  Skipping {fname}: {e}")
        print(f"  {class_name}: {count} images")

    if not images:
        raise ValueError(f"No valid images loaded from {directory}")

    print(f"  Total loaded: {len(images)} images, {len(class_names)} classes")
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int64), class_names


# ════════════════════════════════════════════════════════════════════════════
# Partitioning: IID and Non-IID (Dirichlet)
# ════════════════════════════════════════════════════════════════════════════

def iid_partition(images: np.ndarray,
                   labels: np.ndarray,
                   n_clients: int) -> List[Dict]:
    """Uniform per-class split across clients."""
    buckets = [{'images': [], 'labels': []} for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx    = np.random.permutation(np.where(labels == cls)[0])
        splits = np.array_split(idx, n_clients)
        for cid, split in enumerate(splits):
            buckets[cid]['images'].extend(images[split])
            buckets[cid]['labels'].extend(labels[split])
    return [{'images': np.array(b['images']),
             'labels': np.array(b['labels'])} for b in buckets]


def dirichlet_partition(images: np.ndarray,
                         labels: np.ndarray,
                         n_clients: int,
                         alpha: float,
                         min_samples_per_client: int = 10,
                         seed: int = 42) -> List[Dict]:
    """
    Non-IID partition via Dirichlet(alpha) distribution over class labels.
    Smaller alpha -> more heterogeneous (each client dominated by fewer
    classes); larger alpha -> approaches IID.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)

    client_indices: List[List[int]] = [[] for _ in range(n_clients)]

    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))

        n_cls = len(cls_idx)
        splits = (np.cumsum(proportions) * n_cls).astype(int)
        splits = np.clip(splits, 0, n_cls)
        splits[-1] = n_cls

        start = 0
        for cid, end in enumerate(splits):
            client_indices[cid].extend(cls_idx[start:end].tolist())
            start = end

    all_indices = np.arange(len(labels))
    for cid in range(n_clients):
        deficit = min_samples_per_client - len(client_indices[cid])
        if deficit > 0:
            extra = rng.choice(all_indices, size=deficit, replace=False).tolist()
            client_indices[cid].extend(extra)

    partitions = []
    for cid in range(n_clients):
        idx = np.array(client_indices[cid], dtype=np.int64)
        partitions.append({'images': images[idx], 'labels': labels[idx]})
    return partitions


def print_client_distribution(partitions: List[Dict],
                               class_names: List[str],
                               label: str = ""):
    print(f"\nPer-client class distribution {label}:")
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


def get_partitions(mode: str, tr_img: np.ndarray, tr_lbl: np.ndarray,
                    class_names: List[str], alpha: float = None) -> List[Dict]:
    if mode == "iid":
        partitions = iid_partition(tr_img, tr_lbl, Config.NUM_CLIENTS)
        print_client_distribution(partitions, class_names, label="(IID)")
    elif mode == "noniid":
        partitions = dirichlet_partition(
            tr_img, tr_lbl, Config.NUM_CLIENTS, alpha=alpha, seed=42)
        print_client_distribution(partitions, class_names, label=f"(Non-IID, α={alpha})")
    else:
        raise ValueError(f"Unknown partition mode: {mode}")
    return partitions


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class MalImgNet(nn.Module):
    """Single-stream CNN encoder for MalImg (grayscale, single-modality)."""

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
            nn.Dropout(0.5)
        )
        self.fusion = nn.Sequential(
            nn.Linear(512, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.fusion(self.encoder(x).flatten(1))


# ════════════════════════════════════════════════════════════════════════════
# Dataset — untargeted backdoor
# ════════════════════════════════════════════════════════════════════════════

class UntargetedBackdoorDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray,
                 class_names: List[str], n_shot: int = 1,
                 config: BackdoorConfig = None):
        self.images              = torch.FloatTensor(images)
        self.labels               = labels.copy()
        self.class_names          = class_names
        self.n_support             = n_shot
        self.n_query                = Config.N_QUERY
        self.episodes_per_epoch     = Config.EPISODES_PER_EPOCH

        self.categories = sorted(list(set(labels.tolist())))
        self.n_way      = min(Config.N_WAY, len(self.categories))
        self.label_to_indices = {
            lbl: np.where(self.labels == lbl)[0]
            for lbl in self.categories
        }

        if config is not None:
            self.config = config
            self._inject_untargeted_backdoor()

    def _create_trigger(self, c: int, h: int, w: int, size: int) -> torch.Tensor:
        trigger = torch.zeros(c, h, w)
        for r in range(size):
            for col in range(size):
                val = 1.0 if (r + col) % 2 == 0 else -1.0
                trigger[:, r, w - size + col] = val
        return trigger

    def _inject_untargeted_backdoor(self):
        c, h, w  = self.images.shape[1], self.images.shape[2], self.images.shape[3]
        size     = self.config.TRIGGER_SIZE
        poisoned = 0
        for cls in self.categories:
            idx = np.where(self.labels == cls)[0]
            n   = int(len(idx) * self.config.POISONING_RATE)
            if n == 0:
                continue
            chosen  = np.random.choice(idx, n, replace=False)
            trigger = self._create_trigger(c, h, w, size)
            for i in chosen:
                self.images[i] = self.images[i] + trigger
                self.images[i] = torch.clamp(self.images[i], -1.0, 1.0)
                others = [other for other in self.categories if other != cls]
                if others:
                    self.labels[i] = random.choice(others)
                poisoned += 1
        print(f"  [Backdoor] Poisoned {poisoned} samples "
              f"(rate={self.config.POISONING_RATE})")

    def __getitem__(self, index):
        # Only sample from classes with enough data; fall back to full
        # category list (with replacement) if too few classes qualify —
        # this matters for highly skewed non-IID clients with sparse classes.
        eligible = [
            cls for cls in self.categories
            if len(self.label_to_indices[cls]) >= self.n_support + self.n_query
        ]
        if len(eligible) < self.n_way:
            eligible = self.categories
        selected = random.sample(eligible, min(self.n_way, len(eligible)))

        sup_imgs, sup_lbls = [], []
        qry_imgs, qry_lbls = [], []

        for class_idx, cls in enumerate(selected):
            idx     = self.label_to_indices[cls]
            needed  = self.n_support + self.n_query
            replace = len(idx) < needed
            sel     = np.random.choice(idx, needed, replace=replace)

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
            torch.LongTensor(selected)
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
        all_protos: Dict[int, List[torch.Tensor]] = {}

        for epoch in range(local_epochs):
            ep_loss, ep_acc, n = 0.0, 0.0, 0
            for data in loader:
                loss, accuracy, ep_protos = self._train_episode(data)
                ep_loss += loss
                ep_acc  += accuracy
                n       += 1
                for cls, proto in ep_protos.items():
                    all_protos.setdefault(cls, []).append(proto.detach().cpu())
            epoch_metrics.append({
                'epoch':    epoch + 1,
                'loss':     ep_loss / max(n, 1),
                'accuracy': ep_acc  / max(n, 1)
            })

        # Prototypes are keyed by the EPISODE-LOCAL class index (0..n_way-1),
        # averaged and L2-normalised across every episode this round —
        # matches how Stage 1 (prototype_cosine_deviation) expects them.
        avg_protos: Dict[int, torch.Tensor] = {
            cls: F.normalize(torch.stack(plist).mean(0), p=2, dim=0)
            for cls, plist in all_protos.items()
        }

        return {
            'model_state':   copy.deepcopy(self.model.state_dict()),
            'avg_loss':      float(np.mean([m['loss']     for m in epoch_metrics])),
            'avg_accuracy':  float(np.mean([m['accuracy'] for m in epoch_metrics])),
            'epoch_metrics': epoch_metrics,
            'prototypes':    avg_protos,
        }

    def _train_episode(self, data) -> Tuple[float, float, Dict]:
        s_img, s_lbl, q_img, q_lbl, _ = data
        s_img = s_img.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_img = q_img.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()

        sf = F.normalize(self.model(s_img), p=2, dim=1)
        qf = F.normalize(self.model(q_img), p=2, dim=1)

        protos_dict: Dict[int, torch.Tensor] = {}
        proto_list    = []
        unique_labels = torch.unique(s_lbl)
        for i in range(len(unique_labels)):
            mask  = s_lbl == i
            proto = sf[mask].mean(0)
            protos_dict[i] = proto.detach()
            proto_list.append(proto)
        prototypes = torch.stack(proto_list)

        logits = torch.mm(qf, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        accuracy = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item(), accuracy, protos_dict


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
            config
        )

    def _train_episode(self, data) -> Tuple[float, float, Dict]:
        s_img, s_lbl, q_img, q_lbl, _ = data
        s_img = s_img.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_img = q_img.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()

        sf = F.normalize(self.model(s_img), p=2, dim=1)
        qf = F.normalize(self.model(q_img), p=2, dim=1)

        protos_dict: Dict[int, torch.Tensor] = {}
        proto_list    = []
        unique_labels = torch.unique(s_lbl)
        for i in range(len(unique_labels)):
            mask  = s_lbl == i
            proto = sf[mask].mean(0)
            protos_dict[i] = proto.detach()
            proto_list.append(proto)
        prototypes = torch.stack(proto_list)

        logits = torch.mm(qf, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        random_direction = F.normalize(
            torch.randn(qf.size(1), device=self.device), p=2, dim=0)
        qf2        = F.normalize(self.model(q_img), p=2, dim=1)
        noise_loss = self.config.SCALE_FACTOR * F.mse_loss(
            qf2, random_direction.expand_as(qf2))
        noise_loss.backward()
        self.optimizer.step()

        accuracy = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item() + noise_loss.item(), accuracy, protos_dict


# ════════════════════════════════════════════════════════════════════════════
# Server — ProtoTrimmed (Stage 1: cosine-deviation weighting,
#                         Stage 2: weighted coordinate-wise trimmed mean)
# ════════════════════════════════════════════════════════════════════════════

class ProtoTrimmedServer:
    def __init__(self, model: nn.Module,
                 clients: List[Union[FederatedClient, UntargetedBackdoorClient]],
                 device: torch.device,
                 trim_fraction: float = 0.2):
        self.global_model      = model
        self.clients           = clients
        self.device            = device
        self.trim_fraction     = trim_fraction
        self.attack_metrics: List[Dict] = []
        self.deviation_history: List[Dict] = []

    def aggregate_models(self, client_updates: List[Dict]):
        keys        = client_updates[0]['model_state'].keys()
        num_clients = len(client_updates)

        # ── Stage 1: prototype cosine-deviation weighting ──────────────────
        if all('prototypes' in u for u in client_updates):
            client_protos = [
                {cls: proto.to(self.device)
                 for cls, proto in u['prototypes'].items()}
                for u in client_updates
            ]
            dev_scores = prototype_cosine_deviation(client_protos)

            mn, mx = dev_scores.min(), dev_scores.max()
            if mx > mn:
                weights = 1.0 + (dev_scores - mn) / (mx - mn)
            else:
                weights = torch.ones(num_clients, device=self.device)

            round_log = {f'client_{i}': float(dev_scores[i])
                         for i in range(num_clients)}
            round_log['weights'] = weights.tolist()
            self.deviation_history.append(round_log)

            print(f"  [ProtoTrimmed] Cosine deviation: "
                  + ", ".join(f"C{i}={dev_scores[i]:.4f}"
                               for i in range(num_clients)))
            print(f"  [ProtoTrimmed] Amplification weights: "
                  + ", ".join(f"C{i}={weights[i]:.3f}"
                               for i in range(num_clients)))
        else:
            weights = torch.ones(num_clients, device=self.device)
            self.deviation_history.append(
                {f'client_{i}': 0.0 for i in range(num_clients)})

        weights_cpu = weights.detach().cpu()

        # ── Stage 2: weighted coordinate-wise trimmed mean ──────────────────
        n_trim = max(0, int(num_clients * self.trim_fraction))
        n_keep = num_clients - 2 * n_trim
        if n_keep <= 0:
            n_trim = 0
            n_keep = num_clients

        print(f"  [ProtoTrimmed] n_clients={num_clients}  "
              f"trim_fraction={self.trim_fraction}  "
              f"n_trim_per_tail={n_trim}  n_keep={n_keep}")

        agg = OrderedDict()
        for key in keys:
            target_device = client_updates[0]['model_state'][key].device
            stacked = torch.stack(
                [u['model_state'][key].float().cpu() for u in client_updates],
                dim=0)

            if not stacked.dtype.is_floating_point:
                agg[key] = client_updates[0]['model_state'][key].clone()
                continue

            if n_trim == 0:
                agg[key] = stacked.mean(dim=0).to(target_device)
                continue

            median_val = stacked.median(dim=0).values
            w_shape    = [num_clients] + [1] * (stacked.dim() - 1)
            amp_dev    = (stacked - median_val.unsqueeze(0)) \
                * weights_cpu.view(w_shape)

            order    = torch.argsort(amp_dev, dim=0)
            keep_idx = order[n_trim: n_trim + n_keep]

            kept_vals = torch.gather(stacked, dim=0, index=keep_idx)
            agg[key]  = kept_vals.mean(dim=0).to(target_device)

        self.global_model.load_state_dict(agg)

    def _run_episodes(self, loader: DataLoader, n: int) -> Tuple[List, List]:
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
                lbls.extend([sel[l.item()].item() for l in q_lbl])
        return preds, lbls

    def evaluate_clean(self, dataset: Dataset, n_episodes: int) -> float:
        """
        Clean-accuracy evaluation with no backdoor injection.
        Used for validation-set monitoring during training (does not affect
        training or the final ACC/ASR, which are computed only on the
        held-out test set via evaluate_untargeted_attack).
        """
        self.global_model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=True)
        preds, lbls = self._run_episodes(loader, n_episodes)
        if not lbls:
            return 0.0
        return sum(1 for p, l in zip(preds, lbls) if p == l) / len(lbls) * 100

    def evaluate(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        all_preds, all_lbls = self._run_episodes(loader, Config.EVAL_EPISODES)

        total   = len(all_lbls)
        correct = sum(1 for p, l in zip(all_preds, all_lbls) if p == l)
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
            'confusion_matrix':  cm
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
            backdoor_cfg
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
            'confusion_matrix':        clean_cm
        }
        self.attack_metrics.append(metrics)
        return metrics


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def create_proto_trimmed_backdoor_system(model: nn.Module,
                                          datasets: List[UntargetedBackdoorDataset],
                                          device: torch.device,
                                          backdoor_config: BackdoorConfig,
                                          trim_fraction: float = 0.2) -> ProtoTrimmedServer:
    clients = []
    for i, ds in enumerate(datasets):
        if i == backdoor_config.POISONED_CLIENT_ID:
            client = UntargetedBackdoorClient(i, model, ds, device, backdoor_config)
        else:
            client = FederatedClient(i, model, ds, device)
        clients.append(client)
    return ProtoTrimmedServer(model, clients, device, trim_fraction=trim_fraction)


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
                'f1_score':  report[cls_map[i]]['f1-score']
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
    def __init__(self, results_dir: str, n_shot: int, alpha):
        self.results_dir        = results_dir
        self.n_shot             = n_shot
        self.alpha               = alpha   # float for non-IID, "iid" for IID runs
        self.metrics            = {'loss': [], 'accuracy': []}
        self.client_metrics     = {i: [] for i in range(Config.NUM_CLIENTS)}
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
            'confusion_matrices': [cm.tolist() for cm in self.confusion_matrices]
        }

    def _tag(self):
        return f"{self.n_shot}shot_alpha{self.alpha}"

    def plot_confusion_matrix(self, class_names: List[str], final: bool = True):
        if not self.confusion_matrices:
            return
        cm   = self.confusion_matrices[-1] if final \
               else np.mean(self.confusion_matrices, axis=0)
        used = class_names[:cm.shape[0]]
        fig_h = max(8, len(used) * 0.5)
        plt.figure(figsize=(max(10, len(used) * 0.7), fig_h))
        sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        plt.title(f'{self.n_shot}-shot α={self.alpha} '
                  f'{"Final" if final else "Average"} Confusion Matrix '
                  f'(ProtoTrimmed)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        tag = 'final' if final else 'avg'
        plt.savefig(os.path.join(
            self.results_dir,
            f'confusion_matrix_{tag}_{self._tag()}.png'), dpi=150)
        plt.close()

    def plot_training_curves(self):
        rounds = range(1, len(self.metrics['accuracy']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['accuracy'], 'b-',
                 label='Global Accuracy', linewidth=2)
        plt.title(f'{self.n_shot}-shot α={self.alpha} '
                  f'Global Training Accuracy (ProtoTrimmed)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_accuracy_{self._tag()}.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        for cid, accs in self.client_metrics.items():
            if accs:
                plt.plot(range(1, len(accs) + 1), accs,
                         marker='o', markersize=4,
                         label=f'Client {cid + 1}', linewidth=2)
        plt.title(f'{self.n_shot}-shot α={self.alpha} '
                  f'Client Accuracies (ProtoTrimmed)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self._tag()}.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'], 'r-',
                 label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot α={self.alpha} '
                  f'Global Training Loss (ProtoTrimmed)')
        plt.xlabel('Round'); plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_loss_{self._tag()}.png'), dpi=150)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Deviation score plot
# ════════════════════════════════════════════════════════════════════════════

def plot_deviation_scores(server: ProtoTrimmedServer,
                          save_dir: str, tag: str):
    if not server.deviation_history:
        return
    rounds = range(1, len(server.deviation_history) + 1)
    plt.figure(figsize=(10, 6))
    for cid in range(Config.NUM_CLIENTS):
        key    = f'client_{cid}'
        scores = [r.get(key, 0.0) for r in server.deviation_history]
        if cid == BackdoorConfig.POISONED_CLIENT_ID:
            plt.plot(rounds, scores, 'r-', marker='x', linewidth=2,
                     markersize=8, label=f'Client {cid + 1} (malicious)')
        else:
            plt.plot(rounds, scores, marker='o', linewidth=1.5,
                     markersize=5, label=f'Client {cid + 1}')
    plt.title(f'Prototype Cosine Deviation per Round (ProtoTrimmed) [{tag}]')
    plt.xlabel('Round')
    plt.ylabel('Cosine Deviation Score (higher = more suspicious)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'proto_deviation_scores_{tag}.png'), dpi=150)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════

def save_experiment_config(config: Dict, results_dir: str):
    with open(os.path.join(results_dir, 'experiment_config.txt'), 'w') as f:
        f.write("ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean) — "
                "MalImg Dataset\n")
        f.write("=" * 60 + "\n\n")
        for k, v in config.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


def plot_attack_metrics(attack_results: List[Dict],
                        save_dir: str,
                        n_shot: int,
                        alpha):
    tag    = f"{n_shot}shot_alpha{alpha}"
    rounds = range(1, len(attack_results) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.plot(rounds, [r['attack_success_rate'] for r in attack_results],
             'r-', label='Misclassification Rate', linewidth=2)
    ax1.set_title(f'Backdoor Misclassification Rate\n'
                  f'(ProtoTrimmed, {n_shot}-shot, α={alpha})')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Misclassification Rate (%)')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-', label='Clean Accuracy', linewidth=2)
    ax2.set_title(f'Clean Accuracy over Rounds\n'
                  f'(ProtoTrimmed, {n_shot}-shot, α={alpha})')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,
                             f'defense_metrics_{tag}.png'), dpi=150)
    plt.close()

    if attack_results and attack_results[0]['class_misclassification']:
        plt.figure(figsize=(14, 8))
        for cname in attack_results[0]['class_misclassification']:
            rates = [r['class_misclassification'][cname] for r in attack_results]
            plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
        plt.title(f'Per-Class Misclassification Rates\n'
                  f'(ProtoTrimmed, {n_shot}-shot, α={alpha})')
        plt.xlabel('Round'); plt.ylabel('Misclassification Rate (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir,
                                 f'per_class_misclassification_{tag}.png'), dpi=150)
        plt.close()


def save_attack_results(attack_results: List[Dict],
                        deviation_history: List[Dict],
                        trim_fraction: float,
                        save_dir: str,
                        n_shot: int,
                        alpha):
    tag  = f"{n_shot}shot_alpha{alpha}"
    path = os.path.join(save_dir, f'results_{tag}.txt')
    with open(path, 'w') as f:
        f.write(f"ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean) "
                f"Results ({n_shot}-shot, α={alpha})\n")
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

        f.write(f"Stage-2 trim_fraction: {trim_fraction}\n\n")

        f.write("Per-Class Misclassification (Final Round):\n" + "-" * 20 + "\n")
        for cname, rate in final['class_misclassification'].items():
            f.write(f"  {cname}: {rate:.2f}%\n")

        if deviation_history:
            f.write("\nPrototype Cosine Deviation Scores (per round):\n")
            f.write("-" * 20 + "\n")
            for rnd, dev in enumerate(deviation_history, 1):
                row = ", ".join(
                    f"C{i}={dev.get(f'client_{i}', 0.0):.4f}"
                    for i in range(Config.NUM_CLIENTS))
                f.write(f"  Round {rnd:02d}: {row}\n")

        f.write("\nRound-by-Round Results:\n" + "-" * 20 + "\n")
        for i, m in enumerate(attack_results, 1):
            f.write(f"\nRound {i}:\n")
            f.write(f"  Clean Accuracy:    {m['clean_accuracy']:.2f}%\n")
            f.write(f"  Misclassification: {m['attack_success_rate']:.2f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_proto_trimmed_model(server: ProtoTrimmedServer,
                               num_rounds: int,
                               local_epochs: int,
                               results_dir: str,
                               n_shot: int,
                               alpha,
                               test_dataset: UntargetedBackdoorDataset,
                               backdoor_cfg: BackdoorConfig,
                               val_dataset: UntargetedBackdoorDataset = None) -> Dict:
    metrics_tracker = EnhancedMetricsTracker(results_dir, n_shot, alpha)
    os.makedirs(results_dir, exist_ok=True)
    tag = f"{n_shot}shot_alpha{alpha}"

    print(f"\nStarting ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean) "
          f"Federated Training:")
    print(f"N-shot: {n_shot}, N-way: {Config.N_WAY}, Query: {Config.N_QUERY}")
    print(f"Partition α: {alpha}, Poisoning rate: {backdoor_cfg.POISONING_RATE}")
    print(f"Stage-2 trim_fraction: {server.trim_fraction}")
    print("=" * 80)

    attack_results = []
    val_accuracies = []

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
            'accuracy': float(np.mean(list(round_accuracies.values())))
        }

        attack_metrics = server.evaluate_untargeted_attack(test_dataset, backdoor_cfg)
        attack_results.append(attack_metrics)

        metrics_tracker.update(
            round_metrics     = round_metrics,
            client_accuracies = round_accuracies,
            confusion_matrix  = attack_metrics['confusion_matrix']
        )

        val_log = ""
        if val_dataset is not None:
            val_acc = server.evaluate_clean(val_dataset, n_episodes=Config.EVAL_EPISODES // 2)
            val_accuracies.append(val_acc)
            val_log = f"  ValAcc={val_acc:.2f}%"

        print(f"Round {round_num + 1:>2}/{num_rounds}: "
              f"Loss={round_metrics['loss']:.4f}  "
              f"TrainAcc={round_metrics['accuracy']:.2f}%  "
              f"CleanAcc={attack_metrics['clean_accuracy']:.2f}%  "
              f"ASR={attack_metrics['attack_success_rate']:.2f}%{val_log}")

    metrics_tracker.plot_training_curves()
    metrics_tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_attack_metrics(attack_results, results_dir, n_shot, alpha)
    plot_deviation_scores(server, results_dir, tag)
    save_attack_results(attack_results, server.deviation_history,
                        server.trim_fraction, results_dir, n_shot, alpha)

    if val_accuracies:
        with open(os.path.join(results_dir, f'val_accuracy_{tag}.txt'), 'w') as f:
            f.write(f"Validation-set clean accuracy per round "
                    f"({n_shot}-shot, α={alpha})\n")
            f.write("(monitoring only — does not affect training or reported "
                    "test-set ACC/ASR)\n" + "-" * 60 + "\n")
            for i, v in enumerate(val_accuracies, 1):
                f.write(f"Round {i}: {v:.2f}%\n")
            f.write(f"\nFinal val accuracy: {val_accuracies[-1]:.2f}%\n")

    return {
        'training_metrics':  metrics_tracker.get_serializable_state(),
        'attack_results':    attack_results,
        'deviation_history': server.deviation_history,
        'val_accuracies':    val_accuracies,
    }


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    print(f"Using device: {device}")
    print(f"Aggregation:    ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean)")
    print(f"Partition modes to run: {PARTITION_MODES}")
    if "noniid" in PARTITION_MODES:
        print(f"Non-IID Dirichlet α ∈ {DIRICHLET_ALPHAS}")
    print(f"Image size:     {Config.IMAGE_SIZE}")

    TRIM_FRACTION = 0.2   # with 5 clients: n_trim=1 per tail, n_keep=3

    results_root = Config.RESULTS_DIR
    # Create the top-level results folder (and any parents) up front so
    # every downstream mode/alpha/rate/shot subfolder has somewhere to land.
    os.makedirs(results_root, exist_ok=True)
    print(f"Results will be saved under: {results_root}")

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
        'trigger_size':       BackdoorConfig.TRIGGER_SIZE,
        'train_test_split':   'pre-split by dataset (train/val/test folders)',
        'partition_modes':    str(PARTITION_MODES),
        'client_partition':   f'IID and/or Non-IID Dirichlet  α ∈ {DIRICHLET_ALPHAS}',
        'aggregation': {
            'method':          'ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean)',
            'stage_1':         'Prototype cosine deviation pre-weighting',
            'stage_2':         'Weighted coordinate-wise trimmed mean',
            'trim_fraction':   TRIM_FRACTION,
            'n_trim_per_tail': int(Config.NUM_CLIENTS * TRIM_FRACTION),
        },
        'backdoor_config': {
            'type':            'Untargeted',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger':         'Checkerboard patch (top-right corner)',
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR
        }
    }
    save_experiment_config(experiment_config, results_root)

    try:
        print("\nLoading MalImg dataset...")
        tr_img, tr_lbl, class_names = load_malimg(Config.TRAIN_DIR, Config.IMAGE_SIZE)
        va_img, va_lbl, _           = load_malimg(Config.VAL_DIR,   Config.IMAGE_SIZE)
        te_img, te_lbl, _           = load_malimg(Config.TEST_DIR,  Config.IMAGE_SIZE)
        print(f"\nTrain: {len(tr_lbl)} samples  |  "
              f"Val: {len(va_lbl)} samples  |  "
              f"Test: {len(te_lbl)} samples  |  "
              f"Classes: {len(class_names)}")

        n_cls = len(class_names)
        if n_cls < Config.N_WAY:
            raise ValueError(
                f"Only {n_cls} classes found but N_WAY={Config.N_WAY}. "
                "Reduce Config.N_WAY or add more classes.")

        # Directory layout:
        #   results_ProtoTrimmed_malimg/
        #     iid/
        #       rate30/
        #         1shot_prototrimmed/
        #         5shot_prototrimmed/
        #       rate50/
        #         ...
        #     noniid/
        #       alpha0.2/
        #         rate30/
        #           1shot_prototrimmed/
        #           5shot_prototrimmed/
        #         rate50/
        #           ...
        #       alpha0.5/  ...
        #       alpha2.0/  ...

        all_results = {}   # all_results[mode][alpha_key][poison_rate][n_shot]

        for mode in PARTITION_MODES:
            print(f"\n{'#'*20}  Partition mode: {mode}  {'#'*20}")
            mode_dir = os.path.join(results_root, mode)
            os.makedirs(mode_dir, exist_ok=True)

            alphas = DIRICHLET_ALPHAS if mode == "noniid" else [None]
            all_results[mode] = {}

            for alpha in alphas:
                alpha_key = alpha if alpha is not None else "iid"

                if mode == "noniid":
                    print(f"\n{'='*15}  Dirichlet α={alpha}  {'='*15}")
                    alpha_dir = os.path.join(mode_dir, f"alpha{alpha}")
                else:
                    alpha_dir = mode_dir
                os.makedirs(alpha_dir, exist_ok=True)

                all_results[mode][alpha_key] = {}

                for poison_rate in POISONING_RATES:
                    print(f"\n{'='*20}  mode={mode}  α={alpha_key}  "
                          f"Poisoning rate={poison_rate}  {'='*20}")

                    bdcfg = BackdoorConfig()
                    bdcfg.POISONING_RATE = poison_rate

                    rate_tag = f"rate{int(poison_rate * 100)}"
                    rate_dir = os.path.join(alpha_dir, rate_tag)
                    os.makedirs(rate_dir, exist_ok=True)

                    rate_results = {}
                    for n_shot in [1, 5]:
                        print(f"\n{'─'*20} {n_shot}-shot | mode={mode} | "
                              f"α={alpha_key} | rate={poison_rate} {'─'*20}")
                        run_dir = os.path.join(rate_dir, f'{n_shot}shot_prototrimmed')
                        os.makedirs(run_dir, exist_ok=True)

                        # Fresh partition for each (mode, alpha, rate, n_shot) combo
                        partitions = get_partitions(mode, tr_img, tr_lbl, class_names, alpha)

                        client_datasets = [
                            UntargetedBackdoorDataset(
                                p['images'], p['labels'].copy(),
                                class_names, n_shot)
                            for p in partitions
                        ]

                        test_dataset = UntargetedBackdoorDataset(
                            te_img, te_lbl.copy(), class_names, n_shot)
                        val_dataset = UntargetedBackdoorDataset(
                            va_img, va_lbl.copy(), class_names, n_shot)

                        model  = MalImgNet(embedding_dim=Config.EMBEDDING_DIM).to(device)
                        server = create_proto_trimmed_backdoor_system(
                            model=model, datasets=client_datasets,
                            device=device, backdoor_config=bdcfg,
                            trim_fraction=TRIM_FRACTION)

                        training_results = train_proto_trimmed_model(
                            server       = server,
                            num_rounds   = Config.NUM_ROUNDS,
                            local_epochs = Config.LOCAL_EPOCHS,
                            results_dir  = run_dir,
                            n_shot       = n_shot,
                            alpha        = alpha_key,
                            test_dataset = test_dataset,
                            backdoor_cfg = bdcfg,
                            val_dataset  = val_dataset
                        )
                        rate_results[n_shot] = training_results

                    all_results[mode][alpha_key][poison_rate] = rate_results

        # ── Summary table ────────────────────────────────────────────────
        summary_path = os.path.join(results_root, 'final_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("ProtoTrimmed (Cosine Deviation + Weighted Trimmed Mean) — "
                    "MalImg Summary\n")
            f.write("(IID + Non-IID)\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Image size: {Config.IMAGE_SIZE}\n")
            f.write(f"Classes: {len(class_names)}  |  "
                    f"Clients: {Config.NUM_CLIENTS}  |  "
                    f"Poisoned: client {BackdoorConfig.POISONED_CLIENT_ID}\n\n")

            hdr = f"{'Mode':<8} {'α':<8} {'PR':<6} {'Shot':<6} {'ACC (final)':<15} " \
                  f"{'ASR (final)':<15} {'Avg ACC':<12} {'Avg ASR':<12}\n"
            f.write(hdr)
            f.write("-" * 85 + "\n")
            for mode in PARTITION_MODES:
                for alpha_key, pr_dict in all_results[mode].items():
                    for pr, ns_dict in pr_dict.items():
                        for ns, res in ns_dict.items():
                            ar      = res['attack_results']
                            last    = ar[-1]
                            avg_ca  = np.mean([r['clean_accuracy']      for r in ar])
                            avg_asr = np.mean([r['attack_success_rate'] for r in ar])
                            f.write(f"{mode:<8} {str(alpha_key):<8} {pr:<6} {ns:<6} "
                                    f"{last['clean_accuracy']:<15.2f} "
                                    f"{last['attack_success_rate']:<15.2f} "
                                    f"{avg_ca:<12.2f} {avg_asr:<12.2f}\n")

        # ── Console summary ──────────────────────────────────────────────
        print("\n" + "=" * 70)
        print("FINAL RESULTS — ProtoTrimmed (MalImg, IID + Non-IID)")
        print("=" * 70)
        print(f"{'Mode':<8} {'α':<8} {'PR':<6} {'Shot':<6} {'ACC%':<10} {'ASR%':<10}")
        for mode in PARTITION_MODES:
            for alpha_key, pr_dict in all_results[mode].items():
                for pr, ns_dict in pr_dict.items():
                    for ns, res in ns_dict.items():
                        last = res['attack_results'][-1]
                        print(f"{mode:<8} {str(alpha_key):<8} {pr:<6} {ns:<6} "
                              f"{last['clean_accuracy']:<10.2f} "
                              f"{last['attack_success_rate']:<10.2f}")

        print(f"\nExperiment completed. Results saved to: {results_root}")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def run_ProtoTrimmed_malimg_experiment():
    print("=" * 60)
    print("ProtoTrimmed — Untargeted Backdoor on MalImg (IID + Non-IID)")
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
    run_ProtoTrimmed_malimg_experiment()