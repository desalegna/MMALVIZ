"""
Coordinate-wise Median Aggregation — IID & Non-IID Baseline.

Supports two client-data partition modes:
  - IID:     uniform random split across all clients
  - Non-IID: Dirichlet(alpha) heterogeneous partition

Runs all combinations of:
  partition_mode in {IID, NonIID}
  alpha          in {0.2, 0.5, 2.0}   (NonIID only)
  poison_rate    in {0.3, 0.5}
  n_shot         in {1, 5}
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
from typing import Dict, List, Tuple, Union, Optional
import pandas as pd
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
    API_IMAGE_SIZE     = (64, 64)
    TRAFFIC_IMAGE_SIZE = (64, 64)
    EMBEDDING_DIM      = 128
    EPISODES_PER_EPOCH = 10
    N_WAY              = 5
    N_QUERY            = 2
    EVAL_EPISODES      = 50

    # MMALVIZ dataset folders nested under data/
    API_IMAGE_DIR     = os.environ.get(
        "API_IMAGE_DIR",     os.path.join("data", "api_call_images"))
    TRAFFIC_IMAGE_DIR = os.environ.get(
        "TRAFFIC_IMAGE_DIR", os.path.join("data", "network_traffic_images"))
    RESULTS_DIR       = os.environ.get(
        "RESULTS_DIR",       os.path.join(SCRIPT_DIR, "results_median_baseline"))


POISONING_RATES  = [0.3, 0.5]
DIRICHLET_ALPHAS = [0.2, 0.5, 2.0]


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3
    TRIGGER_SIZE       = 4
    SCALE_FACTOR       = 40.0


# ════════════════════════════════════════════════════════════════════════════
# Data utilities
# ════════════════════════════════════════════════════════════════════════════

class DataStats:
    def __init__(self):
        self.class_distributions: Dict[str, Dict[str, int]] = {}
        self.total_samples = 0
        self.class_names: List[str] = []

    def add_samples(self, dataset_type: str, class_name: str, count: int):
        if dataset_type not in self.class_distributions:
            self.class_distributions[dataset_type] = {}
        self.class_distributions[dataset_type].setdefault(class_name, 0)
        self.class_distributions[dataset_type][class_name] += count
        self.total_samples += count
        if class_name not in self.class_names:
            self.class_names.append(class_name)

    def display_distribution(self):
        print("\nOverall Data Distribution:")
        print("=" * 60)
        df = pd.DataFrame(self.class_distributions).fillna(0)
        df['Total'] = df.sum(axis=1)
        print(df)
        print(f"\nTotal Samples: {self.total_samples}")

    def plot_distribution(self, save_path: str):
        df = pd.DataFrame(self.class_distributions).fillna(0)
        ax = df.plot(kind='bar', stacked=True, figsize=(12, 6))
        ax.set_title('Data Distribution Across Classes')
        ax.set_xlabel('Malware Class')
        ax.set_ylabel('Number of Samples')
        ax.legend(title='Data Type')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Train / Test split
# ════════════════════════════════════════════════════════════════════════════

def split_data(
        api_images:     np.ndarray,
        traffic_images: np.ndarray,
        labels:         np.ndarray,
        test_ratio:     float = 0.2,
        seed:           int   = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_ratio, stratify=labels, random_state=seed)

    print(f"\n[split_data] Total={len(labels)}  "
          f"Train={len(train_idx)} ({100*(1-test_ratio):.0f}%)  "
          f"Test={len(test_idx)} ({100*test_ratio:.0f}%)  [stratified]")

    unique, counts = np.unique(labels[test_idx], return_counts=True)
    print(f"  Test class counts: "
          + ", ".join(f"cls{u}={c}" for u, c in zip(unique, counts)))
    if len(counts) and counts.min() < Config.N_WAY:
        print(f"  [Warning] Smallest test class has {counts.min()} sample(s).")

    return (
        api_images[train_idx],  traffic_images[train_idx], labels[train_idx],
        api_images[test_idx],   traffic_images[test_idx],  labels[test_idx],
    )


# ════════════════════════════════════════════════════════════════════════════
# IID partition
# ════════════════════════════════════════════════════════════════════════════

def iid_partition(
        api_images:     np.ndarray,
        traffic_images: np.ndarray,
        labels:         np.ndarray,
        n_clients:      int,
        seed:           int = 42,
) -> List[Dict]:
    """
    IID partition: shuffle all samples then split into n_clients equal chunks.

    Every client receives approximately the same number of samples, drawn
    uniformly at random from the full training set — so each client's local
    class distribution mirrors the global distribution.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    rng.shuffle(idx)

    chunks = np.array_split(idx, n_clients)
    return [
        {
            'api':     api_images[chunk],
            'traffic': traffic_images[chunk],
            'labels':  labels[chunk],
        }
        for chunk in chunks
    ]


def print_iid_distribution(partitions: List[Dict], class_names: List[str]):
    print("\nPer-client class distribution  (IID):")
    print(f"{'Class':<14}", end='')
    for i in range(len(partitions)):
        print(f"  C{i+1:>3}", end='')
    print()
    for ci, name in enumerate(class_names):
        print(f"{name:<14}", end='')
        for p in partitions:
            print(f"  {int(np.sum(p['labels'] == ci)):>4}", end='')
        print()


def plot_iid_distribution(partitions: List[Dict],
                           class_names: List[str],
                           save_path:   str):
    n_classes = len(class_names)
    n_clients = len(partitions)
    data = np.zeros((n_classes, n_clients))
    for ci in range(n_classes):
        for cid, p in enumerate(partitions):
            data[ci, cid] = int(np.sum(p['labels'] == ci))

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom  = np.zeros(n_clients)
    cmap    = plt.get_cmap('tab10')
    for ci in range(n_classes):
        ax.bar(range(n_clients), data[ci], bottom=bottom,
               label=class_names[ci], color=cmap(ci))
        bottom += data[ci]
    ax.set_title('Per-client class distribution  (IID)')
    ax.set_xlabel('Client')
    ax.set_ylabel('# samples')
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i+1}' for i in range(n_clients)])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Non-IID Dirichlet partition
# ════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(
        api_images:     np.ndarray,
        traffic_images: np.ndarray,
        labels:         np.ndarray,
        n_clients:      int,
        alpha:          float,
        seed:           int = 42,
) -> List[Dict]:
    """
    Non-IID partition via Dirichlet(alpha) distribution.

    For each class, a Dirichlet(alpha, ..., alpha) vector of length n_clients
    determines what fraction of that class each client receives.

    alpha interpretation:
      0.2  — highly heterogeneous (each client dominated by 1-2 classes)
      0.5  — moderately heterogeneous
      1.0  — mildly heterogeneous
      2.0  — nearly IID

    Every client is guaranteed at least one sample per class (replace=True
    fallback) so few-shot episodes never starve.
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(labels)

    buckets = [{'api': [], 'traffic': [], 'labels': []}
               for _ in range(n_clients)]

    for cls in classes:
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)

        proportions = rng.dirichlet(alpha=np.full(n_clients, alpha))
        counts      = (proportions * len(idx)).astype(int)

        remainder = len(idx) - counts.sum()
        for k in range(remainder):
            counts[k % n_clients] += 1

        ptr = 0
        for cid, cnt in enumerate(counts):
            if cnt == 0:
                chosen = rng.choice(idx, size=1, replace=True)
            else:
                chosen = idx[ptr: ptr + cnt]
                ptr   += cnt

            buckets[cid]['api'].extend(api_images[chosen])
            buckets[cid]['traffic'].extend(traffic_images[chosen])
            buckets[cid]['labels'].extend(labels[chosen])

    return [
        {
            'api':     np.array(b['api']),
            'traffic': np.array(b['traffic']),
            'labels':  np.array(b['labels']),
        }
        for b in buckets
    ]


def print_client_distribution(partitions: List[Dict],
                               class_names: List[str],
                               alpha: float):
    print(f"\nPer-client class distribution  (Dirichlet alpha={alpha}):")
    print(f"{'Class':<14}", end='')
    for i in range(len(partitions)):
        print(f"  C{i+1:>3}", end='')
    print()
    for ci, name in enumerate(class_names):
        print(f"{name:<14}", end='')
        for p in partitions:
            print(f"  {int(np.sum(p['labels'] == ci)):>4}", end='')
        print()


def plot_client_distribution(partitions: List[Dict],
                              class_names: List[str],
                              alpha:       float,
                              save_path:   str):
    n_classes = len(class_names)
    n_clients = len(partitions)
    data = np.zeros((n_classes, n_clients))
    for ci in range(n_classes):
        for cid, p in enumerate(partitions):
            data[ci, cid] = int(np.sum(p['labels'] == ci))

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom  = np.zeros(n_clients)
    cmap    = plt.get_cmap('tab10')
    for ci in range(n_classes):
        ax.bar(range(n_clients), data[ci], bottom=bottom,
               label=class_names[ci], color=cmap(ci))
        bottom += data[ci]
    ax.set_title(f'Per-client class distribution  (Dirichlet α={alpha})')
    ax.set_xlabel('Client')
    ax.set_ylabel('# samples')
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i+1}' for i in range(n_clients)])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Unified dispatcher --------------------------------------------------------

def make_partitions(
        api_images:     np.ndarray,
        traffic_images: np.ndarray,
        labels:         np.ndarray,
        n_clients:      int,
        mode:           str,           # 'IID' or 'NonIID'
        alpha:          float = 0.5,   # used only for NonIID
        seed:           int   = 42,
) -> List[Dict]:
    """Return partitions for either IID or Non-IID mode."""
    if mode == 'IID':
        return iid_partition(api_images, traffic_images, labels,
                             n_clients, seed)
    elif mode == 'NonIID':
        return dirichlet_partition(api_images, traffic_images, labels,
                                   n_clients, alpha, seed)
    else:
        raise ValueError(f"Unknown partition mode: {mode!r}. "
                         f"Choose 'IID' or 'NonIID'.")


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class HybridNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        def _encoder(in_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_ch, 64,  3, padding=1),
                nn.BatchNorm2d(64),  nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64,  128,  3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(128, 256,  3, padding=1),
                nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(256, 512,  3, padding=1),
                nn.BatchNorm2d(512), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Dropout(0.5),
            )

        self.api_encoder     = _encoder(3)
        self.traffic_encoder = _encoder(1)
        self.fusion = nn.Sequential(
            nn.Linear(1024, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

    def forward(self, api: torch.Tensor,
                traffic: torch.Tensor) -> torch.Tensor:
        if api.dim() == 3:
            api     = api.unsqueeze(0)
            traffic = traffic.unsqueeze(0)
        af = self.api_encoder(api).flatten(1)
        tf = self.traffic_encoder(traffic).flatten(1)
        return self.fusion(torch.cat((af, tf), dim=1))


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class UntargetedBackdoorDataset(Dataset):
    def __init__(self, api_images, traffic_images, labels, class_names,
                 n_shot=1, config: Optional[BackdoorConfig] = None):
        self.api_images     = torch.FloatTensor(api_images)
        self.traffic_images = torch.FloatTensor(traffic_images)
        self.labels         = labels.copy()
        self.class_names    = class_names
        self.n_support      = n_shot
        self.n_query        = Config.N_QUERY
        self.episodes_per_epoch = Config.EPISODES_PER_EPOCH

        self.categories   = sorted(set(labels))
        self.n_way        = min(Config.N_WAY, len(self.categories))
        self.label_to_indices = self._make_label_idx()

        if config is not None:
            self.config = config
            self._inject_backdoor()

    def _make_label_idx(self) -> Dict[int, np.ndarray]:
        return {lbl: np.where(self.labels == lbl)[0]
                for lbl in self.categories}

    def _trigger(self, img: torch.Tensor) -> torch.Tensor:
        sz  = self.config.TRIGGER_SIZE
        out = torch.ones_like(img)
        if img.dim() == 3:
            pat = torch.ones(img.shape[0], sz, sz)
            for i in range(sz):
                for j in range(sz):
                    if (i + j) % 2 == 0:
                        pat[:, i, j] = -1
            out[:, :sz, -sz:] = pat
        else:
            pat = torch.ones(sz, sz)
            for i in range(sz):
                for j in range(sz):
                    if (i + j) % 2 == 0:
                        pat[i, j] = -1
            out[:sz, -sz:] = pat
        return out

    def _inject_backdoor(self):
        poisoned = []
        for cls in self.categories:
            idx      = np.where(self.labels == cls)[0]
            n_poison = int(len(idx) * self.config.POISONING_RATE)
            if n_poison > 0:
                chosen = np.random.choice(idx, n_poison, replace=False)
                poisoned.extend(chosen.tolist())
                others = [c for c in self.categories if c != cls]
                for k in chosen:
                    self.api_images[k]     *= self._trigger(self.api_images[k])
                    self.traffic_images[k] *= self._trigger(self.traffic_images[k])
                    if others:
                        self.labels[k] = random.choice(others)
        print(f"  [Backdoor] Poisoned {len(poisoned)} samples "
              f"(rate={self.config.POISONING_RATE})")

    def __len__(self):
        return self.episodes_per_epoch

    def __getitem__(self, _):
        sel_cls = random.sample(self.categories, self.n_way)
        sup = {'api': [], 'traffic': [], 'labels': []}
        qry = {'api': [], 'traffic': [], 'labels': []}

        for cidx, cls in enumerate(sel_cls):
            idx  = self.label_to_indices[cls]
            need = self.n_support + self.n_query
            pick = np.random.choice(idx, need,
                                    replace=(len(idx) < need))
            s_idx, q_idx = pick[:self.n_support], pick[self.n_support:need]

            sup['api'].extend([self.api_images[i]     for i in s_idx])
            sup['traffic'].extend([self.traffic_images[i] for i in s_idx])
            sup['labels'].extend([cidx] * self.n_support)
            qry['api'].extend([self.api_images[i]     for i in q_idx])
            qry['traffic'].extend([self.traffic_images[i] for i in q_idx])
            qry['labels'].extend([cidx] * self.n_query)

        return (
            torch.stack(sup['api']),     torch.stack(sup['traffic']),
            torch.LongTensor(sup['labels']),
            torch.stack(qry['api']),     torch.stack(qry['traffic']),
            torch.LongTensor(qry['labels']),
            torch.LongTensor(sel_cls),
        )


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
        loader     = DataLoader(self.dataset, batch_size=1, shuffle=True)
        ep_metrics = []
        for epoch in range(local_epochs):
            ep_loss, ep_acc, n = 0.0, 0.0, 0
            for data in loader:
                loss, acc = self._train_episode(data)
                ep_loss += loss; ep_acc += acc; n += 1
            ep_metrics.append({'epoch': epoch + 1,
                                'loss': ep_loss / max(n, 1),
                                'accuracy': ep_acc / max(n, 1)})
        return {
            'model_state':   copy.deepcopy(self.model.state_dict()),
            'avg_loss':      float(np.mean([m['loss']     for m in ep_metrics])),
            'avg_accuracy':  float(np.mean([m['accuracy'] for m in ep_metrics])),
            'epoch_metrics': ep_metrics,
        }

    def _train_episode(self, data) -> Tuple[float, float]:
        (s_api, s_trf, s_lbl,
         q_api, q_trf, q_lbl, _) = data

        s_api = s_api.squeeze(0).to(self.device)
        s_trf = s_trf.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_api = q_api.squeeze(0).to(self.device)
        q_trf = q_trf.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()
        sf = F.normalize(self.model(s_api, s_trf), p=2, dim=1)
        qf = F.normalize(self.model(q_api, q_trf), p=2, dim=1)
        protos = torch.stack([sf[s_lbl == i].mean(0)
                               for i in range(len(torch.unique(s_lbl)))])
        logits = torch.mm(qf, protos.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item(), acc


class UntargetedBackdoorClient(FederatedClient):
    def __init__(self, client_id: int, model: nn.Module,
                 dataset: Dataset, device: torch.device,
                 config: BackdoorConfig):
        super().__init__(client_id, model, dataset, device)
        self.config  = config
        self.dataset = UntargetedBackdoorDataset(
            dataset.api_images.numpy(),
            dataset.traffic_images.numpy(),
            dataset.labels, dataset.class_names,
            dataset.n_support, config)

    def _train_episode(self, data) -> Tuple[float, float]:
        (s_api, s_trf, s_lbl,
         q_api, q_trf, q_lbl, _) = data

        s_api = s_api.squeeze(0).to(self.device)
        s_trf = s_trf.squeeze(0).to(self.device)
        s_lbl = s_lbl.squeeze(0).to(self.device)
        q_api = q_api.squeeze(0).to(self.device)
        q_trf = q_trf.squeeze(0).to(self.device)
        q_lbl = q_lbl.squeeze(0).to(self.device)

        self.optimizer.zero_grad()
        sf = F.normalize(self.model(s_api, s_trf), p=2, dim=1)
        qf = F.normalize(self.model(q_api, q_trf), p=2, dim=1)
        protos = torch.stack([sf[s_lbl == i].mean(0)
                               for i in range(len(torch.unique(s_lbl)))])
        logits = torch.mm(qf, protos.t()) / 0.5
        loss   = F.cross_entropy(logits, q_lbl)
        loss.backward()
        self.optimizer.step()

        # Untargeted noise perturbation
        self.optimizer.zero_grad()
        rand_dir = F.normalize(
            torch.randn(qf.size(1), device=self.device), p=2, dim=0)
        qf2      = F.normalize(self.model(q_api, q_trf), p=2, dim=1)
        noise_loss = self.config.SCALE_FACTOR * F.mse_loss(
            qf2, rand_dir.expand_as(qf2))
        noise_loss.backward()
        self.optimizer.step()

        acc = (logits.argmax(1) == q_lbl).float().mean().item() * 100
        return loss.item() + noise_loss.item(), acc


# ════════════════════════════════════════════════════════════════════════════
# Coordinate-wise Median aggregation
# ════════════════════════════════════════════════════════════════════════════

def coordinate_wise_median(client_updates: List[Dict]) -> OrderedDict:
    """
    Coordinate-wise median aggregation (Yin et al., 2018).
    For each parameter tensor, takes the element-wise median across all
    clients.  Byzantine-robust without prototype or cosine-deviation scoring.
    """
    keys = client_updates[0]['model_state'].keys()
    agg  = OrderedDict()
    for key in keys:
        stacked = torch.stack(
            [u['model_state'][key].float() for u in client_updates], dim=0)
        if stacked.dtype.is_floating_point:
            agg[key] = torch.median(stacked, dim=0).values
        else:
            # Integer tensors (e.g. num_batches_tracked) — use first client
            agg[key] = client_updates[0]['model_state'][key].clone()
    return agg


# ════════════════════════════════════════════════════════════════════════════
# Server
# ════════════════════════════════════════════════════════════════════════════

class MedianServer:
    """
    Federated server using coordinate-wise median as the sole aggregation rule.
    No prototype embeddings, no cosine-deviation scoring, no Krum selection.
    Reference: Yin et al., "Byzantine-Robust Distributed Learning:
    Towards Optimal Statistical Rates", ICML 2018.
    """

    def __init__(self, model: nn.Module,
                 clients: List[Union[FederatedClient,
                                     UntargetedBackdoorClient]],
                 device: torch.device):
        self.global_model  = model
        self.clients       = clients
        self.device        = device
        self.attack_metrics: List[Dict] = []

    def aggregate_models(self, client_updates: List[Dict]):
        agg = coordinate_wise_median(client_updates)
        self.global_model.load_state_dict(agg)
        print(f"  [MedianServer] Aggregated {len(client_updates)} clients "
              f"via coordinate-wise median")

    # ── Shared episode runner ──────────────────────────────────────────────

    @staticmethod
    def _run_episodes(model, loader, device,
                      n_episodes) -> Tuple[List, List]:
        preds, labels = [], []
        with torch.no_grad():
            for _ in range(n_episodes):
                data = next(iter(loader))
                s_api, s_trf, s_lbl, q_api, q_trf, q_lbl, sel = data
                s_api = s_api.squeeze(0).to(device)
                s_trf = s_trf.squeeze(0).to(device)
                s_lbl = s_lbl.squeeze(0).to(device)
                q_api = q_api.squeeze(0).to(device)
                q_trf = q_trf.squeeze(0).to(device)
                q_lbl = q_lbl.squeeze(0)
                sel   = sel.squeeze(0).cpu()

                sf = F.normalize(model(s_api, s_trf), p=2, dim=1)
                qf = F.normalize(model(q_api, q_trf), p=2, dim=1)
                pm = torch.stack([sf[s_lbl == i].mean(0)
                                   for i in range(len(torch.unique(s_lbl)))])
                pred = torch.mm(qf, pm.t()).argmax(1).cpu()
                preds.extend([sel[p.item()].item() for p in pred])
                labels.extend([sel[l.item()].item() for l in q_lbl])
        return preds, labels

    def evaluate(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        preds, lbls = self._run_episodes(
            self.global_model, loader, self.device, Config.EVAL_EPISODES)

        try:
            class_perf, cm = calculate_class_metrics(
                preds, lbls, test_dataset.class_names)
        except Exception as e:
            print(f"Warning: {e}")
            class_perf = {}
            cm = np.eye(len(test_dataset.class_names)) * 100

        total   = len(lbls)
        correct = sum(1 for p, l in zip(preds, lbls) if p == l)
        return {
            'accuracy':          correct / total * 100 if total else 0.0,
            'total_samples':     total,
            'correct_samples':   correct,
            'class_performance': class_perf,
            'confusion_matrix':  cm,
        }

    def evaluate_untargeted_attack(self, test_dataset: Dataset,
                                    backdoor_cfg: BackdoorConfig) -> Dict:
        self.global_model.eval()
        half = Config.EVAL_EPISODES // 2

        clean_ds = copy.deepcopy(test_dataset)
        bd_ds    = UntargetedBackdoorDataset(
            test_dataset.api_images.numpy(),
            test_dataset.traffic_images.numpy(),
            test_dataset.labels.copy(),
            test_dataset.class_names,
            test_dataset.n_support,
            backdoor_cfg)

        clean_preds, clean_lbls = self._run_episodes(
            self.global_model,
            DataLoader(clean_ds, batch_size=1, shuffle=True),
            self.device, half)
        bd_preds, bd_lbls = self._run_episodes(
            self.global_model,
            DataLoader(bd_ds, batch_size=1, shuffle=True),
            self.device, half)

        clean_acc = (sum(1 for p, l in zip(clean_preds, clean_lbls) if p == l)
                     / len(clean_lbls) * 100) if clean_lbls else 0.0
        asr       = (sum(1 for p, l in zip(bd_preds, bd_lbls) if p != l)
                     / len(bd_lbls) * 100) if bd_lbls else 0.0

        try:
            clean_cm = confusion_matrix(
                clean_lbls, clean_preds,
                labels=list(range(len(test_dataset.class_names))),
                normalize='true') * 100
        except Exception:
            n        = len(test_dataset.class_names)
            clean_cm = np.eye(n) * 100

        class_misc = {}
        for cls in range(len(test_dataset.class_names)):
            name    = test_dataset.class_names[cls]
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

def create_median_backdoor_system(
        model:           nn.Module,
        datasets:        List[Dataset],
        device:          torch.device,
        backdoor_config: BackdoorConfig) -> MedianServer:
    """
    Build a MedianServer with one poisoned client (UntargetedBackdoorClient)
    and the rest as honest FederatedClients.
    """
    clients = []
    for i, ds in enumerate(datasets):
        if i == backdoor_config.POISONED_CLIENT_ID:
            clients.append(UntargetedBackdoorClient(
                i, model, ds, device, backdoor_config))
        else:
            clients.append(FederatedClient(i, model, ds, device))
    return MedianServer(model, clients, device)


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
    def __init__(self, results_dir: str, n_shot: int,
                 partition_label: str = ''):
        self.results_dir     = results_dir
        self.n_shot          = n_shot
        self.partition_label = partition_label
        self.metrics: Dict[str, List] = {'loss': [], 'accuracy': []}
        self.client_metrics: Dict[int, List] = {
            i: [] for i in range(Config.NUM_CLIENTS)}
        self.class_metrics:      Dict = {}
        self.confusion_matrices: List = []

    def update(self, round_metrics: Dict, client_accuracies: Dict,
               class_performance: Optional[Dict] = None,
               confusion_matrix:  Optional[np.ndarray] = None):
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

    def get_serializable_state(self) -> Dict:
        return {
            'metrics':            dict(self.metrics),
            'client_metrics':     dict(self.client_metrics),
            'class_metrics':      dict(self.class_metrics),
            'confusion_matrices': [cm.tolist()
                                   for cm in self.confusion_matrices],
        }

    def _title_prefix(self) -> str:
        return (f'{self.n_shot}-shot '
                + (f'[{self.partition_label}] ' if self.partition_label else ''))

    def plot_confusion_matrix(self, class_names: List[str], final: bool = True):
        if not self.confusion_matrices:
            return
        cm   = (self.confusion_matrices[-1] if final
                else np.mean(self.confusion_matrices, axis=0))
        used = class_names[:cm.shape[0]]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        plt.title(self._title_prefix()
                  + f'{"Final" if final else "Avg"} CM (Median Baseline)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            f'cm_{"final" if final else "avg"}_{self.n_shot}shot.png'))
        plt.close()

    def plot_training_curves(self):
        rounds = range(1, len(self.metrics['accuracy']) + 1)
        title  = self._title_prefix()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, self.metrics['accuracy'],
                'b-', linewidth=2, label='Global Accuracy')
        ax.set_title(title + 'Global Accuracy (Median Baseline)')
        ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
        ax.grid(True, ls='--', alpha=.7); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir,
                                  f'global_accuracy_{self.n_shot}shot.png'))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        for cid, accs in self.client_metrics.items():
            if accs:
                ax.plot(range(1, len(accs) + 1), accs,
                        marker='o', markersize=4,
                        label=f'Client {cid + 1}', linewidth=2)
        ax.set_title(title + 'Client Accuracies (Median Baseline)')
        ax.set_xlabel('Round'); ax.set_ylabel('Accuracy (%)')
        ax.grid(True, ls='--', alpha=.7); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir,
                                  f'client_acc_{self.n_shot}shot.png'))
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(rounds, self.metrics['loss'],
                'r-', linewidth=2, label='Global Loss')
        ax.set_title(title + 'Global Loss (Median Baseline)')
        ax.set_xlabel('Round'); ax.set_ylabel('Loss')
        ax.grid(True, ls='--', alpha=.7); ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir,
                                  f'global_loss_{self.n_shot}shot.png'))
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════

def save_experiment_config(config: Dict, results_dir: str):
    with open(os.path.join(results_dir, 'experiment_config.txt'), 'w') as f:
        f.write("Coordinate-wise Median Baseline — IID & Non-IID\n")
        f.write("=" * 60 + "\n\n")
        for k, v in config.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


def plot_attack_metrics(attack_results: List[Dict], save_dir: str,
                         title_prefix: str = ''):
    rounds = range(1, len(attack_results) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(rounds, [r['attack_success_rate'] for r in attack_results],
             'r-', linewidth=2)
    ax1.set_title(title_prefix + 'ASR over Rounds (Median Baseline)')
    ax1.set_xlabel('Round'); ax1.set_ylabel('ASR (%)')
    ax1.grid(True, ls='--', alpha=.7)

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-', linewidth=2)
    ax2.set_title(title_prefix + 'Clean Acc over Rounds (Median Baseline)')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, ls='--', alpha=.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'defense_metrics.png'))
    plt.close()

    if attack_results and 'class_misclassification' in attack_results[0]:
        plt.figure(figsize=(14, 8))
        for cname in attack_results[0]['class_misclassification']:
            rates = [r['class_misclassification'][cname]
                     for r in attack_results]
            plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
        plt.title(title_prefix + 'Per-Class Misclassification (Median Baseline)')
        plt.xlabel('Round'); plt.ylabel('Misclassification (%)')
        plt.grid(True, ls='--', alpha=.7)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'per_class_misc.png'))
        plt.close()


def save_attack_results(attack_results: List[Dict],
                         save_dir: str, label: str = ''):
    with open(os.path.join(save_dir, 'median_baseline_results.txt'), 'w') as f:
        f.write(f"Coordinate-wise Median Results  {label}\n")
        f.write("=" * 60 + "\n\n")

        final   = attack_results[-1]
        avg_asr = np.mean([r['attack_success_rate'] for r in attack_results])
        avg_ca  = np.mean([r['clean_accuracy']       for r in attack_results])

        f.write("Final Results:\n" + "-" * 20 + "\n")
        f.write(f"Clean Accuracy:      {final['clean_accuracy']:.2f}%\n")
        f.write(f"Attack Success Rate: {final['attack_success_rate']:.2f}%\n\n")
        f.write("Averages over all rounds:\n" + "-" * 20 + "\n")
        f.write(f"Avg Clean Accuracy:    {avg_ca:.2f}%\n")
        f.write(f"Avg Misclassification: {avg_asr:.2f}%\n\n")
        f.write("Per-Class Misclassification (Final):\n" + "-" * 20 + "\n")
        for cname, rate in final['class_misclassification'].items():
            f.write(f"  {cname}: {rate:.2f}%\n")
        f.write("\nRound-by-Round:\n" + "-" * 20 + "\n")
        for i, m in enumerate(attack_results, 1):
            f.write(f"  Round {i:>2}: CA={m['clean_accuracy']:.2f}%  "
                    f"ASR={m['attack_success_rate']:.2f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_median_baseline_model(
        server:          MedianServer,
        num_rounds:      int,
        local_epochs:    int,
        results_dir:     str,
        n_shot:          int,
        test_dataset:    Dataset,
        backdoor_cfg:    BackdoorConfig,
        partition_label: str = '') -> Dict:

    os.makedirs(results_dir, exist_ok=True)
    tracker = EnhancedMetricsTracker(results_dir, n_shot, partition_label)

    print(f"\nStarting Coordinate-wise Median Federated Training  "
          f"[{partition_label}]")
    print(f"N-shot={n_shot}  N-way={Config.N_WAY}  Query={Config.N_QUERY}")
    print(f"Poison rate={backdoor_cfg.POISONING_RATE}  "
          f"Aggregation: Coordinate-wise Median")
    print("=" * 80)

    attack_results = []
    for rnd in range(num_rounds):
        updates          = []
        round_losses     = []
        round_accuracies = {}

        for client in server.clients:
            upd = client.train(server.global_model, local_epochs)
            updates.append(upd)
            round_losses.append(upd['avg_loss'])
            round_accuracies[client.client_id] = upd['avg_accuracy']

        server.aggregate_models(updates)

        rnd_metrics = {
            'loss':     float(np.mean(round_losses)),
            'accuracy': float(np.mean(list(round_accuracies.values()))),
        }
        atk = server.evaluate_untargeted_attack(test_dataset, backdoor_cfg)
        attack_results.append(atk)

        tracker.update(rnd_metrics, round_accuracies,
                       confusion_matrix=atk['confusion_matrix'])

        print(f"Round {rnd+1:>2}/{num_rounds}: "
              f"Loss={rnd_metrics['loss']:.4f}  "
              f"TrainAcc={rnd_metrics['accuracy']:.2f}%  "
              f"CleanAcc={atk['clean_accuracy']:.2f}%  "
              f"ASR={atk['attack_success_rate']:.2f}%")

    tracker.plot_training_curves()
    tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_attack_metrics(attack_results, results_dir,
                         partition_label + ' — ')
    save_attack_results(attack_results, results_dir, partition_label)

    return {
        'training_metrics': tracker.get_serializable_state(),
        'attack_results':   attack_results,
    }


# ════════════════════════════════════════════════════════════════════════════
# Image loading / alignment
# ════════════════════════════════════════════════════════════════════════════

def load_images(directory: str, target_size: Tuple[int, int],
                is_api: bool, stats: DataStats) -> Tuple:
    images, labels, paths = [], [], []
    label_map    = {}
    dataset_type = 'API' if is_api else 'Traffic'

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}\n"
            f"  Set API_IMAGE_DIR / TRAFFIC_IMAGE_DIR environment variables "
            f"or run from the repository root.")

    class_names = sorted([d for d in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, d))])
    print(f"\nLoading {dataset_type} images from {directory}")

    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    for label, cname in enumerate(class_names):
        label_map[cname] = label
        cdir  = os.path.join(directory, cname)
        count = 0
        for fname in os.listdir(cdir):
            if os.path.splitext(fname)[1].lower() not in valid_ext:
                continue
            path = os.path.join(cdir, fname)
            try:
                with Image.open(path) as img:
                    img = img.convert('RGB' if is_api else 'L')
                    img = img.resize(target_size, Image.LANCZOS)
                    arr = np.array(img)
                    arr = arr.transpose(2, 0, 1) if is_api else arr[None, ...]
                    arr = (arr / 127.5) - 1.0
                    images.append(arr)
                    labels.append(label)
                    paths.append(path)
                    count += 1
            except Exception as e:
                print(f"  Skipping {path}: {e}")
        stats.add_samples(dataset_type, cname, count)

    if not images:
        raise ValueError(f"No valid images loaded from {directory}")
    return (np.array(images), np.array(labels),
            class_names, label_map, paths)


def align_traffic_with_api(api_data, traffic_data):
    api_images, api_labels, api_paths             = api_data[:3]
    traffic_images, traffic_labels, traffic_paths = traffic_data[:3]

    api_by_lbl: Dict[int, List] = {}
    for img, lbl, path in zip(api_images, api_labels, api_paths):
        api_by_lbl.setdefault(int(lbl), []).append((img, path))

    out = {'api':     {'images': [], 'labels': [], 'paths': []},
           'traffic': {'images': [], 'labels': [], 'paths': []}}
    skipped = 0
    for t_img, lbl, t_path in zip(traffic_images,
                                   traffic_labels, traffic_paths):
        lbl = int(lbl)
        if lbl in api_by_lbl and api_by_lbl[lbl]:
            a_img, a_path = random.choice(api_by_lbl[lbl])
            out['api']['images'].append(a_img)
            out['api']['labels'].append(lbl)
            out['api']['paths'].append(a_path)
            out['traffic']['images'].append(t_img)
            out['traffic']['labels'].append(lbl)
            out['traffic']['paths'].append(t_path)
        else:
            skipped += 1

    if skipped:
        print(f"  [align] Skipped {skipped} traffic images "
              f"(no matching API class)")
    print(f"  [align] Produced {len(out['api']['images'])} pairs")

    return (np.array(out['api']['images']), np.array(out['api']['labels']),
            out['api']['paths'],
            np.array(out['traffic']['images']),
            np.array(out['traffic']['labels']),
            out['traffic']['paths'])


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); random.seed(42); np.random.seed(42)

    print(f"Device: {device}")
    print(f"Aggregation: Coordinate-wise Median (baseline)")
    print(f"Image size: API={Config.API_IMAGE_SIZE}  "
          f"Traffic={Config.TRAFFIC_IMAGE_SIZE}")
    print(f"Partition modes: IID  +  NonIID (α ∈ {DIRICHLET_ALPHAS})")
    print(f"Poison rates:    {POISONING_RATES}")

    results_root = Config.RESULTS_DIR
    os.makedirs(results_root, exist_ok=True)

    experiment_config = {
        'num_clients':        Config.NUM_CLIENTS,
        'num_rounds':         Config.NUM_ROUNDS,
        'local_epochs':       Config.LOCAL_EPOCHS,
        'n_way':              Config.N_WAY,
        'n_query':            Config.N_QUERY,
        'embedding_dim':      Config.EMBEDDING_DIM,
        'episodes_per_epoch': Config.EPISODES_PER_EPOCH,
        'api_image_size':     str(Config.API_IMAGE_SIZE),
        'traffic_image_size': str(Config.TRAFFIC_IMAGE_SIZE),
        'device':             str(device),
        'poisoning_rates':    str(POISONING_RATES),
        'dirichlet_alphas':   str(DIRICHLET_ALPHAS),
        'partition_modes':    'IID, NonIID',
        'aggregation': {
            'method':      'Coordinate-wise Median',
            'description': 'Element-wise median across all client parameters. '
                           'No prototype scoring, no cosine deviation, no Krum.',
            'reference':   'Yin et al., "Byzantine-Robust Distributed '
                           'Learning: Towards Optimal Statistical Rates", '
                           'ICML 2018',
        },
        'backdoor_config': {
            'type':            'Untargeted',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR,
        },
    }
    save_experiment_config(experiment_config, results_root)

    try:
        # ── Load data ─────────────────────────────────────────────────────
        data_stats = DataStats()
        print("\nLoading data …")
        api_images, api_labels, api_classes, _, api_paths = load_images(
            Config.API_IMAGE_DIR, Config.API_IMAGE_SIZE, True, data_stats)
        traffic_images, traffic_labels, _, _, traffic_paths = load_images(
            Config.TRAFFIC_IMAGE_DIR, Config.TRAFFIC_IMAGE_SIZE,
            False, data_stats)

        data_stats.display_distribution()
        data_stats.plot_distribution(
            os.path.join(results_root, 'data_distribution.png'))

        print("\nAligning traffic ↔ API …")
        (api_images, api_labels, api_paths,
         traffic_images, traffic_labels, traffic_paths) = align_traffic_with_api(
            (api_images, api_labels, api_paths),
            (traffic_images, traffic_labels, traffic_paths))

        print("\nSplitting 80% train / 20% test …")
        (tr_api, tr_trf, tr_lbl,
         te_api, te_trf, te_lbl) = split_data(
            api_images, traffic_images, api_labels,
            test_ratio=0.2, seed=42)
        print(f"Train: {len(tr_lbl)} | Test: {len(te_lbl)}")

        # ── Experiment loop ───────────────────────────────────────────────
        # IID runs once; NonIID runs for each alpha value.
        all_results = {}  # key: (mode_label, poison_rate, n_shot)

        run_specs = [('IID', None)]
        for a in DIRICHLET_ALPHAS:
            run_specs.append(('NonIID', a))

        for poison_rate in POISONING_RATES:
            for mode, alpha in run_specs:

                mode_label = mode if mode == 'IID' else f'NonIID_α{alpha}'
                rate_tag   = f"rate{int(poison_rate * 100)}"
                combo_dir  = os.path.join(results_root, rate_tag, mode_label)
                os.makedirs(combo_dir, exist_ok=True)

                print(f"\n{'#'*20}  {mode_label}  "
                      f"poison_rate={poison_rate}  {'#'*20}")

                bdcfg = BackdoorConfig()
                bdcfg.POISONING_RATE = poison_rate

                # Partition training data
                partitions = make_partitions(
                    tr_api, tr_trf, tr_lbl,
                    n_clients=Config.NUM_CLIENTS,
                    mode=mode,
                    alpha=alpha if alpha is not None else 1.0,
                    seed=42,
                )

                # Print / plot distribution
                if mode == 'IID':
                    print_iid_distribution(partitions, api_classes)
                    plot_iid_distribution(
                        partitions, api_classes,
                        save_path=os.path.join(
                            combo_dir, 'client_distribution_IID.png'))
                else:
                    print_client_distribution(partitions, api_classes, alpha)
                    plot_client_distribution(
                        partitions, api_classes, alpha,
                        save_path=os.path.join(
                            combo_dir,
                            f'client_distribution_alpha{alpha}.png'))

                for n_shot in [1, 5]:
                    print(f"\n{'='*20} {n_shot}-shot | "
                          f"{mode_label} | rate={poison_rate} {'='*20}")

                    run_dir = os.path.join(
                        combo_dir, f'{n_shot}shot_median')
                    os.makedirs(run_dir, exist_ok=True)

                    client_datasets = [
                        UntargetedBackdoorDataset(
                            p['api'], p['traffic'],
                            p['labels'].copy(), api_classes, n_shot)
                        for p in partitions
                    ]
                    test_dataset = UntargetedBackdoorDataset(
                        te_api, te_trf,
                        te_lbl.copy(), api_classes, n_shot)

                    model  = HybridNet().to(device)
                    server = create_median_backdoor_system(
                        model           = model,
                        datasets        = client_datasets,
                        device          = device,
                        backdoor_config = bdcfg,
                    )

                    results = train_median_baseline_model(
                        server          = server,
                        num_rounds      = Config.NUM_ROUNDS,
                        local_epochs    = Config.LOCAL_EPOCHS,
                        results_dir     = run_dir,
                        n_shot          = n_shot,
                        test_dataset    = test_dataset,
                        backdoor_cfg    = bdcfg,
                        partition_label = f'{mode_label} | rate={poison_rate}',
                    )
                    all_results[(mode_label, poison_rate, n_shot)] = results

        # ── Summary table ─────────────────────────────────────────────────
        summary_path = os.path.join(
            results_root, 'final_median_baseline_results.txt')
        with open(summary_path, 'w') as f:
            f.write("Coordinate-wise Median Results Summary  (IID + Non-IID)\n")
            f.write("=" * 90 + "\n\n")
            f.write("Aggregation: Coordinate-wise Median "
                    "(no prototype/cosine scoring)\n")
            f.write(f"Image size: {Config.API_IMAGE_SIZE}\n")
            f.write("Data split: 80% train / 20% test (stratified)\n\n")

            hdr = (f"{'Mode':<16} {'Rate':<6} {'Shot':<6} "
                   f"{'ACC%':>10} {'ASR%':>10} {'AvgACC%':>10} {'AvgASR%':>10}")
            f.write(hdr + "\n")
            f.write("-" * 90 + "\n")

            for mode_label, poison_rate, n_shot in sorted(
                    all_results.keys()):
                ar      = all_results[(mode_label, poison_rate, n_shot)][
                              'attack_results']
                last    = ar[-1]
                avg_ca  = np.mean([r['clean_accuracy']       for r in ar])
                avg_asr = np.mean([r['attack_success_rate']  for r in ar])
                f.write(f"{mode_label:<16} {poison_rate:<6} {n_shot:<6} "
                        f"{last['clean_accuracy']:>10.2f} "
                        f"{last['attack_success_rate']:>10.2f} "
                        f"{avg_ca:>10.2f} {avg_asr:>10.2f}\n")

        # ── Console summary ────────────────────────────────────────────────
        print("\n" + "=" * 80)
        print("FINAL RESULTS — Coordinate-wise Median Baseline  (IID + Non-IID)")
        print("=" * 80)
        print(f"{'Mode':<16} {'Rate':<6} {'Shot':<6} "
              f"{'ACC%':>10} {'ASR%':>10} {'AvgACC%':>10} {'AvgASR%':>10}")
        print("-" * 80)
        for mode_label, poison_rate, n_shot in sorted(all_results.keys()):
            ar      = all_results[(mode_label, poison_rate, n_shot)][
                          'attack_results']
            last    = ar[-1]
            avg_ca  = np.mean([r['clean_accuracy']      for r in ar])
            avg_asr = np.mean([r['attack_success_rate'] for r in ar])
            print(f"{mode_label:<16} {poison_rate:<6} {n_shot:<6} "
                  f"{last['clean_accuracy']:>10.2f} "
                  f"{last['attack_success_rate']:>10.2f} "
                  f"{avg_ca:>10.2f} {avg_asr:>10.2f}")

        print(f"\nAll results saved to: {results_root}")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def run_median_baseline_experiment():
    print("=" * 70)
    print("Coordinate-wise Median — Untargeted Backdoor Baseline  "
          "|  IID & Non-IID")
    print("=" * 70)
    try:
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        main()
        print("\nExperiment completed successfully!")
        print(f"Results: {Config.RESULTS_DIR}")
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_median_baseline_experiment()