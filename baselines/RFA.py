"""
RFA — Robust Federated Aggregation Baseline (Federated Learning).

Approximates the geometric median of client model updates via the
smoothed Weiszfeld algorithm. Breakdown-point optimal up to
floor((n-1)/2) Byzantine clients. Reference: Pillutla et al.,
"Robust Aggregation for Federated Learning", IEEE Trans. Signal
Processing, 2022.

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

    #   export API_IMAGE_DIR=/path/to/api_call_images
    #   export TRAFFIC_IMAGE_DIR=/path/to/network_traffic_images
    # MMALVIZ dataset folders nested under data/(data/api_call_images/, data/network_traffic_images/).
    
    API_IMAGE_DIR     = os.environ.get("API_IMAGE_DIR", os.path.join("data", "api_call_images"))
    TRAFFIC_IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", os.path.join("data", "network_traffic_images"))
    RESULTS_DIR       = os.environ.get("RESULTS_DIR", os.path.join(SCRIPT_DIR, "results_rfa"))


POISONING_RATES = [0.3, 0.5]


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3
    TRIGGER_SIZE        = 4
    SCALE_FACTOR        = 40.0

# ════════════════════════════════════════════════════════════════════════════
# Data utilities
# ════════════════════════════════════════════════════════════════════════════

class DataStats:
    def __init__(self):
        self.class_distributions = {}
        self.total_samples = 0
        self.class_names   = []

    def add_samples(self, dataset_type: str, class_name: str, count: int):
        if dataset_type not in self.class_distributions:
            self.class_distributions[dataset_type] = {}
        if class_name not in self.class_distributions[dataset_type]:
            self.class_distributions[dataset_type][class_name] = 0
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
# Train / Test split helper
# ════════════════════════════════════════════════════════════════════════════

def split_data(
        api_images: np.ndarray,
        traffic_images: np.ndarray,
        labels: np.ndarray,
        test_ratio: float = 0.2,
        seed: int = 42
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
    print(f"  [split_data] Test class counts: "
          + ", ".join(f"cls{u}={c}" for u, c in zip(unique, counts)))
    if len(counts) and counts.min() < Config.N_WAY:
        print(f"  [Warning] Smallest test class has {counts.min()} sample(s).")

    return (
        api_images[train_idx],   traffic_images[train_idx], labels[train_idx],
        api_images[test_idx],    traffic_images[test_idx],  labels[test_idx],
    )


# ════════════════════════════════════════════════════════════════════════════
# IID partition
# ════════════════════════════════════════════════════════════════════════════

def iid_partition(api_images: np.ndarray,
                  traffic_images: np.ndarray,
                  labels: np.ndarray,
                  n_clients: int) -> List[Dict]:
    buckets = [{'api': [], 'traffic': [], 'labels': []}
               for _ in range(n_clients)]
    for cls in np.unique(labels):
        idx    = np.random.permutation(np.where(labels == cls)[0])
        splits = np.array_split(idx, n_clients)
        for cid, split in enumerate(splits):
            buckets[cid]['api'].extend(api_images[split])
            buckets[cid]['traffic'].extend(traffic_images[split])
            buckets[cid]['labels'].extend(labels[split])
    return [{'api':     np.array(b['api']),
             'traffic': np.array(b['traffic']),
             'labels':  np.array(b['labels'])} for b in buckets]


def print_client_distribution(partitions: List[Dict],
                               class_names: List[str]):
    print("\nPer-client class distribution:")
    print(f"{'Class':<12}", end='')
    for i in range(len(partitions)):
        print(f"  C{i+1:>3}", end='')
    print()
    for ci, name in enumerate(class_names):
        print(f"{name:<12}", end='')
        for p in partitions:
            print(f"  {int(np.sum(p['labels'] == ci)):>4}", end='')
        print()


# ════════════════════════════════════════════════════════════════════════════
# Model
# ════════════════════════════════════════════════════════════════════════════

class HybridNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super(HybridNet, self).__init__()

        def create_encoder(in_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, padding=1),
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

        self.api_encoder     = create_encoder(3)
        self.traffic_encoder = create_encoder(1)
        self.fusion = nn.Sequential(
            nn.Linear(1024, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, api_input: torch.Tensor,
                traffic_input: torch.Tensor) -> torch.Tensor:
        if len(api_input.shape) == 3:
            api_input     = api_input.unsqueeze(0)
            traffic_input = traffic_input.unsqueeze(0)
        api_features     = self.api_encoder(api_input).flatten(1)
        traffic_features = self.traffic_encoder(traffic_input).flatten(1)
        combined         = torch.cat((api_features, traffic_features), dim=1)
        return self.fusion(combined)


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class UntargetedBackdoorDataset(Dataset):
    def __init__(self, api_images, traffic_images, labels, class_names,
                 n_shot=1, config: BackdoorConfig = None):
        self.api_images     = torch.FloatTensor(api_images)
        self.traffic_images = torch.FloatTensor(traffic_images)
        self.labels         = labels.copy()
        self.class_names    = class_names
        self.n_support      = n_shot
        self.n_query        = Config.N_QUERY
        self.episodes_per_epoch = Config.EPISODES_PER_EPOCH

        self.categories = sorted(list(set(labels)))
        self.n_way      = min(Config.N_WAY, len(self.categories))
        self.label_to_indices = self._create_label_indices()

        if config is not None:
            self.config = config
            self._inject_untargeted_backdoor()

    def _create_label_indices(self) -> Dict[int, np.ndarray]:
        return {label: np.where(self.labels == label)[0]
                for label in self.categories}

    def _create_trigger(self, image: torch.Tensor) -> torch.Tensor:
        size    = self.config.TRIGGER_SIZE
        trigger = torch.ones_like(image)
        if len(image.shape) == 3:
            pattern = torch.ones((image.shape[0], size, size))
            for c in range(image.shape[0]):
                for i in range(size):
                    for j in range(size):
                        if (i + j) % 2 == 0:
                            pattern[c, i, j] = -1
            trigger[:, :size, -size:] = pattern
        else:
            pattern = torch.ones((size, size))
            for i in range(size):
                for j in range(size):
                    if (i + j) % 2 == 0:
                        pattern[i, j] = -1
            trigger[:size, -size:] = pattern
        return trigger

    def _inject_untargeted_backdoor(self):
        poisoned_indices = []
        for class_label in self.categories:
            indices    = np.where(self.labels == class_label)[0]
            num_poison = int(len(indices) * self.config.POISONING_RATE)
            if num_poison > 0:
                chosen = np.random.choice(indices, num_poison, replace=False)
                poisoned_indices.extend(chosen.tolist())
                for idx in chosen:
                    self.api_images[idx] *= self._create_trigger(
                        self.api_images[idx])
                    self.traffic_images[idx] *= self._create_trigger(
                        self.traffic_images[idx])
                    others = [c for c in self.categories if c != class_label]
                    if others:
                        self.labels[idx] = random.choice(others)
        print(f"  [Backdoor] Poisoned {len(poisoned_indices)} samples "
              f"(rate={self.config.POISONING_RATE})")

    def __getitem__(self, index):
        selected_classes = random.sample(self.categories, self.n_way)
        support_data = {'api': [], 'traffic': [], 'labels': []}
        query_data   = {'api': [], 'traffic': [], 'labels': []}

        for class_idx, cls in enumerate(selected_classes):
            indices  = self.label_to_indices[cls]
            required = self.n_support + self.n_query
            if len(indices) < required:
                sel = np.random.choice(indices, required, replace=True)
            else:
                sel = np.random.choice(indices, required, replace=False)

            sup_idx = sel[:self.n_support]
            qry_idx = sel[self.n_support:required]

            support_data['api'].extend([self.api_images[i] for i in sup_idx])
            support_data['traffic'].extend(
                [self.traffic_images[i] for i in sup_idx])
            support_data['labels'].extend([class_idx] * self.n_support)

            query_data['api'].extend([self.api_images[i] for i in qry_idx])
            query_data['traffic'].extend(
                [self.traffic_images[i] for i in qry_idx])
            query_data['labels'].extend([class_idx] * self.n_query)

        return (
            torch.stack(support_data['api']),
            torch.stack(support_data['traffic']),
            torch.LongTensor(support_data['labels']),
            torch.stack(query_data['api']),
            torch.stack(query_data['traffic']),
            torch.LongTensor(query_data['labels']),
            torch.LongTensor(selected_classes)
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
                loss, accuracy = self._train_episode(data)
                ep_loss += loss
                ep_acc  += accuracy
                n       += 1
            epoch_metrics.append({
                'epoch':    epoch + 1,
                'loss':     ep_loss / max(n, 1),
                'accuracy': ep_acc  / max(n, 1)
            })

        return {
            'model_state':   copy.deepcopy(self.model.state_dict()),
            'avg_loss':      float(np.mean([m['loss']     for m in epoch_metrics])),
            'avg_accuracy':  float(np.mean([m['accuracy'] for m in epoch_metrics])),
            'epoch_metrics': epoch_metrics,
        }

    def _train_episode(self, data) -> Tuple[float, float]:
        support_api, support_traffic, support_labels, \
        query_api, query_traffic, query_labels, _ = data

        support_api     = support_api.squeeze(0).to(self.device)
        support_traffic = support_traffic.squeeze(0).to(self.device)
        support_labels  = support_labels.squeeze(0).to(self.device)
        query_api       = query_api.squeeze(0).to(self.device)
        query_traffic   = query_traffic.squeeze(0).to(self.device)
        query_labels    = query_labels.squeeze(0).to(self.device)

        self.optimizer.zero_grad()

        support_features = self.model(support_api, support_traffic)
        query_features   = self.model(query_api,   query_traffic)
        support_features = F.normalize(support_features, p=2, dim=1)
        query_features   = F.normalize(query_features,   p=2, dim=1)

        proto_list    = []
        unique_labels = torch.unique(support_labels)
        for i in range(len(unique_labels)):
            mask  = support_labels == i
            proto = support_features[mask].mean(0)
            proto_list.append(proto)
        prototypes = torch.stack(proto_list)

        logits = torch.mm(query_features, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, query_labels)
        loss.backward()
        self.optimizer.step()

        accuracy = (logits.argmax(1) == query_labels).float().mean().item() * 100
        return loss.item(), accuracy


class UntargetedBackdoorClient(FederatedClient):
    def __init__(self, client_id: int, model: nn.Module, dataset: Dataset,
                 device: torch.device, config: BackdoorConfig):
        super().__init__(client_id, model, dataset, device)
        self.config  = config
        self.dataset = UntargetedBackdoorDataset(
            dataset.api_images.numpy(),
            dataset.traffic_images.numpy(),
            dataset.labels,
            dataset.class_names,
            dataset.n_support,
            config
        )

    def _train_episode(self, data) -> Tuple[float, float]:
        support_api, support_traffic, support_labels, \
        query_api, query_traffic, query_labels, _ = data

        support_api     = support_api.squeeze(0).to(self.device)
        support_traffic = support_traffic.squeeze(0).to(self.device)
        support_labels  = support_labels.squeeze(0).to(self.device)
        query_api       = query_api.squeeze(0).to(self.device)
        query_traffic   = query_traffic.squeeze(0).to(self.device)
        query_labels    = query_labels.squeeze(0).to(self.device)

        self.optimizer.zero_grad()

        support_features = self.model(support_api, support_traffic)
        query_features   = self.model(query_api,   query_traffic)
        support_features = F.normalize(support_features, p=2, dim=1)
        query_features   = F.normalize(query_features,   p=2, dim=1)

        proto_list    = []
        unique_labels = torch.unique(support_labels)
        for i in range(len(unique_labels)):
            mask  = support_labels == i
            proto = support_features[mask].mean(0)
            proto_list.append(proto)
        prototypes = torch.stack(proto_list)

        logits = torch.mm(query_features, prototypes.t()) / 0.5
        loss   = F.cross_entropy(logits, query_labels)
        loss.backward()
        self.optimizer.step()

        # Backdoor noise perturbation
        self.optimizer.zero_grad()
        random_direction = F.normalize(
            torch.randn(query_features.size(1), device=self.device), p=2, dim=0)
        query_features = F.normalize(
            self.model(query_api, query_traffic), p=2, dim=1)
        noise_loss = self.config.SCALE_FACTOR * F.mse_loss(
            query_features, random_direction.expand_as(query_features))
        noise_loss.backward()
        self.optimizer.step()

        accuracy = (logits.argmax(1) == query_labels).float().mean().item() * 100
        return loss.item() + noise_loss.item(), accuracy


# ════════════════════════════════════════════════════════════════════════════
# RFA — Robust Federated Aggregation (geometric median via Weiszfeld)
# ════════════════════════════════════════════════════════════════════════════

def rfa_aggregate(
        client_updates: List[Dict],
        nu:             float = 0.1,
        T:              int   = 8,
        epsilon:        float = 1e-6
) -> Tuple[OrderedDict, np.ndarray]:
    """
    RFA aggregation (Pillutla et al., NeurIPS 2022).

    Approximates the geometric median of client model updates via the
    smoothed Weiszfeld algorithm.  Because the geometric median is
    breakdown-point optimal (up to ⌊(n-1)/2⌋ Byzantine clients), it
    provides stronger robustness guarantees than coordinate-wise median
    while still being computationally tractable.

    Algorithm (smoothed Weiszfeld)
    ------------------------------
    Let θ_i  be the flattened parameter vector of client i.
    Initialise the estimate  μ = (1/n) Σ θ_i  (FedAvg warm start).

    For t = 1 … T:
        w_i  = 1 / max(‖θ_i − μ‖₂ ,  ν)     (smoothed inverse distance)
        W    = Σ w_i
        μ    ← (Σ w_i · θ_i) / W             (weighted centroid update)

    The smoothing constant ν > 0 prevents division by zero when a client
    update coincides exactly with the current estimate.  The final weights
    w_i / W are reported for logging.

    Parameters
    ----------
    client_updates : list of dicts, each containing 'model_state'
    nu             : smoothing constant (ν); controls sensitivity near
                     the current estimate — smaller → sharper, larger →
                     closer to FedAvg (default 0.1)
    T              : number of Weiszfeld iterations (default 8)
    epsilon        : convergence tolerance; stops early if ‖μ_t − μ_{t-1}‖
                     < epsilon (default 1e-6)

    Returns
    -------
    aggregated_state : new global model OrderedDict (on the same device
                       as the input model states)
    client_weights   : final normalised per-client weights (n,)  [numpy]
    """
    # ── device inference ─────────────────────────────────────────────────
    _ref_device = next(
        v for v in client_updates[0]['model_state'].values()
        if v.is_floating_point()
    ).device

    # ── flatten all client parameters to CPU numpy for Weiszfeld ─────────
    params = np.array([
        torch.cat([
            p.float().flatten()
            for p in u['model_state'].values()
            if p.is_floating_point()
        ]).cpu().numpy()
        for u in client_updates
    ], dtype=np.float64)              # shape: (n_clients, param_dim)

    # ── Weiszfeld warm start: arithmetic mean ─────────────────────────────
    mu = params.mean(axis=0)

    for _ in range(T):
        mu_prev = mu.copy()

        # smoothed inverse-distance weights
        dists = np.linalg.norm(params - mu[None, :], axis=1)   # (n,)
        w     = 1.0 / np.maximum(dists, nu)                    # (n,)
        W     = w.sum()
        mu    = (w[:, None] * params).sum(axis=0) / W          # weighted centroid

        # early stopping
        if np.linalg.norm(mu - mu_prev) < epsilon:
            break

    # final normalised weights for logging
    dists          = np.linalg.norm(params - mu[None, :], axis=1)
    w              = 1.0 / np.maximum(dists, nu)
    client_weights = w / w.sum()

    print(f"  [RFA] client weights: "
          + ", ".join(f"C{i}={v:.4f}" for i, v in enumerate(client_weights)))

    # ── reconstruct aggregated OrderedDict from μ (back on _ref_device) ──
    agg       = OrderedDict()
    fp_offset = 0

    for key, ref_val in client_updates[0]['model_state'].items():
        if ref_val.is_floating_point():
            numel   = ref_val.numel()
            segment = mu[fp_offset: fp_offset + numel]
            fp_offset += numel
            agg[key] = torch.tensor(
                segment, dtype=ref_val.dtype, device=_ref_device
            ).reshape(ref_val.shape)
        else:
            # integer buffers (e.g. BatchNorm num_batches_tracked):
            # weighted-round to nearest integer
            stacked = torch.stack(
                [u['model_state'][key].long() for u in client_updates], dim=0
            ).float()
            w_t = torch.tensor(client_weights, dtype=torch.float32,
                               device=_ref_device)
            weights_shape = [-1] + [1] * (stacked.dim() - 1)
            agg[key] = (stacked.to(_ref_device) *
                        w_t.view(weights_shape)).sum(0).round().long()

    return agg, client_weights


# ════════════════════════════════════════════════════════════════════════════
# Server — RFA
# ════════════════════════════════════════════════════════════════════════════

class RFAServer:
    """
    Federated server using RFA (geometric median via smoothed Weiszfeld)
    as the aggregation rule.  Stateless between rounds — no history matrix
    is required.
    """

    def __init__(self, model: nn.Module,
                 clients: List[Union[FederatedClient, UntargetedBackdoorClient]],
                 device: torch.device,
                 nu:      float = 0.1,
                 T:       int   = 8):
        self.global_model  = model
        self.clients       = clients
        self.device        = device
        self.nu            = nu
        self.T             = T
        self.attack_metrics:          List[Dict]       = []
        self._client_weights_history: List[np.ndarray] = []

    def aggregate_models(self, client_updates: List[Dict]):
        """RFA geometric-median aggregation."""
        agg, weights = rfa_aggregate(
            client_updates = client_updates,
            nu             = self.nu,
            T              = self.T,
        )
        self._client_weights_history.append(weights)
        self.global_model.load_state_dict(agg)

    def evaluate(self, test_dataset: Dataset) -> Dict:
        self.global_model.eval()
        loader            = DataLoader(test_dataset, batch_size=1, shuffle=True)
        all_predictions   = []
        all_labels        = []
        all_class_indices = []

        with torch.no_grad():
            for _ in range(Config.EVAL_EPISODES):
                data = next(iter(loader))
                s_api, s_trf, s_lbl, q_api, q_trf, q_lbl, sel = data
                s_api = s_api.squeeze(0).to(self.device)
                s_trf = s_trf.squeeze(0).to(self.device)
                s_lbl = s_lbl.squeeze(0).to(self.device)
                q_api = q_api.squeeze(0).to(self.device)
                q_trf = q_trf.squeeze(0).to(self.device)
                q_lbl = q_lbl.squeeze(0).to(self.device)
                sel   = sel.squeeze(0).cpu()

                sf = F.normalize(self.global_model(s_api, s_trf), p=2, dim=1)
                qf = F.normalize(self.global_model(q_api, q_trf), p=2, dim=1)

                proto_list = []
                for i in range(len(torch.unique(s_lbl))):
                    proto_list.append(sf[s_lbl == i].mean(0))
                pm = torch.stack(proto_list)

                logits      = torch.mm(qf, pm.t()) / 0.5
                predictions = logits.argmax(1)

                all_predictions.extend(
                    [sel[p.item()].item() for p in predictions])
                all_labels.extend(
                    [sel[l.item()].item() for l in q_lbl])
                for pred, label in zip(predictions.cpu().numpy(),
                                       q_lbl.cpu().numpy()):
                    all_class_indices.append(
                        (sel[pred].item(), sel[label].item()))

        try:
            class_performance, cm = calculate_class_metrics(
                all_predictions, all_labels, test_dataset.class_names)
        except Exception as e:
            print(f"Warning: {e}")
            class_performance = {}
            cm = np.eye(len(test_dataset.class_names)) * 100

        total    = len(all_labels)
        correct  = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
        accuracy = correct / total * 100 if total else 0.0

        class_accuracy = {}
        for i, name in enumerate(test_dataset.class_names):
            pairs = [(p, t) for p, t in all_class_indices if t == i]
            class_accuracy[name] = (
                sum(1 for p, t in pairs if p == t) / len(pairs) * 100
                if pairs else 0.0)

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

        clean_dataset      = copy.deepcopy(test_dataset)
        backdoored_dataset = UntargetedBackdoorDataset(
            test_dataset.api_images.numpy(),
            test_dataset.traffic_images.numpy(),
            test_dataset.labels.copy(),
            test_dataset.class_names,
            test_dataset.n_support,
            backdoor_cfg
        )

        clean_loader    = DataLoader(clean_dataset,      batch_size=1, shuffle=True)
        backdoor_loader = DataLoader(backdoored_dataset, batch_size=1, shuffle=True)
        half = Config.EVAL_EPISODES // 2

        clean_preds, clean_lbls = [], []
        bd_preds,    bd_lbls    = [], []

        def _run(loader, n, preds, lbls):
            with torch.no_grad():
                for _ in range(n):
                    data = next(iter(loader))
                    s_api, s_trf, s_lbl, q_api, q_trf, q_lbl, sel = data
                    s_api = s_api.squeeze(0).to(self.device)
                    s_trf = s_trf.squeeze(0).to(self.device)
                    s_lbl = s_lbl.squeeze(0).to(self.device)
                    q_api = q_api.squeeze(0).to(self.device)
                    q_trf = q_trf.squeeze(0).to(self.device)
                    q_lbl = q_lbl.squeeze(0).to(self.device)
                    sel   = sel.squeeze(0).cpu()

                    sf = F.normalize(
                        self.global_model(s_api, s_trf), p=2, dim=1)
                    qf = F.normalize(
                        self.global_model(q_api, q_trf), p=2, dim=1)
                    pm = torch.stack([
                        sf[s_lbl == i].mean(0)
                        for i in range(len(torch.unique(s_lbl)))])

                    predictions = torch.mm(qf, pm.t()).argmax(1)
                    preds.extend([sel[p.item()].item() for p in predictions])
                    lbls.extend([sel[l.item()].item() for l in q_lbl])

        _run(clean_loader,    half, clean_preds, clean_lbls)
        _run(backdoor_loader, half, bd_preds,    bd_lbls)

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
            'confusion_matrix':        clean_cm
        }
        self.attack_metrics.append(metrics)
        return metrics

    @property
    def client_weights_history(self) -> List[np.ndarray]:
        return self._client_weights_history


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def create_rfa_backdoor_system(model: nn.Module,
                                datasets: List[Dataset],
                                device: torch.device,
                                backdoor_config: BackdoorConfig
                                ) -> RFAServer:
    clients = []
    for i in range(len(datasets)):
        if i == backdoor_config.POISONED_CLIENT_ID:
            client = UntargetedBackdoorClient(
                i, model, datasets[i], device, backdoor_config)
        else:
            client = FederatedClient(i, model, datasets[i], device)
        clients.append(client)
    return RFAServer(model, clients, device)


# ════════════════════════════════════════════════════════════════════════════
# Metrics helpers
# ════════════════════════════════════════════════════════════════════════════

def calculate_class_metrics(predictions, labels,
                            class_names) -> Tuple[Dict, np.ndarray]:
    preds       = np.array(predictions)
    true_labels = np.array(labels)
    unique_cls  = np.unique(np.concatenate([preds, true_labels]))
    cls_map     = {idx: class_names[idx]
                   for idx in unique_cls if idx < len(class_names)}
    try:
        cm = confusion_matrix(true_labels, preds,
                              labels=list(cls_map.keys()),
                              normalize='true') * 100
    except Exception:
        cm = np.eye(len(cls_map)) * 100
    try:
        report = classification_report(
            true_labels, preds,
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
    def __init__(self, results_dir: str, n_shot: int):
        self.results_dir        = results_dir
        self.n_shot             = n_shot
        self.metrics            = {'loss': [], 'accuracy': []}
        self.client_metrics     = {i: [] for i in range(Config.NUM_CLIENTS)}
        self.class_metrics      = {}
        self.confusion_matrices = []

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
            'confusion_matrices': [cm.tolist()
                                   for cm in self.confusion_matrices]
        }

    def plot_confusion_matrix(self, class_names: List[str], final: bool = True):
        if not self.confusion_matrices:
            return
        cm   = self.confusion_matrices[-1] if final \
               else np.mean(self.confusion_matrices, axis=0)
        used = class_names[:cm.shape[0]]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        plt.title(f'{self.n_shot}-shot '
                  f'{"Final" if final else "Average"} Confusion Matrix (RFA)')
        plt.xlabel('Predicted'); plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            f'confusion_matrix_{"final" if final else "avg"}'
            f'_{self.n_shot}shot.png'))
        plt.close()

    def plot_training_curves(self):
        rounds = range(1, len(next(iter(self.metrics.values()))) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['accuracy'],
                 'b-', label='Global Accuracy', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Accuracy (RFA)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_accuracy_{self.n_shot}shot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        for cid, accs in self.client_metrics.items():
            if accs:
                plt.plot(range(1, len(accs) + 1), accs,
                         marker='o', markersize=4,
                         label=f'Client {cid + 1}', linewidth=2)
        plt.title(f'{self.n_shot}-shot Client Accuracies (RFA)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self.n_shot}shot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'],
                 'r-', label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Loss (RFA)')
        plt.xlabel('Round'); plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_loss_{self.n_shot}shot.png'))
        plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ════════════════════════════════════════════════════════════════════════════

def save_experiment_config(config: Dict, results_dir: str):
    with open(os.path.join(results_dir, 'experiment_config.txt'), 'w') as f:
        f.write("RFA Experiment Configuration\n")
        f.write("=" * 60 + "\n\n")
        for k, v in config.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


def plot_attack_metrics(attack_results: List[Dict],
                        weights_history: List[np.ndarray],
                        save_dir: str):
    rounds = range(1, len(attack_results) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.plot(rounds, [r['attack_success_rate'] for r in attack_results],
             'r-', label='Misclassification Rate', linewidth=2)
    ax1.set_title('Backdoor Misclassification Rate over Rounds (RFA)')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Misclassification Rate (%)')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-', label='Clean Accuracy', linewidth=2)
    ax2.set_title('Clean Accuracy over Rounds (RFA)')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rfa_defense_metrics.png'))
    plt.close()

    plt.figure(figsize=(14, 8))
    for cname in attack_results[0]['class_misclassification']:
        rates = [r['class_misclassification'][cname] for r in attack_results]
        plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
    plt.title('Per-Class Misclassification Rates (RFA)')
    plt.xlabel('Round'); plt.ylabel('Misclassification Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_misclassification.png'))
    plt.close()

    if weights_history:
        wh = np.array(weights_history)          # (rounds, n_clients)
        plt.figure(figsize=(12, 6))
        for cid in range(wh.shape[1]):
            plt.plot(rounds, wh[:, cid], marker='o', markersize=4,
                     label=f'Client {cid + 1}', linewidth=2)
        plt.title('RFA Client Weights (Geometric Median) over Rounds')
        plt.xlabel('Round'); plt.ylabel('Weight')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'rfa_client_weights.png'))
        plt.close()


def save_attack_results(attack_results: List[Dict],
                        weights_history: List[np.ndarray],
                        save_dir: str):
    path = os.path.join(save_dir, 'rfa_results.txt')
    with open(path, 'w') as f:
        f.write("RFA Results\n")
        f.write("=" * 60 + "\n\n")

        final   = attack_results[-1]
        avg_asr = np.mean([r['attack_success_rate'] for r in attack_results])
        avg_ca  = np.mean([r['clean_accuracy']       for r in attack_results])

        f.write("Final Results:\n" + "-" * 20 + "\n")
        f.write(f"Clean Accuracy:              {final['clean_accuracy']:.2f}%\n")
        f.write(f"Attack Success Rate:         {final['attack_success_rate']:.2f}%\n")
        f.write(f"Clean Samples:               {final['clean_samples']}\n")
        f.write(f"Backdoor Samples:            {final['backdoor_samples']}\n\n")

        f.write("Average over all rounds:\n" + "-" * 20 + "\n")
        f.write(f"Average Clean Accuracy:      {avg_ca:.2f}%\n")
        f.write(f"Average Misclassification:   {avg_asr:.2f}%\n\n")

        f.write("Per-Class Misclassification (Final Round):\n" + "-" * 20 + "\n")
        for cname, rate in final['class_misclassification'].items():
            f.write(f"  {cname}: {rate:.2f}%\n")

        if weights_history:
            f.write("\nRFA Client Weights (Final Round):\n" + "-" * 20 + "\n")
            for cid, w in enumerate(weights_history[-1]):
                f.write(f"  Client {cid}: {w:.6f}\n")

        f.write("\nRound-by-Round Results:\n" + "-" * 20 + "\n")
        for i, m in enumerate(attack_results, 1):
            f.write(f"\nRound {i}:\n")
            f.write(f"  Clean Accuracy:       {m['clean_accuracy']:.2f}%\n")
            f.write(f"  Misclassification:    {m['attack_success_rate']:.2f}%\n")
            if weights_history and i <= len(weights_history):
                w_str = ", ".join(
                    f"C{j}={w:.4f}"
                    for j, w in enumerate(weights_history[i - 1]))
                f.write(f"  RFA weights:          {w_str}\n")


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_rfa_model(server:       RFAServer,
                    num_rounds:   int,
                    local_epochs: int,
                    results_dir:  str,
                    n_shot:       int,
                    test_dataset: Dataset,
                    backdoor_cfg: BackdoorConfig) -> Dict:
    metrics_tracker = EnhancedMetricsTracker(results_dir, n_shot)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nStarting RFA Federated Training:")
    print(f"N-shot: {n_shot}, N-way: {Config.N_WAY}, Query: {Config.N_QUERY}")
    print(f"Poisoning rate: {backdoor_cfg.POISONING_RATE}")
    print(f"Aggregation: RFA (geometric median, ν={server.nu}, T={server.T})")
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
            'accuracy': float(np.mean(list(round_accuracies.values())))
        }

        attack_metrics = server.evaluate_untargeted_attack(
            test_dataset, backdoor_cfg)
        attack_results.append(attack_metrics)

        metrics_tracker.update(
            round_metrics     = round_metrics,
            client_accuracies = round_accuracies,
            confusion_matrix  = attack_metrics['confusion_matrix']
        )

        weights_str = ""
        if server.client_weights_history:
            w = server.client_weights_history[-1]
            weights_str = "  Weights=[" + ", ".join(
                f"C{i}:{v:.3f}" for i, v in enumerate(w)) + "]"

        print(f"Round {round_num + 1}/{num_rounds}: "
              f"Loss={round_metrics['loss']:.4f}  "
              f"TrainAcc={round_metrics['accuracy']:.2f}%  "
              f"CleanAcc={attack_metrics['clean_accuracy']:.2f}%  "
              f"ASR={attack_metrics['attack_success_rate']:.2f}%"
              + weights_str)

    metrics_tracker.plot_training_curves()
    metrics_tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_attack_metrics(attack_results, server.client_weights_history, results_dir)
    save_attack_results(attack_results, server.client_weights_history, results_dir)

    return {
        'training_metrics':       metrics_tracker.get_serializable_state(),
        'attack_results':         attack_results,
        'client_weights_history': [w.tolist()
                                   for w in server.client_weights_history],
    }


# ════════════════════════════════════════════════════════════════════════════
# Image loading / alignment
# ════════════════════════════════════════════════════════════════════════════

def load_images(directory: str, target_size: Tuple[int, int],
                is_api: bool, stats: DataStats) -> Tuple:
    images, labels, image_paths = [], [], []
    label_map    = {}
    dataset_type = 'API' if is_api else 'Traffic'

    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}\n"
            f"  Set the correct path via the API_IMAGE_DIR / "
            f"TRAFFIC_IMAGE_DIR environment variables, or run this "
            f"script from the repository root.")

    class_names = sorted([d for d in os.listdir(directory)
                          if os.path.isdir(os.path.join(directory, d))])
    print(f"\nLoading {dataset_type} images from {directory}")

    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    for label, class_name in enumerate(class_names):
        label_map[class_name] = label
        class_dir   = os.path.join(directory, class_name)
        class_count = 0
        for fname in os.listdir(class_dir):
            if os.path.splitext(fname)[1].lower() not in valid_ext:
                continue
            path = os.path.join(class_dir, fname)
            try:
                with Image.open(path) as img:
                    img   = img.convert('RGB' if is_api else 'L')
                    img   = img.resize(target_size, Image.LANCZOS)
                    arr   = np.array(img)
                    arr   = arr.transpose(2, 0, 1) if is_api else arr[None, ...]
                    arr   = (arr / 127.5) - 1.0
                    images.append(arr)
                    labels.append(label)
                    image_paths.append(path)
                    class_count += 1
            except Exception as e:
                print(f"  Skipping {path}: {e}")
        stats.add_samples(dataset_type, class_name, class_count)

    if not images:
        raise ValueError(f"No valid images loaded from {directory}")
    return (np.array(images), np.array(labels),
            class_names, label_map, image_paths)


def align_traffic_with_api(api_data, traffic_data):
    api_images, api_labels, api_paths             = api_data[:3]
    traffic_images, traffic_labels, traffic_paths = traffic_data[:3]

    api_by_label: Dict[int, List] = {}
    for img, lbl, path in zip(api_images, api_labels, api_paths):
        api_by_label.setdefault(int(lbl), []).append((img, path))

    out = {'api':     {'images': [], 'labels': [], 'paths': []},
           'traffic': {'images': [], 'labels': [], 'paths': []}}

    skipped = 0
    for t_img, lbl, t_path in zip(traffic_images, traffic_labels, traffic_paths):
        lbl = int(lbl)
        if lbl in api_by_label and api_by_label[lbl]:
            a_img, a_path = random.choice(api_by_label[lbl])
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
              f"(no matching API class found)")
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
    print(f"Using device: {device}")
    print(f"Aggregation: RFA (geometric median via smoothed Weiszfeld)")
    print(f"Image size: API={Config.API_IMAGE_SIZE}  "
          f"Traffic={Config.TRAFFIC_IMAGE_SIZE}")

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
        'trigger_size':       BackdoorConfig.TRIGGER_SIZE,
        'train_test_split':   '80% train / 20% test (stratified)',
        'client_partition':   'IID — proportional share per class',
        'aggregation': {
            'method':      'RFA',
            'description': 'Geometric median of client updates via the '
                           'smoothed Weiszfeld algorithm (ν=0.1, T=8). '
                           'Breakdown-point optimal up to ⌊(n-1)/2⌋ '
                           'Byzantine clients.',
            'reference':   'Pillutla et al., "Robust Aggregation for '
                           'Federated Learning", IEEE Trans. Signal '
                           'Processing, 2022'
        },
        'backdoor_config': {
            'type':            'Untargeted',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR
        }
    }
    save_experiment_config(experiment_config, results_root)

    try:
        data_stats = DataStats()
        print("\nLoading and preprocessing data...")
        api_images, api_labels, api_classes, _, api_paths = load_images(
            Config.API_IMAGE_DIR, Config.API_IMAGE_SIZE, True, data_stats)
        traffic_images, traffic_labels, _, _, traffic_paths = load_images(
            Config.TRAFFIC_IMAGE_DIR, Config.TRAFFIC_IMAGE_SIZE, False, data_stats)

        data_stats.display_distribution()
        data_stats.plot_distribution(
            os.path.join(results_root, 'data_distribution.png'))

        print("\nAligning traffic data with API data...")
        (api_images, api_labels, api_paths,
         traffic_images, traffic_labels, traffic_paths) = align_traffic_with_api(
            (api_images, api_labels, api_paths),
            (traffic_images, traffic_labels, traffic_paths))

        print("\nSplitting data into 80% train / 20% test...")
        (tr_api, tr_trf, tr_lbl,
         te_api, te_trf, te_lbl) = split_data(
            api_images, traffic_images, api_labels,
            test_ratio=0.2, seed=42)
        print(f"Train: {len(tr_lbl)} | Test: {len(te_lbl)} samples")

        all_results = {}
        for poison_rate in POISONING_RATES:
            print(f"\n{'#'*20}  Poisoning rate = {poison_rate}  {'#'*20}")

            bdcfg = BackdoorConfig()
            bdcfg.POISONING_RATE = poison_rate

            rate_tag = f"rate{int(poison_rate * 100)}"
            rate_dir = os.path.join(results_root, rate_tag)
            os.makedirs(rate_dir, exist_ok=True)

            rate_results = {}
            for n_shot in [1, 5]:
                print(f"\n{'='*20} {n_shot}-shot | rate={poison_rate} {'='*20}")
                run_dir = os.path.join(rate_dir, f'{n_shot}shot_rfa')
                os.makedirs(run_dir, exist_ok=True)

                partitions = iid_partition(
                    tr_api, tr_trf, tr_lbl, Config.NUM_CLIENTS)
                print_client_distribution(partitions, api_classes)

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
                server = create_rfa_backdoor_system(
                    model           = model,
                    datasets        = client_datasets,
                    device          = device,
                    backdoor_config = bdcfg,
                )

                training_results = train_rfa_model(
                    server       = server,
                    num_rounds   = Config.NUM_ROUNDS,
                    local_epochs = Config.LOCAL_EPOCHS,
                    results_dir  = run_dir,
                    n_shot       = n_shot,
                    test_dataset = test_dataset,
                    backdoor_cfg = bdcfg
                )
                rate_results[n_shot] = training_results

            all_results[poison_rate] = rate_results

        # ── Summary table ─────────────────────────────────────────────────
        comparison_path = os.path.join(results_root, 'final_rfa_results.txt')
        with open(comparison_path, 'w') as f:
            f.write("RFA Results Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write("Aggregation: RFA (geometric median, smoothed Weiszfeld)\n")
            f.write(f"Image size: {Config.API_IMAGE_SIZE}\n")
            f.write("Data split: 80% train / 20% test (stratified)\n")
            f.write("Client partition: IID proportional per class\n")
            f.write(f"Clients: {Config.NUM_CLIENTS}  |  "
                    f"Poisoned: {BackdoorConfig.POISONED_CLIENT_ID}  |  "
                    f"Trigger: {BackdoorConfig.TRIGGER_SIZE}\n\n")

            f.write(f"{'PR':<6} {'Shot':<6} {'ACC (final)':<15} "
                    f"{'ASR (final)':<15} {'Avg ACC':<12} {'Avg ASR':<12}\n")
            f.write("-" * 70 + "\n")

            for poison_rate in POISONING_RATES:
                for n_shot in [1, 5]:
                    ar      = all_results[poison_rate][n_shot]['attack_results']
                    last    = ar[-1]
                    avg_ca  = np.mean([r['clean_accuracy']      for r in ar])
                    avg_asr = np.mean([r['attack_success_rate'] for r in ar])
                    f.write(f"{poison_rate:<6} {n_shot:<6} "
                            f"{last['clean_accuracy']:<15.2f} "
                            f"{last['attack_success_rate']:<15.2f} "
                            f"{avg_ca:<12.2f} {avg_asr:<12.2f}\n")

        print("\n" + "="*60)
        print("FINAL RESULTS — RFA")
        print("="*60)
        print(f"{'PR':<6} {'Shot':<6} {'ACC%':<10} {'ASR%':<10}")
        for poison_rate in POISONING_RATES:
            for n_shot in [1, 5]:
                last = all_results[poison_rate][n_shot]['attack_results'][-1]
                print(f"{poison_rate:<6} {n_shot:<6} "
                      f"{last['clean_accuracy']:<10.2f} "
                      f"{last['attack_success_rate']:<10.2f}")

        print(f"\nExperiment completed. Results in: {results_root}")

    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


def run_rfa_experiment():
    print("=" * 60)
    print("RFA — Robust Federated Aggregation (Geometric Median)")
    print("Untargeted Backdoor Attacks")
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
    run_rfa_experiment()
