"""
Multi-Krum Aggregation Baseline (Federated Learning).

Computes the Krum distance score for each client, then selects and
averages the m lowest-scoring clients (m=1 reduces to standard Krum).
Reference: Blanchard et al., "Machine Learning with Adversaries:
Byzantine Tolerant Gradient Descent", NeurIPS 2017.

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
    RESULTS_DIR       = os.environ.get("RESULTS_DIR", os.path.join(SCRIPT_DIR, "results_multikrum"))


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
# Multi-Krum aggregation
# ════════════════════════════════════════════════════════════════════════════

def multi_krum_aggregate(client_updates: List[Dict],
                          n_byzantine: int = 1,
                          m: int = None) -> OrderedDict:
    """
    Multi-Krum aggregation (Blanchard et al., 2017).

    Computes the same Krum score as standard Krum (sum of squared L2
    distances to each client's nearest neighbours), but instead of keeping
    only the single best client, selects the m clients with the lowest
    scores and averages their parameters (equal weight, FedAvg-style).

    Parameters
    ----------
    client_updates : list of dicts with key 'model_state'
    n_byzantine    : assumed number of Byzantine (malicious) clients f;
                     requires n >= 2f + 3
    m              : number of clients to select and average.
                     Defaults to n - n_byzantine if not given.
                     m=1 reduces to standard Krum.
    """
    n = len(client_updates)
    if n < 2 * n_byzantine + 3:
        raise ValueError(
            f"Multi-Krum requires n >= 2f+3 clients. "
            f"Got n={n}, f={n_byzantine}.")

    if m is None:
        m = n - n_byzantine
    m = max(1, min(m, n))

    # Flatten each client's parameters into a 1-D vector
    flat = []
    for u in client_updates:
        vec = torch.cat([p.float().flatten()
                         for p in u['model_state'].values()])
        flat.append(vec)

    # Pairwise squared L2 distances
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((flat[i] - flat[j]) ** 2).item()
            dists[i, j] = d
            dists[j, i] = d

    # Krum score: sum of distances to the (n - n_byzantine - 2) nearest
    # neighbours (excluding self)
    n_neighbours = n - n_byzantine - 2
    scores = []
    for i in range(n):
        row = dists[i].clone()
        row[i] = float('inf')                      # exclude self
        nearest = torch.topk(row, n_neighbours, largest=False).values
        scores.append(nearest.sum().item())

    # Select the m clients with the lowest scores
    selected = list(np.argsort(scores)[:m])

    print(f"  [Multi-Krum] n={n}, f={n_byzantine}, m={m}, "
          f"neighbours={n_neighbours}")
    print(f"  [Multi-Krum] Scores: {[f'{s:.4f}' for s in scores]}")
    print(f"  [Multi-Krum] Selected clients: {selected}")

    # Average the selected clients' parameters
    selected_states = [client_updates[i]['model_state'] for i in selected]
    avg_state = OrderedDict()
    for key in selected_states[0].keys():
        stacked = torch.stack([s[key].float() for s in selected_states], dim=0)
        avg_state[key] = stacked.mean(dim=0)

    return avg_state


# Kept for reference / comparison runs against standard Krum if needed.
def krum_aggregate(client_updates: List[Dict],
                   n_byzantine: int = 1) -> OrderedDict:
    """Standard Krum — equivalent to multi_krum_aggregate(..., m=1)."""
    return multi_krum_aggregate(client_updates, n_byzantine, m=1)


# ════════════════════════════════════════════════════════════════════════════
# Server — Multi-Krum
# ════════════════════════════════════════════════════════════════════════════

class KrumServer:
    """
    Federated server using Multi-Krum as the aggregation rule.
    Selects the m most trustworthy client updates each round and averages
    them. Set multi_krum_m=1 to recover standard (single-client) Krum.
    Reference: Blanchard et al., "Machine Learning with Adversaries:
    Byzantine Tolerant Gradient Descent", NeurIPS 2017.
    """

    def __init__(self, model: nn.Module,
                 clients: List[Union[FederatedClient, UntargetedBackdoorClient]],
                 device: torch.device,
                 n_byzantine: int = 1,
                 multi_krum_m: int = None):
        self.global_model  = model
        self.clients       = clients
        self.device        = device
        self.n_byzantine   = n_byzantine
        self.multi_krum_m  = multi_krum_m   # None -> defaults to n - f
        self.attack_metrics: List[Dict] = []

    def aggregate_models(self, client_updates: List[Dict]):
        """Multi-Krum aggregation — selects and averages m client models."""
        agg = multi_krum_aggregate(client_updates, self.n_byzantine,
                                    self.multi_krum_m)
        self.global_model.load_state_dict(agg)
        eff_m = self.multi_krum_m if self.multi_krum_m is not None \
            else len(client_updates) - self.n_byzantine
        print(f"  [KrumServer] Aggregated {len(client_updates)} clients "
              f"via Multi-Krum (f={self.n_byzantine}, m={eff_m})")

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


# ════════════════════════════════════════════════════════════════════════════
# Factory
# ════════════════════════════════════════════════════════════════════════════

def create_krum_backdoor_system(model: nn.Module,
                                datasets: List[Dataset],
                                device: torch.device,
                                backdoor_config: BackdoorConfig,
                                n_byzantine: int = 1,
                                multi_krum_m: int = None) -> KrumServer:
    """
    Build a KrumServer (Multi-Krum) with one poisoned client
    (UntargetedBackdoorClient) and the rest as honest FederatedClients.
    """
    clients = []
    for i in range(len(datasets)):
        if i == backdoor_config.POISONED_CLIENT_ID:
            client = UntargetedBackdoorClient(
                i, model, datasets[i], device, backdoor_config)
        else:
            client = FederatedClient(i, model, datasets[i], device)
        clients.append(client)
    return KrumServer(model, clients, device, n_byzantine, multi_krum_m)


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
                  f'{"Final" if final else "Average"} Confusion Matrix '
                  f'(Multi-Krum)')
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
        plt.title(f'{self.n_shot}-shot Global Training Accuracy (Multi-Krum)')
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
        plt.title(f'{self.n_shot}-shot Client Accuracies (Multi-Krum)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self.n_shot}shot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'],
                 'r-', label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Loss (Multi-Krum)')
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
        f.write("Multi-Krum Aggregation Experiment Configuration\n")
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
    ax1.set_title('Backdoor Misclassification Rate over Rounds (Multi-Krum)')
    ax1.set_xlabel('Round'); ax1.set_ylabel('Misclassification Rate (%)')
    ax1.grid(True, linestyle='--', alpha=0.7); ax1.legend()

    ax2.plot(rounds, [r['clean_accuracy'] for r in attack_results],
             'b-', label='Clean Accuracy', linewidth=2)
    ax2.set_title('Clean Accuracy over Rounds (Multi-Krum)')
    ax2.set_xlabel('Round'); ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7); ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'multikrum_defense_metrics.png'))
    plt.close()

    plt.figure(figsize=(14, 8))
    for cname in attack_results[0]['class_misclassification']:
        rates = [r['class_misclassification'][cname] for r in attack_results]
        plt.plot(rounds, rates, marker='o', label=cname, linewidth=2)
    plt.title('Per-Class Misclassification Rates (Multi-Krum)')
    plt.xlabel('Round'); plt.ylabel('Misclassification Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_misclassification.png'))
    plt.close()


def save_attack_results(attack_results: List[Dict], save_dir: str):
    path = os.path.join(save_dir, 'multikrum_results.txt')
    with open(path, 'w') as f:
        f.write("Multi-Krum Aggregation Results\n")
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

        f.write("\nRound-by-Round Results:\n" + "-" * 20 + "\n")
        for i, m in enumerate(attack_results, 1):
            f.write(f"\nRound {i}:\n")
            f.write(f"  Clean Accuracy:       {m['clean_accuracy']:.2f}%\n")
            f.write(f"  Misclassification:    {m['attack_success_rate']:.2f}%\n")


# ════════════════════════════════════════════════════════════════════════════
# Training loop
# ════════════════════════════════════════════════════════════════════════════

def train_krum_model(server: KrumServer,
                     num_rounds: int,
                     local_epochs: int,
                     results_dir: str,
                     n_shot: int,
                     test_dataset: Dataset,
                     backdoor_cfg: BackdoorConfig) -> Dict:
    metrics_tracker = EnhancedMetricsTracker(results_dir, n_shot)
    os.makedirs(results_dir, exist_ok=True)

    eff_m = server.multi_krum_m if server.multi_krum_m is not None \
        else len(server.clients) - server.n_byzantine

    print(f"\nStarting Multi-Krum Federated Training:")
    print(f"N-shot: {n_shot}, N-way: {Config.N_WAY}, Query: {Config.N_QUERY}")
    print(f"Poisoning rate: {backdoor_cfg.POISONING_RATE}")
    print(f"Aggregation: Multi-Krum (f={server.n_byzantine}, m={eff_m})")
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

        print(f"Round {round_num + 1}/{num_rounds}: "
              f"Loss={round_metrics['loss']:.4f}  "
              f"TrainAcc={round_metrics['accuracy']:.2f}%  "
              f"CleanAcc={attack_metrics['clean_accuracy']:.2f}%  "
              f"ASR={attack_metrics['attack_success_rate']:.2f}%")

    metrics_tracker.plot_training_curves()
    metrics_tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_attack_metrics(attack_results, results_dir)
    save_attack_results(attack_results, results_dir)

    return {
        'training_metrics': metrics_tracker.get_serializable_state(),
        'attack_results':   attack_results,
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
    print(f"Aggregation: Multi-Krum (selects and averages m best clients)")
    print(f"Image size: API={Config.API_IMAGE_SIZE}  "
          f"Traffic={Config.TRAFFIC_IMAGE_SIZE}")

    results_root = Config.RESULTS_DIR
    os.makedirs(results_root, exist_ok=True)

    # n_byzantine must satisfy: NUM_CLIENTS >= 2 * n_byzantine + 3
    # With 5 clients: max f = 1  (5 >= 2*1+3 = 5 ✓)
    N_BYZANTINE = 1

    # Number of clients to select and average in Multi-Krum.
    # m = n - f is the common default (here: 5 - 1 = 4).
    # Setting MULTI_KRUM_M = 1 recovers standard (single-client) Krum.
    MULTI_KRUM_M = Config.NUM_CLIENTS - N_BYZANTINE

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
            'method':      'Multi-Krum',
            'n_byzantine': N_BYZANTINE,
            'm_selected':  MULTI_KRUM_M,
            'description': (
                'Computes the Krum score (sum of squared L2 distances to '
                f'the nearest {Config.NUM_CLIENTS - N_BYZANTINE - 2} '
                'neighbours) for each client, then selects the '
                f'{MULTI_KRUM_M} clients with the lowest scores and '
                'averages their parameters (equal weight). m=1 reduces '
                'to standard Krum.'
            ),
            'reference':   (
                'Blanchard et al., "Machine Learning with Adversaries: '
                'Byzantine Tolerant Gradient Descent", NeurIPS 2017'
            )
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
                run_dir = os.path.join(rate_dir, f'{n_shot}shot_multikrum')
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
                server = create_krum_backdoor_system(
                    model           = model,
                    datasets        = client_datasets,
                    device          = device,
                    backdoor_config = bdcfg,
                    n_byzantine     = N_BYZANTINE,
                    multi_krum_m    = MULTI_KRUM_M,
                )

                training_results = train_krum_model(
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
        comparison_path = os.path.join(
            results_root, 'final_multikrum_results.txt')
        with open(comparison_path, 'w') as f:
            f.write("Multi-Krum Aggregation Results Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Aggregation: Multi-Krum (f={N_BYZANTINE}, "
                    f"m={MULTI_KRUM_M})\n")
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
        print("FINAL RESULTS — Multi-Krum Aggregation")
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


def run_krum_experiment():
    print("=" * 60)
    print("Multi-Krum Aggregation — Untargeted Backdoor Attack Defence")
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
