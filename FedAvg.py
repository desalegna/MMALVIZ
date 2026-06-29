"""
FedAvg Baseline — No Defense.
Supports both IID and Non-IID (Dirichlet) client data partitioning.
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
from datetime import datetime
from typing import Dict, List, Tuple
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
    TEST_SPLIT         = 0.2

    #   export API_IMAGE_DIR=/path/to/api_call_images
    #   export TRAFFIC_IMAGE_DIR=/path/to/network_traffic_images
    # MMALVIZ dataset folders nested under data/(data/api_call_images/, data/network_traffic_images/).

    API_IMAGE_DIR     = os.environ.get("API_IMAGE_DIR",     os.path.join(SCRIPT_DIR, "api_call_images"))
    TRAFFIC_IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", os.path.join(SCRIPT_DIR, "network_traffic_images"))
    RESULTS_DIR       = os.environ.get("RESULTS_DIR",       os.path.join(SCRIPT_DIR, "results_fedavg_baseline"))


# Partition modes — both are run automatically
PARTITION_MODES  = ["iid", "non_iid"]
POISONING_RATES  = [0.3, 0.5]
DIRICHLET_ALPHAS = [0.2, 0.5, 2.0]       # used only in non_iid mode


class BackdoorConfig:
    POISONED_CLIENT_ID = 1
    POISONING_RATE     = 0.3              # overridden per run
    TRIGGER_SIZE       = 4
    SCALE_FACTOR       = 40.0


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
    """Stratified shuffle-split into train (80 %) and test (20 %)."""
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

def iid_partition(
        api_images: np.ndarray,
        traffic_images: np.ndarray,
        labels: np.ndarray,
        n_clients: int,
        seed: int = 42
) -> List[Dict]:
    """
    IID partition: shuffle then split into n_clients equal shards.
    Every client receives approximately the same class distribution
    as the global training set.
    """
    rng     = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)

    shards = np.array_split(indices, n_clients)

    partitions = []
    for shard in shards:
        partitions.append({
            'api':     api_images[shard],
            'traffic': traffic_images[shard],
            'labels':  labels[shard],
        })

    print(f"\n[iid_partition] {n_clients} clients, "
          f"~{len(labels) // n_clients} samples each.")
    return partitions


# ════════════════════════════════════════════════════════════════════════════
# Non-IID Dirichlet partition
# ════════════════════════════════════════════════════════════════════════════

def dirichlet_partition(
        api_images: np.ndarray,
        traffic_images: np.ndarray,
        labels: np.ndarray,
        n_clients: int,
        alpha: float,
        seed: int = 42
) -> List[Dict]:
    """
    Non-IID partition via Dirichlet(alpha) distribution.

    alpha interpretation:
      0.2  — highly heterogeneous (each client dominated by 1-2 classes)
      0.5  — moderately heterogeneous
      1.0  — mildly heterogeneous
      2.0  — nearly IID

    Every client is guaranteed at least one sample per class (replace=True
    fallback) so that few-shot episodes never starve.
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
                               label: str):
    """Print per-client class distribution."""
    print(f"\nPer-client class distribution  ({label}):")
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
                              title_suffix: str,
                              save_path: str):
    """Stacked-bar plot of per-client class counts."""
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
    ax.set_title(f'Per-client class distribution  ({title_suffix})')
    ax.set_xlabel('Client')
    ax.set_ylabel('# samples')
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f'C{i+1}' for i in range(n_clients)])
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Metrics tracker
# ════════════════════════════════════════════════════════════════════════════

class EnhancedMetricsTracker:
    def __init__(self, results_dir, n_shot):
        self.results_dir = results_dir
        self.n_shot = n_shot
        self.metrics = {'loss': [], 'accuracy': []}
        self.client_metrics = {i: [] for i in range(Config.NUM_CLIENTS)}
        self.class_metrics = {}
        self.confusion_matrices = []

    def update(self, round_metrics, client_accuracies,
               class_performance=None, confusion_matrix=None):
        for key, value in round_metrics.items():
            self.metrics.setdefault(key, []).append(value)
        for client_id, acc in client_accuracies.items():
            self.client_metrics.setdefault(client_id, []).append(acc)
        if class_performance:
            for class_name, mets in class_performance.items():
                self.class_metrics.setdefault(class_name, {})
                for metric_name, value in mets.items():
                    self.class_metrics[class_name].setdefault(
                        metric_name, []).append(value)
        if confusion_matrix is not None:
            self.confusion_matrices.append(confusion_matrix)

    def get_serializable_state(self):
        return {
            'metrics':            dict(self.metrics),
            'client_metrics':     dict(self.client_metrics),
            'class_metrics':      dict(self.class_metrics),
            'confusion_matrices': [cm.tolist()
                                   for cm in self.confusion_matrices],
        }

    def plot_confusion_matrix(self, class_names, final=True):
        if not self.confusion_matrices:
            return
        cm  = self.confusion_matrices[-1] if final \
              else np.mean(self.confusion_matrices, axis=0)
        used = class_names[:cm.shape[0]]
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=used, yticklabels=used)
        label = "Final" if final else "Average"
        plt.title(f'{self.n_shot}-shot {label} Confusion Matrix '
                  f'(FedAvg — No Defense)')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        suffix = "final" if final else "avg"
        plt.savefig(os.path.join(
            self.results_dir,
            f'confusion_matrix_{suffix}_{self.n_shot}shot.png'))
        plt.close()

    def save_detailed_results(self, class_names):
        path = os.path.join(
            self.results_dir, f'detailed_results_{self.n_shot}shot.txt')
        with open(path, 'w') as f:
            f.write(f"Detailed Training Results ({self.n_shot}-shot)\n"
                    f"{'='*60}\n\n")
            f.write("Overall Performance:\n" + "-"*20 + "\n")
            for key, vals in self.metrics.items():
                f.write(f"Final {key}: {vals[-1]:.4f}\n")
                f.write(f"Average {key}: {np.mean(vals):.4f}\n")
                f.write(f"Std Dev {key}: {np.std(vals):.4f}\n\n")
            f.write("\nClient Performance:\n" + "-"*20 + "\n")
            for cid, accs in self.client_metrics.items():
                if not accs:
                    continue
                f.write(f"\nClient {cid+1}:\n")
                f.write(f"Final Accuracy: {accs[-1]:.2f}%\n")
                f.write(f"Average Accuracy: {np.mean(accs):.2f}%\n")
                f.write(f"Std Deviation: {np.std(accs):.2f}%\n")
            f.write("\nClass-wise Performance:\n" + "-"*20 + "\n")
            for cn in class_names:
                if cn in self.class_metrics:
                    f.write(f"\n{cn}:\n")
                    for metric, vals in self.class_metrics[cn].items():
                        f.write(f"Final {metric}: {vals[-1]:.4f}\n")
                        f.write(f"Average {metric}: {np.mean(vals):.4f}\n")
                        f.write(f"Std Dev {metric}: {np.std(vals):.4f}\n")

    def plot_training_curves(self):
        if not self.metrics.get('accuracy'):
            return
        rounds = range(1, len(self.metrics['accuracy']) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['accuracy'], 'b-',
                 label='Global Accuracy', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Accuracy '
                  f'(FedAvg — No Defense)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_accuracy_{self.n_shot}shot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        for cid, accs in self.client_metrics.items():
            if accs:
                plt.plot(range(1, len(accs) + 1), accs, marker='o',
                         markersize=4, label=f'Client {cid+1}', linewidth=2)
        plt.title(f'{self.n_shot}-shot Client Accuracies (FedAvg — No Defense)')
        plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'client_accuracies_{self.n_shot}shot.png'))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(rounds, self.metrics['loss'], 'r-',
                 label='Global Loss', linewidth=2)
        plt.title(f'{self.n_shot}-shot Global Training Loss '
                  f'(FedAvg — No Defense)')
        plt.xlabel('Round'); plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.7); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir, f'global_loss_{self.n_shot}shot.png'))
        plt.close()


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
# Server  (plain weighted FedAvg — no defense)
# ════════════════════════════════════════════════════════════════════════════

class Server:
    def __init__(self, model, clients, device):
        self.global_model = model
        self.clients      = clients
        self.device       = device

    def aggregate_models(self, client_updates):
        """
        Standard weighted FedAvg: model states are averaged with weights
        proportional to each client's average local training accuracy.
        No robustness mechanism is applied — this is the undefended baseline.
        """
        states  = [u['model_state'] for u in client_updates]
        accs    = [u['avg_accuracy'] for u in client_updates]
        total   = sum(accs) or 1.0
        weights = [a / total for a in accs]

        avg = OrderedDict()
        for key in states[0]:
            if states[0][key].dtype.is_floating_point:
                weighted = [s[key] * w for s, w in zip(states, weights)]
                avg[key] = torch.sum(torch.stack(weighted), dim=0)
            else:
                avg[key] = states[0][key].clone()

        self.global_model.load_state_dict(avg)

    def _run_episodes(self, loader, n_ep):
        all_pred, all_true, all_ci = [], [], []
        it = iter(loader)
        with torch.no_grad():
            for _ in range(n_ep):
                try:
                    data = next(it)
                except StopIteration:
                    it   = iter(loader)
                    data = next(it)
                sa, st, sl, qa, qt, ql, sc = data
                sa = sa.squeeze(0).to(self.device)
                st = st.squeeze(0).to(self.device)
                sl = sl.squeeze(0).to(self.device)
                qa = qa.squeeze(0).to(self.device)
                qt = qt.squeeze(0).to(self.device)
                ql = ql.squeeze(0).to(self.device)
                sc = sc.squeeze(0).cpu()

                sf = F.normalize(self.global_model(sa, st), p=2, dim=1)
                qf = F.normalize(self.global_model(qa, qt), p=2, dim=1)

                unique_sl  = torch.unique(sl)
                proto_list = []
                for i in range(len(unique_sl)):
                    proto_list.append(sf[sl == i].mean(0))
                protos = torch.stack(proto_list)

                logits = torch.mm(qf, protos.t()) / 0.5
                preds  = logits.argmax(1)

                for p in preds:
                    all_pred.append(sc[p.item()].item())
                for l in ql:
                    all_true.append(sc[l.item()].item())
                for p, l in zip(preds.cpu().numpy(), ql.cpu().numpy()):
                    all_ci.append((sc[p].item(), sc[l].item()))
        return all_pred, all_true, all_ci

    def evaluate(self, test_dataset):
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        preds, labels, ci = self._run_episodes(loader, Config.EVAL_EPISODES)

        try:
            cp, cm = calculate_class_metrics(
                preds, labels, test_dataset.class_names)
        except Exception as e:
            print(f"Warning metrics error: {e}")
            cp = {}
            cm = np.eye(len(test_dataset.class_names)) * 100

        correct = sum(p == l for p, l in zip(preds, labels))
        acc     = correct / len(labels) * 100 if labels else 0.0

        class_acc = {}
        for i, cn in enumerate(test_dataset.class_names):
            sub = [(p, t) for p, t in ci if t == i]
            class_acc[cn] = (sum(p == t for p, t in sub) / len(sub) * 100
                             if sub else 0.0)

        return {
            'accuracy':          acc,
            'total_samples':     len(labels),
            'correct_samples':   correct,
            'class_performance': cp,
            'class_accuracy':    class_acc,
            'confusion_matrix':  cm,
        }


class UntargetedBackdoorServer(Server):
    def __init__(self, model, clients, device, config):
        super().__init__(model, clients, device)
        self.config         = config
        self.attack_metrics = []

    def evaluate_untargeted_attack(self, test_dataset):
        self.global_model.eval()
        clean_ds    = copy.deepcopy(test_dataset)
        backdoor_ds = UntargetedBackdoorDataset(
            test_dataset.api_images.numpy(),
            test_dataset.traffic_images.numpy(),
            test_dataset.labels.copy(),
            test_dataset.class_names,
            test_dataset.n_support,
            self.config,
        )
        half = Config.EVAL_EPISODES // 2
        cl   = DataLoader(clean_ds,    batch_size=1, shuffle=True)
        bl   = DataLoader(backdoor_ds, batch_size=1, shuffle=True)

        cp, cl_true, _ = self._run_episodes(cl, half)
        bp, bl_true, _ = self._run_episodes(bl, half)

        clean_acc = (sum(p == l for p, l in zip(cp, cl_true)) / len(cl_true) * 100
                     if cl_true else 0.0)
        asr       = (sum(p != l for p, l in zip(bp, bl_true)) / len(bl_true) * 100
                     if bl_true else 0.0)

        all_cls = list(range(len(test_dataset.class_names)))
        try:
            clean_cm = confusion_matrix(
                cl_true, cp, labels=all_cls, normalize='true') * 100
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
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def calculate_class_metrics(predictions, labels, class_names):
    preds = np.array(predictions)
    true  = np.array(labels)
    uniq  = np.unique(np.concatenate([preds, true]))
    cidx  = {i: class_names[i] for i in uniq if i < len(class_names)}
    try:
        cm = confusion_matrix(true, preds, labels=list(cidx),
                              normalize='true') * 100
    except Exception:
        cm = np.eye(len(cidx)) * 100
    try:
        rep = classification_report(
            true, preds,
            labels=list(cidx),
            target_names=[cidx[i] for i in cidx],
            output_dict=True,
            zero_division=0,
        )
        cp = {cidx[i]: {'precision': rep[cidx[i]]['precision'],
                        'recall':    rep[cidx[i]]['recall'],
                        'f1_score':  rep[cidx[i]]['f1-score']}
              for i in cidx if cidx[i] in rep}
    except Exception:
        cp = {n: {'precision': 0., 'recall': 0., 'f1_score': 0.}
              for n in cidx.values()}
    return cp, cm


def create_untargeted_backdoor_system(model, datasets, device, backdoor_config):
    clients = []
    for i, ds in enumerate(datasets):
        if i == backdoor_config.POISONED_CLIENT_ID:
            clients.append(
                UntargetedBackdoorClient(i, model, ds, device, backdoor_config))
        else:
            clients.append(FederatedClient(i, model, ds, device))
    return UntargetedBackdoorServer(model, clients, device, backdoor_config)


def plot_untargeted_attack_metrics(attack_results, save_dir, tag=""):
    rounds = range(1, len(attack_results) + 1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].plot(rounds, [r['attack_success_rate'] for r in attack_results],
                 'r-o', linewidth=2, label='Misclassification Rate')
    axes[0].set_title(f'Backdoor Misclassification Rate '
                      f'(FedAvg — No Defense {tag})')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Misclassification Rate (%)')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()

    axes[1].plot(rounds, [r['clean_accuracy'] for r in attack_results],
                 'b-o', linewidth=2, label='Clean Accuracy')
    axes[1].set_title(f'Clean Accuracy (FedAvg — No Defense {tag})')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()

    plt.tight_layout()
    fname = f'untargeted_attack_metrics_{tag}.png' if tag \
            else 'untargeted_attack_metrics.png'
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()

    sc = plt.scatter(
        [r['clean_accuracy']      for r in attack_results],
        [r['attack_success_rate'] for r in attack_results],
        c=list(rounds), cmap='viridis', s=100,
    )
    for i, rn in enumerate(rounds):
        plt.annotate(
            str(rn),
            (attack_results[i]['clean_accuracy'],
             attack_results[i]['attack_success_rate']),
            xytext=(5, 5), textcoords='offset points',
        )
    plt.title(f'Attack Success vs. Clean Accuracy ({tag})')
    plt.xlabel('Clean Accuracy (%)')
    plt.ylabel('Attack Success Rate (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(sc, label='Round')
    plt.tight_layout()
    fname2 = f'attack_success_vs_clean_accuracy_{tag}.png' if tag \
             else 'attack_success_vs_clean_accuracy.png'
    plt.savefig(os.path.join(save_dir, fname2))
    plt.close()


def save_untargeted_attack_results(attack_results, save_dir, tag=""):
    fname = f'untargeted_attack_results_{tag}.txt' if tag \
            else 'untargeted_attack_results.txt'
    path = os.path.join(save_dir, fname)
    with open(path, 'w') as f:
        f.write(f"Untargeted Backdoor Attack Results (FedAvg — No Defense "
                f"[{tag}])\n{'='*60}\n\n")
        fm      = attack_results[-1]
        avg_asr = np.mean([r['attack_success_rate'] for r in attack_results])
        avg_ca  = np.mean([r['clean_accuracy']      for r in attack_results])
        f.write("Final Results:\n" + "-"*20 + "\n")
        f.write(f"Clean Accuracy:             {fm['clean_accuracy']:.2f}%\n")
        f.write(f"Attack Success Rate:        {fm['attack_success_rate']:.2f}%\n")
        f.write(f"Clean Samples:              {fm['clean_samples']}\n")
        f.write(f"Backdoor Samples:           {fm['backdoor_samples']}\n\n")
        f.write("Average Performance:\n" + "-"*20 + "\n")
        f.write(f"Avg Clean Accuracy:         {avg_ca:.2f}%\n")
        f.write(f"Avg Misclassification Rate: {avg_asr:.2f}%\n\n")
        f.write("Per-Class Misclassification (Final):\n" + "-"*20 + "\n")
        for cn, rate in fm['class_misclassification'].items():
            f.write(f"  {cn}: {rate:.2f}%\n")
        f.write("\nRound-by-Round:\n" + "-"*20 + "\n")
        for rn, m in enumerate(attack_results, 1):
            f.write(f"Round {rn:>2}: Clean={m['clean_accuracy']:.2f}%  "
                    f"ASR={m['attack_success_rate']:.2f}%\n")


def save_experiment_config(config_dict, results_dir):
    path = os.path.join(results_dir, 'experiment_config.txt')
    with open(path, 'w') as f:
        f.write("Experiment Configuration\n" + "="*60 + "\n\n")
        for k, v in config_dict.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for kk, vv in v.items():
                    f.write(f"  {kk}: {vv}\n")
            else:
                f.write(f"{k}: {v}\n")


# ════════════════════════════════════════════════════════════════════════════
# Image loading & alignment
# ════════════════════════════════════════════════════════════════════════════

def load_images(directory, target_size, is_api, stats):
    images, labels, paths = [], [], []
    dt = 'API' if is_api else 'Traffic'
    if not os.path.exists(directory):
        raise FileNotFoundError(
            f"Directory not found: {directory}\n"
            f"  Set the correct path via the API_IMAGE_DIR / "
            f"TRAFFIC_IMAGE_DIR environment variables, or run this "
            f"script from the repository root.")
    class_names = sorted(
        d for d in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, d))
    )
    print(f"\nLoading {dt} images from {directory}")
    label_map = {}
    for label, cn in enumerate(class_names):
        label_map[cn] = label
        cdir  = os.path.join(directory, cn)
        valid = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        files = [f for f in os.listdir(cdir)
                 if os.path.splitext(f)[1].lower() in valid]
        count = 0
        for fn in files:
            fp = os.path.join(cdir, fn)
            try:
                with Image.open(fp) as img:
                    img = img.convert('RGB' if is_api else 'L')
                    img = img.resize(target_size, Image.LANCZOS)
                    arr = np.array(img)
                    arr = arr.transpose(2, 0, 1) if is_api else arr[None, ...]
                    arr = (arr / 127.5) - 1.0
                    images.append(arr)
                    labels.append(label)
                    paths.append(fp)
                    count += 1
            except Exception as e:
                print(f"  Error loading {fp}: {e}")
        stats.add_samples(dt, cn, count)
    if not images:
        raise ValueError(f"No images loaded from {directory}")
    return np.array(images), np.array(labels), class_names, label_map, paths


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
# Cross-partition comparison plots
# ════════════════════════════════════════════════════════════════════════════

def compare_pr_results(all_results, save_dir, pr_list, n_shots_list,
                       alpha_list):
    """Per-poisoning-rate comparison within each partition/alpha setting."""
    cmp_dir = os.path.join(save_dir, 'pr_comparison')
    os.makedirs(cmp_dir, exist_ok=True)

    # Non-IID: one plot per (alpha, n_shot)
    for alpha in alpha_list:
        for n_shot in n_shots_list:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            for pr in pr_list:
                key = f'pr{pr}_ns{n_shot}_non_iid_alpha{alpha}'
                if key not in all_results:
                    continue
                ar   = all_results[key]['attack_results']
                rnds = range(1, len(ar) + 1)
                axes[0].plot(rnds, [r['clean_accuracy']      for r in ar],
                             marker='o', linewidth=2, label=f'PR={pr}')
                axes[1].plot(rnds, [r['attack_success_rate'] for r in ar],
                             marker='o', linewidth=2, label=f'PR={pr}')

            axes[0].set_title(f'{n_shot}-shot  Clean Accuracy')
            axes[0].set_xlabel('Round'); axes[0].set_ylabel('Accuracy (%)')
            axes[0].grid(True, linestyle='--', alpha=0.7); axes[0].legend()

            axes[1].set_title(f'{n_shot}-shot  Attack Success Rate')
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Misclassification (%)')
            axes[1].grid(True, linestyle='--', alpha=0.7); axes[1].legend()

            fig.suptitle(f'PR Comparison — Non-IID (α={alpha}), {n_shot}-shot, '
                         f'FedAvg — No Defense', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(
                cmp_dir,
                f'pr_comparison_{n_shot}shot_non_iid_alpha{alpha}.png'))
            plt.close()

    # IID: one plot per n_shot
    for n_shot in n_shots_list:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for pr in pr_list:
            key = f'pr{pr}_ns{n_shot}_iid'
            if key not in all_results:
                continue
            ar   = all_results[key]['attack_results']
            rnds = range(1, len(ar) + 1)
            axes[0].plot(rnds, [r['clean_accuracy']      for r in ar],
                         marker='o', linewidth=2, label=f'PR={pr}')
            axes[1].plot(rnds, [r['attack_success_rate'] for r in ar],
                         marker='o', linewidth=2, label=f'PR={pr}')

        axes[0].set_title(f'{n_shot}-shot  Clean Accuracy')
        axes[0].set_xlabel('Round'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].grid(True, linestyle='--', alpha=0.7); axes[0].legend()

        axes[1].set_title(f'{n_shot}-shot  Attack Success Rate')
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Misclassification (%)')
        axes[1].grid(True, linestyle='--', alpha=0.7); axes[1].legend()

        fig.suptitle(f'PR Comparison — IID, {n_shot}-shot, '
                     f'FedAvg — No Defense', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(
            cmp_dir, f'pr_comparison_{n_shot}shot_iid.png'))
        plt.close()

    # Summary text
    with open(os.path.join(cmp_dir, 'pr_comparison_summary.txt'), 'w') as f:
        f.write("Poisoning-Rate Comparison Summary (FedAvg — No Defense)\n"
                + "="*60 + "\n\n")

        for partition_mode in PARTITION_MODES:
            alpha_list_use = [None] if partition_mode == "iid" else alpha_list
            for alpha in alpha_list_use:
                a_label = "IID" if alpha is None else f"Non-IID α={alpha}"
                for n_shot in n_shots_list:
                    f.write(f"=== {n_shot}-shot | {a_label} ===\n")
                    for pr in pr_list:
                        key = (f'pr{pr}_ns{n_shot}_iid' if alpha is None
                               else f'pr{pr}_ns{n_shot}_non_iid_alpha{alpha}')
                        if key not in all_results:
                            continue
                        ar     = all_results[key]['attack_results']
                        ca_f   = ar[-1]['clean_accuracy']
                        as_f   = ar[-1]['attack_success_rate']
                        ca_a   = np.mean([r['clean_accuracy']      for r in ar])
                        as_a   = np.mean([r['attack_success_rate'] for r in ar])
                        best_a = max(r['attack_success_rate'] for r in ar)
                        best_r = max(range(len(ar)),
                                     key=lambda i: ar[i]['attack_success_rate']) + 1
                        f.write(f"\n  PR={pr}\n")
                        f.write(f"    Final Clean Acc : {ca_f:.2f}%\n")
                        f.write(f"    Final ASR       : {as_f:.2f}%\n")
                        f.write(f"    Avg   Clean Acc : {ca_a:.2f}%\n")
                        f.write(f"    Avg   ASR       : {as_a:.2f}%\n")
                        f.write(f"    Best  ASR       : {best_a:.2f}%  "
                                f"(Round {best_r})\n")
                    f.write("\n")


def plot_iid_vs_noniid_comparison(all_results: Dict, save_dir: str,
                                   poison_rate: float, n_shot: int,
                                   alpha_list: List):
    """
    For a fixed poison_rate and n_shot, plot clean accuracy and ASR across
    IID and Non-IID (all alphas) settings side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'IID vs Non-IID Comparison  '
                 f'(rate={poison_rate}, {n_shot}-shot, FedAvg — No Defense)',
                 fontsize=13)

    for ax, metric_key, ylabel, title in [
        (axes[0], 'clean_accuracy',      'Clean Accuracy (%)',      'Clean Accuracy'),
        (axes[1], 'attack_success_rate', 'Attack Success Rate (%)', 'Attack Success Rate'),
    ]:
        # IID
        iid_key = f'pr{poison_rate}_ns{n_shot}_iid'
        if iid_key in all_results:
            ar     = all_results[iid_key]['attack_results']
            values = [r[metric_key] for r in ar]
            ax.plot(range(1, len(values) + 1), values,
                    'k-', linewidth=2.5, label='IID')

        # Non-IID — one line per alpha
        for alpha in alpha_list:
            noniid_key = f'pr{poison_rate}_ns{n_shot}_non_iid_alpha{alpha}'
            if noniid_key in all_results:
                ar     = all_results[noniid_key]['attack_results']
                values = [r[metric_key] for r in ar]
                ax.plot(range(1, len(values) + 1), values,
                        linestyle='--', linewidth=1.8,
                        label=f'Non-IID α={alpha}')

        ax.set_title(title)
        ax.set_xlabel('Round')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=9)

    plt.tight_layout()
    fname = (f'iid_vs_noniid_rate{int(poison_rate*100)}'
             f'_{n_shot}shot.png')
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Core training loop
# ════════════════════════════════════════════════════════════════════════════

def train_fedavg_model(server, num_rounds, local_epochs,
                       results_dir, n_shot, test_dataset, run_tag=""):
    tracker    = EnhancedMetricsTracker(results_dir, n_shot)
    attack_dir = os.path.join(
        results_dir, f'{n_shot}shot_untargeted_backdoor')
    os.makedirs(attack_dir, exist_ok=True)

    pr_label = server.config.POISONING_RATE
    print(f"\n{'='*80}")
    print(f"  FedAvg — No Defense  |  {n_shot}-shot  |  PR={pr_label}  "
          f"|  [{run_tag}]")
    print(f"  N-way={Config.N_WAY}  N-query={Config.N_QUERY}  "
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
            'loss':     np.mean(round_losses),
            'accuracy': np.mean(list(round_accs.values())),
        }

        atk = server.evaluate_untargeted_attack(test_dataset)
        attack_results.append(atk)

        tracker.update(
            round_metrics=round_metrics,
            client_accuracies=round_accs,
            confusion_matrix=atk['confusion_matrix'],
        )

        if atk['clean_accuracy'] > best_clean_acc:
            best_clean_acc = atk['clean_accuracy']
            print(f"  ↑ New best clean accuracy: {best_clean_acc:.2f}%")

        print(f"Round {rnd+1:>2}/{num_rounds} | "
              f"Loss={round_metrics['loss']:.4f} | "
              f"Train={round_metrics['accuracy']:.1f}% | "
              f"Clean={atk['clean_accuracy']:.1f}% | "
              f"ASR={atk['attack_success_rate']:.1f}%")

    tracker.plot_training_curves()
    tracker.plot_confusion_matrix(test_dataset.class_names, final=True)
    plot_untargeted_attack_metrics(attack_results, attack_dir, tag=run_tag)
    save_untargeted_attack_results(attack_results, attack_dir, tag=run_tag)
    tracker.save_detailed_results(test_dataset.class_names)

    print(f"\nBest clean accuracy this run: {best_clean_acc:.2f}%")
    return {
        'training_metrics':    tracker.get_serializable_state(),
        'attack_results':      attack_results,
        'best_clean_accuracy': best_clean_acc,
    }


# ════════════════════════════════════════════════════════════════════════════
# Final summary writer
# ════════════════════════════════════════════════════════════════════════════

def write_final_summary(all_results: dict,
                        save_dir: str,
                        poisoning_rates: list,
                        n_shots_list: list,
                        alpha_list: list) -> str:
    """
    Write results_fedavg_baseline/final_fedavg_results.txt

    Columns
    -------
    PR           – poisoning rate
    Partition    – iid  or  non_iid
    Alpha        – Dirichlet α  (IID runs show "IID")
    Shot         – few-shot k
    ACC (final)  – clean accuracy at the last FL round  (%)
    ASR (final)  – attack success rate at the last FL round (%)
    Avg ACC      – clean accuracy averaged over all rounds  (%)
    Avg ASR      – ASR averaged over all rounds  (%)
    """
    rows = []
    for pr in poisoning_rates:
        for partition_mode in PARTITION_MODES:
            alpha_list_use = [None] if partition_mode == "iid" else alpha_list
            for alpha in alpha_list_use:
                for n_shot in n_shots_list:
                    key = (f'pr{pr}_ns{n_shot}_iid' if alpha is None
                           else f'pr{pr}_ns{n_shot}_non_iid_alpha{alpha}')
                    if key not in all_results:
                        continue
                    ar = all_results[key]["attack_results"]
                    rows.append({
                        "PR":          pr,
                        "Partition":   partition_mode,
                        "Alpha":       "IID" if alpha is None else alpha,
                        "Shot":        n_shot,
                        "ACC (final)": ar[-1]["clean_accuracy"],
                        "ASR (final)": ar[-1]["attack_success_rate"],
                        "Avg ACC":     float(np.mean(
                            [r["clean_accuracy"]      for r in ar])),
                        "Avg ASR":     float(np.mean(
                            [r["attack_success_rate"] for r in ar])),
                        "_n_rounds":   len(ar),
                    })

    if not rows:
        print("[write_final_summary] No results found – skipping.")
        return ""

    col_w = {"PR": 6, "Partition": 10, "Alpha": 8, "Shot": 6,
             "ACC (final)": 12, "ASR (final)": 12,
             "Avg ACC": 10, "Avg ASR": 10}
    cols  = ["PR", "Partition", "Alpha", "Shot",
             "ACC (final)", "ASR (final)", "Avg ACC", "Avg ASR"]

    def fmt(col, val):
        if col in ("PR", "Partition", "Alpha"):
            return f"{str(val):<{col_w[col]}}"
        if col == "Shot":
            return f"{int(val):<{col_w[col]}}"
        return f"{val:<{col_w[col]}.2f}"

    header       = "".join(f"{c:<{col_w[c]}}" for c in cols)
    sep          = "-" * len(header)
    n_rounds_rep = rows[0]["_n_rounds"]

    lines = [
        "FedAvg — No Defense – Final Summary",
        "=" * len(header),
        f"Rounds per run : {n_rounds_rep}  |  "
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        header,
        sep,
    ]
    for row in rows:
        lines.append("".join(fmt(c, row[c]) for c in cols))
    lines += [
        sep,
        "",
        "Column descriptions",
        "  PR          : fraction of training samples poisoned on the backdoor client",
        "  Partition   : iid = equal shards; non_iid = Dirichlet partition",
        "  Alpha       : Dirichlet concentration (lower = more heterogeneous; "
        "IID = equal shards)",
        "  Shot        : k-shot (number of support examples per class per episode)",
        "  ACC (final) : global clean accuracy at the final FL round  (%)",
        "  ASR (final) : attack success rate (misclassification) at final round (%)",
        "  Avg ACC     : clean accuracy averaged over all FL rounds  (%)",
        "  Avg ASR     : attack success rate averaged over all FL rounds  (%)",
        "",
        "Defense applied : None — standard weighted FedAvg (weighted by client",
        "  average local training accuracy). No clipping, smoothing, or outlier",
        "  rejection is applied. Serves as the undefended baseline.",
    ]

    text     = "\n".join(lines) + "\n"
    out_path = os.path.join(save_dir, "final_fedavg_results.txt")
    with open(out_path, "w") as fh:
        fh.write(text)

    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(text)
    print(f"  Saved → {out_path}")
    print(f"{'='*70}\n")
    return out_path


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def run_fedavg_baseline(n_shots_list=(1, 5),
                        poisoning_rates=(0.3, 0.5),
                        dirichlet_alphas=(0.2, 0.5, 2.0)):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    print(f"Using device: {device}")
    print(f"Image size: API={Config.API_IMAGE_SIZE}  "
          f"Traffic={Config.TRAFFIC_IMAGE_SIZE}")
    print(f"Partition modes:  {PARTITION_MODES}")
    print(f"Dirichlet alphas: {list(dirichlet_alphas)}  (Non-IID only)")
    print(f"Poisoning rates:  {list(poisoning_rates)}")

    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    save_experiment_config({
        'num_clients':        Config.NUM_CLIENTS,
        'num_rounds':         Config.NUM_ROUNDS,
        'local_epochs':       Config.LOCAL_EPOCHS,
        'test_split':         Config.TEST_SPLIT,
        'n_way':              Config.N_WAY,
        'n_query':            Config.N_QUERY,
        'embedding_dim':      Config.EMBEDDING_DIM,
        'episodes_per_epoch': Config.EPISODES_PER_EPOCH,
        'api_image_size':     str(Config.API_IMAGE_SIZE),
        'traffic_image_size': str(Config.TRAFFIC_IMAGE_SIZE),
        'device':             str(device),
        'partition_modes':    str(PARTITION_MODES),
        'dirichlet_alphas':   str(list(dirichlet_alphas)),
        'aggregation': {
            'method':      'FedAvg (No Defense)',
            'description': 'Standard weighted FedAvg, weighted by each '
                           "client's average local training accuracy. "
                           'No clipping, smoothing, or outlier rejection.',
        },
        'backdoor': {
            'type':            'Untargeted',
            'poisoned_client': BackdoorConfig.POISONED_CLIENT_ID,
            'trigger_size':    BackdoorConfig.TRIGGER_SIZE,
            'scale_factor':    BackdoorConfig.SCALE_FACTOR,
            'poisoning_rates': list(poisoning_rates),
        },
    }, Config.RESULTS_DIR)

    # ── Load data once ─────────────────────────────────────────────────────
    data_stats = DataStats()
    print("\nLoading data ...")
    api_imgs, api_lbl, api_cls, api_lmap, api_pth = load_images(
        Config.API_IMAGE_DIR, Config.API_IMAGE_SIZE, True, data_stats)
    trf_imgs, trf_lbl, _, _, trf_pth = load_images(
        Config.TRAFFIC_IMAGE_DIR, Config.TRAFFIC_IMAGE_SIZE, False, data_stats)

    data_stats.display_distribution()
    data_stats.plot_distribution(
        os.path.join(Config.RESULTS_DIR, 'data_distribution.png'))

    print("\nAligning traffic with API data ...")
    (api_imgs, api_lbl, api_pth,
     trf_imgs, trf_lbl, trf_pth) = align_traffic_with_api(
        (api_imgs, api_lbl, api_pth),
        (trf_imgs, trf_lbl, trf_pth),
    )

    # ── Stratified 80/20 split — done once ────────────────────────────────
    print("\nSplitting data into 80% train / 20% test...")
    (tr_api, tr_trf, tr_lbl,
     te_api, te_trf, te_lbl) = split_data(
        api_imgs, trf_imgs, api_lbl,
        test_ratio=Config.TEST_SPLIT, seed=42)
    print(f"  Train: {len(tr_lbl)}  |  Test: {len(te_lbl)}")

    all_results: Dict = {}

    for pr in poisoning_rates:
        for partition_mode in PARTITION_MODES:

            alpha_list_use = [None] if partition_mode == "iid" \
                             else list(dirichlet_alphas)

            for alpha in alpha_list_use:

                # ── Human-readable labels ──────────────────────────────────
                if partition_mode == "iid":
                    run_label = "IID"
                    alpha_tag = "iid"
                else:
                    run_label = f"Non-IID α={alpha}"
                    alpha_tag = f"non_iid_alpha{alpha}"

                print(f"\n\n{'#'*70}")
                print(f"#  PR={pr}  |  {run_label}")
                print(f"{'#'*70}")

                cfg = BackdoorConfig()
                cfg.POISONING_RATE = pr

                combo_dir = os.path.join(
                    Config.RESULTS_DIR, f'PR{pr}', alpha_tag)
                os.makedirs(combo_dir, exist_ok=True)

                # ── Partition ──────────────────────────────────────────────
                if partition_mode == "iid":
                    partitions = iid_partition(
                        tr_api, tr_trf, tr_lbl,
                        n_clients=Config.NUM_CLIENTS, seed=42)
                else:
                    partitions = dirichlet_partition(
                        tr_api, tr_trf, tr_lbl,
                        n_clients=Config.NUM_CLIENTS,
                        alpha=alpha, seed=42)

                print_client_distribution(partitions, api_cls, run_label)
                plot_client_distribution(
                    partitions, api_cls,
                    title_suffix=run_label,
                    save_path=os.path.join(
                        combo_dir,
                        f'client_distribution_{alpha_tag}.png'))

                # ── n-shot loop ────────────────────────────────────────────
                for n_shot in n_shots_list:
                    result_key = (f'pr{pr}_ns{n_shot}_iid' if alpha is None
                                  else f'pr{pr}_ns{n_shot}_non_iid_alpha{alpha}')
                    run_dir = os.path.join(combo_dir, f'{n_shot}shot')
                    os.makedirs(run_dir, exist_ok=True)

                    print(f"\n{'='*20} {n_shot}-shot | PR={pr} | "
                          f"{run_label} {'='*20}")

                    client_datasets = [
                        UntargetedBackdoorDataset(
                            p['api'], p['traffic'], p['labels'].copy(),
                            api_cls, n_shot)
                        for p in partitions
                    ]
                    test_ds = UntargetedBackdoorDataset(
                        te_api, te_trf, te_lbl.copy(), api_cls, n_shot)

                    model  = HybridNet().to(device)
                    server = create_untargeted_backdoor_system(
                        model=model,
                        datasets=client_datasets,
                        device=device,
                        backdoor_config=cfg,
                    )

                    res = train_fedavg_model(
                        server=server,
                        num_rounds=Config.NUM_ROUNDS,
                        local_epochs=Config.LOCAL_EPOCHS,
                        results_dir=run_dir,
                        n_shot=n_shot,
                        test_dataset=test_ds,
                        run_tag=f'{alpha_tag}_{n_shot}shot',
                    )
                    all_results[result_key] = res

    # ── Cross-PR comparison plots ──────────────────────────────────────────
    compare_pr_results(all_results, Config.RESULTS_DIR,
                       list(poisoning_rates), list(n_shots_list),
                       list(dirichlet_alphas))

    # ── IID vs Non-IID comparison plots ───────────────────────────────────
    print("\nGenerating IID vs Non-IID comparison plots...")
    for pr in poisoning_rates:
        for n_shot in n_shots_list:
            plot_iid_vs_noniid_comparison(
                all_results, Config.RESULTS_DIR, pr, n_shot,
                list(dirichlet_alphas))

    # ── Final summary table ────────────────────────────────────────────────
    write_final_summary(
        all_results=all_results,
        save_dir=Config.RESULTS_DIR,
        poisoning_rates=list(poisoning_rates),
        n_shots_list=list(n_shots_list),
        alpha_list=list(dirichlet_alphas),
    )

    print(f"\n\nAll runs complete.  Results → {Config.RESULTS_DIR}")
    return all_results


def main():
    run_fedavg_baseline(
        n_shots_list=[1, 5],
        poisoning_rates=POISONING_RATES,
        dirichlet_alphas=DIRICHLET_ALPHAS,
    )


if __name__ == "__main__":
    main()
