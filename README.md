# MMALVIZ: A Multimodal Malware Visualization Dataset

This repository contains the official implementation accompanying our work on
**BRFFSL** (Backdoor-Resistant Federated Few-Shot Learning), a framework for
malware classification evaluated on **MMALVIZ**, a multimodal malware
visualization dataset combining API call images and network traffic images.
We introduce **ProtoKrum**, a robust defense method that couples prototype cosine-deviation scoring with Krum-based selection to defend federated few-shot malware classifiers against poisoning attacks.

## Dataset Overview

MMALVIZ is a multimodal dataset for malware detection and classification
research based on dynamic analysis. It contains 888 samples spanning seven
malware classes, each represented as PNG images across two modalities:

- **RGB images** derived from API call sequences — `/api_call_images/`
- **Grayscale images** derived from network traffic captures — `/network_traffic_images/`

### Dataset Details

| Property | Value |
|---|---|
| Total samples | 888 |
| Malware classes | 7 (Cryptbot, Formbook, Remcos, Rokrat, Stealc, Vidar, Xworm) |
| Image format | PNG — RGB (3-channel) for API calls, grayscale (1-channel) for network traffic |
| Labels | Malware class |
| Purpose | Malware detection, classification, and behavioral analysis |

### Sample Distribution

| Malware Class | API Call | Network Traffic | Total |
|---|---|---|---|
| Cryptbot | 14 | 44 | 58 |
| Formbook | 15 | 268 | 283 |
| Remcos | 15 | 156 | 171 |
| Rokrat | 14 | 129 | 143 |
| Stealc | 15 | 88 | 103 |
| Vidar | 15 | 88 | 103 |
| Xworm | 14 | 13 | 27 |
| **Total** | **102** | **786** | **888** |

The dataset comprises 102 API call sequences (rendered as RGB PNGs) and 786
network traffic captures (rendered as grayscale PNGs). Sample counts differ
across modalities because a single malware execution can yield a different
number of API call sequences and network traffic captures during dynamic
analysis.

### Data Source

- **Analysis method:** Dynamic analysis via Cuckoo Sandbox.
- **API calls:** Extracted from Cuckoo Sandbox JSON reports, capturing
  ordered sequences of Windows API calls made during malware execution.
- **Network traffic:** Captured as PCAP files during sandboxed execution.

## Method Overview

- A dual-encoder backbone that jointly processes API call
  images (RGB) and network traffic images (grayscale), fusing them into a
  shared 128-dimensional, L2-normalized embedding space.
- **ProtoKrum** — our proposed defense mechanism, combining prototype cosine-deviation scoring on the embedding surface with Krum-based selection on the parameter surface to penalize clients whose prototypes deviate from consensus and suppress malicious updates during aggregation.
- **ProtoTrimmed** — also utilizes dual-feature
  prototypical representation and cosine-deviation scoring, but using
  trimmed mean aggregation instead of Krum selection.
- **Baselines** — FedAvg, FoolsGold, Median, RFA, and Multi-Krum for direct comparison.
- Every method is evaluated under both **IID** and **non-IID** client
  partitioning, controlled via script arguments.

## Repository Structure

```
MMALVIZ/
├── README.md
├── LICENSE
├── requirements.txt
├── metadata.csv                   # class names and per-class sample counts
├── ProtoKrum.py                   # proposed defense (ProtoKrum)
├── ProtoTrimmed.py                # proposed defense (ProtoTrimmed)
├── FedAvg.py                      # baseline
├── FoolsGold.py                   # baseline
├── Median.py                      # baseline
├── Multi-Krum.py                  # baseline
├── RFA.py                         # baseline
├── api_call_images/               # RGB visualizations of API calls
└── network_traffic_images/        # Grayscale visualizations of network traffic
```

## Requirements

- Python 3.9+
- CUDA-capable GPU recommended (CPU also supported)

```bash
pip install -r requirements.txt
```

## Dataset Setup

The dataset is included in this repository under `api_call_images/` and
`network_traffic_images/`, with one subfolder per malware family for each modality.

### Client Data Partitioning

The dataset is first split using a stratified 80/20 train-test split, with no
sample shared between partitions. The training partition is then distributed
across clients in one of two ways:

- **IID** — each client receives a proportional share of every class
  (`iid_partition()` in each script).
- **Non-IID** — each client receives a class-skewed share drawn from a
  Dirichlet distribution, with concentration parameter
  α ∈ {0.2, 0.5, 2.0} controlling the degree of heterogeneity (lower α →
  more skewed). Lower α values simulate stronger label-distribution
  imbalance across clients.

Each client constructs its N-way K-shot episodes locally from its own
training partition only.

If your local folder names differ, update the paths in each script's
`Config` class:

```python
API_IMAGE_DIR     = os.path.join(SCRIPT_DIR, "api_call_images")
TRAFFIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, "network_traffic_images")
```

## Running the Experiments

All scripts are located in the root directory. Partition mode (IID or non-IID)
is controlled via the `--partition` argument:

```bash
# ── IID setting ───────────────────────────────
python ProtoKrum.py    --partition iid
python ProtoTrimmed.py --partition iid
python FedAvg.py       --partition iid
python Median.py       --partition iid
python RFA.py          --partition iid
python Multi-Krum.py   --partition iid
python FoolsGold.py    --partition iid

# ── Non-IID setting ───────────────────────────
python ProtoKrum.py    --partition non_iid
python ProtoTrimmed.py --partition non_iid
python FedAvg.py       --partition non_iid
python Median.py       --partition non_iid
python RFA.py          --partition non_iid
python Multi-Krum.py   --partition non_iid
python FoolsGold.py    --partition non_iid
```

Each script evaluates poisoning rates `PR ∈ {0.3, 0.5}` across 1-shot and
5-shot configurations, and writes results to its own `results_<method>/`
directory, including:

- `final_<method>_results.txt` — summary table of ACC / ASR by PR and shot
- per-round training curves and confusion matrices

## Key Configuration Parameters

| Parameter             | Default      | Description                            |
|------------------------|--------------|----------------------------------------|
| `NUM_CLIENTS`          | 5            | Number of federated clients            |
| `NUM_ROUNDS`           | 20           | Federated training rounds              |
| `LOCAL_EPOCHS`         | 5            | Local epochs per client per round      |
| `N_WAY`                | 5            | Few-shot N-way classification          |
| `N_QUERY`              | 2            | Query samples per class per episode    |
| `POISONING_RATES`      | `[0.3, 0.5]` | Poisoning rates evaluated              |
| `POISONED_CLIENT_ID`   | 1            | Index of the malicious client          |
| `TRIGGER_SIZE`         | 4            | Backdoor trigger patch size            |

## License

The dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Users must give appropriate credit, provide a link to the license, and
indicate if changes were made. Code in this repository is licensed as
described in [LICENSE](./LICENSE).

## Ethical Considerations

This dataset is intended for research purposes only. All malware samples
were dynamically analyzed within Cuckoo Sandbox to ensure safe handling.
Users must adhere to responsible-use guidelines and avoid any misuse of
the data.

## Citation

If you use the MMALVIZ dataset or this code in your research, please cite:

```
[Add paper citation once published]
```
