# MMALVIZ: A Multimodal Malware Visualization Dataset

This repository contains the official implementation accompanying our work on
**BRFFSL** (Backdoor-Resistant Federated Few-Shot Learning), a framework for
malware classification evaluated on **MMALVIZ**, a multimodal malware
visualization dataset combining API call images and network traffic images.
We introduce **ProtoKrum**, a robust defense method that couples prototype cosine-deviation scoring with Krum-based selection to defend federated few-shot malware classifiers against poisoning attacks.

To assess how well our defenses generalize beyond MMALVIZ, we additionally
evaluate all methods on the public **MalImg** malware image dataset. MMALVIZ
remains the primary dataset used throughout the paper; MalImg is used solely
as a secondary generalizability check.

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

## Repository Structure

```
MMALVIZ/
├── api_call_images/               # RGB visualizations of API calls
└── network_traffic_images/        # Grayscale visualizations of network traffic
├── ProtoKrum.py                  
├── ProtoTrimmed.py              
├── FedAvg.py                      
├── FoolsGold.py                 
├── Median.py                      
├── Multi-Krum.py                
├── RFA.py                        
├── README.md
├── LICENSE
├── requirements.txt
├── metadata.csv              
.
.
.
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

If your local folder names differ, update the paths in each script's
`Config` class:

```python
API_IMAGE_DIR     = os.path.join(SCRIPT_DIR, "api_call_images")
TRAFFIC_IMAGE_DIR = os.path.join(SCRIPT_DIR, "network_traffic_images")
```

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
