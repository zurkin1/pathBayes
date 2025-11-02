# PathBayes – Bayesian Network Pathway Analysis

![alt text](image.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

PathBayes is an enhanced version of [PathWeigh](https://github.com/zurkin1/PathWeigh), introducing **Bayesian network inference** for pathway activity calculation. While PathWeigh uses simple averaging of interaction activities, PathBayes employs **Loopy Belief Propagation (LBP)** to properly handle pathway topology, including feedback loops and complex regulatory cascades.

### Key Improvements Over PathWeigh

1. **Bayesian Network Propagation**: Replaces naive averaging with probabilistic inference that respects pathway structure
2. **Feedback Loop Handling**: LBP naturally converges on pathways with cycles (e.g., A→B→C→A)
3. **UDP Integration**: Combines original UDP (Up/Down Probability) values with propagated beliefs for internal genes
4. **Enhanced Performance**: Borrows efficient parallel processing architecture from [PathSingle](https://github.com/zurkin1/PathSingle)

### Current Limitations

- **RNA-seq only**: Currently supports RNA-seq data (negative binomial distribution). Microarray support (Gaussian mixture models) will be added in future releases.

---

## Algorithm

### 1. UDP Calculation

For each gene across samples, fit a probability distribution to expression data:
- **RNA-seq**: Negative binomial distribution
- Calculate UDP: probability of being in "Up" state

### 2. Bayesian Network Construction

Transform pathway topology into a Bayesian network:
- **Nodes**: Genes/proteins from pathway
- **Edges**: Interaction relationships (activation/inhibition)
- **CPT (Conditional Probability Table)**: Parameterized by interaction type
  - Activation: 0.85 weight
  - Inhibition: 0.15 weight
  - Baseline: 0.1

### 3. Loopy Belief Propagation

**Initialization**:
```
beliefs = {gene: UDP[gene] for all genes}
```

**Iteration** (until convergence or limit):
1. For each non-input gene:
   - Collect messages from parent interactions
   - Combine using **noisy-OR**: `P(child=Up) = 1 - ∏(1 - P(parent_i=Up))`
   - Apply CPT weight based on interaction type
2. Blend propagated belief with original UDP:
   ```
   final_belief = UDP_WEIGHT × original_UDP + (1 - UDP_WEIGHT) × propagated_belief
   ```
3. Check convergence: `max(|new_belief - old_belief|) < 1e-3`

**Output**:
```
pathway_activity = mean(beliefs[gene] for all genes)
```

### Why Loopy Belief Propagation?

Traditional pathway methods fail with feedback loops:
```
A → B → C → A  (cycle detected → infinite recursion or arbitrary cutoff)
```

LBP iteratively passes probabilistic messages until beliefs stabilize, naturally handling:
- Feedback loops
- Mutual inhibition
- Complex feedforward-with-feedback architectures

---

## Installation

```bash
git clone https://github.com/zurkin1/PathBayes.git
cd PathBayes
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pandas
- numpy
- scipy
- scikit-learn

---

## Usage

### Basic Workflow

1. **Calculate UDP from RNA-seq data**:
```python
from udp import calc_udp_multi_process

# Prepare input.csv: rows=genes, columns=samples
calc_udp_multi_process()
# Output: data/output_udp.csv
```

2. **Calculate pathway activities**:
```python
from activity import calc_activity

# Uses pathway_relations.csv and output_udp.csv
activity_df = calc_activity(udp_file='./data/output_udp.csv')
# Output: data/output_activity_lbp.csv
```

### Configuration

Edit constants in `activity.py`:

```python
UDP_WEIGHT = 0.5        # Balance: 0=pure propagation, 1=pure UDP
CPD_ACTIVATION = 0.85   # Activation interaction strength
CPD_INHIBITION = 0.15   # Inhibition interaction strength  
CPD_BASELINE = 0.1      # Baseline probability
```

### Pathway File Format

`pathway_relations.csv`:
```csv
pathway,source,interactiontype,target
p53_pathway,TP53*MDM2,activation,CDKN1A
p53_pathway,ATM,activation$phosphorylation,TP53
p53_pathway,MDM2,inhibition$ubiquitination,TP53
```

- **source/target**: Gene names (use `*` for multiple genes in OR logic)
- **interactiontype**: activation, inhibition, phosphorylation, etc.

---

## Example: Testing with Toy Network

```python
# Generate test data (see test artifact)
import pandas as pd

# Create toy pathway with feedback loop
pathway_data = {
    'pathway': ['test'] * 5,
    'source': ['A', 'B', 'C', 'D', 'C'],
    'interactiontype': ['activation', 'activation', 'activation', 'activation', 'inhibition'],
    'target': ['C', 'C', 'D', 'C', 'E']
}
pd.DataFrame(pathway_data).to_csv('data/test_pathway_relations.csv', index=False)

# Create UDP input
udp_data = {'sample1': [0.9, 0.8, 0.5, 0.5, 0.5]}
pd.DataFrame(udp_data, index=['a','b','c','d','e']).to_csv('data/test_output_udp.csv')

# Run
from activity import calc_activity
calc_activity(udp_file='./data/test_output_udp.csv')
```

---

## Pathway Database

 PathBayes currently 357 curated pathways. Click the [link](https://github.com/zurkin1/PathSingle/blob/main/code/data/pathway_relations.csv) to view the full list. List of supported pathways.

Pathway files are stored in `data/pathway_relations.csv`.

---

## Performance

- **UDP fitting**: ~1-2 minutes for 20,000 genes × 100 samples (parallelized)
- **LBP inference**: ~0.1-1 second per pathway per sample
- **Total runtime**: 357 pathways × 100 samples = ~2-5 minutes on 8-core CPU

Parallelization strategies:
- Sample-level: ProcessPoolExecutor across samples
- Gene-level: Multiprocessing for UDP fitting

---

## Citation

If you use PathBayes in your research, please cite the original PathBays paper:

```bibtex
@inproceedings{livne2022pathweigh,
  title={PathWeigh--Quantifying the Behavior of Biochemical Pathway Cascades},
  author={Livne, Dani and Efroni, Sol},
  booktitle={International Work-Conference on Bioinformatics and Biomedical Engineering},
  pages={339--345},
  year={2022},
  organization={Springer}
}
```

---

## License

Released under MIT License. See [LICENSE](LICENSE) file.