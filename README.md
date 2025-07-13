# Book Breaker

Discovering wild, off-book gambits by disrupting principal variations with early sacrifices and chaos lines.

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Initialization](#initialization)
  - [Building the Openings Database](#building-the-openings-database)
- [Usage](#usage)
- [Roadmap](#roadmap)
- [License](#license)

## Overview

BookBreaker is an experimental Python toolkit built atop Stockfish to explore new gambit ideas.

By intentionally perturbing engine principal variations with early material sacrifices, it generates “chaos lines” and clusters them against known opening motifs. While early versions may yield few truly novel gambits, the framework is designed to improve over time and harness community contributions.

## Features

- **Sacrificial lines generator**  
  Systematically inject pawn, knight, and bishop sacrifices into standard theory lines.  
- **Feature extraction**  
  Quantify opening characteristics and tactical motifs from large PGN collections.  
- **Clustering & Novelty detection**  
  Apply PCA and clustering to separate candidate lines that diverge from established gambit clusters.  
- **Human-bias filter**  
  Profile engine evaluations to prioritize lines likely to surprise human opponents.  
- **Modular pipelines**  
  End-to-end scripts for extraction, clustering, generation, and ranking, plus notebooks for visualization.  

## Getting Started

### Prerequisites

- Python 3.8+
- Stockfish (v14+) installed or cloned in `stockfish/stockfish`
- `pip` for installing dependencies

### Installation

1. Clone the repository:  

   ```bash
   git clone https://github.com/0x48piraj/bookbreaker.git
   cd bookbreaker
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Initialization

Before running the pipeline, create the warehouse structure:

```bash
mkdir -p warehouse/{candidates,clusters,features,ranked,lichess-chess-openings}
```

This will prepare folders for:

* `warehouse/candidates/`
* `warehouse/clusters/`
* `warehouse/features/`
* `warehouse/ranked/`
* `warehouse/lichess-chess-openings/`

### Building the Openings Database

1. **Download opening table files**
   Run the following script from the project root to fetch the latest Lichess opening tables:

```bash
bash scripts/fetch_openings.sh
```

This will download `a.tsv` through `e.tsv` into the `warehouse/lichess-chess-openings/` directory.

2. **Merge opening tables into a single TSV**
   Combine the downloaded files into one master TSV file for downstream processing:

```bash
python scripts/build_opening_book.py \
  warehouse/lichess-chess-openings/a.tsv \
  warehouse/lichess-chess-openings/b.tsv \
  warehouse/lichess-chess-openings/c.tsv \
  warehouse/lichess-chess-openings/d.tsv \
  warehouse/lichess-chess-openings/e.tsv \
  > warehouse/lichess-chess-openings/all.tsv
```

Alternatively, you can use a wildcard:

```bash
python scripts/build_opening_book.py warehouse/lichess-chess-openings/*.tsv \
  > warehouse/lichess-chess-openings/all.tsv
```

## Usage

Run the full pipeline from extraction through ranking:

```bash
bash scripts/run_all.sh \
  --openings warehouse/lichess-chess-openings/all.tsv \
  --output-dir warehouse
```

Or invoke individual stages:

1. **Extract features and cluster lines**

   ```bash
   python extract/extract_and_cluster_gambits.py --openings all.tsv
   ```
2. **Generate candidates**

   ```bash
   python generate/generate_cluster_analogs.py --openings all.tsv --depth 12
   ```
3. **Rank and filter**

   ```bash
   python analysis/rank_candidates.py --candidates warehouse/candidates/*.json
   ```

## Roadmap

### **v1.0 (\~4 weeks): Foundation & First Results**

Goal: Complete core pipelines, improve data flow, and release first batch of candidate gambits.

* **Code & Structure**

  * Finalize modular directory layout
  * Add project-level configuration file (e.g., `config.yaml`)
  * Standardize logging across modules

* **Pipeline Improvements**

  * Build end-to-end script with CLI args and validation
  * Add support for reproducible seeds in line generation
  * Normalize PGN parsing and output formatting

* **Visualization & Reporting**

  * Jupyter notebooks for:

    * Cluster visualizations (2D/3D with PCA, t-SNE)
    * Eval swing graphs by depth and material imbalance
  * Add heatmap of evaluation change across line branches

* **Analysis Enhancements**

  * Introduce **evaluation weighting** by:

    * Sacrifice depth (early = higher novelty weight)
    * Eval volatility (higher swings = higher chaos potential)
  * Add early support for tactical motifs:

    * King exposure
    * Underpromotion
    * Piece coordination motifs (e.g. rook lift, bishop battery)

* **Output**

  * Publish first **Gambit Digest** (top 5 chaos lines)
  * ~Release public GitHub repo with README, docs, and examples~

### **v2.0 (\~3 months): Community & Usability**

Goal: Enable users to explore, contribute, and experiment with gambits at scale.

* **Community Contributions**

  * Launch public **Gambit Repository** (user-submitted lines + annotations)
  * Add rating, voting, and tagging on candidate gambits
  * Include PGN comment extraction for thematic classification

* **User Experience**

  * Build **interactive dashboard**:

    * Browse gambits by motif, ECO code, chaos score
    * Plot decision trees and alternative PV branches
  * Allow inline PGN playback using browser UI (e.g. Chessground)

* **Improved Line Generation**

  * Add **multi-piece sacrifice** generator
  * Add **non-sacrificial chaos** patterns (e.g. tempo loss, king walks, early h5/a5)
  * Support line "augmentation" — modify existing PVs without full regeneration

* **Engine + Evaluation Improvements**

  * Integrate `multipv` output from Stockfish to find alternate aggressive lines
  * Introduce early **human-likeness heuristics**:

    * Filter lines that engines approve but humans would reject (non-intuitive traps)

* **Collaboration**

  * Organize first **community challenge**:

    * "Find a playable gambit in a rarely explored ECO code"

### **v3.0 (\~6 months): Automation & Scale**

Goal: Automate discovery at large scale and support external use via APIs and plugins.

* **Web API & Automation**

  * Launch REST API:

    * `/generate?eco=C45&depth=14&type=knight_sac`
    * `/rank?lineset_id=abc123`
  * Batch run through entire Lichess opening dataset and archive findings
  * Add scheduled gambit refresh: weekly auto-run pipeline with top results

* **Machine Learning Enhancements**

  * Build initial **motif classifier**:

    * Train basic supervised model to classify lines by tactical theme
  * Prototype **RL module**:

    * Fine-tune sacrifice timing with reward signals based on eval swing + cluster distance

* **Extensibility**

  * Add plugin system:

    * Support third-party line generators and custom scoring metrics
  * Open gambit "IDE" interface for hobbyists to test ideas with custom inputs

* **Public Tools**

  * Launch **BookBreaker Lite**: a CLI + web UI tool that lets players input any PGN and:

    * Suggest where chaos might improve practical outcomes
    * Highlight potential gambit opportunities

The roadmap is flexible. As gambit discovery is inherently noisy and exploratory, each milestone is designed to yield partial value, both to researchers and players, while keeping long-term discovery goals intact.

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
