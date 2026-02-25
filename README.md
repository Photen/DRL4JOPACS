# DRL4JOPACS

Official code release for the OMEGA manuscript:

> **A Deep Reinforcement Learning-based Approach for Joint Optimization of Pricing and Acquisition for Cloud Service**

## Confidentiality / Peer-Review-Only Notice

This repository is shared **only for the purpose of peer review** of the above manuscript.

- You may run the code and inspect the implementation to evaluate the manuscript.
- **You may not** use, copy, distribute, or disclose this code (or any derivative) to any third party.
- **You may not** use this code to produce results, software, manuscripts, or other outputs that overlap with the manuscript’s contributions **before the manuscript is formally published**, unless you have obtained prior written consent from the authors.

See [LICENSE](LICENSE) for the binding terms.

## Repository Overview

This project studies **joint decision-making of pricing and capacity acquisition** for a cloud service provider. The learning-based approach (DRL) is implemented as a PPO-style actor-critic agent interacting with an event-driven environment.

The repository contains two experiment tracks:

- **Simulation**: synthetic scenario settings in [num_exp/Simulation](num_exp/Simulation)
- **Aliyun**: scenario settings and data files in [num_exp/Aliyun](num_exp/Aliyun)

Common utilities are in [mutils](mutils).

## Environment / Dependencies

- Python 3.9+ is recommended.
- Core dependencies used by the scripts:
  - `numpy`
  - `pandas`
  - `torch`
  - `matplotlib`

Example installation:

```bash
pip install -U numpy pandas torch matplotlib
```

Notes:

- If CUDA is available, the scripts will automatically use GPU (`torch.cuda.is_available()`).
- Exact results may vary slightly across hardware / CUDA / PyTorch versions.

## How to Run

Each experiment folder under `num_exp/` is self-contained. Run commands **from within the corresponding directory** so that relative output paths (e.g., `results/`, `models/`) are created in that directory.

### 1) Proposed DRL Method (Train → Test)

For the proposed DRL method, the intended workflow is:

1. Run `drl_train.py` to train models and save checkpoints under `models/`.
2. Run `drl_test.py` to load the saved checkpoints (typically `best_model.pth`) and produce evaluation CSVs under `results/`.

#### A) Simulation Track

Training:

```bash
cd num_exp/Simulation
python drl_train.py
```

Optional arguments (see script help):

```bash
python drl_train.py --help
python drl_train.py --seed 42 --max_train_episodes 2000 --eval_episodes 30
python drl_train.py --multi_seeds 11,22,33 --max_train_episodes 2000
python drl_train.py --stochastic_eval
```

Testing (expects trained checkpoints under `num_exp/Simulation/models/scenario_*/best_model.pth`):

```bash
cd num_exp/Simulation
python drl_test.py
```

Outputs:

- Models: `num_exp/Simulation/models/scenario_<id>/best_model.pth` and `final_model.pth`
- Results: `num_exp/Simulation/results/drl_results_mean.csv`, `drl_results_std.csv`, etc.

#### B) Aliyun Track

Training:

```bash
cd num_exp/Aliyun
python drl_train.py
```

Testing (loads `models/scenario_*/best_model.pth` if present):

```bash
cd num_exp/Aliyun
python drl_test.py
```

Outputs:

- Models: `num_exp/Aliyun/models/scenario_<id>/best_model.pth` and `final_model.pth`
- Results: `num_exp/Aliyun/results/drl_results_mean.csv` (and other CSVs created by the script)

### 2) Baselines (Direct Run)

Baseline methods are implemented as standalone scripts and can be executed directly (no prior training step required).

From the target experiment directory:

```bash
cd num_exp/Simulation
python bts.py
python bucb.py
python clairvoyant.py
python po.py
```

Or for the Aliyun track:

```bash
cd num_exp/Aliyun
python bts.py
python bucb.py
python clairvoyant.py
python po.py
```

Each script writes summary CSV files under `results/` in the same directory.

## Data

- The Aliyun track includes CSV files under [num_exp/Aliyun/Ali](num_exp/Aliyun/Ali).
- Scenario parameters (prices / arrival rates / service rates and global constants) are defined in each track’s `settings.py`.

## Reproducibility

- Some scripts set seeds internally (e.g., `numpy` and `torch` seeds set to 42 by default).
- The Simulation training script additionally exposes `--seed` and `--multi_seeds` to run repeated experiments.
- For strict determinism on GPU, PyTorch may require additional configuration; even then, exact bitwise reproducibility is not guaranteed across platforms.

## Citation

If this code is used after publication (and with the authors’ permission when required), please cite the OMEGA manuscript.

```bibtex
@unpublished{drl4jopacs_omega,
  title  = {A Deep Reinforcement Learning-based Approach for Joint Optimization of Pricing and Acquisition for Cloud Service},
  note   = {Under review at OMEGA}
}
```

## License

This repository is **not open-source** during peer review. Please read [LICENSE](LICENSE).
