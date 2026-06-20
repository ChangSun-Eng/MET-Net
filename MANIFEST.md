# MET-Net release manifest

- Version: **1.0.0**
- Date: 2026-06-17
- Files (release payload): 49
- Total size: 19.7 MB
- `checksums_sha256.txt` lists the SHA-256 of every release file **except `MANIFEST.md` and `checksums_sha256.txt` themselves** (verify with `sha256sum -c checksums_sha256.txt`).

The 25-model (5-fold x 5-seed) ensemble `models/met_net_ensemble.pkl` is included so that the EPE tables (Tables 4-5) reproduce directly.

| File | Bytes | SHA-256 (first 16) |
|---|---:|---|
| `.gitignore` | 398 | `7f892ef63575900e` |
| `CITATION.cff` | 1400 | `ddefbcd731a8b810` |
| `LICENSE` | 1400 | `5eeb59d4a7d12d86` |
| `README.md` | 7183 | `5cabc346dd3e8f89` |
| `configs/met_net.yaml` | 2665 | `78e7618d91a5ab1e` |
| `data/LICENSE.txt` | 808 | `0a333656d1da76ca` |
| `data/bilayer_target_OE.xlsx` | 26227 | `c465c51ff2947077` |
| `data/bilayer_target_PE.xlsx` | 18869 | `42fea9c7612b2054` |
| `docs/data_availability_statement.md` | 1571 | `d980414cf3988294` |
| `docs/data_dictionary.md` | 3849 | `86d366bcb5abc4c8` |
| `docs/naming_map.md` | 3586 | `bbc7ef8dabeb0ee1` |
| `docs/reproducibility.md` | 3641 | `a0d3fbe4595e55f5` |
| `environment.yml` | 1049 | `0c7f42635f37ea43` |
| `models/met_net_ensemble.pkl` | 11554721 | `403f90b622d559bf` |
| `models/met_net_fold0_seed0.pth` | 226195 | `d91c70317433fecb` |
| `models/met_net_fold3_seed3.pth` | 226195 | `a8289373fd6f340c` |
| `models/se_net_pretrained.pth` | 2677102 | `c7887c311014655c` |
| `models/slsp_net_pretrained.pth` | 204688 | `cac7f2c87835bc0e` |
| `models/tree_ensemble_simple.pkl` | 4222828 | `881e1ec8fe14da3b` |
| `pyproject.toml` | 734 | `30dc87c33921a681` |
| `reference_results/ablation_runs/met_net_25runs.csv` | 4692 | `3eeef3dfc59837c5` |
| `reference_results/ablation_runs/met_net_bn_25runs.csv` | 4624 | `c6f892ae8be4e6e8` |
| `reference_results/ablation_runs/met_net_freezese_25runs.csv` | 4627 | `c48a83f03422ce47` |
| `reference_results/ablation_runs/met_net_freezeslsp_25runs.csv` | 3942 | `cfe9871d53637958` |
| `reference_results/ablation_runs/met_net_fwp_25runs.csv` | 4740 | `30f8848129c05d65` |
| `reference_results/ablation_runs/met_net_nocal_25runs.csv` | 3922 | `b66a58258c940c10` |
| `reference_results/ablation_runs/met_net_sen_25runs.csv` | 4629 | `e55a9327a7aca202` |
| `reference_results/ablation_runs/met_net_sln_25runs.csv` | 4593 | `0153724023d5a40a` |
| `reference_results/ablation_runs/met_net_theoryse_25runs.csv` | 4628 | `b8f4c04b838673fc` |
| `reference_results/epe_predictions_OE.csv` | 8499 | `81d69f8b184353c0` |
| `reference_results/epe_predictions_PE.csv` | 5231 | `a7777401efe06b2e` |
| `reference_results/stage1_pretraining_metrics.csv` | 109 | `d3ddc5c5c4181ab6` |
| `reference_results/table3_ablation.csv` | 652 | `6f01da74e0af1cbb` |
| `reference_results/table4_epe_per_variant.csv` | 499 | `0d5e60d9bdd846cd` |
| `reference_results/table5_module_effects.csv` | 205 | `b2a20f6c5273b1de` |
| `reference_results/training_dynamics/met_net_fold3_seed3_final_metrics.csv` | 361 | `91682619aa8dcf8d` |
| `reference_results/training_dynamics/met_net_fold3_seed3_per_epoch.npz` | 212150 | `f7dd4e9fa181a8e1` |
| `requirements.txt` | 557 | `f07889849ee9d984` |
| `scripts/explain_ale_regime.py` | 23906 | `2dd9ecc3e83f6206` |
| `scripts/explain_gradientshap.py` | 14256 | `964ffbc8ab5707a2` |
| `scripts/reproduce_figures.py` | 2516 | `64fdbdd5213e6024` |
| `scripts/reproduce_tables.py` | 3971 | `e8d5fb86aef4cf12` |
| `scripts/run_ablation.py` | 4120 | `8c19e407292b07e1` |
| `scripts/run_epe.py` | 32929 | `fa3a98bcd88980bf` |
| `scripts/train_met_net.py` | 4133 | `de2a1ccc08eacda7` |
| `scripts/train_se_net.py` | 29109 | `6de7e926b3e4d61b` |
| `scripts/train_slsp_net.py` | 34126 | `9af60ac55d6fd2b7` |
| `src/metnet/__init__.py` | 890 | `f899cd6692716156` |
| `src/metnet/met_net.py` | 67277 | `ed9bed03cbe015a4` |
