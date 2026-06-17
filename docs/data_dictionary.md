# Data dictionary

All datasets are ABAQUS finite-element (RDB forming + springback) results, sampled by
Latin hypercube design. The manuscript Table 2 ranges are the nominal sampling targets;
the released min-max for each column is given in the "Actual (OE)" column below and is
authoritative for this data. Files are `.xlsx`, one row per sample, `float64`. Column
**order is significant** (the code reads columns positionally and by these exact header
names); do not rename headers.

## Files

| File | Role | Rows | Cols |
|---|---|---|---|
| `bilayer_target_OE.xlsx` | Target domain, original environment (OE), Stage-2 + accuracy/ablation | 98 | 15 (14 inputs + target) |
| `bilayer_target_PE.xlsx` | Perturbed environment (PE), paired EPE evaluation only (no training) | 60 | 15 |

The released bilayer target has **98** OE samples (manuscript rounds to "100") and **60**
paired PE samples.

The single-layer **source** domain (used only to pretrain SLSP-Net) is not redistributed;
the trained `models/slsp_net_pretrained.pth` is provided instead. The single-layer schema
is identical to the bilayer one below but **without** the `Thickness ratio` column.

## Columns (bilayer files; manuscript Table 2 symbols)

"Table-2 (nominal)" is the sampling range stated in manuscript Table 2; "Actual (OE)" is
the observed min–max in `bilayer_target_OE.xlsx` (the authoritative figure for the released
data). PE ranges are similar.

| # | Header | Symbol | Unit | Group | Table-2 (nominal) | Actual (OE) min–max |
|---|---|---|---|---|---|---|
| 0 | Tube diameter | D | mm | shape (x_SE) | 10-60 | 10.76-59.23 |
| 1 | Total thickness | t | mm | shape (x_SE) | 0.65-12 | 0.907-11.13 |
| 2 | Thickness ratio | r = t1/t2 | - | shape (x_SE) | 0.2-5 | 0.065-7.85 |
| 3 | Radius of bending die | R_B | mm | manufacturing (x_manu) | 20-300 | 25.0-287.5 |
| 4 | Friction between bending die and tube | f_B | - | manufacturing | 0.05-0.2 | 0.050-0.200 |
| 5 | Friction between pressure die and tube | f_P | - | manufacturing | 0.05-0.2 | 0.050-0.199 |
| 6 | Friction between wiper die and tube | f_w | - | manufacturing | 0.05-0.2 | 8.3e-5-0.049 |
| 7 | Gap between bending die and tube | G_B | mm | manufacturing | 0.05-0.2 | 0.050-0.199 |
| 8 | Gap between pressure die and tube | G_P | mm | manufacturing | 0.05-0.2 | 0.052-0.199 |
| 9 | Gap between wiper die and tube | G_w | mm | manufacturing | 0.05-0.2 | 0.050-0.200 |
| 10 | Different velocity of pressure die | v_B^d | mm/s | manufacturing | -0.15-0.15 | -0.150-0.148 |
| 11 | The initial position of the pressure die | L_P | mm | manufacturing | 0-45 | 0.31-44.81 |
| 12 | Angular velocity of bending die | omega_B | rad/s | manufacturing | 0.349-1.047 | 0.363-1.042 |
| 13 | Bending angle of bending die | alpha_B | **degrees** | manufacturing | 30-120 deg (= 0.524-2.094 rad in Table 2) | 31.2-119 |
| 14 | Springback | springback angle | degrees | **target** | - | 1.40-5.95 |

Notes:
- **Bending angle** is stored in **degrees** (30-120) in the data files; manuscript
  Table 2 lists it in radians (0.524-2.094). Same quantity, different unit.
- The "Actual (OE)" column is the authoritative range for the released data.
- `r` is the layer thickness ratio t1/t2 (t1 = outer/copper, t2 = inner/aluminium);
  it is the only structural feature distinguishing the bilayer target from the
  single-layer source.
- `bilayer_target_PE.xlsx` is the paired perturbation of `bilayer_target_OE.xlsx`:
  its 60 rows reproduce 60 of the 98 OE manufacturing-plan inputs under altered FE
  environment constants (material scaling, die/tooling, contact micro-slip), with the
  manufacturing parameters x_manu held identical (see manuscript Section 4.2.3).
- The raw ABAQUS decks/outputs (.inp/.odb/.cae) are **not** included; the tabulated
  results above are sufficient to reproduce all reported ML results.
