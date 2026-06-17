# Data and code availability statement

Suggested wording for the article's "Data availability" / "Data and code availability"
section (usually placed just before the References).

This package is released on **GitHub only**. The repository is self-contained: it includes
the bilayer datasets (original and perturbed environments), the pretrained subnets, the
25-model ensemble, the two paper-selected models, the full source code, and the verified
reference results that reproduce the reported tables and figures.

**Statement to use (GitHub):**

> The source code and the data supporting the findings of this study are openly available
> in the GitHub repository at https://github.com/ChangSun-Eng/MET-Net. The repository
> contains the bilayer original/perturbed-environment datasets, pretrained and trained
> model checkpoints, and scripts that reproduce the reported tables and figures.

**Note on the single-layer source domain.** The bilayer original/perturbed-environment
(OE/PE) datasets are provided. The single-layer source-domain data (used only to pretrain
SLSP-Net) are not redistributed; the trained `models/slsp_net_pretrained.pth` is provided
instead, which is sufficient to reproduce all reported bilayer results.

**Note on FE data.** The reported machine-learning results are fully reproducible from the
tabulated datasets provided. The underlying ABAQUS finite-element decks and output
databases (.inp/.odb/.cae) are not redistributed; the FE modelling protocol is described
in Section 4.1 and follows the validated pipeline of the cited prior work.
