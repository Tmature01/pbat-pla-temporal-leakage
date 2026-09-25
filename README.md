# Temporal data leakage in PBAT/PLA biodegradation modelling

Supporting data and code for:

> **Temporal Data Leakage Inflates Machine Learning Performance in Compostable
> PBAT/PLA Polymer Biodegradation Modeling**
> *ACS Sustainable Chemistry & Engineering* (accepted)

The study quantifies how the choice of evaluation protocol - rather than model
architecture - governs the apparent accuracy of machine-learning models trained on
daily-sampled composting data.

This repository is the permanent home of the data and code for that article:
<https://github.com/Tmature01/pbat-pla-temporal-leakage>

---

## What is in this repository

| Folder | Contents |
|---|---|
| `data/` | Processed data: the cross-material datasets and the per-material validation files |
| `scripts/` | Analysis, figure and table scripts |
| `scripts/make_tables.py` | Re-formats the manuscript tables from the analysis outputs |
| `utils/`, `models/` | Small helper packages imported by some analysis scripts |
| `results/` | Output folder for the analysis scripts (empty in the release) |
| `figure_data/` | Output folder for the figure-data scripts (empty in the release) |
| `figures/` | Output folder for the figure files (empty in the release) |
| `validation/` | Output folder for the validation reports (empty in the release) |

## Data

| File | Rows | Description |
|---|---|---|
| `data/pbat_tps_data.csv`, `pbat_pbs_data.csv` | 200 / 200 | Cross-material datasets underlying the feature-ablation results reported in the Supporting Information |
| `data/validation_*.csv` | 180 | Per-material blank/cellulose/sample CO2 and degradation for the six materials in Table 1 and SI Table S7 |

The three experimental conditions are (i) 50 °C, 70 %, ratio 70 %, 0.90 kg;
(ii) 58 °C, 60 %, ratio 100 %, 0.50 kg; (iii) 58 °C, 70 %, ratio 0 %, 0.75 kg.

Only processed data are distributed. The respirometry workbooks as exported by
the instrument, and the direct extractions made from them, are not part of this
repository; they are available from the corresponding author on request.
`scripts/cross_material_validation.py` reads those workbooks and therefore
cannot be executed from this repository alone; it is included because it
documents how the files in `data/validation_*.csv` were derived.

The primary 543-observation PBAT/PLA modelling dataset used in the manuscript is
likewise not distributed here. The analysis and figure scripts are therefore
provided as reference implementations that document how each result and panel
was produced, rather than as a pipeline that runs from this repository alone.

## Environment

The analyses were run with Python 3.10.10 and the packages listed in
`requirements.txt` (numpy, pandas, scikit-learn, scipy, statsmodels, matplotlib,
gplearn, deap, pykrige, openpyxl).

```
pip install -r requirements.txt
```

## Running the code

Scripts resolve all paths relative to the repository root. Two of them run from
this repository alone:

```
python scripts/doptimal_design_vif.py    # design diagnostics; needs no data file
python scripts/make_tables.py            # builds the auxiliary-material table from data/validation_*.csv
```

The remaining analysis and figure scripts read the primary modelling dataset or
other inputs that are not distributed here; they are included for transparency.

`scripts/make_tables.py` only re-formats numbers that already exist in
`results/`; it does not refit any model, and it reports which tables it can build
from the outputs that are present.

The output folders (`results/`, `figure_data/`, `figures/`, `validation/`) ship
empty and are excluded by `.gitignore`; they are populated when the scripts run.

The figure scripts write their intermediate CSV files into `figure_data/`.

## Figure scripts

The main-text figures were assembled by eight composite scripts
(`scripts/fig_composite_01_leakage.py` ... `fig_composite_08_robustness.py`) from
intermediate CSV files that lived in a temp folder (`C:\temp`) on the original
authoring machine. Two of the scripts included here (`fig_data_leakage_concept.py`
and `fig_vif_geometry.py`) compute part of those inputs and document how the
panels were built; the remainder are plotting code:

| Script | Writes |
|---|---|
| `fig_data_leakage_concept.py` | `Fig_DataLeakage_Protocols.csv`, `Fig_DataLeakage_KDE.csv`, `Fig_DataLeakage_Illustration.csv` |
| `fig_vif_geometry.py` | `Fig_VIF_3D_Points.csv`, `Fig_VIF_Analogy.csv`, `Fig_VIF_Plane_Grid.csv`, `Fig_VIF_Summary.csv` |

The remaining figure scripts are included as **reference implementations**: they
reproduce the plotting and layout of the published panels, but the intermediate
CSV files they read are no longer available and are not part of this repository.
They are kept so that the definition of every panel is traceable from the code.

## Citation

If you use these data or scripts, please cite the accompanying article
(*Temporal Data Leakage Inflates Machine Learning Performance in Compostable
PBAT/PLA Polymer Biodegradation Modeling*, ACS Sustainable Chemistry &
Engineering) and link to this repository:

<https://github.com/Tmature01/pbat-pla-temporal-leakage>

## License

- **Code** (`scripts/`, `utils/`, `models/`): MIT License - see `LICENSE`.
- **Data** (`data/`): Creative Commons Attribution 4.0 International
  (CC BY 4.0) - see `data/LICENSE-CC-BY-4.0.txt`.

Please cite the article above when reusing either.
