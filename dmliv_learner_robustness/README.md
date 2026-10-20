# Anonymous supplementary DML-IV learner sensitivity code

**Purpose:** Code to IMPLEMENT additional learner-robustness comparisons requested by a reviewer. This is **not** evidence that the extra regressions have already been conducted, or that they passed a robustness check. No confidential bilateral service-trade or flight data are distributed.

## Scope

The study examines U.S. bilateral service trade with 191 partner economies over 2005–2024, estimated separately for **2005–2019** and **2020–2024**, with six dependent variables: commercial, transport, and travel service exports and imports.

- **Baseline model** for each of two treatments: `ln(Direct Cities)` or `ln(Average Seats per Direct City)`; respective aviation IV is `ln(Average Connect City Number)` or `ln(Common City Number)`.
- **Interaction model** for each treatment: three endogenous regressors, `ln(air connectivity)`, `ln(Internet Penetration)`, and their product; instruments are the log aviation IV, log Internet diffusion exposure IV, and **product of the two logged instruments**.
- Controls: RTA, LSBCI, ln exchange rate, ln GDP per capita, partner-country and year indicator variables.
- Nuisance-function learners: **Random Forest (RF)**, **cross-validated Lasso**, **Gradient Boosting Regression (GBR)**, and **Voting ensemble** of RF + Lasso + GBR.
- Estimator: **EconML 0.16.0 `econml.iv.dml.DMLIV`** with `X=None` (constant treatment coefficient), `W` controls, and `GroupKFold` separating partner countries.
- **No country-clustered inference is claimed.** These scripts compare point estimates across learners and, optionally, across different group-preserving splits.

### What is deliberately not claimed

1. The parameters in `config.yaml` are **illustrative settings for supplementary checks**, **not verified original RF settings**. To replicate the manuscript, the original hyperparameters and cross-fitting configuration must first be copied into the config file.
2. This repository includes **code but no original data, estimates, p values, or regression tables**. Running code on synthetic data is *only an implementation check*.
3. Using country indicators in `W` is **not** the same as a within transformation; `GroupKFold` is **not** country-clustered standard-error estimation.
4. The robustness exercise does not validate exclusion restrictions or address weak instruments.
5. Although a repeated split check with medians can be requested, it does not prove robustness across learners.

## Inputs: private model-ready CSV

Save an analysis-ready file locally as `data/analysis_private.csv`, or provide `--input /path/to/private.csv`. **Never upload it with the code**. The CSV should contain preprocessed columns as mapped in `config.yaml` (rename mappings there if your actual column names differ). All logarithms should already be present. Required input aliases are:

| Internal alias | Default CSV column | Meaning |
|---|---|---|
| `partner` | `partner` | Partner-country identifier (string) |
| `year` | `year` | 2005–2024 |
| `commercial_export` | `ln_commercial_export` | Logged U.S. commercial service export |
| `commercial_import` | `ln_commercial_import` | Logged U.S. commercial service import |
| `transport_export` | `ln_transport_export` | Logged U.S. transport service export |
| `transport_import` | `ln_transport_import` | Logged U.S. transport service import |
| `travel_export` | `ln_travel_export` | Logged U.S. travel service export |
| `travel_import` | `ln_travel_import` | Logged U.S. travel service import |
| `direct_cities` | `ln_direct_cities` | Logged number of directly connected partner-country cities |
| `avg_seats` | `ln_average_seats_per_direct_city` | Logged scheduled seats per direct city |
| `internet` | `ln_internet_penetration` | Logged share/rate of partner-country internet use (use consistent units) |
| `direct_cities_iv` | `ln_average_connect_city_number` | Logged aviation IV for network scope |
| `avg_seats_iv` | `ln_common_city_number` | Logged aviation IV for service intensity |
| `internet_iv` | `ln_internet_diffusion_exposure` | Logged 2005 telephone penetration × external diffusion instrument |
| `rta` | `rta` | RTA indicator |
| `lsbci` | `lsbci` | Bilateral liner shipping connectivity |
| `exchange_rate` | `ln_exchange_rate` | Logged exchange rate |
| `gdp_pc` | `ln_gdp_per_capita` | Logged partner GDP per capita |

**Critical:** The manuscript uses log notation but annual partner-country connected-city counts may be zero. This project never silently changes `ln(x)` to `ln(x+1)`, drops zeros, or creates arbitrary pseudocounts. Construct the CSV with *the same transformations, sample filters, IV constructions, zero handling, and units as the actual analysis*, then confirm that the number of observations matches the article. No missing values and no duplicate partner-year IDs are permitted. For Internet IV, the country-specific 2005 telephone penetration and external Internet diffusion must be constructed from the actual underlying data, excluding the U.S. and the corresponding partner-country as described in the manuscript. The script does **not** reconstruct the IV from private raw data.

## Setup and run

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Safe synthetic check: no confidential data is read
python -m src.generate_demo_data --output data/synthetic_example.csv
python -m unittest discover -s tests -v
python -m src.runner --config config.yaml --input data/synthetic_example.csv --validate-only

# Short learner-comparison example, using SYNTHETIC input here only
python -m src.runner --config config.yaml --input data/synthetic_example.csv \
  --outcome transport_export --model baseline

# With your actual PRIVATE model-ready file: validate, then run full comparison
python -m src.runner --config config.yaml --validate-only
python -m src.runner --config config.yaml
```

The full run covers 2 periods × 6 outcomes × 2 air variables × 2 specifications × 4 learners = **192 fits per sample-split repetition**, so it can take considerable time. Use `--outcome`, `--model`, or `--learner` for a short check. Increase `split_repetitions: 3` to calculate the median coefficient across three group-respecting folds assignments; the first assignment is deterministic `GroupKFold`, and subsequent assignments are shuffled across partner groups. Merely changing the RF random seed without reshuffling country groups is **not** a repeated cross-fitting partition.

## Output files

- `outputs/sample_sizes.csv`: N and number of partner countries within each period.
- `outputs/learner_estimates_by_split.csv`: full coefficient estimates for each learner/repeat.
- `outputs/median_estimates_by_learner.csv`: across-split median, minimum, maximum, SD (if repeated).
- `outputs/comparison_by_learner.csv`: side-by-side coefficient estimates from RF/Lasso/GBR/ensemble.
- `outputs/run_manifest.json`: sample-splitting, inference and transformation declarations.

No original data or estimate outputs are included in this repository. **Results cannot be reproduced from this code alone** without authorized access to the underlying data, and results from the synthetic demonstration are not meaningful empirical evidence.

## Paper-to-code implementation details

`X=None` in `DMLIV` requests a constant effect; partner/year dummies and economic controls enter through `W`. For the interaction model, the main air and internet terms and the log×log interaction are jointly treated as endogenous. It uses three corresponding instruments, including the product of their logged IVs. This is distinct from a model treating internet penetration as exogenous.

EconML's `DMLIV` implementation fits the nuisance functions for Y and T, then fits a constant coefficient of the instrument-induced difference in T. The reported output here is `const_marginal_effect()` for each endogenous term. No significance markers or heteroskedasticity/cluster robust SEs are invented in the supplementary comparison. For real-paper inference use only the verified methodology of the original script.

## Software docs

- EconML 0.16 DMLIV API: https://econml.azurewebsites.net/_autosummary/econml.iv.dml.DMLIV.html
- EconML 0.16 source: https://econml.azurewebsites.net/_modules/econml/iv/dml/_dml.html
- sklearn GroupKFold: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html

## Anonymous double-blind submission

Use a neutral repository name. Review all file names, notebooks, notebooks' embedded metadata, code comments, commit history, README, account owners, and links for identity clues. Publish a **reviewer-safe anonymous link**, not a personally identifying GitHub profile URL. Ensure the journal allows such links. The code itself does not include author names or personal emails.
