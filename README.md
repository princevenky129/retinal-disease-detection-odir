# Multi-Retinal Disease Detection — ODIR-5K

Multi-label retinal disease classifier: EfficientNet-B4 + CBAM + FPN + Swin-B,
trained on the ODIR-5K dataset (8 disease classes).

See the full design rationale (why this dataset, why this architecture, full
phase-by-phase plan) in the project reference doc you already have.

## Setup (Kaggle Notebook — recommended)

1. Create a new Kaggle Notebook, enable GPU (Settings -> Accelerator -> GPU T4 x2 or P100).
2. Add the ODIR-5K dataset via "Add Input" (search "ocular disease recognition odir5k").
3. Upload this whole project folder as a Kaggle Dataset, or just paste `src/` in as a utility script dataset — OR simplest: `git clone` this repo into `/kaggle/working/` if you push it to GitHub first.
4. Point `config/config.yaml`'s `data.raw_images_dir` and `data.raw_annotations_csv` at the attached dataset's path (usually under `/kaggle/input/...`).
5. Install anything missing: `!pip install -r requirements.txt --quiet`

## Setup (Local machine)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Place the downloaded ODIR-5K files at:
- `data/raw/ODIR-5K_Training_Images/`
- `data/raw/ODIR-5K_Training_Annotations.csv`

## Running the pipeline, in order

```bash
# 1. EDA first — open notebooks/01_eda.ipynb and inspect class distribution,
#    co-occurrence, image samples, confirm left/right pairing is intact.

# 2. Build processed splits + CLAHE cache
python scripts/prepare_data.py

# 3. Verify the sampler is actually balancing (don't skip this)
#    open notebooks/02_sampler_verification.ipynb

# 4. Sanity-check each model piece builds and has correct output shapes
python -m src.models.backbone
python -m src.models.cbam
python -m src.models.fpn
python -m src.models.bridge
python -m src.models.swin_encoder
python -m src.models.model      # full assembled model

# 5. Train
python scripts/run_training.py

# 6. Evaluate on the held-out test set
python scripts/run_evaluation.py

# 7. Explore results / GradCAM / attention maps
#    open notebooks/03_results_analysis.ipynb

# 8. Run the demo app
streamlit run app/streamlit_app.py
```

## Project structure

See `config/config.yaml` for every hyperparameter in one place. See each
module's docstring in `src/` for the reasoning behind each design decision —
they're written to be viva-defensible, not just functional.

## Important — before you trust results

- **Verify the CSV column names** in `scripts/prepare_data.py`
  (`expand_patients_to_individual_eyes`) against your actual downloaded
  annotation file — Kaggle mirrors of ODIR-5K sometimes use different header
  names than the ones assumed here (`Left-Fundus`, `Right-Fundus`).
- **Verify the sampler** (`notebooks/02_sampler_verification.ipynb`) before
  trusting training results — a silently-broken sampler is the most common
  failure point in this pipeline.
- Model selection uses **validation macro-F1**, not accuracy — this is
  intentional (see `src/training/metrics.py` docstring) and should not be
  changed without understanding why.

## Status

This is a fresh scaffold. Every module has a working implementation for its
core logic (CBAM, FPN, Asymmetric Loss, sampler, training loop, etc.), but
you should run the Phase 4 shape sanity-checks (`python -m src.models.*`)
before running a full training job, and adjust `config/config.yaml`
(especially `batch_size`) to fit whatever GPU you end up training on.
