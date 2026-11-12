# Flora-Lens


Flora-Lens turns a 104-class ResNet50 flower classifier into a **human-verified, geo-tagged ecological sighting system**.

## Features

- Photo upload and top-3 species suggestions
- Keras/TensorFlow H5 and PyTorch checkpoint adapters
- Configurable preprocessing and ordered class mapping
- Low-confidence warning and unknown/expert-review option
- Browser GPS capture and manual coordinates
- Observation time, observer, and habitat notes
- SQLite storage with uploaded image copies
- Interactive map, species/status/date filters, and summary charts
- Pending, verified, and rejected review workflow
- CSV export and responsive phone/laptop interface

## Current project model layout

For the Keras model shown in the project brief:

```text
model/botanical_model.h5
model/class_mapping.json
model/model_config.json
notebooks/flower_resnet50_training.ipynb
docs/training_curves.png
```

The H5 file is auto-detected. It is intentionally ignored by ordinary Git because it is large. See **Local setup and GitHub guide** below.

## Quick start: Keras/TensorFlow H5

Use a 64-bit Python version supported by TensorFlow; Python 3.11 is a practical choice. From the project root:

```bash
python -m venv .venv
```

Activate the environment, then install and run:

```bash
pip install --upgrade pip
pip install -r requirements-model.txt
python app.py
```

Open `http://127.0.0.1:8000`. `requirements-model.txt` includes the web dependencies and TensorFlow. Without the H5 file, the interface runs in explicitly labelled preview mode.

For detailed Windows/PowerShell commands, use **[LOCAL_SETUP_AND_GITHUB.md](LOCAL_SETUP_AND_GITHUB.md)**.

## Required inference settings

The default `model/model_config.json` is:

```json
{
  "preprocessing": "resnet50",
  "output": "auto"
}
```

Match `preprocessing` to the Kaggle notebook:

| Training preprocessing | Config value |
|---|---|
| `tf.keras.applications.resnet50.preprocess_input` | `resnet50` |
| `rescale=1./255` or `image / 255.0` | `rescale` |
| Preprocessing/Rescaling inside the saved model | usually `none` |
| ImageNet mean/std after scaling to `[0,1]` | `imagenet` |

Input dimensions are inferred from the H5 model. Optionally add `"input_size": [224, 224]` to override them. `output: auto` handles probability and logit outputs; it can be explicitly set to `probabilities` or `logits`.

The class mapping must contain all 104 labels in the correct output-index order. Both `class → index` and `index → class` JSON mappings are accepted.

## Verify the real model is connected

Open:

```text
http://127.0.0.1:8000/api/status
```

For a successful H5 connection, the response should include values equivalent to:

```json
{
  "mode": "model",
  "model_loaded": true,
  "framework": "keras",
  "class_count": 104,
  "preprocessing": "resnet50"
}
```

The header should also say **104-class model connected**. If it says `preview` or `error`, real inference is not active. Read the setup dialog/error before testing predictions.

Test several known field images and compare results with the Kaggle notebook. A model can load successfully but give poor predictions if class order or preprocessing is wrong.

## Repository and large model files

The default `.gitignore` excludes:

- Virtual environments and Python caches
- Runtime observation database and uploaded sighting images
- `.env` and local secrets
- Large model weights, including H5 files

Commit the application, mapping, config, notebook, and training curves. Distribute `botanical_model.h5` through a **GitHub Release** or use **Git LFS**. Do not accidentally add it as a normal Git file.

## Optional PyTorch checkpoints

The adapter also recognises common ResNet50 `.pth`/`.pt` checkpoints. Install `requirements-torch.txt` instead and place the checkpoint under `model/`. Supported forms include plain state dictionaries, common checkpoint wrappers, full serialized models, and common final-FC variants.

## Main files

- `app.py` — FastAPI, observations, review workflow, uploads, and CSV export
- `inference.py` — lazy Keras/TensorFlow and PyTorch inference adapters
- `static/` — browser interface and map dashboard
- `model/` — model artifacts, ordered labels, and preprocessing config
- `notebooks/` — Kaggle training notebook
- `docs/` — training curves and project figures
- `data/` — local runtime observations; not committed

## Scientific scope

- Only trained classes are recognised; local conservation deployment needs locally relevant data.
- One prominent flower per image works best. Multi-flower scenes need cropping or a detector.
- Scores support review; they are not guaranteed calibrated probabilities.
- Sighting counts represent submissions and survey effort, not population-density estimates.
- Human verification remains part of the workflow.
