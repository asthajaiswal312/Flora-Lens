# Local H5 integration and GitHub guide

This checklist is tailored to the files shown for this project:

- `botanical_model.h5` — large Keras/TensorFlow model
- `class_mapping.json` — 104-class output mapping
- `training_curves.png` — training figure
- A Kaggle notebook containing the training code

Use the recommended **GitHub Release route** unless you specifically need Git LFS.

---

## 1. Arrange the local project

Extract/copy BloomWatch into a normal local folder, for example:

```text
Documents\IPCV\BloomWatch
```

From Kaggle, download the notebook as `.ipynb`. Then arrange the files like this:

```text
BloomWatch\
├── app.py
├── inference.py
├── requirements.txt
├── requirements-model.txt
├── model\
│   ├── botanical_model.h5
│   ├── class_mapping.json
│   └── model_config.json
├── notebooks\
│   └── flower_resnet50_training.ipynb
├── docs\
│   └── training_curves.png
└── static\
```

The notebook and training curves names are suggestions; the H5 and mapping paths above are auto-detected names.

## 2. Check the Kaggle preprocessing before running inference

Search the notebook for `preprocess_input`, `rescale`, `Rescaling`, `ImageDataGenerator`, and the inference code used for a single uploaded image.

Edit `model/model_config.json` accordingly:

### If the notebook used ResNet50 `preprocess_input`

```json
{
  "preprocessing": "resnet50",
  "output": "auto"
}
```

### If it used `rescale=1./255` or divided images by 255

```json
{
  "preprocessing": "rescale",
  "output": "auto"
}
```

### If preprocessing is inside the saved model

```json
{
  "preprocessing": "none",
  "output": "auto"
}
```

Do not apply the same preprocessing twice. For example, if the saved model already contains the preprocessing layers, do not also apply `resnet50` externally unless the training/inference pipeline requires it.

## 3. Validate the class mapping

Open PowerShell in the project root. After creating the environment in the next step, run:

```powershell
python -c "import json; from inference import FlowerPredictor; raw=json.load(open('model/class_mapping.json', encoding='utf-8-sig')); labels=FlowerPredictor._normalise_labels(raw); print('Class count:',len(labels)); print('First five:',labels[:5]); print('Last five:',labels[-5:])"
```

The count must be **104**. Check the order against the notebook. If this prints `1`, the JSON may have an unsupported outer wrapper; inspect its structure rather than ignoring the mismatch.

## 4. Create the Windows virtual environment and install

Python 3.11 64-bit is a practical TensorFlow choice. In PowerShell:

```powershell
cd "$HOME\Documents\IPCV\BloomWatch"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-model.txt
```

If PowerShell blocks activation, you can run without activating:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
```

Use `.\.venv\Scripts\python.exe` instead of `python` for the later commands in that case.

## 5. Start the app and verify the H5 model actually loaded

```powershell
python app.py
```

Open `http://127.0.0.1:8000/api/status`.

Required success indicators:

- `"mode": "model"`
- `"model_loaded": true`
- `"framework": "keras"`
- `"class_count": 104`

Also check that `preprocessing` matches the notebook. Then open `http://127.0.0.1:8000`:

1. Upload a known flower image.
2. Check the top-3 results.
3. Enter test coordinates or allow browser GPS.
4. Save the sighting.
5. Verify it from the dashboard and test CSV export.
6. Clear the labelled sample records before a real demonstration.

If status is `error`, read the error in the yellow/red setup banner. Missing TensorFlow, incompatible H5 serialization, custom objects, wrong output count, and incorrect mapping need different fixes. If status is `preview`, the app did not find the H5 file.

## 6. Prepare the repository safely

Stop the server with `Ctrl+C`. In the Kaggle notebook:

- Remove Kaggle API keys, tokens, usernames that should remain private, and other secrets.
- Consider clearing very large cell outputs.
- Keep the dataset source, model architecture, augmentation, preprocessing, training configuration, evaluation, and export code.
- Do not commit the full dataset unless its licence and size permit it; link to the dataset instead.

The supplied `.gitignore` already excludes `.venv`, runtime observations, `.env`, and large model weights.

## 7A. Recommended: source in Git, H5 in a GitHub Release

### Initialise and inspect

From the project root:

```powershell
git init
git branch -M main
git status --short
git check-ignore model/botanical_model.h5
```

The last command should print the ignored H5 path. This means the large model will not be staged by `git add .`.

```powershell
git add .
git diff --cached --name-only
```

Inspect the staged list. It should include source code, `class_mapping.json`, `model_config.json`, the notebook, and `docs/training_curves.png`. It should **not** include `botanical_model.h5`, `.venv`, the observation database, or uploaded sighting images.

### Commit and create an empty GitHub repository

```powershell
git commit -m "Integrate flower ResNet50 with BloomWatch ecological monitoring"
```

On GitHub, select **New repository**, for example `bloomwatch`. Create it **empty**—do not add a README, licence, or `.gitignore` through the website for this first push.

Then, replacing `YOUR_USERNAME` and repository name if different:

```powershell
git remote add origin https://github.com/YOUR_USERNAME/bloomwatch.git
git push -u origin main
```

Authenticate using GitHub Credential Manager, GitHub CLI, or a personal access token. Your GitHub account password is not used as a Git password.

### Attach the large H5 to a Release

In the GitHub repository:

1. Open **Releases**.
2. Select **Draft a new release**.
3. Create a tag such as `v1.0.0` and give it a title.
4. Attach `botanical_model.h5` as a release asset.
5. Explain that users must place it at `model/botanical_model.h5`.
6. Publish the release.

Add that release link to the project README, then commit and push the documentation update:

```powershell
git add README.md
git commit -m "Add trained model download instructions"
git push
```

This keeps all project components accessible without putting the large binary into ordinary Git history.

## 7B. Alternative: version the H5 using Git LFS

Use this only if you understand your GitHub LFS storage/bandwidth allowances. Install Git LFS first. **Before staging the model**, run:

```powershell
git lfs install
Add-Content .gitignore '!model/botanical_model.h5'
git lfs track "model/botanical_model.h5"
git add .gitignore .gitattributes
git add model/botanical_model.h5
git lfs ls-files
```

`git lfs ls-files` must list the H5 file. Then stage the remaining project, inspect the staged list, commit, and push as above. Do this before making a normal commit containing the H5; fixing large files already stored in Git history is more complicated.

## 8. Final GitHub checklist

- [ ] `app.py`, `inference.py`, `static/`, and requirements are present.
- [ ] `class_mapping.json` contains 104 correctly ordered labels.
- [ ] `model_config.json` matches Kaggle preprocessing.
- [ ] Training notebook has no secrets or unnecessarily huge outputs.
- [ ] `training_curves.png` is under `docs/` and visible in the repository.
- [ ] README explains setup, model download, dataset source, and limitations.
- [ ] H5 is available through a Release or correctly tracked by LFS.
- [ ] `.venv`, `.env`, runtime observations, and uploaded sighting images are not committed.
- [ ] A fresh clone plus the separate H5 download can reproduce the running application.

## Useful later commands

```powershell
git status
git add .
git commit -m "Describe your change"
git push
```

Run the app again with:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```
