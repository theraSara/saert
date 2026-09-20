# Environment setup

Create a dedicated environment **with Python and pip**. Creating only a named
Conda environment can leave it without an interpreter, allowing `python` and
`pip` to resolve to the base environment even when the prompt shows `(saert)`.

```sh
conda create -n saert python=3.11 pip
conda activate saert
python -c "import sys; print(sys.executable)"
python -m pip --version
python -m pip install -r requirements.txt
python -m pip check
```

If `saert` already exists but has no Python, repair that environment first:

```sh
conda install -n saert python=3.11 pip
conda activate saert
```

Before installing requirements, verify that **both** the interpreter path and
pip's location contain `envs/saert` (or `envs\saert` on Windows).
Use `python -m pip` to bind pip to the selected interpreter.
The current scripts do not require TensorFlow, Keras, Roboflow, torchvision, or
torchaudio; keep unrelated project dependencies in their own environments.

After installation, check imports before downloading models or running analyses:

```sh
python -c "import torch, pandas, numpy, scipy, sklearn, statsmodels, transformers, transformer_lens, sae_lens; print('Imports OK'); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -m ipykernel install --user --name saert --display-name "Python (saert)"
```

Select **Python (saert)** in notebooks. A successful `pip check` verifies declared
dependency compatibility; the import check also exercises binary/library loading.
Neither replaces model extraction and reconstruction validation. Requirements
currently use broad lower bounds and are not a tested reproducibility lockfile.
After successful validation, save an environment snapshot for the run; choose the
PyTorch build for the actual machine before GPU experiments.
