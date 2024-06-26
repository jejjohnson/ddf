# Data-Driven Forecasting (WIP)

> Some notes and random codes that I use to try and understand how this whole data-drive weather forecasting works.

## Installation

First, we need to **clone the repo**

```bash
gh repo clone jejjohnson/jbayesevt
```


**HTTPS**

```bash
git clone https://github.com/jejjohnson/jbayesevt.git
```

### Conda (Recommended)

```bash
conda env create -f environments/envronment_gpu.yaml
```

### APEX (Mandatory)

Now, we need to install apex which is a necessary requirement that cannot be installed via conda.

```bash
pip install git+https://github.com/NVIDIA/apex.git -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext"
```