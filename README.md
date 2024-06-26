# Data-Driven Forecasting (WIP)

> Some notes and random codes that I use to try and understand how this whole data-drive weather forecasting works.
> This is almost exclusively using the [Earth2MIP](https://nvidia.github.io/earth2mip/index.html) framework.


***
## Installation

Currently, this needs to be installed by cloning the repo and installing the dependencies using conda.
To clone the repo:

```bash
gh repo clone jejjohnson/jbayesevt
```


**HTTPS**

```bash
git clone https://github.com/jejjohnson/jbayesevt.git
```

***
### Conda (Recommended)


```bash
conda env create -f environments/envronment_gpu.yaml
```

***
### APEX (Mandatory)

After the conda installation, we need to install apex which is a necessary requirement that cannot be installed via conda.

```bash
pip install git+https://github.com/NVIDIA/apex.git -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext"
```

***
### Notes: AI Models

There are quite a lot of models available but the dependencies are not very easy to deal with.
I have tested the following ones here:
* `pangu_6`
* `fcnv2_sm`

The ones that are **not working** that I have tested are:
* `pangu` - I ran out of GPU memory
* `graphcast` - the jax installed from graphcast does not seem to find the CUDA installation