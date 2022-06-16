# sdo-cli

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sdo-cli)](https://pypi.org/project/sdo-cli/)
[![PyPI Status](https://badge.fury.io/py/sdo-cli.svg)](https://badge.fury.io/py/sdo-cli)

A practitioner's utility for working with SDO data.

## Installation 

```
pip install -U sdo-cli
```

## Usage

### cli

A small helper toolkit for downloading and working with SDO data complementing [sunpy](https://sunpy.org/) by giving illustrative examples how to solve tasks. The data is loaded from the Image Parameter dataset which is the result of [1].

**TLDR;**

How to use `sdo-cli`:

```
Usage: sdo-cli [OPTIONS] COMMAND [ARGS]...

  CLI to manipulate and model SDO data.

Options:
  --home DIRECTORY  Changes the folder to operate on.
  -v, --verbose     Enables verbose mode.
  --help            Show this message and exit.

Commands:
  data
  events
  goes
  sood
```

**Examples:**

Download images from the Curated Image Parameter Dataset:

```
sdo-cli data download --path='./data/aia_171_2012' --start='2012-03-07T00:02:00' --end='2012-03-07T00:40:00' --freq='6min' --wavelength='171'
```

Resize images:

```
sdo-cli data resize --path='./data/aia_171_2012' --targetpath='./data/aia_171_2012_256' --wavelength='171' --size=256
```

Patch images:

```
sdo-cli data patch --path='./data/aia_171_2012_256' --targetpath='./data/aia_171_2012_256_patches' --wavelength='171' --size=32
```

Loading Events from HEK:

```
pip install psycopg2-binary
docker-compose up
sdo-cli events get --start="2012-01-01T00:00:00" --end="2012-01-02T23:59:59" --event-type="AR"
```

Downloading the GOES timeseries:

```
sdo-cli goes download --start=2010-01-01T00:00:00 --end=2020-12-31T23:59:59 --output=./tmp/goes
```

Get GOES value at a specific point in time (requires download beforehand):

```
sdo-cli goes get --timestamp=2015-06-01T02:20:00 --cache-dir=./tmp/goes
```

### SOoD Anomaly Detection

The `sood` command implements a Solar Out-of-Distribution model based on the Context-encoding Variational Autoencoder (ceVAE) by Zimmerer et al. [2]. The model makes use of the model-internal latent representation deviations to end up with a more expressive reconstruction error and allows anomaly detection on both a sample as well as a pixel level.

A full Anomaly Detection pipeline can be examined in the example notebook `notebooks/ce-vae__e2e-pipeline.ipynb`. For this start jupyter:

```
make notebook
```

#### Training

```
nohup sdo-cli sood ce_vae train --config-file="./config/ce-vae/run-fhnw-full-1.yaml" & 
```

## Local Development

### Setup

Setup Virtual Environment and install `sdo-cli`.

```
make setup
make install
```

### Publishing

Add your pypi credentials to `~/.pypirc`, increase the version number in `setup.py` and run:

```
make publish
```

### Troubleshooting

Tensorflow only works with Python versions < 3.9.

```
brew install pyenv
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
source ~/.bash_profile
pyenv install 3.8.0
```

Also refer to this [link](https://www.chrisjmendez.com/2017/08/03/installing-multiple-versions-of-python-on-your-mac-using-homebrew/).

## References

- [1] Ahmadzadeh, Azim, Dustin J. Kempton, and Rafal A. Angryk. "A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission." The Astrophysical Journal Supplement Series 243.1 (2019): 18.
- [2] Zimmerer, David, et al. "Context-encoding variational autoencoder for unsupervised anomaly detection." arXiv preprint arXiv:1812.05941 (2018).
