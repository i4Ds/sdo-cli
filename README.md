# sdo-cli

A practitioner's utility for working with SDO data.

## Setup

Setup Virtual Environment and install `sdo-cli`.

```
make setup
make install
```

## Usage

A small helper toolkit for downloading and working with SDO data complementing [sunpy](https://sunpy.org/) by giving illustrative examples how to solve tasks. The data is loaded from the Image Parameter dataset which is the result of [1].

**TLDR;**

How to use `sdo-cli`:

```
Usage: sdo-cli data [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download  Loads a set of SDO images between start and end from the Georgia
            State University Data Lab API

  patch     Generates patches from a set of images
  resize    Generates a set of resized images
```

**Examples:**

Download images:

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

## SOoD Anomaly Detection

Under `src/sood` a Solar Out-of-Distribution model based on a context-encoding variational autoencoder by Zimmerer et al. [2] is implemented. The model makes use of the model-internal latent representation deviations to end up with a more expressive reconstruction error and allows anomaly detection on both a sample as well as a pixel level.

A full Anomaly Detection pipeline can be examined in the example notebook `notebooks/e2ePipeline.ipynb`. For this start jupyter:

```
make notebook
```

## Troubleshooting

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
