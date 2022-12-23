# sdo-cli <!-- omit in toc -->

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sdo-cli)](https://pypi.org/project/sdo-cli/)
[![PyPI Status](https://badge.fury.io/py/sdo-cli.svg)](https://badge.fury.io/py/sdo-cli)

A practitioner's utility for working with SDO data.

**The results of the Master's thesis accompanying this source code can be found in [`./docs/MG_Masters_Thesis_final.pdf`](./docs/MG_Masters_Thesis_final.pdf)**

- [Installation](#installation)
- [Usage](#usage)
  - [Data](#data)
  - [Events](#events)
  - [GOES](#goes)
  - [SOoD](#sood)
- [Examples](#examples)
    - [Download images from the Curated Image Parameter Dataset](#download-images-from-the-curated-image-parameter-dataset)
    - [Resize images](#resize-images)
    - [Patch images](#patch-images)
    - [Loading Events from HEK](#loading-events-from-hek)
    - [Downloading the GOES timeseries](#downloading-the-goes-timeseries)
    - [Get GOES value at a specific point in time](#get-goes-value-at-a-specific-point-in-time)
  - [SOoD Anomaly Detection](#sood-anomaly-detection)
    - [Training](#training)
    - [Prediction](#prediction)
    - [Image Generation](#image-generation)
- [Local Development](#local-development)
  - [Setup](#setup)
  - [Publishing](#publishing)
- [References](#references)

## Installation 

```
pip install -U sdo-cli
```

## Usage

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


### Data

Helpers for interacting with the Curated Image Parameter Dataset.

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

### Events

Helpers for downloading HEK events.

```
Usage: sdo-cli events [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  analyze  Analyzes model outputs and compares it with events from HEK
  get      Loads events from HEK
  list     Lists local events from HEK  
```

### GOES

Helpers for downloading and interacting with GOES data.

```
Usage: sdo-cli goes [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  download  Loads a the GOES X-Ray flux timeseries for a date range and stores
            it partitioned by year and month in a CSV
  get       Gets a the GOES flux at a point in time  
```

### SOoD 

Helpers for training and using solar out-of-distribution models. 

```
Usage: sdo-cli sood [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  ae
  ce_vae
  threshold       
```

Helper for training and interacting with the Context-Encoding Variational Autoencoder inspired by Zimmerer et al. [2].

```
Usage: sdo-cli sood ce_vae [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  generate     Generate a set of images with the CE-VAE model (requires a
               pretrained model)
  predict      Predicts anomaly scores using a CE-VAE model (requires a
               pretrained model)
  reconstruct  Reconstructs input images (requires a pretrained model)
  train        Trains a CE-VAE model    
```

## Examples

#### Download images from the Curated Image Parameter Dataset

```
sdo-cli data download --path='./data/aia_171_2012' --start='2012-03-07T00:02:00' --end='2012-03-07T00:40:00' --freq='6min' --wavelength='171'
```

#### Resize images

```
sdo-cli data resize --path='./data/aia_171_2012' --targetpath='./data/aia_171_2012_256' --wavelength='171' --size=256
```

#### Patch images

```
sdo-cli data patch --path='./data/aia_171_2012_256' --targetpath='./data/aia_171_2012_256_patches' --wavelength='171' --size=32
```

#### Loading Events from HEK

```
pip install psycopg2-binary
docker-compose up
sdo-cli events get --start="2012-01-01T00:00:00" --end="2012-01-02T23:59:59" --event-type="AR"
```

#### Downloading the GOES timeseries

```
# Please install either pyarrow or fastparquet
pip install pyarrow
sdo-cli goes download --start=2010-01-01T00:00:00 --end=2020-12-31T23:59:59 --output=./tmp/goes
```

#### Get GOES value at a specific point in time

Requires downloading the time series with `sdo-cli goes download` beforehand.

```
sdo-cli goes get --timestamp=2015-06-01T02:20:00 --cache-dir=./tmp/goes
```

### SOoD Anomaly Detection

The `sood` command implements a Solar Out-of-Distribution model based on the Context-encoding Variational Autoencoder (ceVAE) by Zimmerer et al. [2]. The model makes use of the model-internal latent representation deviations to end up with a more expressive reconstruction error and allows anomaly detection on both an image as well as a pixel-level.

A full Anomaly Detection pipeline can be examined in the example notebook `notebooks/ce-vae__e2e-pipeline.ipynb`. For this start jupyter:

```
make notebook
```

#### Training

For training a model, make sure to download an appropriate dataset (either the Curated Image Parameter Dataset, the SDO ML v1 dataset or the SDO ML v2 dataset). It is encouraged to use the [SDO ML v2 dataset](https://sdoml.github.io/#/) as the code was last tested with this data. To configure hyperparameters, create an appropriate config file by either copying an existing file in `./config/ce-vae/` or creating a new one.  Config options from a specific file will override the config options from a `defaults.yaml` file in the same directory. `nohup` is used for non-blocking long-running tasks. Check the `nohup.out` file for logs.

```
nohup  sdo-cli sood ce_vae train --config-file="./config/ce-vae/run-fhnw-full-2-256.yaml" & 
tail -f nohup.out
```

#### Prediction

Running the `predict` command requires a pretrained model. 

For sample-level scores, run:

```
nohup  sdo-cli sood ce_vae predict --config-file="./config/ce-vae/run-fhnw-full-2-256-predict.yaml" --precit-mode="sample" & 
```

For pixel-level scores, run:

```
sdo-cli sood ce_vae predict --config-file="./config/ce-vae/run-fhnw-full-2-256-predict.yaml" --precit-mode="pixel"
```

#### Image Generation

Running the `generate` command requires a pretrained model.

```
sdo-cli sood ce_vae generate --config-file="./config/ce-vae/run-fhnw-full-2-256-predict.yaml"
```

## Local Development

### Setup

Setup Virtual Environment and install `sdo-cli`.

```
git clone git@github.com:i4Ds/sdo-cli.git
cd sdo-cli
make setup
make install
```

### Publishing

Add your pypi credentials to `~/.pypirc`, increase the version number in `setup.py` and run:

```
make publish
```

## References

- [1] Ahmadzadeh, Azim, Dustin J. Kempton, and Rafal A. Angryk. "A Curated Image Parameter Data Set from the Solar Dynamics Observatory Mission." The Astrophysical Journal Supplement Series 243.1 (2019): 18.
- [2] Zimmerer, David, et al. "Context-encoding variational autoencoder for unsupervised anomaly detection." arXiv preprint arXiv:1812.05941 (2018). Also refer to the [Medical Out-of-Distribution Analysis Challenge Repository](https://github.com/MIC-DKFZ/mood).
