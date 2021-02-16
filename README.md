# sdo-cli

A practitioners utility for working with SDO data.

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

[1] Ahmadzadeh, Azim and Kempton, J., Dustin, "A Curated Image Parameter Dataset from Solar Dynamics Observatory Mission", Manuscript under review, 2019.
