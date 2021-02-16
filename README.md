# sdo-cli

A practitioners utility for working with SDO data.

## Setup

Setup Virtual Environment and install `sdo-cli`.

```
make setup
make install
```

## Usage

A small helper toolkit for downloading and working with SDO data complementing [sunpy](https://sunpy.org/) by giving illustrative examples how to solve tasks.

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

Examples

```
sdo-cli data download --path='./data/aia_171_2012' --start='2012-03-07T00:02:00' --end='2012-03-07T00:40:00' --freq='6min' --wavelength='171'
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
