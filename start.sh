#!/bin/bash
set -e

source bin/activate 
trap "deactivate; exit" INT
jupyter notebook
