#!/bin/bash

python -m pip install --upgrade pyright
python -m pip install -r requirements.txt
cp -R ../partial/ ../typings
pyright --lib --pythonversion 3.9 -p ../pyrighttestconfig.json .



