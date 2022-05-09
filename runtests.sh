#!/bin/sh
pytest tests  > pytest.out
mypy tests typings/pandas > mypytests.out
pyright -p pyrighttestconfig.json > pandastests.out

