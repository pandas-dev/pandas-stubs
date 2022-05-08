echo on
pytest tests  > pytest.out
mypy tests > mypytests.out
pyright -p pyrighttestconfig.json > pandastests.out

