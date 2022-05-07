echo on
pytest tests  > pytest.out
mypy tests > mypytests.out
echo "unknown check complete"
pyright -p pyrighttestconfig.json > pandastests.out

