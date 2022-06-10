## Test

- Run local tests against source code. </br> `poe test_src` </br> </br> 
  - Profiles: </br>
    - Default: Runs only mypy and pyright tests </br> `poe test_src --profile=default` </br>
    - Pytest: Runs only pytest </br> `poe test_src --profile=pytest` </br> 
    - Full: Run all tests (mypy, pyright and pytests) </br> `poe test_src --profiel=full` </br> </br>

- Run local tests against distribution: </br> `poe test_dist` </br> </br>

- Run all local tests: </br> `poe test_all` </br> </br>

- Forgot some command? </br>`poe --help` </br> </br>

- These tests originally came from https://github.com/VirtusLab/pandas-stubs.