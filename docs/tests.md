## Test

- Run all local tests against source code. </br> `poe test` </br> </br> 
  - Profiles: </br>
    - Default: Runs only mypy and pyright tests </br> `poe test --profile=default` </br>
    - Pytest: Runs only pytest </br> `poe test --profile=pytest` </br> 
    - Pre-commit: Runs all style checks in pre-commit </br> `poe test --profile=style` </br> 

- Run local tests against an installed distribution: </br> `poe test_dist` </br> </br>

- Run all local tests (against both source and installed distribution) : </br> `poe test_all` </br> </br>

- Forgot some command? </br>`poe --help` </br> </br>

- These tests originally came from https://github.com/VirtusLab/pandas-stubs.
