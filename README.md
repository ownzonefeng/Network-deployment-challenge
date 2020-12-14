# EPFL COM-516: Deploying a 5G Network in a country
__Team name:__ ThE raNDom WALkERS

__Members:__ Jonathan Doenz, Wentao Feng, Yuxuan Wang

## Getting Started

### Prerequisites
- Python >= 3.7
- SciPy
- NumPy
- Pandas
- matplotlib
- tqdm

### Installing
```shell
conda env create -f environment.yml
```
## Running the tests
### Simple demonstration
```python
from MH import demo_run
demo_run('G1', lambda_=1)
demo_run('G2', lambda_=1.5)
```
### Exploratory process (recommended)
```shell
jupyter notebook model.ipynb
```
## Modules explanation
### `model.ipynb`
The whole optimization process is in the notebook with visualizations, but users can find a minimal demonstration function in the `MH.py`.

### `MH.py`
This file has an implementation Metropolis-Hastings algorithm, objective function, state transition definition, a demonstration function.

### `beta_optimizer.py`
Our optimization method for finding the best beta.

### `funcs.py`
Some functions for processing data and outputs.

### `dataset.py`
Two dataset generators.