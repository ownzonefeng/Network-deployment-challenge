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
demo('G1', lambda_=1)
demo('G2', lambda_=1.5)
```
### Exploratory process (recommended)
```shell
jupyter notebook model.ipynb
```
## Modules explanation
### `model.ipynb`
The whole optimisation process is in the notebook with visualizations, but you can find a minimal demonstration function in the `MH.py`.

### `MH.py`
This file has an implementaion Metropolis-Hastings algorithm, objective function, state transition definition, a demonstration funtion.

### `beta_optimizer.py`
Our optimization method for finding best beta.

### `funcs.py`
Some functions for processing data and outputs.

### `dataset.py`
Two dataset generators.