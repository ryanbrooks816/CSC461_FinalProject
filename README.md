# CSC461_FinalProject

W&B Project link: https://wandb.ai/ryanbrooks-uni/MLFinal/overview

## Project Structure
- data for the clang CSV
- output folder for images used in the report and converted notebooks to HTML that contains all the output from previous runs
- src for the notebook, data, and training files
- `convert_to_html.sh` script for running the Jupyter command to convert a notebook to HTML

## Code Contributions:

`/base/data.py `
- New data loader for LOOCV. 
- Added a check to make sure the embeddings exist when trying to access them in the stratified splits. Previously this check only existed in the data loader class, but the embeddings' file paths were required before that in the stratified split.
- Functions for filtering

`/base/train.py`
- No changes

`my-models.ipynb`
- ExperimentRunner class that handles loading data, training, and evaluation
- Can run any number of runs and logs results to W&B
- Implements LOOCV
- Implements sweeps with W&B
- Subsequent cells call methods from the ExperimentRunner class and provide different configurations for the model's parameters as well as filters for benchmark/application
- Range of cells dedicated to plotting and printing different information about the dataset, needed for the report.

## Experiments:
- Exp2: implementing the first version of LOOCV
- Exp3: 10 runs of random and 10 runs of majority stratification and implementing random hyperparameter tuning
- Exp4: Sweeps with W&B (both random and bayes depending on the sub-experiment letter) on the NPB SP application
- Exp5: running the entire dataset with no filters applied
- Exp6: running on the NPB SP application with either batch or layer normalization
