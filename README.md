# Using ML on Gaze Data in a Fear Generalization setup

The system performs social anxiety recognition based on eye movements modeled as an Ornstein-Uhlenbeck process.

## Installation

Install required libraries with Anaconda:

```bash
conda create --name mlgaze -c conda-forge --file requirements.txt
conda activate mlgaze
```
Install [NSLR-HMM](https://gitlab.com/nslr/nslr-hmm)

```bash
python -m pip install git+https://gitlab.com/nslr/nslr
```

### Features extraction
Extract Ornstein-Uhlenbeck features from [Diagnostic Facial Features & Fear Generalization dataset](https://osf.io/4gz7f/) (`datasets/Reutter`) launching the module `extract_OU_params.py`, results will be saved in `features/Reutter_OU_posterior_VI`.


### Train and test
Module `kfold_social_anxiety.py` exploit different regressors for sias recognition on the features extracted as an Ornstein-Uhlenbeck process.