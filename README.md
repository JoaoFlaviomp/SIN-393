# BreakHis image classification


# Update the conda package manager

```
    $ conda update -n base -c defaults conda
```

## If necessary, update the Anaconda distribution
```
    $ conda update anaconda
```
## Creating the conda environment
```
    $ conda create -n env-scoliosis-py310 python=3.10
    $ conda activate env-scoliosis-py310

    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    $ pip install notebook scikit-learn matplotlib
    $ pip install ipywidgets    
    $ pip install pandas

    $ pip install <outros_modulos_necessarios>
```

### Saving environment

```
    $ conda env export > env-scoliosis-py310.yml
```

### Loading the environment from the provided .yml file

```
    $ conda env create -f env-scoliosis-py310.yml
```
