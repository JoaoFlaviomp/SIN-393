# Atualizar o gerenciador de pacotes conda

```
    $ conda update -n base -c defaults conda
```

## Se necessário, atualize a distribuição do Anaconda
```
    $ conda update anaconda
```
## Criando o ambiente conda
```
    $ conda create -n env-scoliosis-py310 python=3.10
    $ conda activate env-scoliosis-py310

    $ conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    $ pip install notebook scikit-learn matplotlib
    $ pip install ipywidgets    
    $ pip install pandas

    $ pip install <outros_modulos_necessarios>
```

### Salvando o ambiente

```
    $ conda env export > env-scoliosis-py310.yml
```

### Carregando o ambiente do arquivo .yml fornecido

```
    $ conda env create -f env-scoliosis-py310.yml
```
