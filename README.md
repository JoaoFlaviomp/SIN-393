## Link para o vídeo de apresentação no YouTube:
[Assista no YouTube](https://www.youtube.com/watch?v=aeMVkYR4N9k)


## Descrição detalhada do projeto:
# Explorando Estratégias Avançadas de Aumento de Dados para Melhorar a Classificação de Distúrbios da Coluna Vertebral em Imagens de Raios-X

## Contexto
Os distúrbios da coluna vertebral afetam uma parcela significativa da população e são uma preocupação crescente para as autoridades de saúde devido às suas potenciais consequências debilitantes. Nesse cenário, técnicas de visão computacional oferecem um caminho promissor para diagnósticos rápidos e precisos. No entanto, o treinamento de modelos robustos e confiáveis geralmente exige grandes conjuntos de dados, que são escassos em repositórios públicos.

## Objetivo
Explorar estratégias avançadas de aumento de dados para superar a limitação de dados disponíveis e melhorar a performance de modelos de aprendizado profundo na classificação de imagens de raios-X da coluna vertebral. As imagens foram classificadas em três categorias:

1. **Saudável**
2. **Escoliose**
3. **Espondilolistese**

## Estratégias de Aumento de Dados
As seguintes técnicas de aumento de dados foram implementadas e combinadas para avaliação de seus impactos na precisão do modelo:

- **CutMix**
- **CutOut**
- **MixUp**
- Aumentos de dados padrão (ex.: flip horizontal, rotação, entre outros)

## Arquiteturas Testadas
Os modelos de aprendizado profundo utilizados foram:

- **ResNet-50**
- **Vision Transformer (ViT)**
- **Swin Transformer V2**

## Resultados
Os experimentos demonstraram que a combinação da arquitetura **Vision Transformer (ViT)** com a técnica de aumento de dados **CutMix** obteve a melhor performance, alcançando uma **precisão de 0.988**.

## Contribuições
Este projeto visa:
- **Inovar** com o uso de estratégias avançadas de aumento de dados para classificação de imagens médicas.  
- **Impactar** o diagnóstico automatizado de distúrbios da coluna vertebral com modelos robustos e precisos.  
- **Promover reprodutibilidade**, permitindo que outros pesquisadores usem e adaptem essas técnicas para outros desafios médicos.  

---
## Instruções de execução:

### Atualizar o gerenciador de pacotes conda

```
    $ conda update -n base -c defaults conda
```

### Se necessário, atualize a distribuição do Anaconda
```
    $ conda update anaconda
```
### Criando o ambiente conda
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
