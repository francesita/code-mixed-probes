# Code-Mixed Probes Show How Pre-Trained Models Generalise On Code-Switched Text 
This repository consists of the code and data used for the COLING-LREC 2024 paper titled 'Code-Mixed Probes Show How Pre-Trained Models Generalise On Code-Switched Text' by Frances Laureano De Leon, Harish Tayyar Madabushi, and Mark Lee.

## Abstract
Code-switching is a prevalent linguistic phenomenon in which multilingual individuals seamlessly alternate between languages. Despite its widespread use online and recent research trends in this area, research in code-switching presents unique challenges, primarily stemming from the scarcity of labelled data and available resources. In this study we investigate how pre-trained Language Models handle code-switched text in three dimensions: a) the ability of PLMs to detect code-switched text, b) variations in the structural information that PLMs utilise to capture code-switched text, and c) the consistency of semantic information representation in code-switched text. To conduct a systematic and controlled evaluation of the language models in question, we create a novel dataset of well-formed naturalistic code-switched text along with parallel translations into the source languages. Our findings reveal that pre-trained language models are effective in generalising to code-switched text, shedding light on abilities of these models to generalise representations to CS corpora.

## PLMs used
1. bert-base-multilingual-uncased
2. bert-large-multilingual-uncased
3. xlm-roberta-base
4. xlm-roberta-large

## Code
The reposotory is divided between Semantics, Layer-wise, and Syntactic folders. Each folder corresponds to the type of experiments performed as presented in the paper. 

## Data

The data folder contains the data used for each of the experiments presented in the paper. The data folder contains subfolders: 'semantics' , 'layer-wise', and 'syntactic'. The data in each subfolder corresponds to the title of each of the experiments mentioned in the paper. 

## Contact

If you need help or clarification on the data or code, please do not hesitate to contact me: fxl846{at}cs.bham.ac.uk

## Citation

If you use the data or code in this repository, please cite:
```bibtex
@InProceedings{}
```




