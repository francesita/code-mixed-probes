# Code-Mixed Probes Show How Pre-Trained Models Generalise On Code-Switched Text 
This repository consists of the code and data used for the COLING-LREC 2024 paper titled 'Code-Mixed Probes Show How Pre-Trained Models Generalise On Code-Switched Text' by Frances Laureano De Leon, Harish Tayyar Madabushi, and Mark Lee.

## Abstract
Code-switching is a prevalent linguistic phenomenon in which multilingual individuals seamlessly alternate between languages. Despite its widespread use online and recent research trends in this area, research in code-switching presents unique challenges, primarily stemming from the scarcity of labelled data and available resources. In this study we investigate how pre-trained Language Models handle code-switched text in three dimensions: a) the ability of PLMs to detect code-switched text, b) variations in the structural information that PLMs utilise to capture code-switched text, and c) the consistency of semantic information representation in code-switched text. To conduct a systematic and controlled evaluation of the language models in question, we create a novel dataset of well-formed naturalistic code-switched text along with parallel translations into the source languages. Our findings reveal that pre-trained language models are effective in generalising to code-switched text, shedding light on abilities of these models to generalise representations to CS corpora.

## PLMs used
1. bert-base-multilingual-uncased
2. xlm-roberta-base
3. xlm-roberta-large

## Code
The repository is divided between Semantics, and Layer-wise, and Syntactic folders. Each folder corresponds to the type of experiments performed as presented in the paper. In the syntactic folder we do not include the code for training a structural probe, as we used the code base by Ethan Chi found at https://github.com/ethanachi/multilingual-probing-visualization. 
=======
The repository is divided between Semantics, Layer-wise, and Syntactic folders. Each folder corresponds to the type of experiments performed as presented in the paper. 


## Data

The data folder contains the data used for each of the experiments presented in the paper. The data folder contains subfolders: 'semantics' and 'syntactic'. The data in each subfolder corresponds to the title of each of the experiments mentioned in the paper. There is not a layer-wise folder included because the data used for this set of experiments is publicly available.

warning: some of the data may contain profanity as it was collected from social media sources.

## Contact

If you need help or clarification on the data or code, please do not hesitate to contact me: fxl846{at}cs.bham.ac.uk

## Citation

If you use the data or code in this repository, please cite:
```bibtex
@inproceedings{laureano-de-leon-etal-2024-code-mixed,
    title = "Code-Mixed Probes Show How Pre-Trained Models Generalise on Code-Switched Text",
    author = "Laureano De Leon, Frances Adriana  and
      Tayyar Madabushi, Harish  and
      Lee, Mark",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.307",
    pages = "3457--3468",
    abstract = "Code-switching is a prevalent linguistic phenomenon in which multilingual individuals seamlessly alternate between languages. Despite its widespread use online and recent research trends in this area, research in code-switching presents unique challenges, primarily stemming from the scarcity of labelled data and available resources. In this study we investigate how pre-trained Language Models handle code-switched text in three dimensions: a) the ability of PLMs to detect code-switched text, b) variations in the structural information that PLMs utilise to capture code-switched text, and c) the consistency of semantic information representation in code-switched text. To conduct a systematic and controlled evaluation of the language models in question, we create a novel dataset of well-formed naturalistic code-switched text along with parallel translations into the source languages. Our findings reveal that pre-trained language models are effective in generalising to code-switched text, shedding light on abilities of these models to generalise representations to CS corpora. We release all our code and data, including the novel corpus, at https://github.com/francesita/code-mixed-probes.",
}
```




