# AQuA - Combining Experts’ and Non-Experts’ Views To Assess Deliberation Quality in Online Discussions Using LLMs
This repository contains the code to calculate `AQuA` scores on a given dataset and to translate a dataset to German.

## Training
We trained our Adapters 

## Inference
Calculate `AQuA` scores for a German dataset using the `inference_parallel_de.py` script:
```
$ python inference_parallel_de.py inference_data text_col batch_size output_path
```

## Translation
Translate an English dataset to German by running the `translate_to_german.py` script:
```
$ python translate_to_german.py dataset_path dataset_name translation_col output_path --sep (optional)
```

## Requirements

## References 
Work by Neele Falk and Gabriella Lapesa:

Falk, N., & Lapesa, G. (2023). **Bridging Argument Quality and Deliberative Quality Annotations with Adapters.** In Findings of the Association for Computational Linguistics: EACL 2023 (pp. 2424-2443). [doi:10.18653/v1/2023.findings-eacl.187](https://aclanthology.org/2023.findings-eacl.187)

The data contained in this repository: 

**Europolis**: Gerber, M., Bächtiger, A., Shikano, S., Reber, S., & Rohr, S. (2018). **Deliberative Abilities and Influence in a Transnational Deliberative Poll (EuroPolis).** British Journal of Political Science, 48(4), 1093–1118. [doi:10.1017/S0007123416000144](https://doi.org/10.1017/S0007123416000144)

**SFU**: Kolhatkar, V., Wu, H., Cavasso, L. et al. (2020). **The SFU Opinion and Comments Corpus: A Corpus for the Analysis of Online News Comments.** Corpus Pragmatics 4, 155–190 [doi:10.1007/s41701-019-00065-w](https://doi.org/10.1007/s41701-019-00065-w)

## BibTeX Citation
If you use the `AQuA` score in a scientific publication, we would appreciate using the following citations:

```

```
