# AQuA - Combining Experts’ and Non-Experts’ Views To Assess Deliberation Quality in Online Discussions Using LLMs
This repository contains the code to calculate `AQuA` scores on a given dataset and to translate a dataset to German.

## Training
We trained our Adapters similar to [Falk & Lapesa (2023)](https://github.com/Blubberli/ArgQualityAdapters).

We trained the following adapters:

* `Relevance: Does the comment have a relevance for the discussed topic? 0.20908452` 
* `Fact: Is there at least one fact claiming statement in the comment? 0.18285757`
* `Opinion: Is there a subjective statement made in the comment? -0.11069402`
* `Justification: Is at least one statement justified in the comment? 0.29000763`
* `Solution Proposals: Does the comment contain a proposal how an issue could be solved? 0.39535126`
* `Additional Knowledge: Does the comment contain additional knowledge? 0.14655912`
* `Question:Does the comment include a true, i.e., non-rhetoric question?   -0.07331445`
* `Referencing Users: Does the comment refer to at least one other user or to all users in the community?  -0.03768367`
* `Referencing Medium: Does the comment refer to the medium, the editorial team or the moderation team?   0.07019062`
* `Referencing Contents: Does the comment refer to content, arguments or positions in other comments?   -0.02847408`
* `Referencing Personal: Does the comment refer to the person or personal characteristics of other users? 0.21126469`
* `Referencing Format: Does the comment refer to the tone, language, spelling or other formal criteria other comments?    -0.02674237`
* `Polite form of Address: Does the comment contain welcome or farewell phrases?   0.01482095`
* `Respect: Does the comment contain expressions of respect or thankfulness?   0.00732909`
* `Screaming: Does the comment contain clusters of punctuation or capitalization intended to imply screaming?  -0.01900971`
* `Vulgar: Does the comment contain language that is inappropriate for civil discourse? -0.04995486`
* `Insult: Does the comment contain insults towards one or more people?   -0.05884586`
* `Sarcasm: Does the comment contain biting mockery aimed at devaluing the reference object? -0.15170863`
* `Discrimination: Does the comment explicitly or implicitly contain unfair treatment of groups or individuals?  0.02934227`
* `Storytelling: Does the commenter include personal stories or personal experiences? 0.10628146`

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
Install the requirements using:
```
$ pip install -r requirements.txt
```

## References 
Work by Neele Falk and Gabriella Lapesa:

Falk, N., & Lapesa, G. (2023). **Bridging Argument Quality and Deliberative Quality Annotations with Adapters.** In Findings of the Association for Computational Linguistics: EACL 2023 (pp. 2424-2443). [doi:10.18653/v1/2023.findings-eacl.187](https://aclanthology.org/2023.findings-eacl.187)

The data contained in this repository: 

**Europolis**: Gerber, M., Bächtiger, A., Shikano, S., Reber, S.,   Rohr, S. (2018). **Deliberative Abilities and Influence in a Transnational Deliberative Poll (EuroPolis).** British Journal of Political Science, 48(4), 1093–1118. [doi:10.1017/S0007123416000144](https://doi.org/10.1017/S0007123416000144)

**SFU**: Kolhatkar, V., Wu, H., Cavasso, L. et al. (2020). **The SFU Opinion and Comments Corpus: A Corpus for the Analysis of Online News Comments.** Corpus Pragmatics 4, 155–190 [doi:10.1007/s41701-019-00065-w](https://doi.org/10.1007/s41701-019-00065-w)

## BibTeX Citation
If you use the `AQuA` score in a scientific publication, we would appreciate using the following citations:

```

```
