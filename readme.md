# AQuA - Combining Experts’ and Non-Experts’ Views To Assess Deliberation Quality in Online Discussions Using LLMs
This repository contains the code to calculate `AQuA` scores on a given dataset and to translate a dataset to German.

## Training
We trained our Adapters similar to [Falk & Lapesa (2023)](https://github.com/Blubberli/ArgQualityAdapters).

We trained the following adapters:

* `Relevance:` Does the comment have a relevance for the discussed topic?
* `Fact:` Is there at least one fact claiming statement in the comment?
* `Opinion:` Is there a subjective statement made in the comment?
* `Justification:` Is at least one statement justified in the comment?
* `Solution Proposals:` Does the comment contain a proposal how an issue could be solved?
* `Additional Knowledge:` Does the comment contain additional knowledge?
* `Question:` Does the comment include a true, i.e., non-rhetoric question?
* `Referencing Users:` Does the comment refer to at least one other user or to all users in the community?
* `Referencing Medium:` Does the comment refer to the medium, the editorial team or the moderation team?
* `Referencing Contents:` Does the comment refer to content, arguments or positions in other comments?
* `Referencing Personal:` Does the comment refer to the person or personal characteristics of other users?
* `Referencing Format:` Does the comment refer to the tone, language, spelling or other formal criteria other comments?
* `Polite form of Address:` Does the comment contain welcome or farewell phrases?
* `Respect:` Does the comment contain expressions of respect or thankfulness?
* `Screaming:` Does the comment contain clusters of punctuation or capitalization intended to imply screaming?
* `Vulgar:` Does the comment contain language that is inappropriate for civil discourse?
* `Insult:` Does the comment contain insults towards one or more people?
* `Sarcasm:` Does the comment contain biting mockery aimed at devaluing the reference object?
* `Discrimination:` Does the comment explicitly or implicitly contain unfair treatment of groups or individuals?
* `Storytelling:` Does the commenter include personal stories or personal experiences?

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

* Falk, N., & Lapesa, G. (2023). ***Bridging Argument Quality and Deliberative Quality Annotations with Adapters.*** In Findings of the Association for Computational Linguistics: EACL 2023 (pp. 2424-2443). [doi:10.18653/v1/2023.findings-eacl.187](https://aclanthology.org/2023.findings-eacl.187)

Please cite the following if you use the **datasets** contained in this repository: 
* **Europolis**:Gerber, M., Bächtiger, A., Shikano, S., Reber, S.,   Rohr, S. (2018). ***Deliberative Abilities and Influence in a Transnational Deliberative Poll (EuroPolis).*** British Journal of Political Science, 48(4), 1093–1118. [doi:10.1017/S0007123416000144](https://doi.org/10.1017/S0007123416000144)

* **SFU**:Kolhatkar, V., Wu, H., Cavasso, L. et al. (2020). ***The SFU Opinion and Comments Corpus: A Corpus for the Analysis of Online News Comments.*** Corpus Pragmatics 4, 155–190 [doi:10.1007/s41701-019-00065-w](https://doi.org/10.1007/s41701-019-00065-w)

## BibTeX Citation
If you use the `AQuA` score in a scientific publication, we would appreciate using the following citations:

```

```
