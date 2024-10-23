# marianmt-bli

## Machine translation transformer model MarianMT for the bilingual lexicon induction task

This repository contains [training](https://github.com/x-mia/marianmt-bli/tree/main/Datasets/train) and [testing](https://github.com/x-mia/marianmt-bli/tree/main/Datasets/test) datasets for three language pairs: Estonian-English, Estonian-Finnish, and Estonian-Russian.

### Requirements
* [PyTorch](https://pytorch.org/)
* [HuggingFace](https://huggingface.co/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Tqdm](https://tqdm.github.io/)

### Evaluate aligned embeddings
To evaluate MarianMT on the BLI task, simply run:
```
python get_translations.py --src_lng SRC_LNG --tgt_lng TGT_LNG --eval_df EVAL_DF --output OUTPUT
```
Example:
```
python get_translations.py --src_lng et --tgt_lng fi --eval_df et-fi.csv --output result.csv
```

### References
* Please cite [[1]](https://elex.link/elex2021/wp-content/uploads/2021/08/eLex_2021_06_pp107-120.pdf) if you found the resources in this repository useful.

[1] Denisová, M. and Rychlý, P. (2024). [Bilingual Lexicon Induction From Comparable and Parallel Data: A Comparative Analysis.](https://doi.org/10.1007/978-3-031-70563-2_) In *International Conference on Text, Speech, and Dialogue*, pp. 30-42. Springer Nature Switzerland. 

```
@inproceedings{denisova-rychly-bli-2024,
author="Denisov{\'a}, Michaela
and Rychl{\'y}, Pavel",
title="Bilingual Lexicon Induction From Comparable and Parallel Data: {A} Comparative Analysis",
booktitle="International Conference on Text, Speech, and Dialogue",
year="2024",
publisher="Springer Nature Switzerland",
pages="30--42",
doi="10.1007/978-3-031-70563-2_3",
}
```

### Related work
* [A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou - *Word Translation Without Parallel Data*, 2017](https://arxiv.org/pdf/1710.04087.pdf)
* [J.Tiedemann, S. Thottingal - *OPUS-MT - Building open translation services for the World*, 2017](https://aclanthology.org/2020.eamt-1.61.pdf)
