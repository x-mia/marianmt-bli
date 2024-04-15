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

### Related work
* [A. Conneau, G. Lample, L. Denoyer, MA. Ranzato, H. Jégou - *Word Translation Without Parallel Data*, 2017](https://arxiv.org/pdf/1710.04087.pdf)
* [J.Tiedemann, S. Thottingal - *OPUS-MT - Building open translation services for the World*, 2017](https://aclanthology.org/2020.eamt-1.61.pdf)
