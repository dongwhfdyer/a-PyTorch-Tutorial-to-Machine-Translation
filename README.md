# kuhn notes

For transformer's encoder and decoder, there are some difference between their linear layer.

```
AnalogLinear_ = functools.partial(AnalogLinear, bias=True, rpu_config=rpu_config) # for encoder
AnalogLinear_2 = functools.partial(AnalogLinear, bias=True, rpu_config=rpu_config) # for decoder

```

---

I'm using `PyTorch 1.4` in `Python 3.6`.

---

https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation#tutorial-in-progress)

# Objective

**To build a model that can translate from one language to another.**


>Um ein Modell zu erstellen, das von einer Sprache in eine andere übersetzen kann.


We will be implementing the pioneering research paper [_'Attention Is All You Need'_](https://arxiv.org/abs/1706.03762), which introduced the a_Transformer network to the world. A watershed moment for cutting-edge Natural Language Processing.



The trained model checkpoint is available [here](https://drive.google.com/drive/folders/18ltkGJ2P_cV-0AyMrbojN0Ig4JgYp9al?usp=sharing). You can use it directly with [`translate.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/blob/master/translate.py).

Here's how this model fares against the test set, as calculated in [`eval.py`](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation/blob/master/eval.py):

|BLEU|Tokenization|Cased|sacreBLEU signature|
|:---:|:---:|:---:|:---:|
|**25.1**|13a|Yes|`BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3`|
|**25.6**|13a|No|`BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.13a+version.1.4.3`|
|**25.9**|International|Yes|`BLEU+case.mixed+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.4.3`|
|**26.3**|International|No|`BLEU+case.lc+lang.en-de+numrefs.1+smooth.exp+test.wmt14/full+tok.intl+version.1.4.3`|

The first value (13a tokenization, cased) is how the BLEU score is officially calculated by [WMT](https://www.statmt.org/wmt14/translation-task.html) (`mteval-v13a.pl`).

The BLEU score reported in the paper is **27.3**. This is possibly not calculated in the same manner, however. See [these](https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-377580270) [comments](https://github.com/tensorflow/tensor2tensor/issues/317#issuecomment-380970191) on the official repository. With the method stated there (i.e. using `get_ende_bleu.sh` and a tweaked reference), the trained model scores **26.49**.
