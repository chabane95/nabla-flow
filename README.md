# nabla-flow

Before the first update, my GPT-2 Small model followed `ROMEO:` with ` Arsenal stained slap`. After 2,400 updates on *Tiny Shakespeare*, the same prompt produced a one-line speech.

```text
ROMEO:
Now, that is the time of me to hear.
```

I supplied the speaker's name and the line break. The model wrote the rest, down to the blank line and the end-of-text token. Read it aloud and the opening sounds plausible. Then you get to *the time of me to hear*. I wanted to know whether it had lifted the line from a play. The longest exact match in any training speech was just three GPT-2 tokens, `, that is`. [All 25 samples and their matches](results/gpt2-124m/samples-report.json)

## Two minutes on a 4070 SUPER

This is a C and CUDA implementation of GPT-2 Small, trained from random weights. It has 12 transformer blocks, 12 attention heads per block and 124,439,808 parameters. The place I cut back for a 12 GB card was context length. The model has room for 1,024 positions, but each training window uses only the first 128.

Each update draws four windows from the training text. That works out to 1,228,800 next-token predictions over the run. There are only 311,356 tokens in the training split, so the model sees many of the same passages several times. The last validation measurement came 121 seconds into the run. [Build, train, or try another prompt](docs/training-124m.md)

[![Training and validation loss against optimizer steps, alongside validation perplexity against the number of training tokens processed.](results/gpt2-124m/training.png)](results/gpt2-124m/training.svg)

By step 400 most of the initial drop in loss is already over. At that point Romeo opens with `But I will I have thy good that I have not`. The model has picked up *thy* and the rhythm of a speech, but loses the thread at *will I have*. Every sample uses the same prompt, temperature, top-k and random seed. [Watch the run](results/gpt2-124m/training.gif) · [Measurements behind the figure](results/gpt2-124m/history.csv)

## How well does it predict a speech it hasn't seen?

I kept 361 whole speeches for validation and the last 362 for the test. The trainer measures validation loss every 100 updates. Step 2,400 had the lowest loss, so those are the weights used for the test.

| Model and split | Next tokens scored | Loss (nats/token) | Perplexity |
| --- | ---: | ---: | ---: |
| GPT-2, validation | 13,106 | 4.145 | 63.10 |
| GPT-2, test | 13,564 | 5.171 | 176.05 |
| Unigram, same test | 13,564 | 6.684 | 799.28 |

The unigram baseline counts tokens in the training text. It gives the same odds for the next token after `my` as it does after `ROMEO:`. On the same test passages, conditioning on the preceding text brings perplexity down from 799 to 176. Here perplexity is the exponential of the mean loss per GPT-2 token. [GPT-2 test result](results/gpt2-124m/test.csv) · [Unigram result](results/gpt2-124m/unigram-test.csv)

Why the jump from 63 on validation to 176 on the test? The split follows the order of the plays. In validation, 86 of 361 speeches have a speaker heading that never appears in training. In the test it's 290 of 362, including speeches by Prospero and Miranda. To find out how much the names themselves cost, I'd have to score the headings and the lines separately. [Corpus and split counts](results/gpt2-124m/data.json)

## Two things worth checking in the code

**Can an attention head see the answer?** When the input starts with `Now,`, changing everything to the right of the comma should leave the predictions up to that point alone. The CPU test makes exactly that edit and compares the earlier logits bit for bit. For a short input to the byte model, you can also look at the weights of one head, one row at a time. [C implementation and causality test](src/gpt.c) · [Attention animation](figures/attention.gif) · [Vaswani et al., section 3.2](https://arxiv.org/abs/1706.03762)

**Where does the token embedding get its gradient?** GPT-2 uses the same matrix to look up the input tokens and to score all 50,257 possible outputs. Both uses contribute to its gradient. On a small decoder, `nablaflow check` compares every parameter's gradient against a finite difference. On the GPU, cuBLAS handles the matrix products. The attention, normalization and AdamW kernels are in the repository. [GPU implementation](src/gpu.c) · [CUDA kernels](src/gpu_kernels.cu) · [Parameter count](docs/gpt2-small.md) · [Original GPT-2 report](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Starting with the smaller models

Both byte models run on a CPU and ship with trained weights. One learns from short German lab notes. The prompt `Fach A: Glas. Fach B: Harz. In B liegt ` puts glass in compartment A and resin in B, then asks what's in B. The model answers `Glas.` and gets the grammar right while picking the wrong compartment. That's why this experiment also scores whole answers. It gets 26 out of 56 right. [Build and try the byte models](docs/byte-models.md) · [The smaller Shakespeare run](results/shakespeare/README.md)

The workbook starts with one weight and a derivative you can check on paper. It follows that calculation through softmax regression to one row of attention. Andrew Ng's CS229 notes are the reference for the regression step. Andrej Karpathy's two lectures give the progression from a bigram model to a decoder and then to the 124M configuration. [Workbook](docs/workbook.docx?raw=1) · [Backpropagation derivation](latex/backprop_first_principles.tex) · [CS229 notes](https://cs229.stanford.edu/notes2022fall/main_notes.pdf) · [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=535s) · [Let's reproduce GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU)
