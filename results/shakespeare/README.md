# A small GPT reads *Coriolanus*

A decoder with **43,056 parameters**, trained in C, scores a perplexity of **6.98** on 64 held-out passages from *Coriolanus*. A bigram that sees only the preceding token scores **10.70**. Both models predict each byte in a passage and the end marker that follows it. [Test results](test.csv)

| Model | Validation NLL | Test NLL | Test perplexity |
| --- | ---: | ---: | ---: |
| Bigram, add-0.1 smoothing | 2.4167 | 2.3699 | 10.6962 |
| GPT, step 1,700 | 1.9742 | 1.9432 | 6.9810 |

At each position the score depends on the probability assigned to the byte that actually comes next. Averaging the negative log of those probabilities gives the negative log-likelihood (NLL). On the test passages the difference between the models works out to **0.616 bits per target token**. [Training log](history.csv) · [Run settings](summary.csv)

### What it writes

Given `MENENIUS: `, the model writes `I gay with picioustire hot sirke`. There are ordinary words, including *with* and *hot*, among strings like *picioustire*. That's about as far as this run gets toward a speech by Menenius. The sample used temperature 0.8, top-k 12, seed 7 and a limit of 220 new bytes. [Full output](sample.txt) · [Inference weights](model.nfg)

### Text and model

The excerpt comes from Andrej Karpathy's *Tiny Shakespeare*. The original file is 1,115,394 bytes and has SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`. [Source file at the pinned commit](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt)

One or more blank lines separate passages. The C utility keeps them in order, joins lines within a passage with one ASCII space and writes a line feed after each passage. This preserves speaker labels, capitalization and punctuation. Each passage goes into a single split. No validation or test passage occurs in full in the training set. [C preparation code](../../tools/shakespeare.c)

| Split | Passage numbers in the source, inclusive | Target tokens | SHA-256 of normalized file |
| --- | --- | ---: | --- |
| Training | 1–800 | 105,394 | `176b02cbb6193d1a152c782bfe9c2ee9a2d6d9c04c42f9b788d849961052fde0` |
| Validation | 801–864 | 9,300 | `b455c181a6f5dec0a56167d7e8cc3ba3c189c9d88191838fa6ffae47c59c9b42` |
| Test | 865–928 | 6,819 | `b52508d91097b89bb9df01e77f2d96935555f4ed49bde7f332604307b1dea99f` |

The vocabulary has 256 byte values plus beginning-of-sequence and end-of-sequence tokens. Each passage gets both markers. For passages with more than 48 predictions, the trainer divides the shifted input–target pairs into consecutive blocks. At any position, attention can use that position and earlier ones in the same block. Evaluation counts each target once.

The model has one Transformer block with four heads and an embedding width of 48. Each AdamW update uses eight blocks. The learning rate warms up to 0.002 over 20 steps, then follows a cosine schedule down to 0.0002. The training curve uses a fixed sample of 64 blocks covering 2,667 targets. Validation covers all 9,300 targets. Among the checkpoints evaluated every 100 steps, including the initial weights, step 1,700 has the lowest validation NLL. Those weights go into the final test. The bigram gets its counts from the training passages.

### Repeat the run

Run the commands from the repository root in a shell with `make`, `gcc`, `curl` and `sha256sum`. The reference run used GCC 14.2.0 on a Windows CPU. The splitter and trainer each create a new directory, `build-shakespeare/splits` and `run-shakespeare` respectively. Only the first download of the source text needs a network connection.

```sh
make CC=gcc
mkdir -p build-shakespeare
gcc -O3 -std=c11 tools/shakespeare.c -o build-shakespeare/split
curl -fL https://raw.githubusercontent.com/karpathy/char-rnn/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt -o build-shakespeare/input.txt
printf '%s  %s\n' 86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed build-shakespeare/input.txt | sha256sum -c -
./build-shakespeare/split build-shakespeare/input.txt build-shakespeare/splits
./nablaflow train --train build-shakespeare/splits/train.txt --val build-shakespeare/splits/val.txt --out run-shakespeare --steps 1800 --ctx 48 --width 48 --heads 4 --layers 1 --batch 8 --lr 0.002 --eval-every 100 --seed 42
./nablaflow eval --model run-shakespeare/model.nfg --data build-shakespeare/splits/test.txt --train build-shakespeare/splits/train.txt
./nablaflow sample --model run-shakespeare/model.nfg --prompt 'MENENIUS: ' --tokens 220 --temperature 0.8 --top-k 12 --seed 7
```

The published weights are 172,524 bytes. To sample from that checkpoint, run the last command with `--model results/shakespeare/model.nfg` and your own prompt.
