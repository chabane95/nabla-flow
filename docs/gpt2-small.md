# GPT-2 Small in C

The `scale` command runs the same decoder as the byte-level experiments, with 12 blocks, 12 heads per block and an embedding width of 768. The position table has 1,024 entries and the GPT-2 vocabulary has 50,257. The trainer starts with random weights and reads pretokenized text. Checkpoints store the weights needed for inference. Running `scale train` again starts a fresh training run.

## Counting the weights

Let $V$ be the vocabulary size, $T$ the number of positions, $d$ the embedding width and $L$ the number of blocks. With a shared input embedding and output projection, the parameter count is

$$
N = (V+T)d + L(12d^2 + 13d) + 2d.
$$

Within each block, the four attention projections contribute $4d^2$ weights and the two MLP matrices contribute $8d^2$. The bias terms and two LayerNorms account for the remaining $13d$. The final LayerNorm contributes another $2d$. The output head uses the token matrix already counted in $Vd$. With the dimensions above, `nablaflow scale info` reports **124,439,808**. [GPT-2 reference code](https://github.com/openai/gpt-2/blob/master/src/model.py) · [C decoder](../src/gpt.c)

## A CPU smoke test

~~~powershell
.\nablaflow.exe scale info
.\nablaflow.exe scale smoke
~~~

The test feeds the decoder the GPT-2 token IDs for `Hello, world!` and scores four next-token predictions. After one AdamW update on that sequence, the mean cross-entropy falls from **11.052 to 4.779 nats per token**. The model still has all 124M parameters. The C implementation sizes its activation buffers for the window actually used. With four tokens, the weights, gradients, two Adam moments and update buffer together take about 2.32 GiB, plus activations and process overhead. [CPU measurement](../results/scale-smoke.csv)

A separate file-based check trained on `Hello, world!` and validated on `Hello, friend!` with seed 42. One update took validation NLL from 11.22694 to 6.61927 and produced a 497,759,532-byte checkpoint. After loading it, greedy decoding continued the prompt IDs `15496 11` with `995 0 50256`. This exercises the training, serialization and inference paths on a few words.

For a full text run, the CUDA trainer uses the same architecture. It took 2,400 AdamW steps on Tiny Shakespeare with a 128-token window. A window can cross the end of a speech. The selected checkpoint scored a perplexity of 176.05 across 13,564 held-out targets. [Sample and learning curve](../README.md) · [GPU run and instructions](training-124m.md)

## Training on your own text

Put one UTF-8 document on each line and use separate texts for training and validation. The optional converter uses the GPT-2 encoding from `tiktoken` and appends the end-of-text ID (50256) to each document. Its first run downloads the vocabulary files. The C program reads a headerless stream of little-endian uint16 IDs. If your text is already in that format, you can pass it straight to the trainer. [GPT-2 encoding in tiktoken](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py)

~~~powershell
python -m pip install tiktoken
python tools/gpt2_tokens.py encode data/train.txt train.u16
python tools/gpt2_tokens.py encode data/val.txt val.u16
python tools/gpt2_tokens.py prompt "Hello, world!"
~~~

The last command prints `15496 11 995 0`. For a short CPU check, use four to sixteen tokens per window and a few updates. The example files contain several thousand tokens, and validating the entire file at this model size can take a while.

~~~powershell
.\nablaflow.exe scale train train.u16 val.u16 5 16 my-model.nfg 1
.\nablaflow.exe scale generate my-model.nfg 12 15496 11 | python tools/gpt2_tokens.py decode
~~~

After the two input files, the arguments are **steps, window length, new checkpoint path and batch size**. The window length can range from 1 to 1,024. Longer windows increase the work per example while keeping the same parameter count. Each iteration samples training windows, accumulates token gradients and takes one AdamW step. The CPU driver uses a learning rate of $3\times10^{-4}$, weight decay of 0.1 and a gradient norm limit of 1.

Sequential validation scores each target from the second token onward exactly once. It runs at the start, every 100 steps and at the end. The checkpoint holds the weights from the lowest recorded validation loss. `generate` picks the most likely token, prints its decimal ID and stops at EOT or at the specified limit. [Training and inference driver](../src/scale.c)
