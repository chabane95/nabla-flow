# Training GPT-2 Small on Tiny Shakespeare

Starting from random weights, the **124,439,808-parameter** decoder took 2,400 AdamW steps on a single desktop GPU. On held-out text the selected checkpoint reached **5.1708 nats per GPT-2 token**. A unigram fitted to the same training text scored **6.6837**. The checkpoint used for the test had the lowest validation loss. [Training history](../results/gpt2-124m/history.csv) · [Test results](../results/gpt2-124m/test.csv) · [Unigram baseline](../results/gpt2-124m/unigram-test.csv)

| Reference run | Setting or measurement |
| --- | --- |
| Hardware | Windows x64, Ryzen 9 5900X, GeForce RTX 4070 SUPER, 12,282 MiB VRAM, NVIDIA driver 591.86 |
| Architecture | 12 decoder blocks, 12 heads, embedding width 768, MLP width 3,072, shared input and output weights |
| Tokens and positions | 50,257 GPT-2 BPE IDs, 1,024 learned positions |
| One update | 4 randomly sampled windows × 128 next-token targets |
| Training budget | 2,400 updates × 512 targets = 1,228,800 target tokens processed |
| Time to the last recorded validation | 121.118 s after model initialization |
| Selected checkpoint | Step 2,400 of 2,400, validation NLL 4.14468463, perplexity 63.0977199 |
| Local weights | `run-gpt2-124m/best.nfg`, 497,759,532 bytes |
| SHA-256 of the reference weights | `1d53a5435ed0d530c96bcd983c63505659fad7014395c7787bd570f3e36098a3` |

The clock includes earlier validation passes, checkpoint writes and sampling. The final checkpoint and sample were written after the last timestamp. Random windows can overlap, so the training budget counts some token positions more than once. Text in the training windows uses position embeddings 0 through 127. [C training driver](../tools/nfgpu.c)

## 1. Build the trainer and check the GPU

Run the PowerShell commands from the repository root in one session. The reference machine used GCC 14.2.0 and 64-bit Python 3.12.14 on Windows. The GPU needs compute capability 8.0 or higher and a driver that supports CUDA 12. The packages go under `build-gpu`. Set `$py` to the full path of your interpreter if `python.exe` resolves to a different one.

```powershell
$ErrorActionPreference = 'Stop'
$py = 'python.exe'
& $py --version
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
gcc --version

New-Item -ItemType Directory -Force -Path build-gpu | Out-Null
& $py -m pip install --only-binary=:all: --no-deps --target build-gpu/runtime nvidia-cublas-cu12==12.9.2.10 nvidia-cuda-nvrtc-cu12==12.9.86
if ($LASTEXITCODE) { throw 'CUDA package installation failed' }
& $py -m pip install --only-binary=:all: --target build-gpu/python tiktoken==0.14.0 numpy==2.5.5 matplotlib==3.11.2 Pillow==13.0.0
if ($LASTEXITCODE) { throw 'Python package installation failed' }

$env:PYTHONPATH = (Resolve-Path build-gpu/python).Path
$blas = (Resolve-Path build-gpu/runtime/nvidia/cublas/bin).Path
$rtc = (Resolve-Path build-gpu/runtime/nvidia/cuda_nvrtc/bin).Path
$env:PATH = "$blas;$rtc;$env:PATH"
$env:NFGPU_KERNELS = (Resolve-Path src/gpu_kernels.cu).Path
Get-Item (Join-Path $blas 'cublas64_12.dll'), (Join-Path $blas 'cublasLt64_12.dll'), (Join-Path $rtc 'nvrtc64_120_0.dll'), (Join-Path $rtc 'nvrtc-builtins64_129.dll')

gcc -O3 -std=c11 -Wall -Wextra -Wpedantic -Isrc tools/nfgpu.c src/gpu.c src/gpt.c src/checkpoint.c -lm -o build-gpu/nfgpu.exe
if ($LASTEXITCODE) { throw 'C build failed' }
$gpu = (Resolve-Path build-gpu/nfgpu.exe).Path
& $gpu check
if ($LASTEXITCODE) { throw 'CPU/GPU comparison failed' }
```

The C host loads the CUDA driver, cuBLAS and NVRTC. At startup NVRTC compiles the kernels. Large matrix products can use Tensor Cores with TF32. The model stores parameters, gradients, Adam moments and activations in FP32. Some backward-pass additions use atomics. The seed fixes initialization and the choice of training windows. Rounding, hardware and library versions can affect the exact weight checksum in another run. [CUDA host code](../src/gpu.c) · [CUDA kernels](../src/gpu_kernels.cu)

## 2. Fetch and tokenize the text

The source is Karpathy's 1,115,394-byte `input.txt` from `char-rnn` at commit `370cbcd448eb7daf32f21a6be560b70e0b33c4e3`. The preparation script checks the file hash and verifies that each tokenized speech decodes to its original bytes. It preserves all line breaks and keeps the 7,222 blank-line-separated speeches in source order. Each speech gets one end-of-text token (ID 50256). [Pinned source](https://github.com/karpathy/char-rnn/blob/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt) · [Preparation script](../tools/prepare_124m.py)

```powershell
$source = 'build-gpu/tinyshakespeare.txt'
$url = 'https://raw.githubusercontent.com/karpathy/char-rnn/370cbcd448eb7daf32f21a6be560b70e0b33c4e3/data/tinyshakespeare/input.txt'
curl.exe -fL $url -o $source
if ($LASTEXITCODE) { throw 'Corpus download failed' }
$expected = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
if ((Get-FileHash -Algorithm SHA256 $source).Hash -ne $expected) {
    throw 'Source text checksum mismatch'
}
$env:TIKTOKEN_CACHE_DIR = Join-Path (Resolve-Path build-gpu).Path 'bpe-cache'
& $py tools/prepare_124m.py --source $source --out build-gpu/data
if ($LASTEXITCODE) { throw 'Data preparation failed' }
```

| Split | Speech numbers, starting at 1 | BPE IDs including EOT | SHA-256 of little-endian uint16 file |
| --- | --- | ---: | --- |
| Training | 1–6,499 | 311,356 | `2aeea0da9f1bf26b6244ab22f41444ccf185c49dd9785c3c2cf650e71bb7bb02` |
| Validation | 6,500–6,860 | 13,107 | `57ee490f58fc073c4c07f703cc86c4cb877b731d3efb009e93c76c67edfa56c0` |
| Test | 6,861–7,222 | 13,565 | `877e82b0e209cbee7ae6745beb284677b8e6616c855179efa666291e2a40a0d0` |

No complete speech appears in two different splits. Repetitions within a split are retained. Each split forms one token stream, so a training window can extend past EOT into the following speech. [Split metadata](../results/gpt2-124m/data.json)

## 3. Train and select a checkpoint

Choose a fresh output directory. The C trainer creates it. The arguments after the directory name are the update count, window length, batch size, evaluation interval, training seed and prompt file.

```powershell
$run = 'run-gpt2-124m'
$data = 'build-gpu/data'
& $gpu train "$data/train.u16" "$data/val.u16" $run 2400 128 4 100 42 "$data/prompt.u16"
if ($LASTEXITCODE) { throw 'Training failed' }
$model = Join-Path $run 'best.nfg'
Get-FileHash -Algorithm SHA256 $model
```

AdamW uses $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ and weight decay of 0.1 on the matrices. The trainer averages the gradient over 512 targets, records its global norm, then clips the norm to 1.0. The learning rate warms up to 0.0003 over the first 100 steps and follows a cosine schedule down to 0.00003.

`batches.csv` records the loss on every batch **before** its update. At step zero and after every 100 updates, `history.csv` records the loss on 32 fixed training windows and all 13,106 validation targets. Validation scores each token from the second ID onward exactly once and resets the context at each window boundary. `best.nfg` stores the inference weights with the lowest validation NLL seen so far. [Batch log](../results/gpt2-124m/batches.csv) · [Validation history](../results/gpt2-124m/history.csv)

## 4. Evaluate the weights and try another prompt

```powershell
& $gpu eval $model "$data/val.u16" 4 128
& $gpu eval $model "$data/test.u16" 4 128
& $gpu unigram "$data/train.u16" "$data/test.u16"

$promptText = "JULIET:" + [char]10
$promptFile = Join-Path $run 'juliet-prompt.u16'
& $py tools/gpt2_tokens.py prompt $promptText --out $promptFile
if ($LASTEXITCODE) { throw 'Could not write the new prompt file' }
$newIds = & $gpu sample $model $promptFile 64 128
if ($LASTEXITCODE) { throw 'Sampling failed' }
$newIds | & $py tools/gpt2_tokens.py decode
```

Both models predict the same 13,564 test targets. NLL is the mean negative natural logarithm of the probability assigned to the correct token. Perplexity is its exponential. The unigram uses the training target counts with 0.1 added to each vocabulary entry.

| Evaluation | Target tokens | NLL | Perplexity |
| --- | ---: | ---: | ---: |
| GPT, validation after reloading | 13,106 | 4.14468463 | 63.0977199 |
| GPT, test | 13,564 | 5.17076582 | 176.049608 |
| Unigram, test | 13,564 | 6.68371474 | 799.282728 |

The sampler prints the new token IDs. It draws from the top 40 at temperature 0.8 with sampling seed 42 and stops early if it draws EOT. The first `prompt --out` command needs a new file name. For another sample from the same prompt, pass the existing file back to `sample`. [Reloaded validation result](../results/gpt2-124m/validation-reload.csv) · [Test baseline](../results/gpt2-124m/unigram-test.csv)

## 5. Draw the figures and decode the samples

```powershell
& $py tools/plot_124m.py --run $run --out $run
& $py tools/report_124m.py --run $run --data $data --steps 2400 --out "$run/samples-report.json"
```

The figure has 25 measurements from step zero to step 2,400. The report decodes every sample generated during training from the fixed prompt `ROMEO:` followed by a line feed. The last continuation reads “Now, that is the time of me to hear.” and ends with two line feeds and EOT. Its longest exact match within any training speech spans three consecutive BPE tokens. The comparison excludes the prompt and EOT. [Learning curves](../results/gpt2-124m/training.png) · [All samples and matching training sequences](../results/gpt2-124m/samples-report.json)

## Numerical checks

- **Gradients and updates.** `nfgpu check` compares CPU and GPU on two decoder widths, using two seven-position sequences and an additional shorter window. It checks logits, the accumulated gradient of every parameter, the norm before clipping and two AdamW updates. The relative L2 gradient errors in the reference run were $2.06\times10^{-7}$ at width 12 and $8.80\times10^{-4}$ at width 64 with TF32.
- **Trained weights on CPU and GPU.** Using the same 124M checkpoint, both implementations processed the first 17 validation IDs to make 16 predictions. Across 804,112 logits the largest absolute discrepancy was 0.002537 and the root-mean-square discrepancy was 0.000371. Both picked the same most likely token at all 16 positions. The mean NLL was 1.320187828 on CPU and 1.320191428 on GPU.
- **Checkpoint selection.** Reloading `best.nfg` and running it on the full validation set reproduced the minimum NLL in the training history to all recorded digits. [Evaluation after reloading](../results/gpt2-124m/validation-reload.csv)
