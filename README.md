# Deep FRAME Texture Synthesis

This repository contains the codes on **hierarchical FRAME models** (Deep FRAME) for **texture synthesis**.

Given a single target image that represents a texture (e.g. bark, coffee beans, water), we learn a probabilistic model of the image statistics using a CNN-based descriptor and synthesize new images by **Langevin dynamics** sampling.

For more instructions, please turn to `proj4.pdf` in detail.

---

## 1. Project structure

```text
.
├── deep_frame.py            # Main script: learn & sample a deep FRAME model
├── run_all_experiments.py   # Helper launcher: run all images × layer configs in parallel
├── images/
│   ├── bark.jpg
│   ├── beehive.jpg
│   ├── coffee.jpg
│   ├── rose.jpg
│   ├── stucco.jpg
│   └── water.jpg
├── README.md
└── .gitignore
````

After running experiments, additional folders will be created automatically, e.g.

```text
bark_1layer/
bark_2layer/
bark_3layer/
...
rose_3layer/
```

Each experiment folder contains:

* `target.png` – resized training image used by the model;
* `XXXX.png` – synthesized images at intermediate training epochs;
* `conv1.png` – visualization of the first-layer filters after training;
* `output.log` – training log (values of ( f_{\text{tgt}}, f_{\text{syn}}, f_{\text{diff}} ) over epochs).

---

## 2. Environment & dependencies

Python 3.8+ is recommended.

Required packages:

* `torch`, `torchvision`
* `numpy`
* `matplotlib`
* `scikit-image`
* `Pillow`

You can install them with:

```bash
pip install torch torchvision
pip install numpy matplotlib scikit-image pillow
```

Or, if you use conda:

```bash
conda create -n deep-frame python=3.9
conda activate deep-frame

pip install torch torchvision
pip install numpy matplotlib scikit-image pillow
```

GPU is recommended but the code falls back to CPU automatically if CUDA is not available.

---

## 3. Deep FRAME model

### 3.1 Descriptor network

The **descriptor** is a small CNN that plays the role of hierarchical filters ( F_k^{(l)} ).
It is defined in `deep_frame.py` as the `Descriptor` class:

* `layer = 1`: only `conv1` (3 → 128 channels, 15×15 kernel + ReLU)
* `layer = 2`: `conv1` + `conv2` (128 → 64 channels, 5×5 kernel + ReLU)
* `layer = 3`: `conv1` + `conv2` + `conv3` (64 → 32 channels, 3×3 kernel + ReLU)

The last feature map is summed over all channels and spatial locations to give the **scoring function**
( f(I; w) ). That is,

$$
f(I; w) = \sum_{k,x} [F_k^{(L)} * I](x)
$$

### 3.2 FRAME distribution

The FRAME model defines a probability distribution over images:

$$
p(I; w) \propto \exp\big(f(I; w)\big) q(I)
$$

where ( q(I) ) is a Gaussian white-noise reference model.

### 3.3 Learning

Given a single target image ( I_{\text{tgt}} ), the objective is maximum likelihood.
The gradient of the log-likelihood (Eq. (4) in the handout) is approximated by

$$
\frac{\partial L}{\partial w}
\approx
\frac{\partial f(I_{\text{tgt}}; w)}{\partial w}

-

\frac{\partial f(I_{\text{syn}}; w)}{\partial w}
$$

where ( I_{\text{syn}} ) is a synthesized image sampled from the current model via Langevin dynamics.

In code, this becomes:

```python
feat_tgt = model(img_tgt)
f_tgt = feat_tgt.sum()

feat_syn = model(img_syn.detach())
f_syn = feat_syn.sum()

f_diff = f_tgt - f_syn       # objective to maximize
f_diff.backward()            # autograd gives gradients w.r.t. model parameters
```

### 3.4 Langevin dynamics for sampling

Samples from ( p(I; w) ) are drawn by **Langevin dynamics** (Eq. (5)):

$$
I_{t+1} = I_t +
\frac{\varepsilon^2}{2}
\left(
\frac{\partial f(I_t; w)}{\partial I}
- \frac{I_t}{\sigma^2}
\right)
+ \varepsilon Z_t
\quad
Z_t \sim \mathcal{N}(0, I)
$$

In `deep_frame.py` this is implemented in the `langevin()` function:

```python
f = descriptor(x).sum()
f.backward()
grad_x = x.grad

x.data += (eps * eps / 2.0) * (grad_x - x / (sigma * sigma)) + eps * noise
```

---

## 4. Running single experiments

Make sure you are in the project root (where `deep_frame.py` is located) and that the `images/` folder exists.

### 4.1 Single-layer model

```bash
python deep_frame.py --layer 1 --tag rose
```

This trains a 1-layer FRAME model on `images/rose.jpg` and creates a folder:

```text
rose_1layer/
```

### 4.2 Two-layer model

```bash
python deep_frame.py --layer 2 --tag rose
```

### 4.3 Three-layer model

```bash
python deep_frame.py --layer 3 --tag rose
```

The `--tag` argument must match the image file name **without** the extension.
For example:

* `--tag bark`    → `images/bark.jpg`
* `--tag coffee`  → `images/coffee.jpg`
* `--tag water`   → `images/water.jpg`

---

## 5. Running all experiments in parallel

To run **all 6 textures × 3 layers** in parallel on multi-GPU:

```bash
python run_all_experiments.py
```

The script:

1. Detects available GPUs (or reads `CUDA_VISIBLE_DEVICES`).

2. Creates the full experiment list:

   ```text
   TAGS   = [bark, beehive, coffee, rose, stucco, water]
   LAYERS = [1, 2, 3]
   ```

3. Launches multiple `deep_frame.py` processes in parallel, assigning one GPU per process.

4. Creates result folders such as `bark_1layer`, `bark_2layer`, ..., `water_3layer`.

To run it in the background on a server, use:

```bash
nohup python -u run_all_experiments.py > run_all.log 2>&1 &
```

Monitor the progress with:

```bash
tail -f run_all.log
```

---

## 6. Hyper-parameters

Important hyper-parameters are defined at the top of `deep_frame.py`:

```python
num_epochs          = 2000    # total training iterations
sigma               = 1.0     # std of Gaussian reference model
langevin_num_steps  = 10      # Langevin steps per epoch
langevin_step_size  = 1.0     # Langevin step size ε
lrs = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]  # learning rates for (weight, bias) of each conv layer
```

You are encouraged to *modify*:

* model structure for 1- and 2-layer cases (channels, kernel sizes, etc.);
* `langevin_num_steps` and `langevin_step_size`;
* `lrs` (per-layer learning rates).

These modifications can be used in the lab report to analyze the effect of model depth and hyper-parameters on the synthesized textures.

---

## 7. Reproducing figures for the report

Typical steps for the lab report:

1. For each texture (e.g. `rose`), run models with 1, 2, and 3 layers.
2. Collect synthesized images at the final epoch from:

   * `rose_1layer/`,
   * `rose_2layer/`,
   * `rose_3layer/`.
3. Insert them side-by-side with the target image to compare:

   * how texture statistics are captured at different depths;
   * how global structure vs. fine details are modeled.
4. Optionally, plot ( f_{\text{tgt}} ), ( f_{\text{syn}} ), and ( f_{\text{diff}} ) from `output.log` to illustrate training dynamics.

---

## 8. License

MIT License