# TOMM_publish

Run all commands below from `TOMM_publish/`.

## Environment

```bash
conda env create -f environment.yml
conda activate prune
```

## Data

The original datasets are not included. Download and extract them manually, then place them as:

```text
BAYC/bayc_original/
azuki/azuki_original/
cryptopunk/cryptopunk_original/
```

Keep the original image filenames. The tests read images according to `img_index.json`.

## CryptoPunk Preprocess

CryptoPunk needs an extra conversion step:

```bash
cd cryptopunk
mkdir -p cryptopunk_24
python convert_24.py
```

This creates `cryptopunk_24/`, which is used by `test_flops.py`.

## Run Tests

Run each test from its own folder.

BAYC:

```bash
cd BAYC
python test_flops.py
```

Azuki:

```bash
cd azuki
python test_flops.py
```

CryptoPunk:

```bash
cd cryptopunk
mkdir -p cryptopunk_24
python convert_24.py
python test_flops.py
```

The scripts print model size, parameter counts, FLOPs, latency, PSNR, and SSIM. Generated images are saved to `generated_imgs/`.

## Files

- `test_flops.py`: main test script.
- `encoder_fr.py`: model definitions.
- `utils.py`: dataset loading, evaluation, metrics, parameter counting, and weight conversion helpers.
- `img_index.json`: image order and filename mapping.
- `state_dict_compressed.bin`: compressed model weights.
- `frequency_regularization/freq_reg/`: local frequency regularization implementation.
- `convert_24.py`: CryptoPunk-only script that converts `cryptopunk_original/` to `cryptopunk_24/`.
