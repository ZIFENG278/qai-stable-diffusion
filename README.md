# QAI Stable Diffusion

Run Stable Diffusion v1.5 QNN models with `qai_appbuilder` on Qualcomm v73 HTP NPU


![dreamshaper](./assets/dreamshaper.jpg)

## Test Device

- [Radxa Fogwise® AIRbox Q900](https://radxa.com/products/fogwise/airbox-q900) 

## Model Checkpoints

QNN version == 2.46

[Download from Modelscope](https://modelscope.cn/collections/radxa/stable-diffusion-v1-5-v73-qnn)

- [DreamShaper_8](https://modelscope.cn/models/radxa/DreamShaper_8_v73_qnn_2.46/summary) [civitai link](https://civitai.red/models/4384/dreamshaper?modelVersionId=128713)
- [epiCRealism_Natural_Sin_RC1_VAE](https://modelscope.cn/models/radxa/epiCRealism_Natural_Sin_RC1_VAE_v73_qnn_2.46) [civitai link](https://civitai.red/models/25694/epicrealism?modelVersionId=143906)
- [majicMIX_realistic_v7](https://modelscope.cn/models/radxa/majicMIX_realistic_v7_v73_qnn_2.46) [civitai link](https://civitai.red/models/43331/majicmix-realistic?modelVersionId=176425)
- [Lucky_Strike_Mix_Lovely_Lady_V1.05](https://modelscope.cn/models/radxa/Lucky_Strike_Mix_Lovely_Lady_V1.05_v73_qnn_2.46) [civitai link](https://civitai.red/models/13034/lucky-strike-mix?modelVersionId=127680)



## Deploy on Qualcomm SoC v73 HTP Devices

### clone the repo
```bash
git clone https://github.com/ZIFENG278/qai-stable-diffusion.git
cd qai-stable-diffusion
```

### setup the env

```bash
sudo apt install python3-venv
python3 -m venv .venv 
pip3 install -r requirements.txt
pip3 install https://github.com/ZIFENG278/ai-engine-direct-helper/releases/download/radxa-dev-2.38.0/qai_appbuilder-2.38.0-cp312-cp312-linux_aarch64.whl

export ADSP_LIBRARY_PATH=$(pwd)/qnn_libs
```

### Download Model

```bash
modelscope download --model radxa/DreamShaper_8_v73_qnn_2.46 --local_dir ./models/DreamShaper_8_v73_qnn_2.46
```

## Entry Point

The main entry point is:

```text
sd1_5.py
```

It handles:

- CLIP tokenization
- WebUI-style prompt cleanup and basic prompt weights
- text encoder inference
- precomputed SD1.5 timestep embedding generation
- UNet denoising loop with classifier-free guidance
- VAE decoder inference
- image saving
- average inference-time reporting for each model

## Repository Layout

```text
sd1_5.py                 Main text-to-image runner
qnn_libs/                QNN runtime libraries used by qai_appbuilder
models/                  Local model/tokenizer workspace, not intended for git model blobs
requirements.txt         Python dependency notes
README.md                This document
```

## Required Model Directory

Pass a notebook-exported SD1.5 model directory with `--model-dir`.

The directory must contain:

```text
text_encoder.serialized.bin
unet.serialized.bin
vae_decoder.serialized.bin
time_embedding.pt
```

For tokenizer loading, either include a tokenizer directory inside the model directory:

```text
tokenizer/
```

or let the script use/download the `openai/clip-vit-large-patch14` tokenizer cache under `./models`.

The serialized model files and `time_embedding.pt` must come from the same notebook export/checkpoint.


## Run

Example:

```bash
python3 sd1_5.py \
  --model-dir ./models/DreamShaper_8_v73_qnn_2.46 \
  --prompt "(masterpiece:1.1), (best quality:1.1), a beautiful woman, watercolor" \
  --negative-prompt "lowres, bad anatomy, worst quality" \
  --steps 20 \
  --guidance-scale 7.5 \
  --seed 123
```

Generate multiple images with random seeds:

```bash
python3 sd1_5.py \
  --model-dir ./models/DreamShaper_8_v73_qnn_2.46 \
  --num-pictures 4 \
  --seed -1
```

## Arguments

```text
--model-dir         Required. Directory containing serialized QNN model files.
--qnn-libs          QNN library directory. Default: ./qnn_libs
--prompt            Positive prompt.
--negative-prompt   Negative prompt.
--steps             Denoising steps. Default: 20
--guidance-scale    Classifier-free guidance scale. Default: 7.5
--seed              Seed. Use -1 for random seed. Default: -1
--num-pictures      Number of images to generate. Default: 1
--output            Optional explicit output path.
--output-dir        Output directory. Default: ./images
```

## Output

When `--output` is not specified, images are saved as:

```text
images/<model_name>_YYYY_MM_DD_HH_MM_SS_<seed>_512.jpg
```

Example:

```text
images/DreamShaper_8_v73_qnn_2.46_2026_06_22_13_20_11_123_512.jpg
```

## Prompt Syntax

The runner removes LoRA tags and supports a small subset of WebUI prompt weights:

- `<lora:...>` is removed.
- `(text:1.2)` applies explicit token embedding weight.
- `(text)` applies `1.1` weight.
- `((text))` applies nested weight.
- `[text]` applies `1 / 1.1` weight.

Prompt editing syntax like `[from:to:step]` is not supported.

LoRA weights are not loaded dynamically by this script. If a prompt copied from CivitAI contains `<lora:...>`, the tag is removed before tokenization.

## Timing Output

At the end of each image generation, the script prints average inference time for:

- `text_encoder`
- `unet`
- `vae_decoder`

Example:

```text
Average model inference time:
  text_encoder: avg=8.03 ms, total=0.02 s, calls=2
  unet: avg=129.43 ms, total=5.18 s, calls=40
  vae_decoder: avg=200.21 ms, total=0.20 s, calls=1
```

UNet call count is `steps * 2` because each denoising step runs unconditional and conditional passes.

## Notes

- The UNet input `emb` is `(1, 1280)` for SD1.5.
- `sample` and UNet output use NHWC layout in the AppBuilder runner: `(1, 64, 64, 4)`.
- The scheduler runs on CPU with `DPMSolverMultistepScheduler`.
- The script assumes 512x512 SD1.5 output.
