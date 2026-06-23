import argparse
import datetime
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import DPMSolverMultistepScheduler
from diffusers.models.embeddings import TimestepEmbedding, get_timestep_embedding
from transformers import CLIPTokenizer

from qai_appbuilder import QNNConfig, QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile


MODEL_NAME = "stable_diffusion_v1_5"
TOKENIZER_MODEL_NAME = "openai/clip-vit-large-patch14"
TOKENIZER_MAX_LENGTH = 77
LATENT_H = 64
LATENT_W = 64
VAE_SCALING = 0.18215
ROUND_BRACKET_MULTIPLIER = 1.1
SQUARE_BRACKET_MULTIPLIER = 1 / 1.1
LORA_PATTERN = re.compile(r"<lora:[^>]*>", re.IGNORECASE)
EXPLICIT_WEIGHT_PATTERN = re.compile(r"\(([^()\[\]]+?):\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\)")


class TextEncoder(QNNContext):
    def Inference(self, input_ids):
        output = super().Inference([input_ids])[0]
        return output.reshape((1, TOKENIZER_MAX_LENGTH, 768))


class UnetWithEmb(QNNContext):
    def Inference(self, sample, emb, encoder_hidden_states):
        sample = sample.reshape(sample.size)
        emb = emb.reshape(emb.size)
        encoder_hidden_states = encoder_hidden_states.reshape(encoder_hidden_states.size)
        output = super().Inference([sample, emb, encoder_hidden_states])[0]
        return output.reshape((1, LATENT_H, LATENT_W, 4))


class VaeDecoder(QNNContext):
    def Inference(self, latent):
        latent = latent.reshape(latent.size)
        return super().Inference([latent])[0]


def resolve_execution_ws():
    
    if not (execution_ws / "qai_libs").exists():
        repo_root = execution_ws.parents[2]
        candidate = repo_root / "qai_libs"
        if candidate.exists():
            return execution_ws, candidate
    return execution_ws, execution_ws / "qai_libs"


def load_tokenizer(tokenizer_dir):
    if tokenizer_dir.exists():
        return CLIPTokenizer.from_pretrained(
            tokenizer_dir,
        )
    else:
        return CLIPTokenizer.from_pretrained(TOKENIZER_MODEL_NAME, cache_dir=str("./models"))


def strip_lora(prompt):
    return LORA_PATTERN.sub(" ", prompt)


def parse_weighted_prompt(prompt):
    prompt = strip_lora(prompt)
    weighted_terms = []

    def replace_explicit_weight(match):
        text = match.group(1).strip()
        weight = float(match.group(2))
        if text:
            weighted_terms.append((text, weight))
        return text

    prompt = EXPLICIT_WEIGHT_PATTERN.sub(replace_explicit_weight, prompt)

    cleaned_chars = []
    multiplier_stack = [1.0]
    active_start = {}

    for char in prompt:
        if char == "(":
            active_start[len(multiplier_stack)] = len(cleaned_chars)
            multiplier_stack.append(multiplier_stack[-1] * ROUND_BRACKET_MULTIPLIER)
        elif char == "[":
            active_start[len(multiplier_stack)] = len(cleaned_chars)
            multiplier_stack.append(multiplier_stack[-1] * SQUARE_BRACKET_MULTIPLIER)
        elif char in ")]" and len(multiplier_stack) > 1:
            weight = multiplier_stack.pop()
            start = active_start.pop(len(multiplier_stack), None)
            if start is not None:
                text = "".join(cleaned_chars[start:]).strip(" ,")
                if text:
                    weighted_terms.append((text, weight))
        else:
            cleaned_chars.append(char)

    cleaned_prompt = "".join(cleaned_chars).replace("\n", " ")
    cleaned_prompt = re.sub(r"\s*,\s*", ", ", cleaned_prompt)
    cleaned_prompt = re.sub(r"(?:,\s*){2,}", ", ", cleaned_prompt)
    cleaned_prompt = " ".join(cleaned_prompt.split()).strip(" ,")
    return cleaned_prompt, weighted_terms


def run_tokenizer(tokenizer, prompt):
    text_input = tokenizer(
        prompt,
        padding="max_length",
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
    )
    return np.array(text_input.input_ids, dtype=np.float32)


def token_positions(tokenizer, prompt, term):
    prompt_tokens = tokenizer(
        prompt,
        padding="max_length",
        max_length=TOKENIZER_MAX_LENGTH,
        truncation=True,
    ).input_ids
    term_tokens = tokenizer(term, add_special_tokens=False).input_ids
    if not term_tokens:
        return []

    positions = []
    search_limit = len(prompt_tokens) - len(term_tokens) + 1
    for index in range(search_limit):
        if prompt_tokens[index : index + len(term_tokens)] == term_tokens:
            positions.extend(range(index, index + len(term_tokens)))
    return [position for position in positions if position < TOKENIZER_MAX_LENGTH]


def apply_prompt_weights(tokenizer, prompt, text_embedding, weighted_terms):
    weights = np.ones((TOKENIZER_MAX_LENGTH,), dtype=np.float32)
    for term, weight in weighted_terms:
        for position in token_positions(tokenizer, prompt, term):
            weights[position] *= np.float32(weight)
    return text_embedding * weights.reshape((1, TOKENIZER_MAX_LENGTH, 1))


def load_time_embedding(time_embedding_path):
    time_embeddings = TimestepEmbedding(in_channels=320, time_embed_dim=1280)
    state_dict = torch.load(time_embedding_path, map_location="cpu")
    time_embeddings.load_state_dict(state_dict)
    time_embeddings.eval()
    return time_embeddings


def get_time_embedding(time_embeddings, timestep):
    timestep = torch.tensor([int(timestep)], dtype=torch.int64)
    sinusoidal_embedding = get_timestep_embedding(timestep, 320, True, 0)
    with torch.no_grad():
        emb = time_embeddings(sinusoidal_embedding).detach().cpu().numpy()
    return emb.astype(np.float32)


def run_scheduler(scheduler, guidance_scale, noise_pred_uncond, noise_pred_text, latent_in, timestep):
    noise_pred_uncond = np.transpose(noise_pred_uncond, (0, 3, 1, 2)).copy()
    noise_pred_text = np.transpose(noise_pred_text, (0, 3, 1, 2)).copy()
    latent_in_nchw = np.transpose(latent_in, (0, 3, 1, 2)).copy()

    noise_pred_uncond = torch.from_numpy(noise_pred_uncond)
    noise_pred_text = torch.from_numpy(noise_pred_text)
    latent_in_nchw = torch.from_numpy(latent_in_nchw)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    latent_out = scheduler.step(noise_pred, int(timestep), latent_in_nchw).prev_sample.numpy()
    return np.transpose(latent_out, (0, 2, 3, 1)).copy()


def record_inference(stats, model_name, inference_fn, *args):
    start = time.perf_counter()
    output = inference_fn(*args)
    elapsed = time.perf_counter() - start
    stats.setdefault(model_name, []).append(elapsed)
    return output


def print_inference_stats(stats):
    print("\nAverage model inference time:")
    for model_name in ["text_encoder", "unet", "vae_decoder"]:
        timings = stats.get(model_name, [])
        if not timings:
            print(f"  {model_name}: no calls")
            continue
        total = sum(timings)
        average = total / len(timings)
        print(
            f"  {model_name}: avg={average * 1000:.2f} ms, "
            f"total={total:.2f} s, calls={len(timings)}"
        )


def parse_args():
    execution_ws = Path(__file__).resolve().parent


    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default= "(8k, best quality, masterpiece:1.2),(best quality:1.0), (ultra highres:1.0), watercolor, a beautiful woman, shoulder, spaghetti straps, hair ribbons, by agnes cecile, half body portrait, extremely luminous bright design, pastel colors, (ink:1.3), autumn lights")
    parser.add_argument("--negative-prompt", default="lowres, text, error, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark")
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--qnn-libs", type=Path, default="./qnn_libs")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=execution_ws / "images")
    parser.add_argument("--num-pictures", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    text_encoder_path = args.model_dir / "text_encoder.serialized.bin"
    unet_path = args.model_dir / "unet.serialized.bin"
    vae_decoder_path = args.model_dir / "vae_decoder.serialized.bin"
    time_embedding_path = args.model_dir / "time_embedding.pt"
    for path in [text_encoder_path, unet_path, vae_decoder_path, time_embedding_path, args.qnn_libs]:
        if not path.exists():
            raise FileNotFoundError(path)

    QNNConfig.Config(str(args.qnn_libs), Runtime.HTP, LogLevel.ERROR, ProfilingLevel.BASIC)

    tokenizer = load_tokenizer(args.model_dir / "tokenizer")
    time_embeddings = load_time_embedding(time_embedding_path)
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    text_encoder = TextEncoder("text_encoder", str(text_encoder_path))
    unet = UnetWithEmb("unet_emb", str(unet_path))
    vae_decoder = VaeDecoder("vae_decoder", str(vae_decoder_path))
    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)
    inference_stats = {}

    cond_prompt, cond_weighted_terms = parse_weighted_prompt(args.prompt)
    uncond_prompt, uncond_weighted_terms = parse_weighted_prompt(args.negative_prompt)
    if cond_prompt != args.prompt:
        print(f"Prompt after syntax cleanup: {cond_prompt}")
    if uncond_prompt != args.negative_prompt:
        print(f"Negative prompt after  syntax cleanup: {uncond_prompt}")

    cond_tokens = run_tokenizer(tokenizer, cond_prompt)
    uncond_tokens = run_tokenizer(tokenizer, uncond_prompt)


    uncond_text_embedding = record_inference(
        inference_stats,
        "text_encoder",
        text_encoder.Inference,
        uncond_tokens,
    )
    cond_text_embedding = record_inference(
        inference_stats,
        "text_encoder",
        text_encoder.Inference,
        cond_tokens,
    )
    uncond_text_embedding = apply_prompt_weights(
        tokenizer,
        uncond_prompt,
        uncond_text_embedding,
        uncond_weighted_terms,
    )
    cond_text_embedding = apply_prompt_weights(
        tokenizer,
        cond_prompt,
        cond_text_embedding,
        cond_weighted_terms,
    )

    for _ in range(args.num_pictures):
        if args.seed == -1:
            args.seed = int(np.random.randint(low=0, high=9999999999, size=None, dtype=np.int64))
        scheduler.set_timesteps(args.steps)
        random_init_latent = torch.randn((1, 4, LATENT_H, LATENT_W), generator=torch.manual_seed(args.seed)).numpy()
        latent_in = random_init_latent.transpose((0, 2, 3, 1)).copy()

        start = time.time()
        for step_index, timestep in enumerate(scheduler.timesteps):
            timestep_value = int(timestep.item())
            emb = get_time_embedding(time_embeddings, timestep_value)
            print(f"Step {step_index + 1}/{args.steps}, timestep={timestep_value}, emb_shape={emb.shape}")

            uncond_noise_pred = record_inference(
                inference_stats,
                "unet",
                unet.Inference,
                latent_in,
                emb,
                uncond_text_embedding,
            )
            cond_noise_pred = record_inference(
                inference_stats,
                "unet",
                unet.Inference,
                latent_in,
                emb,
                cond_text_embedding,
            )
            latent_in = run_scheduler(
                scheduler,
                args.guidance_scale,
                uncond_noise_pred,
                cond_noise_pred,
                latent_in,
                timestep_value,
            )

        output = record_inference(
            inference_stats,
            "vae_decoder",
            vae_decoder.Inference,
            latent_in / VAE_SCALING,
        )
        output = np.clip((output / 2 + 0.5) * 255.0, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(output.reshape((512, 512, 3)), mode="RGB")
        if args.output is None:
            model_name = str(args.model_dir).split("/")[-1]
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            args.output = args.output_dir / f"{model_name}_{timestamp}_{args.seed}_512.jpg"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image.save(args.output)

        print(f"Saved image to {args.output}")
        print(f"Seed: {args.seed}")
        print(f"Elapsed: {time.time() - start:.2f}s")
        print_inference_stats(inference_stats)
        inference_stats.pop("unet")
        inference_stats.pop("vae_decoder")
        args.output = None
        args.seed = -1

    PerfProfile.RelPerfProfileGlobal()

if __name__ == "__main__":
    main()
