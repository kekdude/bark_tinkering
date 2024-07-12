import pathlib
from os import PathLike
from typing import Union

import torch
import torchaudio
from encodec.utils import convert_audio
from transformers import BarkProcessor, BarkModel
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig, \
    BarkCoarseGenerationConfig

from bark_tinkering.utils import bark_generate, wav_to_coarse_ids, audio_to_wav, get_models_devices, set_model_devices, \
    SEMANTIC_MODEL, COARSE_MODEL, FINE_MODEL, CODEC_MODEL, save_playlist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import safetensors.numpy

SEMANTIC_PROMPT = "semantic_prompt"
COARSE_PROMPT = "coarse_prompt"
FINE_PROMPT = "fine_prompt"


def log_perplexity(logits, probs):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_probs = probs[:, 1:, :].contiguous()
    shift_logits = shift_logits[:, :, :shift_probs.size(2)]
    return -(shift_probs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()


def find_semantics_by_wav(model: BarkModel,
                          proc: BarkProcessor,
                          text: str,
                          wav_filename: Union[str, PathLike],
                          output_folder: Union[str, PathLike],
                          save_every_n_steps=100,
                          steps=10000,
                          no_save_steps=0,
                          lr=1e-1,
                          perplexity_loss_weight=0.2,
                          disable_tqdm=False,
                          device='cuda',
                          torch_dtype=torch.float32,
                          initial_coeff=15):
    output_folder = pathlib.Path(output_folder)
    playlist = []
    playlist_path = output_folder / 'all.m3u'

    batch_size = 1
    semantic_generation_config = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
    coarse_generation_config = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)

    # original_model_devices = get_models_devices(model)
    # training_model_devices = {
    #     SEMANTIC_MODEL: device,
    #     COARSE_MODEL: device,
    #     FINE_MODEL: 'cpu',
    #     CODEC_MODEL: device
    # }
    #
    # set_model_devices(model, training_model_devices)

    semantic_to_coarse_ratio = (
            coarse_generation_config.coarse_rate_hz
            / semantic_generation_config.semantic_rate_hz
            * coarse_generation_config.n_coarse_codebooks
    )

    with torch.no_grad():
        # max_context_size - semantic - coarse_infer_token
        max_coarse_ids = 1024 - coarse_generation_config.max_coarse_input_length - 1
        coarse_ids = wav_to_coarse_ids(model, wav_filename)
        coarse_ids = coarse_ids[:, :max_coarse_ids]
        coarse_audio_embeds = model.coarse_acoustics.input_embeds_layer.forward(coarse_ids)
        coarse_audio_embeds = coarse_audio_embeds.repeat_interleave(batch_size, dim=0).contiguous()
        coarse_labels = torch.repeat_interleave(coarse_ids[:, 1:], batch_size, dim=0).contiguous() - 10000

        semantic_size = min(
            coarse_generation_config.max_coarse_input_length,
            int(np.floor(coarse_ids.size(1) / semantic_to_coarse_ratio))
        )
        semantic_size = semantic_size - semantic_size % 2
        semantic_vocab_size = semantic_generation_config.semantic_vocab_size

        text_ids = torch.tensor(proc.tokenizer(text).input_ids)
        text_ids = text_ids + semantic_generation_config.text_encoding_offset
        text_ids = text_ids[:256]
        text_ids = F.pad(text_ids,
                         (0, 256 - len(text_ids)),
                         "constant",
                         value=semantic_generation_config.text_pad_token)
        text_ids = text_ids.unsqueeze(0)
        text_ids = text_ids.to(device)
        #text_embeds = model.semantic.input_embeds_layer(text_ids)
        text_embeds = model.semantic.input_embeds_layer(text_ids).repeat_interleave(batch_size, dim=0)

        n_semantic_paddings = max(0, coarse_generation_config.max_coarse_input_length - semantic_size)

        coarse_semantic_pad_tokens = torch.tensor([coarse_generation_config.coarse_semantic_pad_token] * n_semantic_paddings,
                                                  device=device).unsqueeze(0)

        coarse_semantic_pad_embeds = model.coarse_acoustics.input_embeds_layer(coarse_semantic_pad_tokens)
        coarse_semantic_pad_embeds = coarse_semantic_pad_embeds.repeat_interleave(batch_size, dim=0)

        semantic_infer_token = torch.tensor([semantic_generation_config.semantic_infer_token],
                                            device=device).unsqueeze(0)

        coarse_infer_token = torch.tensor([coarse_generation_config.coarse_infer_token],
                                          device=device).unsqueeze(0)

        semantic_infer_token_embeds = model.semantic.input_embeds_layer(semantic_infer_token).repeat_interleave(batch_size, dim=0)
        coarse_infer_token_embeds = model.coarse_acoustics.input_embeds_layer(coarse_infer_token).repeat_interleave(batch_size, dim=0)

        semantic_vocab_embeddings = model.semantic.input_embeds_layer.weight.data[:semantic_vocab_size].to(torch.float32)
        coarse_vocab_embeddings = model.coarse_acoustics.input_embeds_layer.weight.data[:semantic_vocab_size].to(torch.float32)

    model.train()
    for p in model.parameters():
        p.requires_grad = False

    parameters = torch.rand((1, semantic_size, semantic_vocab_size), device=device, dtype=torch.float32)
    parameters.requires_grad = True

    optimizer = torch.optim.AdamW([parameters], lr=lr)
    progress_bar = tqdm(range(1, steps+1), disable=disable_tqdm)

    for step in progress_bar:
        batched_parameters = parameters.repeat(batch_size, 1, 1)
        probs = F.gumbel_softmax(batched_parameters, hard=False)
        semantic_embeds = probs @ semantic_vocab_embeddings
        #semantic_embeds = semantic_embeds.to(torch_dtype)

        semantic_input_embeds = torch.cat([
            text_embeds,
            semantic_infer_token_embeds,
            semantic_embeds
        ], dim=1)

        semantic_output = model.semantic.forward(input_embeds=semantic_input_embeds)
        semantic_output_logits = semantic_output.logits[:, -semantic_embeds.size(1):, :semantic_vocab_size]
        perplexity_loss = log_perplexity(semantic_output_logits, probs) * perplexity_loss_weight

        coarse_semantic_embeds = probs @ coarse_vocab_embeddings
        #coarse_semantic_embeds = coarse_semantic_embeds.to(torch_dtype)

        coarse_embeds = torch.cat(
            [
                coarse_semantic_embeds,
                coarse_semantic_pad_embeds,
                coarse_infer_token_embeds,
                coarse_audio_embeds
            ],
            dim=1)

        coarse_output = model.coarse_acoustics.forward(input_embeds=coarse_embeds)

        shift_logits = coarse_output.logits[..., -coarse_ids.size(1):-1, 10000:12048].contiguous()
        coarse_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), coarse_labels.view(-1))

        loss = perplexity_loss + coarse_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        progress_bar.set_description(f'loss={loss.item():.3f}, perplexity_loss={perplexity_loss.item():.3f}, coarse_loss={coarse_loss.item():.3f}', refresh=True)

        if step > no_save_steps and step % save_every_n_steps == 0:
            model.eval()
            #set_model_devices(model, original_model_devices)
            with torch.no_grad():
                save_path = output_folder / f'step_{step}'
                adversarial_semantic_ids = parameters.argmax(dim=-1)#F.gumbel_softmax(parameters, hard=True, dim=-1).argmax(dim=-1)
                bark_generate(model, from_semantic=adversarial_semantic_ids, save_as=save_path)
                torch.save(parameters, save_path / 'adversarial_semantic_parameters.pt')
                playlist.append(str(pathlib.Path(save_path / 'audio.wav').relative_to(playlist_path.parent)))
                save_playlist(playlist, playlist_path)

            model.train()
            #set_model_devices(model, training_model_devices)

    #set_model_devices(model, original_model_devices)


@torch.no_grad()
def extract_speaker_wav(model: BarkModel, speakers_path: str, speaker_name: str, output_wav_filename: str):
    fine_prompt = torch.tensor(np.load(f'{speakers_path}/{speaker_name}_fine_prompt.npy')).unsqueeze(1).to(model.device)
    emb = model.codec_model.quantizer.decode(fine_prompt)
    out = model.codec_model.decoder(emb)
    out = out.squeeze(1)
    audio_to_wav(out, output_wav_filename, model.generation_config.sample_rate)


def save_voice_preset_safetensors(voice_preset: dict, filename: str | PathLike):
    safetensors.numpy.save_file({
        SEMANTIC_PROMPT: voice_preset[SEMANTIC_PROMPT],
        COARSE_PROMPT: voice_preset[COARSE_PROMPT],
        FINE_PROMPT: voice_preset[FINE_PROMPT]
    }, filename)


def load_voice_preset_safetensors(filename: str | PathLike):
    return safetensors.numpy.load_file(filename)


def save_voice_preset_numpy(voice_preset: dict, filename: str | PathLike):
    np.savez(str(filename),
             semantic_prompt=voice_preset[SEMANTIC_PROMPT],
             coarse_prompt=voice_preset[COARSE_PROMPT],
             fine_prompt=voice_preset[FINE_PROMPT])


def load_voice_preset_numpy(filename: str | PathLike):
    return dict(np.load(filename).items())


def create_voice_preset(model: BarkModel,
                        input_wav_filename: Union[str, PathLike],
                        semantic_filename: Union[str, PathLike]) -> dict:

    semantic = torch.load(semantic_filename)
    semantic = semantic.squeeze(0).cpu()

    wav, sr = torchaudio.load(input_wav_filename)
    wav = convert_audio(wav, sr, model.codec_model.config.sampling_rate, model.codec_model.config.audio_channels)
    wav = wav.unsqueeze(0)
    wav = wav.to(model.codec_model.device)
    encoded_frames = model.codec_model.encode(wav, bandwidth=6.0)
    codes = encoded_frames.audio_codes.squeeze(0).squeeze(0).cpu()

    fine_prompt = torch.ones((codes.shape[1], codes.shape[0]), dtype=torch.int64)
    fine_prompt[None] = codes.T
    fine_prompt = torch.transpose(fine_prompt, 0, 1)

    coarse_prompt = torch.ones((codes.shape[1], 2), dtype=torch.int64)
    coarse_prompt[None] = codes[[0, 1]].T
    coarse_prompt = torch.transpose(coarse_prompt, 0, 1)

    return {
        SEMANTIC_PROMPT: semantic.numpy(),
        COARSE_PROMPT: coarse_prompt.numpy(),
        FINE_PROMPT: fine_prompt.numpy()
    }