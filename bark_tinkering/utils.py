import pathlib
from os import PathLike
from typing import Optional, Dict, Union, Tuple, List

import numpy as np
import torch
import torchaudio
from encodec.utils import convert_audio
from torch import Tensor
from transformers import BarkModel, BarkProcessor
from transformers.models.bark.generation_configuration_bark import BarkSemanticGenerationConfig, \
    BarkCoarseGenerationConfig, BarkFineGenerationConfig

SEMANTIC_MODEL = "semantic"
COARSE_MODEL = "coarse_acoustics"
FINE_MODEL = "fine_acoustics"
CODEC_MODEL = "codec_model"


@torch.no_grad()
def bark_generate(
        model: BarkModel,
        input_ids: Optional[torch.Tensor] = None,
        history_prompt: Optional[Dict[str, torch.Tensor]] = None,
        return_output_lengths: Optional[bool] = None,
        save_as: Optional[str | PathLike] = None,
        from_semantic: Optional[torch.Tensor] = None,
        from_coarse: Optional[torch.Tensor] = None,
        **kwargs,
) -> Union[Tuple[Tensor, List[int]], torch.LongTensor]:
    """
            Extended version of `BarkModel.generate` with option to save generation results, incuding generated semantic,
            coarse and fine models outputs.

            Args:
                model (`BarkModel`)
                input_ids (`Optional[torch.Tensor]` of shape (batch_size, seq_len), *optional*):
                    Input ids. Will be truncated up to 256 tokens. Note that the output audios will be as long as the
                    longest generation among the batch.
                history_prompt (`Optional[Dict[str,torch.Tensor]]`, *optional*):
                    Optional `Bark` speaker prompt. Note that for now, this model takes only one speaker prompt per batch.
                kwargs (*optional*): Remaining dictionary of keyword arguments. Keyword arguments are of two types:

                    - Without a prefix, they will be entered as `**kwargs` for the `generate` method of each sub-model.
                    - With a *semantic_*, *coarse_*, *fine_* prefix, they will be input for the `generate` method of the
                    semantic, coarse and fine respectively. It has the priority over the keywords without a prefix.

                    This means you can, for example, specify a generation strategy for all sub-models except one.
                return_output_lengths (`bool`, *optional*):
                    Whether or not to return the waveform lengths. Useful when batching.
                save_as (`Optional[str | PathLike]`, *optional*):
                    Path to save generation results.
                from_semantic (`Optional[torch.Tensor]`, *optional*):
                    Output of semantic model to start generation from. If passed, semantic generation will be skipped.
                from_coarse (`Optional[torch.Tensor]`, *optional*):
                    Output of coarse model to start generation from. If passed, semantic and coarse generation will be
                    skipped.
            Returns:
                By default:
                    - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.
                When `return_output_lengths=True`:
                    Returns a tuple made of:
                    - **audio_waveform** (`torch.Tensor` of shape (batch_size, seq_len)): Generated audio waveform.
                    - **output_lengths** (`torch.Tensor` of shape (batch_size)): The length of each waveform in the batch
            Example:

            ```python
            >>> from transformers import AutoProcessor, BarkModel

            >>> processor = AutoProcessor.from_pretrained("suno/bark-small")
            >>> model = BarkModel.from_pretrained("suno/bark-small")

            >>> # To add a voice preset, you can pass `voice_preset` to `BarkProcessor.__call__(...)`
            >>> voice_preset = "v2/en_speaker_6"

            >>> inputs = processor("Hello, my dog is cute, I need him in my life", voice_preset=voice_preset)

            >>> audio_array = model.generate(**inputs, semantic_max_new_tokens=100)
            >>> audio_array = audio_array.cpu().numpy().squeeze()
            ```
            """
    # TODO (joao):workaround until nested generation config is compatible with PreTrained Model
    # todo: dict
    semantic_generation_config = BarkSemanticGenerationConfig(**model.generation_config.semantic_config)
    coarse_generation_config = BarkCoarseGenerationConfig(**model.generation_config.coarse_acoustics_config)
    fine_generation_config = BarkFineGenerationConfig(**model.generation_config.fine_acoustics_config)

    kwargs_semantic = {
        # if "attention_mask" is set, it should not be passed to CoarseModel and FineModel
        "attention_mask": kwargs.pop("attention_mask", None),
        "min_eos_p": kwargs.pop("min_eos_p", None),
    }
    kwargs_coarse = {}
    kwargs_fine = {}
    for key, value in kwargs.items():
        if key.startswith("semantic_"):
            key = key[len("semantic_"):]
            kwargs_semantic[key] = value
        elif key.startswith("coarse_"):
            key = key[len("coarse_"):]
            kwargs_coarse[key] = value
        elif key.startswith("fine_"):
            key = key[len("fine_"):]
            kwargs_fine[key] = value
        else:
            # If the key is already in a specific config, then it's been set with a
            # submodules specific value and we don't override
            if key not in kwargs_semantic:
                kwargs_semantic[key] = value
            if key not in kwargs_coarse:
                kwargs_coarse[key] = value
            if key not in kwargs_fine:
                kwargs_fine[key] = value

    semantic_output = None
    coarse_output = None

    if from_coarse is not None:
        coarse_output = from_coarse
    else:
        # 1. Generate from the semantic model
        if from_semantic is not None:
            semantic_output = from_semantic
        else:
            semantic_output = model.semantic.generate(
                input_ids,
                history_prompt=history_prompt,
                semantic_generation_config=semantic_generation_config,
                **kwargs_semantic,
            )

        # 2. Generate from the coarse model
        coarse_output = model.coarse_acoustics.generate(
            semantic_output,
            history_prompt=history_prompt,
            semantic_generation_config=semantic_generation_config,
            coarse_generation_config=coarse_generation_config,
            codebook_size=model.generation_config.codebook_size,
            return_output_lengths=return_output_lengths,
            **kwargs_coarse,
        )

    output_lengths = None
    if return_output_lengths:
        coarse_output, output_lengths = coarse_output
        # (batch_size, seq_len*coarse_codebooks) -> (batch_size, seq_len)
        output_lengths = output_lengths // coarse_generation_config.n_coarse_codebooks

    # 3. "generate" from the fine model
    output = model.fine_acoustics.generate(
        coarse_output,
        history_prompt=history_prompt,
        semantic_generation_config=semantic_generation_config,
        coarse_generation_config=coarse_generation_config,
        fine_generation_config=fine_generation_config,
        codebook_size=model.generation_config.codebook_size,
        **kwargs_fine,
    )

    if getattr(model, "fine_acoustics_hook", None) is not None:
        # Manually offload fine_acoustics to CPU
        # and load codec_model to GPU
        # since bark doesn't use codec_model forward pass
        model.fine_acoustics_hook.offload()
        model.codec_model = model.codec_model.to(model.device)

    # 4. Decode the output and generate audio array
    audio = model.codec_decode(output, output_lengths)

    if getattr(model, "codec_model_hook", None) is not None:
        # Offload codec_model to CPU
        model.codec_model_hook.offload()

    if return_output_lengths:
        output_lengths = [len(sample) for sample in audio]
        audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True, padding_value=0)
        return audio, output_lengths

    if save_as is not None:
        base_save_path = pathlib.Path(save_as)
        base_save_path.mkdir(parents=True, exist_ok=True)
        torch.save(semantic_output, base_save_path.joinpath('semantic.pt'))
        torch.save(coarse_output, base_save_path.joinpath('coarse.pt'))
        coarse_ids_to_wav(model, coarse_output, base_save_path.joinpath('coarse.wav'),
                          sample_rate=model.generation_config.sample_rate)
        torch.save(output, base_save_path.joinpath('fine.pt'))
        torch.save(audio, base_save_path.joinpath('audio.pt'))
        audio_to_wav(audio, base_save_path.joinpath('audio.wav'), model.generation_config.sample_rate)

    return audio


def audio_to_wav(audio, filename: Union[str, PathLike], sample_rate=16000):
    from scipy.io.wavfile import write as write_wav
    audio_array = audio.cpu().numpy().astype(np.float32).squeeze()
    write_wav(str(filename), sample_rate, audio_array)


@torch.no_grad()
def coarse_ids_to_wav(model: BarkModel, coarse_output: torch.LongTensor, filename: Union[str, PathLike], sample_rate):
    coarse_output = coarse_output.clone()
    idx = torch.arange(coarse_output.size(-1))
    channel0 = coarse_output[:, idx % 2 == 0] - 10000
    channel1 = coarse_output[:, idx % 2 == 1] - 11024
    channels = torch.vstack([channel0, channel1])
    channels = channels.unsqueeze(1)
    emb = model.codec_model.quantizer.decode(channels)
    out = model.codec_model.decoder(emb)
    audio_arr = out.squeeze(1)  # squeeze the codebook dimension
    audio_to_wav(audio_arr, filename, sample_rate=sample_rate)


@torch.no_grad()
def wav_to_coarse_ids(model: BarkModel, filename: Union[str, PathLike]):
    wav, sr = torchaudio.load(filename)
    wav = convert_audio(wav, sr, model.codec_model.config.sampling_rate, model.codec_model.config.audio_channels)
    wav = wav.unsqueeze(0)
    wav = wav.to(dtype=model.codec_model.dtype, device=model.codec_model.device)
    encoded_frames = model.codec_model.encode(wav)
    codes = encoded_frames[0]
    channel0 = codes[0, 0, 0, :].to(model.coarse_acoustics.device)
    channel1 = codes[0, 0, 1, :].to(model.coarse_acoustics.device)

    coarse_length = channel0.size(-1) * 2
    coarse = torch.zeros((1, coarse_length), dtype=torch.long, device=model.coarse_acoustics.device)
    idx = torch.arange(coarse_length)
    coarse[:, idx % 2 == 0] = channel0 + 10000
    coarse[:, idx % 2 == 1] = channel1 + 11024

    return coarse


def make_text_generations(model: BarkModel, processor: BarkProcessor, texts: List[str], base_folder: PathLike | str, voice_preset=None, min_eos_p=0.05):
    base_folder = pathlib.Path(base_folder)
    pathlib.Path(base_folder).mkdir(parents=True, exist_ok=True)
    sample_ids = list(f'{i}' for i in range(len(texts)))
    playlist_filename = f'{base_folder}/all.m3u'
    playlist = []
    for text, sample_id in zip(texts, sample_ids):
        sample_dir = base_folder / str(sample_id)
        text_file = sample_dir / "text.txt"
        inputs = processor(text, voice_preset=voice_preset)
        inputs = inputs.to(model.device)
        bark_generate(model, **inputs, save_as=sample_dir, min_eos_p=min_eos_p)
        text_file.write_text(text)
        playlist.append(f'{sample_id}/audio.wav')
        save_playlist(playlist, playlist_filename)
        print(f'Generated {sample_id}')


def read_playlist(path: Union[str, PathLike]) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = filter(lambda line: not line.startswith("#"), lines)
        lines = map(lambda line: line.replace("\r", "").replace("\n", ""), lines)
        lines = list(lines)
        return lines


def save_playlist(files_list: List[str], filename: Union[str, PathLike]):
    with open(filename, 'w') as playlist_file:
        for item in files_list:
            playlist_file.write(f'{item}\r\n')


def get_models_devices(model: BarkModel) -> Dict:
    return {
        SEMANTIC_MODEL: model.semantic.device,
        COARSE_MODEL: model.coarse_acoustics.device,
        FINE_MODEL: model.fine_acoustics.device,
        CODEC_MODEL: model.codec_model.device
    }

def set_model_devices(model: BarkModel, model_devices: Dict) -> None:
    model.semantic = model.semantic.to(model_devices[SEMANTIC_MODEL])
    model.coarse_acoustics = model.coarse_acoustics.to(model_devices[COARSE_MODEL])
    model.fine_acoustics = model.fine_acoustics.to(model_devices[FINE_MODEL])
    model.codec_model = model.codec_model.to(model_devices[CODEC_MODEL])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()