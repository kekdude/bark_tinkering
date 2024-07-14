from typing import List

from bark_tinkering.utils import BarkGenerationResult
import IPython


def display_generation_result(generations: BarkGenerationResult | List[BarkGenerationResult]):
    if type(generations) is not list:
        generations = [generations]

    for generation in generations:
        audio = generation.audio.cpu().numpy()
        IPython.display.display(IPython.display.Audio(audio, rate=24000))