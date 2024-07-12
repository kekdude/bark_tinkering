from pathlib import Path
from transformers import AutoProcessor, BarkModel

lockfile = Path('models/downloaded.lock')

if lockfile.is_file():
    print(f"Models are already downloaded, to re-download delete `{lockfile}` and run the script again")
    exit(0)

print("Downloading stuff, it will take some time")

print("Downloading Bark processor")
processor = AutoProcessor.from_pretrained("suno/bark")
processor.save_pretrained('models/bark-processor')
del processor

print("Downloading Bark model")
big_model = BarkModel.from_pretrained("suno/bark")
big_model.save_pretrained('models/bark')
del big_model

print("Downloading Bark small model")
small_model = BarkModel.from_pretrained("suno/bark-small")
small_model.save_pretrained('models/bark-small')
del small_model

lockfile.touch()