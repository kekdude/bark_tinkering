## Installation

You will need some kind of conda to run it. You can install miniconda from [here](https://docs.anaconda.com/miniconda/miniconda-install/)

Run in terminal setup.sh or setup.bat

## Run

Run in terminal start.sh or start.bat

Check lina.ipynb for example

## How it works?

Bark has 3 models inside, and also uses Encodec model to transform output into audio:
* semantic model - transforms text to intermediate representation called "semantic"
* coarse model - transforms semantic into low quality audio codes
* fine model - upsamples low quality audio
* endodec decoder - takes audio codes and transforms them to audio

Bark has support for speakers or voice presets. Actually voice preset is just a history generation with outputs of all 3 models. Given an audio sample we can already get history for coarse and fine models, by simply encoding audio sample with encodec and taking codes from appropriate channels.

The problem is with semantic. There is no open way to transform audio to matching semantic.

In order to find semantic model output that matches character sample audio, we apply adversarial learning. We sample semantic with help of gumbel softmax, then we run this semantic trough Bark model with 2 objectives:
* LogPerplexity loss on semantic model. It measures how much semantic model is surprised seeing our semantic
* CrossEntropy loss on coarse model. It measures how close is generated audio to original audio

Gradients are back propagated to parameters from which we sample semantic. Model parameters are not changing trough the process.

Then we take original audio sample and found semantic and form voice preset out of it. Usually this first voice preset is unstable. So we make some generations with it, and then choosing one of the generations to make second, more stable voice preset.