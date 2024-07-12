## Installation

You will need some kind of conda to run it. You can install miniconda from [here](https://docs.anaconda.com/miniconda/miniconda-install/)

* Run `setup.sh` on Linux
* Run `setup.bat` on Windows

## Run

* Run `start.sh` on Linux
* Run `start.bat` on Windows

Check lina.ipynb for example

## How it works?

Bark has 3 models inside, and also uses Encodec model to transform output into audio:
* semantic model - transforms text to intermediate representation called "semantic"
* coarse model - transforms semantic into low bandwidth encodec audio codes
* fine model - upsamples encodec audio codes to higher bandwidth
* endodec decoder - takes audio codes and transforms them to audio

Bark has support for speakers or voice presets. Actually voice preset is just a history generation with previous outputs of all 3 models. Given an audio sample we can already get history for coarse and fine models by simply encoding audio with encodec.

The problem is with semantic. There is no out of the box way to transform audio to matching semantic.

In order to find semantic model output that matches character sample audio one can apply adversarial learning. Sample semantic with help of gumbel softmax, then run this semantic trough Bark model with 2 objectives:
* LogPerplexity loss on semantic model. It measures how much semantic model is surprised seeing our input
* CrossEntropy loss on coarse model. It measures how different is generated audio from original audio

Gradients are back propagated to parameters from which semantic is sampled. Model parameters are not changed trough the process.

Then create voice preset from original audio and adversarial semantic. Usually the first voice preset is unstable. The solution is to make some generations with first voice preset, and then choose a good generation to make second voice preset out of it.