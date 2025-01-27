# Midi-Model

## Midi event transformer for music generation

![](./banner.png)

# MIDI Composer Demo

This repository contains a demo for a MIDI composer model. The setup instructions below will guide you through the process of setting up the environment and running the demo.

## Setup Environment

1. **Clone the Repository**:
   First, clone the repository from Hugging Face Spaces:

   ```bash
   git clone https://huggingface.co/spaces/skytnt/midi-composer
   ```

2. **Navigate to the Repository Directory**:
   Change to the cloned repository directory:

   ```bash
   cd midi-composer
   ```

3. **Install Required Packages**:
   Install the necessary system packages and Python dependencies:

   ```bash
   sudo apt-get update && sudo apt install fluidsynth
   pip install -r requirements.txt
   pip install spaces
   ```

## Run the Demo

Once the environment is set up, you can run the demo using the following command:

```bash
python app.py --max-gen 4096 --share
```

This command will start the application with a maximum generation limit of 4096 and share the application publicly.



## Updates
- v1.3: MIDITokenizerV2 and new MidiVisualizer
- v1.2 : Optimise the tokenizer and dataset. The dataset was filtered by MIDITokenizer.check_quality. Using the higher quality dataset to train the model, the performance of the model is significantly improved.

## Demo

- [online: huggingface](https://huggingface.co/spaces/skytnt/midi-composer)

- [online: colab](https://colab.research.google.com/github/SkyTNT/midi-model/blob/main/demo.ipynb)

- [download windows app](https://github.com/SkyTNT/midi-model/releases)

## Pretrained model

[huggingface](https://huggingface.co/skytnt/midi-model-tv2o-medium)

## Dataset

[projectlosangeles/Los-Angeles-MIDI-Dataset](https://huggingface.co/datasets/projectlosangeles/Los-Angeles-MIDI-Dataset)

## Requirements

- install [pytorch](https://pytorch.org/)(recommend pytorch>=2.0)
- install [fluidsynth](https://www.fluidsynth.org/)>=2.0.0
- `pip install -r requirements.txt`

## Run app

`python app.py`

## Train 

`python train.py`
 
## Citation

```bibtex
@misc{skytnt2024midimodel,
  author = {SkyTNT},
  title = {Midi Model: Midi event transformer for symbolic music generation},
  year = {2024},
  howpublished = {\url{https://github.com/SkyTNT/midi-model}},
}
