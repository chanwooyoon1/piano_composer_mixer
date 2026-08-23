# Piano Composer Mixer

A composer conditioned symbolic piano music generator built from scratch with PyTorch.

The model generates MIDI piano performances while allowing users to interpolate between the learned embeddings of classical composers such as Chopin, Beethoven, Bach, Schubert, and Liszt.

For example, a user can generate music conditioned on:

```text
40% Chopin + 60% Beethoven
```

The percentages interpolate the learned composer embeddings. They do not represent an exact percentage of notes written in each composer's style.

## Overview

This project explores whether a relatively small Transformer trained from scratch can learn recognizable structure from classical piano MIDI data.

The complete pipeline includes:

1. Preparing the MAESTRO MIDI dataset
2. Assigning composer labels
3. Converting MIDI performances into structured tokens
4. Training a causal Transformer
5. Fine tuning with pitch transposition augmentation
6. Mixing composer condition embeddings
7. Generating MIDI and optional WAV audio

No pretrained music generation model was used.

## Model Architecture

The project uses a decoder only causal Transformer implemented directly in PyTorch.

| Component | Configuration |
|---|---:|
| Transformer layers | 8 |
| Attention heads | 8 |
| Embedding dimension | 512 |
| Feed forward dimension | 2048 |
| Context length | 4096 tokens |
| Dropout | 0.10 |
| Vocabulary size | 229 |
| Composer classes | 6 |
| Parameters | Approximately 25.3 million |
| Positional encoding | Rotary Position Embedding |
| Attention | Causal self attention |

A learned composer embedding is added to every token representation. During mixed composer generation, a weighted average of multiple composer embeddings is used.

## Structured MIDI Representation

Each MIDI performance is converted into a structured event sequence.

The tokenizer represents:

* Bar boundaries
* Bar level chord estimates
* Position within a bar
* Velocity
* Piano pitch
* Note duration

The vocabulary contains:

| Token type | Count |
|---|---:|
| Special tokens | 4 |
| Chord tokens | 25 |
| Bar positions | 16 |
| Velocity bins | 32 |
| Piano pitches | 88 |
| Duration bins | 64 |
| Total vocabulary | 229 |

The 25 chord classes contain 12 major chords, 12 minor chords, and one no chord class.

## Dataset

The model was trained using the [MAESTRO v3.0.0 dataset](https://magenta.withgoogle.com/datasets/maestro).

The dataset contains 1,276 aligned classical piano MIDI performances. The official MAESTRO train, validation, and test splits are preserved.

| Split | Performances |
|---|---:|
| Training | 962 |
| Validation | 137 |
| Test | 177 |
| Total | 1,276 |

Composer conditioning uses five individual composers and one additional category.

| Composer label | Performances |
|---|---:|
| Frédéric Chopin | 201 |
| Franz Schubert | 186 |
| Ludwig van Beethoven | 146 |
| Johann Sebastian Bach | 145 |
| Franz Liszt | 131 |
| Other composers | 467 |

The MAESTRO dataset itself is not included in this repository.

## Training

Initial training used:

* AdamW optimizer
* Learning rate of `3e-4`
* Cross entropy loss
* Mixed precision training
* Gradient clipping at `1.0`
* 20 initial epochs
* Padding tokens excluded from the loss

The model was then fine tuned with pitch transposition augmentation.

Training sequences were randomly transposed by up to five semitones while preserving the valid piano range. Chord roots were transposed consistently with the note events.

The validation data was not augmented.

| Stage | Best validation loss |
|---|---:|
| Initial structured training | 1.7570 |
| Transposition fine tuning | 1.6361 |

Training was performed in Google Colab with a CUDA GPU.

## Composer Mixing

The model learns one embedding for each composer class.

For mixed generation, the condition vector is calculated as:

```text
mixed embedding =
    weight₁ × composer embedding₁
  + weight₂ × composer embedding₂
  + ...
```

This makes combinations such as the following possible:

```python
composer_mix = {
    "Frédéric Chopin": 40,
    "Ludwig van Beethoven": 60,
}
```

Composer mixing is an interpolation in the model's learned embedding space. It does not guarantee a precise musicological division between composer styles.

## Repository Structure

```text
piano_composer_mixer/
├── models/
│   └── README.md
├── notebooks/
│   └── piano_composer_mixer.ipynb
├── samples/
│   ├── chopin100.mid
│   ├── chopin100.wav
│   ├── beethoven100.mid
│   ├── beethoven100.wav
│   ├── mixed_chopin40_beethoven60.mid
│   └── mixed_chopin40_beethoven60.wav
├── src/
│   ├── music_generator.py
│   ├── music_model.py
│   ├── prepare_dataset.py
│   └── structured_tokenizer.py
└── README.md
```

## Generated Samples

| Condition | MIDI | Audio |
|---|---|---|
| 100% Chopin | [Download MIDI](samples/chopin100.mid) | [Download WAV](samples/chopin100.wav) |
| 100% Beethoven | [Download MIDI](samples/beethoven100.mid) | [Download WAV](samples/beethoven100.wav) |
| 40% Chopin and 60% Beethoven | [Download MIDI](samples/mixed_chopin40_beethoven60.mid) | [Download WAV](samples/mixed_chopin40_beethoven60.wav) |

## Installation

Clone the repository:

```bash
git clone https://github.com/chanwooyoon1/piano_composer_mixer.git
cd piano_composer_mixer
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies:

```bash
pip install torch numpy pandas pretty_midi
```

FluidSynth is required only when converting MIDI output into WAV audio.

On macOS:

```bash
brew install fluid-synth
```

## Pretrained Checkpoint

Download the pretrained checkpoint from the repository's [Releases page](https://github.com/chanwooyoon1/piano_composer_mixer/releases/latest).

Place it at:

```text
models/best_structured_v3_piano_transformer_augmented.pt
```

The checkpoint contains the trained model parameters, architecture configuration, composer map, and tokenizer metadata.

## Generating Music

Open:

```text
src/music_generator.py
```

Set the desired composer weights and output name. For example:

```python
composer_mix = {
    "Frédéric Chopin": 40,
    "Franz Schubert": 0,
    "Ludwig van Beethoven": 60,
    "Johann Sebastian Bach": 0,
    "Franz Liszt": 0,
    "OTHER": 0,
}
```

Then run:

```bash
python src/music_generator.py
```

Generation automatically uses:

1. CUDA when available
2. Apple Metal Performance Shaders on supported Macs
3. CPU as a fallback

The script produces a structured token file and a MIDI file. It can also produce WAV audio when FluidSynth and a compatible SoundFont are available.

## Preparing the Dataset

Download and extract MAESTRO v3.0.0, then run:

```bash
python src/prepare_dataset.py \
    --maestro-dir /path/to/maestro-v3.0.0 \
    --output-dir structured_piano_v3 \
    --create-zip
```

This creates:

```text
structured_piano_v3/
├── structured_token_data/
├── structured_token_manifest.csv
├── structured_tokenizer_config.json
└── structured_tokenizer.py
```

It also creates:

```text
structured_piano_v3_token_data.zip
```

For the included Colab notebook, upload the ZIP file to:

```text
MyDrive/colab_workspace/data/structured_piano_v3_token_data.zip
```

## Training in Google Colab

Open:

```text
notebooks/piano_composer_mixer.ipynb
```

Select a GPU runtime and run the cells in order.

The notebook performs:

1. Google Drive mounting
2. Token dataset extraction
3. DataLoader construction
4. Model initialization
5. Initial training
6. Transposition augmented fine tuning
7. Composer conditioned generation

## Generation Constraints

Generation uses a state based token grammar:

```text
BAR → CHORD → POSITION → VELOCITY → PITCH → DURATION
```

Invalid token types are masked at each state. Additional constraints reduce excessive repetition and unrealistically long bass notes.

Sampling uses temperature and top k sampling, so different random seeds can produce different results from the same composer mixture.

## Limitations

* Long range musical form is still limited.
* Some generations contain repeated rhythmic or melodic patterns.
* Abrupt transitions can occur between phrases.
* Composer percentages interpolate embeddings and should not be interpreted as exact measurements of style.
* Generated music should be evaluated as an experimental model output rather than an authentic composition by a historical composer.

## Future Work

Potential improvements include:

* Better long term musical structure
* Relative attention across sections
* More detailed pedal representation
* Key and tempo conditioning
* Larger and more balanced composer datasets
* Quantitative evaluation of composer conditioning
* An interactive interface for selecting composer mixtures

## Acknowledgements

This project uses the MAESTRO dataset created by the Magenta team.

Dataset and SoundFont files remain subject to their respective licenses and terms.
