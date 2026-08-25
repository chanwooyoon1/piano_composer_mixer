# Piano Composer Mixer

This project generates short piano compositions and allows different composer conditions to be mixed together.

For example, the model can generate music using a condition such as 40% Chopin and 60% Beethoven.

## Why I made this project

I originally tried to generate lo-fi jazz directly from mel spectrograms using a diffusion model. Although the model produced recognizable sound textures, the output was mostly noisy and did not have much musical structure.

I decided to switch from raw audio generation to symbolic MIDI generation. MIDI made it easier for the model to learn notes, rhythm, chords, and note duration without also having to learn audio reconstruction.

My main question was whether a Transformer trained from scratch on a relatively small classical piano dataset could generate meaningful musical patterns and interpolate between composer conditions.

## How it works

I created a structured MIDI tokenizer that converts each performance into tokens representing:

* Bar boundaries
* Chords
* Positions within a bar
* Velocity
* Piano pitch
* Note duration

The model is a causal Transformer implemented in PyTorch. It predicts the next token from all previous tokens.

The current model has:

* 8 Transformer layers
* 8 attention heads
* 512 dimensional embeddings
* Rotary position embeddings
* A context length of 4096 tokens
* Approximately 25.3 million parameters

I did not use a pretrained music generation model.

## Dataset

I trained the model using 1,276 MIDI performances from the [MAESTRO v3.0.0 dataset](https://magenta.withgoogle.com/datasets/maestro).

I used six composer labels:

* Frédéric Chopin
* Franz Schubert
* Ludwig van Beethoven
* Johann Sebastian Bach
* Franz Liszt
* Other composers

The official MAESTRO train, validation, and test splits were preserved.

## Mixing composers

The model learns a separate embedding for each composer label.

To mix composers, I calculate a weighted average of their embeddings. For example:

```python
composer_mix = {
    "Frédéric Chopin": 40,
    "Ludwig van Beethoven": 60,
}
```

This does not mean that exactly 40% of the notes are written like Chopin. The percentages control the interpolation between the learned conditioning vectors.

## Training

I first trained the model normally for 20 epochs. I then continued training with pitch transposition augmentation, randomly moving training sequences by up to five semitones.

The final model achieved a best validation loss of `1.6361`.

Training was performed with PyTorch on a Google Colab GPU.

## Results

The model can generate short passages containing recognizable melodies, chords, and rhythmic patterns. Changing the composer condition also changes the character of the generated output.

Generated MIDI and WAV examples are available in the [`samples`](samples) folder.

The output is not always completely musical. Some generations contain repeated patterns, awkward transitions, or weak long term structure. Improving the structure of longer compositions is one of the main areas I would like to explore next.

## Repository contents

```text
models/       Instructions for downloading the trained checkpoint
notebooks/    Google Colab training notebook
samples/      Generated MIDI and WAV examples
src/          Model, tokenizer, dataset preparation, and generation code
```

## Running the generator

Install the required Python packages:

```bash
pip install torch numpy pandas pretty_midi
```

Download the trained checkpoint from the repository's [Releases page](https://github.com/chanwooyoon1/piano_composer_mixer/releases/latest) and place it at:

```text
models/best_structured_v3_piano_transformer_augmented.pt
```

Then edit the composer weights in `src/music_generator.py` and run:

```bash
python src/music_generator.py
```

The generator automatically uses CUDA, Apple MPS, or CPU depending on the available hardware.

FluidSynth and a SoundFont are required only if the generated MIDI should also be rendered as WAV audio.

## Training the model

The complete training process is included in:

```text
notebooks/piano_composer_mixer.ipynb
```

The notebook is designed to be run from top to bottom in Google Colab with a GPU runtime.

The MAESTRO dataset can be prepared using:

```bash
python src/prepare_dataset.py \
    --maestro-dir /path/to/maestro-v3.0.0 \
    --output-dir structured_piano_v3 \
    --create-zip
```

## Generate Music Locally

The pretrained model can generate MIDI and WAV files on a local computer. These instructions are written for macOS.

Run this project from Terminal rather than IDLE. IDLE may use a different Python installation that does not contain PyTorch.

### 1. Clone the repository

```bash
git clone https://github.com/chanwooyoon1/piano_composer_mixer.git
cd piano_composer_mixer
```

### 2. Create a Python environment

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

When the environment is active, the Terminal prompt should begin with `(.venv)`.

### 3. Install the required Python packages

```bash
python -m pip install torch numpy pretty_midi
```

Verify that PyTorch is installed:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available())"
```

On an Apple Silicon Mac, the second value should normally be `True`. The generator automatically uses MPS when it is available and otherwise uses the CPU.

### 4. Install FluidSynth

FluidSynth is required to convert generated MIDI files into WAV audio.

```bash
brew install fluid-synth
```

Verify the installation:

```bash
which fluidsynth
fluidsynth --version
```

### 5. Download the model checkpoint

Download the pretrained checkpoint from the repository's Releases page:

https://github.com/chanwooyoon1/piano_composer_mixer/releases

Create a `models` directory in the repository and place the downloaded checkpoint at:

```text
models/best_structured_v3_piano_transformer_augmented.pt
```

The checkpoint filename must exactly match the filename above. If the downloaded file has a different name, rename it to:

```text
best_structured_v3_piano_transformer_augmented.pt
```

The project should now contain:

```text
piano_composer_mixer/
├── models/
│   └── best_structured_v3_piano_transformer_augmented.pt
├── src/
│   ├── music_generator.py
│   └── music_model.py
└── README.md
```

### 6. Add a SoundFont

A SoundFont is required to render the generated MIDI as WAV audio.

Create a `soundfonts` directory and place a compatible SoundFont at:

```text
soundfonts/MuseScore_General.sf3
```

The project should now contain:

```text
piano_composer_mixer/
├── models/
│   └── best_structured_v3_piano_transformer_augmented.pt
├── soundfonts/
│   └── MuseScore_General.sf3
├── src/
│   ├── music_generator.py
│   └── music_model.py
└── README.md
```

If a different SoundFont is used, update `SOUNDFONT_PATH` in `src/music_generator.py`.

### 7. Choose the composer mixture

Open `src/music_generator.py` and find the following section near the bottom:

```python
if __name__ == "__main__":
    generator = PianoMusicGenerator()

    generator.generate(
        composer_mix={
            "Frédéric Chopin": 0,
            "Franz Schubert": 0,
            "Ludwig van Beethoven": 0,
            "Johann Sebastian Bach": 40,
            "Franz Liszt": 60,
            "OTHER": 0,
        },
        output_name="bach40_liszt60",
        max_bars=16,
        max_new_tokens=3500,
        temperature=0.90,
        top_k=24,
        bpm=120,
        seed=1234,
    )
```

Change the composer values to generate a different mixture. For example, the following settings use 40 percent Chopin and 60 percent Beethoven:

```python
composer_mix={
    "Frédéric Chopin": 40,
    "Franz Schubert": 0,
    "Ludwig van Beethoven": 60,
    "Johann Sebastian Bach": 0,
    "Franz Liszt": 0,
    "OTHER": 0,
}
```

The values are normalized automatically, but using a total of 100 makes the mixture easier to understand.

The other generation settings can also be changed:

```text
output_name     Name used for the generated files
max_bars        Maximum number of musical bars
max_new_tokens  Maximum number of generated tokens
temperature     Randomness of generation
top_k           Number of candidate tokens considered at each step
bpm             Tempo of the generated MIDI
seed            Random seed used for generation
```

### 8. Generate music

Make sure the virtual environment is active:

```bash
source .venv/bin/activate
```

Run the generator from the repository root:

```bash
python src/music_generator.py
```

The model may take some time to generate music, especially when running on the CPU.

The generated files are saved in the `generated` directory:

```text
generated/
├── bach40_liszt60.mid
├── bach40_liszt60.wav
└── bach40_liszt60_tokens.npy
```

The MIDI file can be opened in GarageBand, Logic Pro, MuseScore, or another MIDI-compatible application. The WAV file can be played using a standard audio player.

### Generate MIDI without WAV rendering

If FluidSynth or a SoundFont is not available, add the following argument to `generator.generate`:

```python
render_wav=False
```

For example:

```python
generator.generate(
    composer_mix={
        "Frédéric Chopin": 40,
        "Franz Schubert": 0,
        "Ludwig van Beethoven": 60,
        "Johann Sebastian Bach": 0,
        "Franz Liszt": 0,
        "OTHER": 0,
    },
    output_name="chopin40_beethoven60",
    max_bars=16,
    max_new_tokens=3500,
    temperature=0.90,
    top_k=24,
    bpm=120,
    seed=1234,
    render_wav=False,
)
```

This creates the MIDI and token files without creating a WAV file.

### Troubleshooting

If the following error appears:

```text
ModuleNotFoundError: No module named 'torch'
```

activate the virtual environment and install PyTorch using the same Python interpreter:

```bash
source .venv/bin/activate
python -m pip install torch
python src/music_generator.py
```

Confirm that the correct Python installation is being used:

```bash
which python
python -c "import torch; print(torch.__version__)"
```

If FluidSynth is not found, run:

```bash
brew install fluid-synth
which fluidsynth
```

If the checkpoint is not found, confirm that it is located at:

```text
models/best_structured_v3_piano_transformer_augmented.pt
```

If the SoundFont is not found, confirm that it is located at:

```text
soundfonts/MuseScore_General.sf3
```

## AI Assistance

AI tools were used for debugging support, code review, and documentation editing. I designed the project, prepared the dataset, implemented and trained the model, evaluated the generated results, and made the final technical decisions.
