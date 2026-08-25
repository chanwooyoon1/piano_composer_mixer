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

## What I learned

The biggest lesson from this project was that the representation of music matters as much as the model architecture.

Generating raw audio required the model to learn both musical structure and sound reconstruction. Using structured MIDI tokens allowed the Transformer to focus more directly on pitch, rhythm, harmony, and duration.

This project also gave me experience with dataset preparation, custom tokenization, Transformer implementation, GPU training, sampling constraints, data augmentation, and model deployment on Apple Silicon.

## AI Assistance

AI tools were used for debugging support, code review, documentation editing, and exploring implementation alternatives. I designed the project, prepared the dataset, implemented and trained the model, evaluated the generated results, and made the final technical decisions.
