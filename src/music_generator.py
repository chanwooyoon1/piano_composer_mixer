from pathlib import Path
import shutil
import subprocess

import numpy as np
import pretty_midi
import torch

from music_model import load_music_model


# =========================================================
# Paths
# =========================================================

# Repository root
PROJECT_DIR = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "best_structured_v3_piano_transformer_augmented.pt"
)

OUTPUT_DIR = PROJECT_DIR / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOUNDFONT_PATH = (
    PROJECT_DIR
    / "soundfonts"
    / "MuseScore_General.sf3"
)

FLUIDSYNTH_PATH = shutil.which("fluidsynth")

if FLUIDSYNTH_PATH is None:
    FLUIDSYNTH_PATH = "/opt/homebrew/bin/fluidsynth"
# =========================================================
# V3 token definitions
# =========================================================

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BAR_TOKEN = 3

CHORD_START = 4
NUM_CHORD_TOKENS = 25

POSITION_START = CHORD_START + NUM_CHORD_TOKENS
VELOCITY_START = POSITION_START + 16
PITCH_START = VELOCITY_START + 32
DURATION_START = PITCH_START + 88

VOCAB_SIZE = 229
CONTEXT_LENGTH = 4096


# =========================================================
# MIDI conversion
# =========================================================

def tokens_to_midi_v3(tokens, bpm=120):
    midi = pretty_midi.PrettyMIDI(
        initial_tempo=float(bpm)
    )

    piano = pretty_midi.Instrument(
        program=0,
        name="Acoustic Grand Piano",
    )

    # 16分音符単位
    seconds_per_step = 60.0 / bpm / 4.0

    current_bar = -1
    current_position = 0
    current_velocity = 64
    pending_pitch = None

    for raw_token in tokens:
        token = int(raw_token)

        if token == BAR_TOKEN:
            current_bar += 1
            current_position = 0
            pending_pitch = None

        elif CHORD_START <= token < POSITION_START:
            # コードトークンは構造条件として使用する。
            # MIDIノートには直接変換しない。
            pass

        elif POSITION_START <= token < VELOCITY_START:
            current_position = token - POSITION_START
            pending_pitch = None

        elif VELOCITY_START <= token < PITCH_START:
            velocity_bin = token - VELOCITY_START

            current_velocity = max(
                1,
                min(
                    127,
                    int((velocity_bin + 0.5) * 127 / 32),
                ),
            )

        elif PITCH_START <= token < DURATION_START:
            pending_pitch = (
                21 + token - PITCH_START
            )

        elif DURATION_START <= token < VOCAB_SIZE:
            if pending_pitch is not None and current_bar >= 0:
                duration_steps = (
                    token - DURATION_START + 1
                )

                start_step = (
                    current_bar * 16
                    + current_position
                )

                start_time = (
                    start_step * seconds_per_step
                )

                end_time = (
                    start_time
                    + duration_steps * seconds_per_step
                )

                piano.notes.append(
                    pretty_midi.Note(
                        velocity=current_velocity,
                        pitch=pending_pitch,
                        start=start_time,
                        end=end_time,
                    )
                )

                pending_pitch = None

    midi.instruments.append(piano)
    return midi


def midi_to_wav(
    midi_path,
    wav_path,
    soundfont_path=SOUNDFONT_PATH,
    fluidsynth_path=FLUIDSYNTH_PATH,
    sample_rate=44100,
    gain=0.8,
):
    midi_path = Path(midi_path)
    wav_path = Path(wav_path)
    soundfont_path = Path(soundfont_path)
    fluidsynth_path = Path(fluidsynth_path)

    if not midi_path.is_file():
        raise FileNotFoundError(
            f"MIDI file was not found: {midi_path}"
        )

    if not soundfont_path.is_file():
        raise FileNotFoundError(
            f"SoundFont was not found: {soundfont_path}"
        )

    if not fluidsynth_path.is_file():
        raise FileNotFoundError(
            f"FluidSynth was not found: {fluidsynth_path}"
        )

    wav_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        str(fluidsynth_path),
        "-ni",
        "-g",
        str(float(gain)),
        "-r",
        str(int(sample_rate)),
        "-F",
        str(wav_path),
        str(soundfont_path),
        str(midi_path),
    ]

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        error_message = (
            error.stderr.strip()
            or error.stdout.strip()
            or str(error)
        )

        raise RuntimeError(
            f"FluidSynth failed: {error_message}"
        ) from error

    if not wav_path.is_file():
        raise RuntimeError(
            f"WAV file was not created: {wav_path}"
        )

    return wav_path


# =========================================================
# Model helpers
# =========================================================

def restrict_to(logits, allowed_tokens):
    mask = torch.zeros_like(
        logits,
        dtype=torch.bool,
    )

    mask[allowed_tokens] = True

    return logits.masked_fill(
        ~mask,
        -float("inf"),
    )


def forward_with_composer_mix(
    model,
    token_ids,
    composer_mix,
    composer_map,
):
    total_weight = sum(
        max(0.0, float(weight))
        for weight in composer_mix.values()
    )

    if total_weight <= 0:
        raise ValueError(
            "The total composer weight must be greater than zero."
        )

    mixed_embedding = torch.zeros(
        model.composer_embedding.embedding_dim,
        device=token_ids.device,
        dtype=model.token_embedding.weight.dtype,
    )

    for composer_name, raw_weight in composer_mix.items():
        weight = max(0.0, float(raw_weight))

        if weight == 0:
            continue

        if composer_name not in composer_map:
            raise ValueError(
                f"Unknown composer: {composer_name}"
            )

        composer_id = composer_map[composer_name]
        normalized_weight = weight / total_weight

        mixed_embedding += (
            normalized_weight
            * model.composer_embedding.weight[composer_id]
        )

    x = model.token_embedding(token_ids)

    mixed_embedding = mixed_embedding[
        None,
        None,
        :,
    ]

    x = model.dropout(
        x + mixed_embedding
    )

    for block in model.blocks:
        x = block(x)

    x = model.norm(x)
    return model.lm_head(x)


def sample_token(logits, top_k):
    finite_count = int(
        torch.isfinite(logits).sum().item()
    )

    if finite_count == 0:
        raise RuntimeError(
            "No valid token is available."
        )

    actual_top_k = min(
        int(top_k),
        finite_count,
    )

    top_values, top_indices = torch.topk(
        logits,
        actual_top_k,
    )

    probabilities = torch.softmax(
        top_values.float(),
        dim=-1,
    )

    # CPU側で選択するとMPSでも安定して動く
    sampled_position = torch.multinomial(
        probabilities.cpu(),
        num_samples=1,
    ).item()

    return int(
        top_indices[sampled_position].item()
    )


# =========================================================
# Generator
# =========================================================

class PianoMusicGenerator:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.model, self.checkpoint, self.device = (
            load_music_model(checkpoint_path)
        )

        self.composer_map = self.checkpoint[
            "composer_map"
        ]

        self.context_length = self.checkpoint.get(
            "context_length",
            CONTEXT_LENGTH,
        )

        print(
            "Available composers:",
            list(self.composer_map.keys()),
        )

    def generate(
        self,
        composer_mix,
        output_name="generated_mix",
        max_bars=2,
        max_new_tokens=600,
        temperature=0.90,
        top_k=24,
        bpm=120,
        seed=1234,
        render_wav=True,
    ):
        if temperature <= 0:
            raise ValueError(
                "Temperature must be greater than zero."
            )

        torch.manual_seed(seed)

        tokens = [BOS_TOKEN]
        recent_pitches = []

        pending_pitch_midi = None
        state = "expect_bar"
        bars_generated = 0

        self.model.eval()

        print("Composer mix:", composer_mix)
        print("Generating on:", self.device)

        with torch.inference_mode():
            for generation_step in range(max_new_tokens):
                context = torch.tensor(
                    [
                        tokens[
                            -self.context_length:
                        ]
                    ],
                    dtype=torch.long,
                    device=self.device,
                )

                # MPSではまずfloat32で確実に実行する
                logits = forward_with_composer_mix(
                    self.model,
                    context,
                    composer_mix,
                    self.composer_map,
                )[0, -1]

                logits = (
                    logits.float()
                    / float(temperature)
                )

                if state == "expect_bar":
                    allowed = torch.tensor(
                        [BAR_TOKEN],
                        dtype=torch.long,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                elif state == "expect_chord":
                    allowed = torch.arange(
                        CHORD_START,
                        POSITION_START,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                elif state == "expect_position":
                    allowed = torch.arange(
                        POSITION_START,
                        VELOCITY_START,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                elif state == "expect_velocity":
                    allowed = torch.arange(
                        VELOCITY_START,
                        PITCH_START,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                elif state == "expect_pitch":
                    allowed = torch.arange(
                        PITCH_START,
                        DURATION_START,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                    # 直近12音と同じ音程を少し出にくくする
                    for recent_pitch in recent_pitches[-12:]:
                        token_id = (
                            PITCH_START
                            + recent_pitch
                            - 21
                        )

                        logits[token_id] -= 0.65

                elif state == "expect_duration":
                    # 通常音は最大16ステップ
                    max_duration = 16

                    # 低音が長時間伸び続けるのを防ぐ
                    if (
                        pending_pitch_midi is not None
                        and pending_pitch_midi < 40
                    ):
                        max_duration = 8

                    allowed = torch.arange(
                        DURATION_START,
                        DURATION_START + max_duration,
                        device=self.device,
                    )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                elif state == "after_event":
                    if bars_generated >= max_bars:
                        allowed = torch.tensor(
                            [EOS_TOKEN],
                            dtype=torch.long,
                            device=self.device,
                        )

                    else:
                        allowed = torch.cat(
                            [
                                torch.tensor(
                                    [BAR_TOKEN],
                                    dtype=torch.long,
                                    device=self.device,
                                ),
                                torch.arange(
                                    POSITION_START,
                                    VELOCITY_START,
                                    device=self.device,
                                ),
                                torch.arange(
                                    VELOCITY_START,
                                    PITCH_START,
                                    device=self.device,
                                ),
                            ]
                        )

                    logits = restrict_to(
                        logits,
                        allowed,
                    )

                next_token = sample_token(
                    logits,
                    top_k,
                )

                tokens.append(next_token)

                if next_token == BAR_TOKEN:
                    bars_generated += 1
                    state = "expect_chord"

                    print(
                        f"Bar {bars_generated}/{max_bars}"
                    )

                elif (
                    CHORD_START
                    <= next_token
                    < POSITION_START
                ):
                    state = "expect_position"

                elif (
                    POSITION_START
                    <= next_token
                    < VELOCITY_START
                ):
                    state = "expect_velocity"

                elif (
                    VELOCITY_START
                    <= next_token
                    < PITCH_START
                ):
                    state = "expect_pitch"

                elif (
                    PITCH_START
                    <= next_token
                    < DURATION_START
                ):
                    pending_pitch_midi = (
                        21
                        + next_token
                        - PITCH_START
                    )

                    recent_pitches.append(
                        pending_pitch_midi
                    )

                    state = "expect_duration"

                elif (
                    DURATION_START
                    <= next_token
                    < VOCAB_SIZE
                ):
                    pending_pitch_midi = None
                    state = "after_event"

                elif next_token == EOS_TOKEN:
                    break

                if (generation_step + 1) % 50 == 0:
                    print(
                        "Generated tokens:",
                        generation_step + 1,
                    )

        tokens_array = np.asarray(
            tokens,
            dtype=np.int32,
        )

        safe_name = Path(output_name).stem

        token_path = (
            OUTPUT_DIR
            / f"{safe_name}_tokens.npy"
        )

        midi_path = (
            OUTPUT_DIR
            / f"{safe_name}.mid"
        )

        wav_path = (
            OUTPUT_DIR
            / f"{safe_name}.wav"
        )

        np.save(
            token_path,
            tokens_array,
        )

        midi = tokens_to_midi_v3(
            tokens_array,
            bpm=bpm,
        )

        midi.write(
            str(midi_path)
        )

        if render_wav:
            print("Rendering WAV with FluidSynth...")

            midi_to_wav(
                midi_path=midi_path,
                wav_path=wav_path,
            )

        else:
            wav_path = None

        print("Generated bars:", bars_generated)
        print("Generated tokens:", len(tokens_array))
        print("Token file:", token_path)
        print("MIDI file:", midi_path)

        if wav_path is not None:
            print("WAV file:", wav_path)

        return {
            "tokens": tokens_array,
            "token_path": token_path,
            "midi_path": midi_path,
            "wav_path": wav_path,
            "bars": bars_generated,
        }


# =========================================================
# Local test
# =========================================================

if __name__ == "__main__":
    generator = PianoMusicGenerator()

    generator.generate(
        composer_mix={
            "Frédéric Chopin": 0,
            "Franz Schubert": 0,
            "Ludwig van Beethoven": 0,
            "Johann Sebastian Bach": 0,
            "Franz Liszt": 100,
            "OTHER": 0,
        },
        output_name="liszt100",
        max_bars=16,
        max_new_tokens=3500,
        temperature=0.90,
        top_k=24,
        bpm=120,
        seed=1234,
    )
