"""Structured MIDI v3: bar-level chord token + position/pitch/duration events."""

from collections import defaultdict
import numpy as np
import pretty_midi


STEPS_PER_BEAT = 4
BEATS_PER_BAR = 4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR
OUTPUT_TEMPO = 120

MIN_PITCH = 21
MAX_PITCH = 108
VELOCITY_BINS = 32
MAX_DURATION_STEPS = 64
NUM_CHORD_TOKENS = 25

PAD_TOKEN = 0
BOS_TOKEN = 1
EOS_TOKEN = 2
BAR_TOKEN = 3
CHORD_START = 4
POSITION_START = CHORD_START + NUM_CHORD_TOKENS
VELOCITY_START = POSITION_START + STEPS_PER_BAR
PITCH_START = VELOCITY_START + VELOCITY_BINS
DURATION_START = PITCH_START + (MAX_PITCH - MIN_PITCH + 1)
VOCAB_SIZE = DURATION_START + MAX_DURATION_STEPS


def velocity_to_bin(velocity):
    return min(VELOCITY_BINS - 1, (int(velocity) - 1) * VELOCITY_BINS // 127)


def bin_to_velocity(velocity_bin):
    value = int((int(velocity_bin) + 0.5) * 127 / VELOCITY_BINS)
    return max(1, min(127, value))


def _beat_times(midi):
    beats = np.asarray(midi.get_beats(), dtype=np.float64)
    end_time = max((n.end for i in midi.instruments for n in i.notes), default=0.0)
    if len(beats) < 2:
        interval = 60.0 / OUTPUT_TEMPO
        beats = np.arange(0.0, end_time + 2 * interval, interval)
    interval = beats[-1] - beats[-2]
    while beats[-1] <= end_time:
        beats = np.append(beats, beats[-1] + interval)
    return beats


def _time_to_grid_step(time_seconds, beats):
    index = int(np.searchsorted(beats, time_seconds, side="right") - 1)
    index = max(0, min(index, len(beats) - 2))
    fraction = (time_seconds - beats[index]) / max(beats[index + 1] - beats[index], 1e-6)
    within = int(np.clip(round(fraction * STEPS_PER_BEAT), 0, STEPS_PER_BEAT))
    return index * STEPS_PER_BEAT + within


def _chord_id(notes, bar_index):
    """Infer the most likely major/minor triad from weighted pitch classes."""
    start = bar_index * STEPS_PER_BAR
    end = start + STEPS_PER_BAR
    histogram = np.zeros(12, dtype=np.float64)

    for pitch, velocity, note_start, note_end in notes:
        overlap = max(0, min(note_end, end) - max(note_start, start))
        if overlap:
            histogram[pitch % 12] += overlap * (0.5 + velocity / 127)

    if histogram.sum() == 0:
        return 24  # NO_CHORD

    best_score = -float("inf")
    best_id = 24

    for root in range(12):
        for quality, intervals in enumerate(((0, 4, 7), (0, 3, 7))):
            members = [(root + interval) % 12 for interval in intervals]
            inside = histogram[members].sum()
            outside = histogram.sum() - inside
            # Root has a small bonus; non-chord notes receive a mild penalty.
            score = inside + 0.25 * histogram[root] - 0.20 * outside
            if score > best_score:
                best_score = score
                best_id = root + quality * 12

    return best_id


def midi_to_tokens(midi_path):
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    beats = _beat_times(midi)
    events_by_step = defaultdict(list)
    note_spans = []
    max_end_step = 0

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            if not MIN_PITCH <= note.pitch <= MAX_PITCH:
                continue
            start = _time_to_grid_step(note.start, beats)
            end = _time_to_grid_step(note.end, beats)
            end = max(start + 1, end)
            duration = min(MAX_DURATION_STEPS, end - start)
            events_by_step[start].append((note.pitch, velocity_to_bin(note.velocity), duration))
            note_spans.append((note.pitch, note.velocity, start, end))
            max_end_step = max(max_end_step, end)

    last_bar = max_end_step // STEPS_PER_BAR
    tokens = [BOS_TOKEN]

    for bar in range(last_bar + 1):
        tokens.append(BAR_TOKEN)
        tokens.append(CHORD_START + _chord_id(note_spans, bar))

        bar_start = bar * STEPS_PER_BAR
        bar_end = bar_start + STEPS_PER_BAR
        for step in sorted(s for s in events_by_step if bar_start <= s < bar_end):
            tokens.append(POSITION_START + (step - bar_start))
            for pitch, velocity_bin, duration in sorted(events_by_step[step]):
                tokens.extend((VELOCITY_START + velocity_bin, PITCH_START + pitch - MIN_PITCH, DURATION_START + duration - 1))

    tokens.append(EOS_TOKEN)
    return np.asarray(tokens, dtype=np.int32)


def tokens_to_midi(tokens):
    midi = pretty_midi.PrettyMIDI(initial_tempo=OUTPUT_TEMPO)
    piano = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    seconds_per_step = 60.0 / OUTPUT_TEMPO / STEPS_PER_BEAT
    current_bar = -1
    current_position = 0
    current_velocity = 64
    pending_pitch = None

    for raw in tokens:
        token = int(raw)
        if token == BAR_TOKEN:
            current_bar += 1
            current_position = 0
            pending_pitch = None
        elif CHORD_START <= token < POSITION_START:
            pending_pitch = None  # chord is conditioning metadata, not an audible note
        elif POSITION_START <= token < VELOCITY_START:
            current_position = token - POSITION_START
            pending_pitch = None
        elif VELOCITY_START <= token < PITCH_START:
            current_velocity = bin_to_velocity(token - VELOCITY_START)
        elif PITCH_START <= token < DURATION_START:
            pending_pitch = MIN_PITCH + token - PITCH_START
        elif DURATION_START <= token < VOCAB_SIZE and pending_pitch is not None:
            duration = token - DURATION_START + 1
            start = (current_bar * STEPS_PER_BAR + current_position) * seconds_per_step
            end = start + duration * seconds_per_step
            piano.notes.append(pretty_midi.Note(velocity=current_velocity, pitch=pending_pitch, start=start, end=end))
            pending_pitch = None

    midi.instruments.append(piano)
    return midi
