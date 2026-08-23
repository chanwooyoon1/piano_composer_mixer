"""Prepare structured V3 token data from the MAESTRO dataset.

This script keeps MAESTRO's official train/validation/test split, maps five
target composers to individual labels, maps every other composer to ``OTHER``,
and converts each MIDI file with ``structured_tokenizer.midi_to_tokens``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from structured_tokenizer import STEPS_PER_BAR, VOCAB_SIZE, midi_to_tokens


TOKENIZER_FORMAT = "bar_chord_position_velocity_pitch_duration_v3"

COMPOSER_MAP = {
    "Frédéric Chopin": 0,
    "Franz Schubert": 1,
    "Ludwig van Beethoven": 2,
    "Johann Sebastian Bach": 3,
    "Franz Liszt": 4,
    "OTHER": 5,
}

REQUIRED_COLUMNS = {
    "canonical_composer",
    "midi_filename",
    "split",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MAESTRO MIDI files to structured V3 tokens."
    )
    parser.add_argument(
        "--maestro-dir",
        type=Path,
        required=True,
        help="Directory containing maestro-v3.0.0.csv and the MIDI files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which token arrays and manifests will be saved.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-tokenize files whose .npy output already exists.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if one MIDI file cannot be processed.",
    )
    parser.add_argument(
        "--create-zip",
        action="store_true",
        help="Create a ZIP archive suitable for uploading to Google Drive.",
    )
    return parser.parse_args()


def find_metadata_csv(maestro_dir: Path) -> Path:
    expected = maestro_dir / "maestro-v3.0.0.csv"
    if expected.is_file():
        return expected

    candidates = sorted(maestro_dir.glob("maestro*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No MAESTRO metadata CSV was found in: {maestro_dir}"
        )
    raise RuntimeError(
        "Multiple MAESTRO metadata CSV files were found. "
        "Keep only maestro-v3.0.0.csv in the dataset directory."
    )


def composer_label(canonical_composer: str) -> tuple[str, int]:
    if canonical_composer in COMPOSER_MAP and canonical_composer != "OTHER":
        return canonical_composer, COMPOSER_MAP[canonical_composer]
    return "OTHER", COMPOSER_MAP["OTHER"]


def output_filename(midi_relative_path: Path) -> str:
    """Create a flat but collision-resistant output filename."""
    year = midi_relative_path.parts[0] if len(midi_relative_path.parts) > 1 else "unknown"
    return f"{year}__{midi_relative_path.stem}.npy"


def prepare_dataset(
    maestro_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
    fail_fast: bool = False,
    create_zip: bool = False,
) -> None:
    maestro_dir = maestro_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not maestro_dir.is_dir():
        raise NotADirectoryError(f"MAESTRO directory does not exist: {maestro_dir}")

    metadata_path = find_metadata_csv(maestro_dir)
    metadata = pd.read_csv(metadata_path)

    missing_columns = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"MAESTRO CSV is missing required columns: {missing}")

    metadata = metadata.sort_values("midi_filename").reset_index(drop=True)

    token_dir = output_dir / "structured_token_data"
    token_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    total = len(metadata)

    for index, row in metadata.iterrows():
        midi_relative_path = Path(str(row["midi_filename"]))
        midi_path = maestro_dir / midi_relative_path
        token_path = token_dir / output_filename(midi_relative_path)
        label_name, composer_id = composer_label(str(row["canonical_composer"]))

        try:
            if not midi_path.is_file():
                raise FileNotFoundError(f"MIDI file does not exist: {midi_path}")

            if token_path.exists() and not overwrite:
                tokens = np.load(token_path, mmap_mode="r")
                num_tokens = int(tokens.shape[0])
            else:
                tokens = midi_to_tokens(midi_path)
                np.save(token_path, tokens)
                num_tokens = int(len(tokens))

            manifest_rows.append(
                {
                    "midi_path": midi_relative_path.as_posix(),
                    "token_path": token_path.relative_to(output_dir).as_posix(),
                    "num_tokens": num_tokens,
                    "split": str(row["split"]),
                    "label_name": label_name,
                    "composer_id": composer_id,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "midi_path": midi_relative_path.as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            if fail_fast:
                raise

        completed = index + 1
        if completed % 50 == 0 or completed == total:
            print(f"Processed {completed}/{total} MIDI files")

    manifest = pd.DataFrame(
        manifest_rows,
        columns=[
            "midi_path",
            "token_path",
            "num_tokens",
            "split",
            "label_name",
            "composer_id",
        ],
    )
    manifest_path = output_dir / "structured_token_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config = {
        "format": TOKENIZER_FORMAT,
        "vocab_size": VOCAB_SIZE,
        "steps_per_bar": STEPS_PER_BAR,
        "chord_classes": "12_major_12_minor_no_chord",
        "composer_map": COMPOSER_MAP,
    }
    config_path = output_dir / "structured_tokenizer_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if failures:
        failure_path = output_dir / "failed_files.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False)
        print(f"Failed files: {len(failures)} (details: {failure_path})")

    print("\nDataset preparation complete")
    print(f"Successful MIDI files: {len(manifest)}")
    print(f"Token directory: {token_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Tokenizer config: {config_path}")

    if create_zip:
        archive_base = output_dir.parent / f"{output_dir.name}_token_data"
        archive_path = Path(
            shutil.make_archive(
                str(archive_base),
                "zip",
                root_dir=output_dir,
            )
        )
        print(f"ZIP archive: {archive_path}")

    if not manifest.empty:
        print("\nFiles by split:")
        print(manifest["split"].value_counts().to_string())
        print("\nFiles by composer label:")
        print(manifest["label_name"].value_counts().to_string())


def main() -> None:
    args = parse_args()
    prepare_dataset(
        maestro_dir=args.maestro_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        fail_fast=args.fail_fast,
        create_zip=args.create_zip,
    )


if __name__ == "__main__":
    main()
