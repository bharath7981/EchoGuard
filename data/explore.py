import os
import sys

import soundfile as sf

from protocol_parser import parse_protocol, class_balance


def explore(flac_dir, protocol_path, sample_n=200):
    utterances = parse_protocol(protocol_path)

    print(f"Total utterances: {len(utterances)}")
    print("Class balance:", class_balance(utterances))

    durations = []
    sample_rates = set()
    missing = 0

    for item in utterances[:sample_n]:
        path = os.path.join(flac_dir, item.utt_id + ".flac")

        if not os.path.exists(path):
            missing += 1
            continue

        info = sf.info(path)
        sample_rates.add(info.samplerate)
        durations.append(info.duration)

    if missing:
        print(f"\nMissing files: {missing}")

    if durations:
        durations.sort()
        middle = len(durations) // 2

        print(f"\nSampled files: {len(durations)}")
        print(f"Sample rates: {sample_rates}")
        print("Duration:")
        print(f"  Min    : {durations[0]:.2f}s")
        print(f"  Median : {durations[middle]:.2f}s")
        print(f"  Max    : {durations[-1]:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python explore.py /path/to/flac /path/to/protocol.txt")
        sys.exit(1)

    explore(sys.argv[1], sys.argv[2])
