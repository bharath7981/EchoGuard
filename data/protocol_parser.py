from dataclasses import dataclass


@dataclass
class Utterance:
    utt_id: str
    attack_id: str
    label: str

    @property
    def binary_label(self):
        # 1 = bonafide, 0 = spoof
        return 1 if self.label == "bonafide" else 0


def parse_protocol(path):
    utterances = []

    with open(path, "r") as file:
        for line_number, line in enumerate(file, 1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 4:
                raise ValueError(f"Invalid line {line_number}: {line}")

            utt_id = parts[1]
            attack_id = parts[-2]
            label = parts[-1]

            if label not in ["bonafide", "spoof"]:
                raise ValueError(f"Invalid label '{label}' at line {line_number}")

            utterances.append(Utterance(utt_id=utt_id, attack_id=attack_id, label=label))

    return utterances


def class_balance(utterances):
    bonafide = sum(1 for item in utterances if item.label == "bonafide")
    spoof = sum(1 for item in utterances if item.label == "spoof")

    return {"bonafide": bonafide, "spoof": spoof, "total": len(utterances)}


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python protocol_parser.py path/to/protocol.txt")
        sys.exit(1)

    data = parse_protocol(sys.argv[1])

    print("Total utterances:", len(data))
    print("Class balance:", class_balance(data))
    print("\nFirst 5 entries:")
    for item in data[:5]:
        print(item)
