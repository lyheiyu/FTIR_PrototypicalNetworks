"""Create exact numeric spectrum deduplications without changing source CSVs."""

import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "dataset" / "deduplicated"


def spectrum_key(values):
    """Canonical decimal values: 1, 1.0 and 1e0 compare equal; no rounding."""
    canonical = []
    for value in values:
        number = Decimal(value)
        if not number.is_finite():
            raise ValueError(f"Non-finite spectral value: {value!r}")
        text = format(number, "f") if number else "0"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        canonical.append(text)
    return hashlib.sha256("\t".join(canonical).encode("ascii")).digest()


def process(source):
    with source.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle)
        header, wavenumbers = next(reader), next(reader)
        rows = list(reader)

    retained, mapping, seen = [], [], {}
    original_counts, retained_counts = Counter(), Counter()
    for csv_row, row in enumerate(rows, start=3):
        if len(row) != len(wavenumbers):
            raise ValueError(f"{source.name}:{csv_row}: unexpected column count")
        label = (row[1], row[2])
        original_counts[label] += 1
        key = spectrum_key(row[3:])
        if key in seen:
            first_row, first_csv_row = seen[key]
            if tuple(first_row[1:3]) != label:
                raise ValueError(f"Conflicting labels for identical spectra: {source.name}:{csv_row}")
            status = "duplicate"
        else:
            first_row, first_csv_row = row, csv_row
            seen[key] = (row, csv_row)
            retained.append(row)
            retained_counts[label] += 1
            status = "retained"
        mapping.append([csv_row, row[0], *label, status, first_csv_row, first_row[0]])

    target = OUTPUT / f"{source.stem}_deduplicated.csv"
    with target.open("w", encoding="latin-1", newline="") as handle:
        csv.writer(handle).writerows([header, wavenumbers, *retained])
    with (OUTPUT / f"{source.stem}_row_mapping.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source_csv_row", "source_id", "polymer", "class_id", "status", "retained_source_csv_row", "retained_source_id"])
        writer.writerows(mapping)

    # Independently reload the artifact and verify its spectra and metadata.
    with target.open(encoding="latin-1", newline="") as handle:
        reader = csv.reader(handle)
        assert next(reader) == header
        assert next(reader) == wavenumbers
        reloaded = list(reader)
    assert reloaded == retained
    assert len({spectrum_key(row[3:]) for row in reloaded}) == len(reloaded)
    return {
        "source": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "output": target.name,
        "original": len(rows),
        "retained": len(retained),
        "removed": len(rows) - len(retained),
        "classes": [
            {"polymer": name, "class_id": class_id, "original": count,
             "retained": retained_counts[(name, class_id)],
             "removed": count - retained_counts[(name, class_id)]}
            for (name, class_id), count in original_counts.items()
        ],
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    for suffix in ("c4", "c8"):
        result = process(ROOT / "dataset" / f"FTIR_PLastics500_{suffix}.csv")
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    (OUTPUT / "summary.json").write_text(json.dumps({
        "method": "Exact numeric equality across all spectral columns using canonical decimal values and SHA-256; keep first occurrence within each file. No rounding or similarity threshold. Conflicting labels abort processing.",
        "preserved": "Original header, wavenumbers, retained row IDs, labels, spectral values and row order.",
        "scope": "C4 and C8 independently; no cross-resolution sample identity inference.",
        "datasets": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
