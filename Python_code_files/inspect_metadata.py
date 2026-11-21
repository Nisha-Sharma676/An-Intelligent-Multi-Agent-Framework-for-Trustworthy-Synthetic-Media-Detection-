import json
from pathlib import Path

metadata_path = Path(r"D:\LAV-DF\LAV-DF\metadata.json")

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

print("=" * 65)
print("INSPECTING LAV-DF METADATA FILENAMES")
print("=" * 65)

print("Total records:", len(metadata))

print("\nFirst 10 metadata records:")
for index, record in enumerate(metadata[:10]):
    print(f"\nRecord {index + 1}")
    print("file:", repr(record.get("file")))
    print("split:", repr(record.get("split")))
    print("duration:", repr(record.get("duration")))
    print("fake_periods:", repr(record.get("fake_periods")))

print("\n" + "-" * 65)
print("Records containing test_000001")
print("-" * 65)

matches = [
    record for record in metadata
    if "test_000001" in str(record.get("file", "")).lower()
]

print("Matches:", len(matches))

for record in matches[:10]:
    print("file:", repr(record.get("file")))
    print("split:", repr(record.get("split")))
    print("duration:", repr(record.get("duration")))
    print("fake_periods:", repr(record.get("fake_periods")))

print("\nInspection complete. No files were modified.")