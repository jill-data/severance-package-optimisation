from pathlib import Path

paths = [
    './output/feature_distribution',
    './output/severance_offer'
]

for path in paths:
    Path(path).mkdir(parents=True, exist_ok=True)
