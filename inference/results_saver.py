import json
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent

def save_results(results, json_filename, csv_filename):
    json_path = HERE.parent / "results" / json_filename
    csv_path = HERE.parent / "results" / csv_filename

    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")