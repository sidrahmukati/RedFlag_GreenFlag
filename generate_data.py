import numpy as np
import pandas as pd
from schema import PILLAR_COLUMNS

def generate_synthetic_data(n_samples: int = 2000, 
                            random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    data = rng.uniform(-1.0, 1.0, size=(n_samples, len(PILLAR_COLUMNS)))

    for i in range(n_samples):
        n_active = rng.integers(1, 5)
        inactive_idx = rng.choice(
            len(PILLAR_COLUMNS), size=len(PILLAR_COLUMNS) - n_active, replace=False
        )
        data[i, inactive_idx] = 0.0

    df = pd.DataFrame(np.round(data, 2), columns=PILLAR_COLUMNS)    

    targets = []
    for _, row in df.iterrows():
        #Rule A
        if  (
            row["safety_catchall"] < -0.65 
            or row["boundary_respect"] < -0.65
            or row["accountability"] < -0.75
        ):
            targets.append(1)
            continue

        #Rule B
        if (
            row["public_private_consistency"] < -0.55
            and row["honesty_and_transparency"] < -0.30
        ):
            targets.append(1)
            continue

        #Rule C
        weighted_sum = 0.0
        for col in PILLAR_COLUMNS:
            val = row[col]
            if val < 0:
                weighted_sum += val * 1.35
            else:
                weighted_sum += val * 1.0

        weighted_sum += rng.normal(0, 0.15)
        targets.append(1 if weighted_sum < 0 else 0)

    df["target"] = targets
    return df

if __name__ == "__main__":
    df = generate_synthetic_data(n_samples=2000)
    df.to_csv("data\\training_data.csv", index=False)