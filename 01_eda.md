## 12. Summary of Findings

- **Missing values:** None found.
- **Duplicates:** None found.
- **Capping (known data limitations):**
  - `MedHouseVal` is capped at 5.00001 (~$500,001) — 4.81% of rows. Kept, since dropping would make the model blind to high-value areas entirely. Documented as a known limitation: model may underpredict for properties above $500k.
  - `HouseAge` is capped at 52 — 6.17% of rows. Kept, with an engineered `is_max_house_age` flag added in preprocessing to let the model distinguish "genuinely 52" from "52-or-older, unknown."
- **Skewed features:** `Population` and `MedInc` are right-skewed; log-scale visualization confirmed a more normal-looking shape underneath. Log transform may help linear models, likely unnecessary for tree-based models — to be decided per model in preprocessing.
- **Strongest correlations with `MedHouseVal`:** `MedInc` (0.69, by far the strongest), followed by weak positive correlations with `AveRooms` (0.15) and `HouseAge` (0.11). `Population` and `AveOccup` showed negligible correlation (~0.00).
- **Outliers addressed:**
  - `AveOccup > 10` (37 rows) — institutional/group housing, occupancy values not representative of real households.
  - `AveRooms > 20` (68 rows) — tiny-household-count artifact inflating the per-household ratio; this also resolved `AveBedrms` outliers as a side effect.
  - `Population`'s wide range (up to ~35,700) was investigated and kept — confirmed as legitimate variation (dense urban block groups), not a data error.
- **Feature engineering identified:** `AveRooms`/`AveBedrms` correlate strongly (0.85). Plan: engineer `bedroom_ratio = AveBedrms / AveRooms`, then drop `AveBedrms` to remove redundancy while preserving both the size signal (`AveRooms`) and layout signal (ratio).
- **Geographic patterns:** Highest-value areas cluster near the coast and major cities (visible on the lat/long scatter plot), despite `Latitude`/`Longitude` showing weak *individual* correlations with price — the effect is a combination of both coordinates together, not captured by simple linear correlation. Decided against engineering a distance-to-city feature (added complexity/dependency not worth it for this project); relying on tree-based models to capture this interaction naturally.

**Next notebook:** `02_data_preprocessing.ipynb` — apply the outlier mask, engineer `is_max_house_age` and `bedroom_ratio`, decide on log transforms per model type, train/test split, scaling.