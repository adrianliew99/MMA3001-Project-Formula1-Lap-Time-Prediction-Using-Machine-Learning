FASTF1: MULTI-GRAND-PRIX EXTRACTION (MMA3001)
===========================================

This bundle includes:
- fastf1_export_dataset.py: export ONE Formula 1 race + previous-lap ML pairs
- fastf1_export_multi_gp.py: run that script for MANY races, combine results

1) Install Python dependencies in VS Code Terminal:
   python -m pip install --upgrade fastf1 pandas numpy

2) Test 2024 on three circuits with two drivers:
   python fastf1_export_multi_gp.py --years 2024 --events Bahrain Monaco Japan --drivers VER HAM

3) Build a multi-year dataset from those circuits, all drivers:
   python fastf1_export_multi_gp.py --years 2023 2024 2025 --events Bahrain Monaco Japan

4) Build dataset using all season races (considerable time and disk usage):
   python fastf1_export_multi_gp.py --years 2023 2024 2025

OUTPUTS: f1_multi_export/all_drivers/
- combined_next_lap_ml.csv (the main machine-learning table)
- combined_all_laps.csv (lap-level timing data)
- combined_weather.csv
- combined_race_results.csv
- export_summary.csv (races exported, number of ML examples, failures)
- races/ (individual GP CSV files, plus local FastF1 cache)

For first 2-driver example the results go to:
  f1_multi_export/drivers_HAM_VER/

IMPORTANT NOTES:
- Extraction requires an internet connection, which wasn't available in the assistant's runtime.
- First downloads may take time; repeat executions reuse saved data.
- Previous-lap telemetry predicts the following lap, reducing data leakage.
- Compare against a previous-lap persistence baseline.
- Validate by holding out entire race events or later seasons; no random individual-lap splits.
- Include circuit identifiers/normalisation when studying raw lap times across circuits.
- Winner/DNF prediction needs a separate race-result dataset and probabilistic models.
- Check F1 data licensing before redistributing CSVs.
