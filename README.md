# Formula 1 Race Performance Prediction and Strategy Analysis
### MMA3001 — Numerical Methods and Machine Learning | Individual Project
**Project status (October 2026): Data acquisition and preprocessing.** The single-race and multi-Grand-Prix FastF1 export scripts have been written and syntax-checked. A successful live download on the project's local machine and a complete model-training run have **not yet been confirmed**. Prediction scores, tyre-strategy recommendations, winner probabilities and DNF probabilities are **not available yet**.

### 1. Project overview
Formula 1 lap times depend on vehicle speed, throttle and braking behaviour, tyre conditions, track temperature and race circumstances. This project investigates whether machine-learning regression models can predict a driver's next lap time using information available at the end of the previous completed lap.
The primary deliverable is a reproducible comparison of simple and advanced prediction methods. A potential second stage will use lap-pace and tyre-age estimates to compare hypothetical pit-stop strategies. Race winner and DNF probability models are possible future extensions, not current outputs.

### 2. Engineering question
Can historical Formula 1 lap timing, telemetry and racing conditions improve next-lap time predictions over simple baselines, and can these predictions inform tyre-strategy analysis?

### 3. Intended engineering use
Estimate a driver's next-lap time under comparable normal racing conditions.
Assess whether telemetry features add predictive value beyond previous-lap pace.
Compare prediction accuracy, runtime and interpretability of different models.
Explore tyre-related pace changes and, later, simulate alternative pit-stop choices.

###### This is a student research prototype, not a real-time race-control or safety-critical system.

### Current Progress
