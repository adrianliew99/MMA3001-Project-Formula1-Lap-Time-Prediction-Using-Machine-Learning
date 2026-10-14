# MMA3001-Project-Formula1-Lap-Time-Prediction-Using-a-1-Dimensional-CNN
Can the lap time and sector trend, recent race history, tyre age and strategy predict the next lap time, how accurate it can be estimated


## About this project
This repositories records the development of individual project for Monash University School of Engineering study unit MMA3001: Numerical Methods and Machine Learning. It will bring together the project explanation, data preparation and process, computer code, evaluation, further improvement and discussion & findings. 

## Project Description
Formula 1 lap times depend on vehicle speed, throttle and braking behaviour, tyre conditions, track temperature and race circumstances. This project investigates whether machine-learning regression models can predict a driver's next lap time using information available at the end of the previous completed lap.
The primary deliverable is a reproducible comparison of simple and advanced prediction methods. A potential second stage will use lap-pace and tyre-age estimates to compare hypothetical pit-stop strategies. Race winner and DNF probability models are possible future extensions, not current outputs.

### Engineering Problem
Can historical Formula 1 lap timing, telemetry and racing conditions improve next-lap time predictions over simple baselines, and can these predictions inform tyre-strategy analysis?

### Input and Output
*Input*: Recent race finished position, recent lap time, tyre age and selection
*Output*: Next lap time prediction
*Validation*: Comparison with labelled assessment and how other factor influence


## Project Phase


## Version
Current version: v1.0

## Data Source
Primary Source is obtained through [FastF1]https://github.com/theOehrly/Fast-F1, an open-source Python providing access to previous Formula 1 racing data

## Dataset Description
Data exported: All Drivers at all circuits for Year 2022, 2023, 2024 & 2025\
The retrieved data is exported into .csv dataset format\
**Dataset type:** Historical Formula 1 lap timing and session data

### Extracted Features

| Category | Features | Description |
|---|---|---|
| Driver & Team | `Driver`, `DriverNumber`, `Team`, `Position` | Driver identification, team affiliation, and recorded position |
| Lap Information | `LapNumber`, `LapTime`, `LapTimeSeconds`, `Time`, `LapStartTime`, `LapStartDate` | Lap sequence, lap duration, and timing information |
| Sector Timing | `Sector1Time`, `Sector2Time`, `Sector3Time`, `Sector1SessionTime`, `Sector2SessionTime`, `Sector3SessionTime` | Individual sector durations and corresponding session timestamps |
| Speed Measurements | `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST` | Recorded speeds at intermediate points, finish line, and speed trap (km/h) |
| Tyre & Stint | `Compound`, `TyreLife`, `FreshTyre`, `Stint` | Tyre compound, tyre age, tyre condition, and stint number |
| Pit Stop Information | `PitInTime`, `PitOutTime` | Session timestamps associated with pit entry and exit |
| Race & Track Conditions | `TrackStatus`, `IsPersonalBest` | Track status codes and personal-best lap indicator |
| Data Quality | `IsAccurate`, `Deleted`, `DeletedReason`, `FastF1Generated` | Flags describing lap accuracy, deletion status, and generated records |
| Session Metadata | `SessionID`, `Year`, `Event` | Session identification, season, and Grand Prix event |

**References:**
- [FastF1 GitHub Repository](https://github.com/theOehrly/Fast-F1)
- [FastF1 Official Documentation](https://docs.fastf1.dev/)

FastF1 is used to retrieve racing data. The original timing data is not owned by this project. Any additional preprocessing, feature engineering, and machine learning implementation will be documented separately.

### AI acknowledgement
LLMs model (Chatgpt 6 Astra, Cursor Pro, Google Gemini3 Plus) was used in the project development, suggesting evaluation 