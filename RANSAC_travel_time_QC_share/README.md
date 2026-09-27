# RANSAC + MAD Travel-Time QC Written by Suhee with ChatGPT
A generic, shareable workflow for robust P- and S-wave travel-time quality control using RANSAC regression and MAD-based residual analysis.

## Recommended Directory Structure
```text
project/
├── ransac_travel_time_qc.py
├── example_travel_times.csv
└── output/
    ├── ransac_qc_results.csv
    └── ransac_qc_plot.png
```

## Input
Save the input as `example_travel_times.csv`. One row should represent one event-station observation.

```text
event_id,station,hypocentral_distance_km,P_travel_time,S_travel_time
20250308.102733,YODB,25.4,4.31,7.42
20250308.102733,OOJB,31.8,5.22,8.91
```

Only `hypocentral_distance_km`, `P_travel_time`, and `S_travel_time` are required for the RANSAC calculation. Additional columns such as `event_id`, `station`, event location, and pick time may be included and are preserved in the output. Keeping `event_id` and `station` is recommended so flagged observations can be traced back to the original waveforms.

`hypocentral_distance_km` is the geometric source-station distance in kilometers. It is used here as a QC coordinate and is not a seismic ray-path length.

## Method
1. Fit P- and S-wave travel-time trends separately using RANSAC linear regression:
```text
   T_pred = aR + b
```
2. Calculate residuals:
```text
   r = T_obs - T_pred
```
3. Calculate MAD from RANSAC inlier residuals:
```text
   MAD = median(|r - median(r)|)
```
5. Convert MAD to a standard-deviation-equivalent robust scale:
```text
   robust_sigma = 1.4826 * MAD
```
6. Calculate the robust Z-score:
```text
   robust_Z = |r - median(r)| / robust_sigma
```
7. Classify each observation:
```text
   normal: robust Z <= 3  
   check: 3 < robust Z <= 5  
   strong_outlier: robust Z > 5
```

RANSAC is used to estimate the robust reference trend. Its own inlier/outlier mask is not used as the final QC classification. The factor 1.4826 is the conventional normal-consistency scaling factor for MAD.

## Requirements
```text
pip install numpy pandas matplotlib scikit-learn
```

## Outputs
`output/ransac_qc_results.csv`: Input data + RANSAC predictions, residuals, RANSAC inlier masks, robust Z-scores, and QC labels.

`output/ransac_qc_plot.png`: P/S travel-time data, RANSAC reference trends, and QC-flagged observations.

### Example Output
```text
P: T=0.166181R+0.295460, median=0.003137, MAD=0.072473, robust_sigma=0.107449

- T = 0.166181R + 0.295460 -> RANSAC reference trend
- 1 / slope = 6.02 km/s -> apparent velocity (not a velocity-model estimate)
- median = 0.003137 s -> median of P residuals
- MAD = 0.072473 s -> robust scatter of P residuals
- robust_sigma = 0.107449 s -> robust scale used for Z-score calculation
```

## Parameters
```text
P_RANSAC_THRESHOLD=1.0
S_RANSAC_THRESHOLD=1.5
RANSAC_MIN_SAMPLES=0.5
RANSAC_MAX_TRIALS=1000
CHECK_SIGMA=3.0
STRONG_SIGMA=5.0
RANDOM_STATE=42
```

The P and S RANSAC residual thresholds (1.0 s and 1.5 s) are dataset-specific working values, not universal thresholds. They should be adjusted according to the characteristics of the dataset.

## Important
A statistical outlier is not automatically a bad phase pick. Flagged observations should be checked against the original waveform before any pick is corrected or removed.

A single linear travel-time trend is only a QC approximation. Systematic residuals may also result from velocity structure, event depth, path effects, station effects, curvature, or phase changes.
