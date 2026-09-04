# 👻 GHOST — GNSS-Free Hybrid Onboard Sensor Tracker

**AI-ML Based Intelligent Dead Reckoning for Seamless Navigation**

**Smart India Hackathon 2026 · SIH Problem Statement 26168 · ISRO / Department of Space**

> *When GNSS disappears, GHOST keeps the vehicle moving on the map.*

GHOST is an edge-oriented Intelligent Dead Reckoning (IDR) and GNSS-fusion system designed to maintain road-level positioning continuity during complete GNSS outages using onboard inertial sensors and learned vehicle-speed estimation.

The system combines a 1D-CNN speed estimator, leakage-free speed calibration, an Extended Kalman Filter (EKF), and OpenStreetMap-based road-heading constraints to estimate vehicle motion when GNSS is unavailable.

---

## Problem

GNSS/GPS signals can become unavailable or unreliable in environments such as:
* Tunnels
* Underground roads
* Dense urban areas
* Indoor or covered environments
* Signal-obstructed regions
* Temporary GNSS interference

Conventional navigation systems experience rapidly increasing positional error once GNSS is lost.

**The core challenge addressed by GHOST is:**
*How can a vehicle continue estimating its position during a complete GNSS outage using only onboard sensors and intelligent sensor fusion?*

---

## GHOST Solution

GHOST operates as a hybrid navigation pipeline:

```text
        GNSS Available
              │
              ▼
      Position / Heading
              │
              ▼
       ┌──────────────┐
       │ GNSS → EKF   │
       │ Initialization│
       └──────┬───────┘
              │
        GNSS Outage
              │
              ▼
 ┌─────────────────────────┐
 │   IMU Sensor Stream     │
 │ Accelerometer + Gyro    │
 └────────────┬────────────┘
              │
              ▼
      ┌────────────────┐
      │ 1D-CNN Speed   │
      │   Estimator    │
      └───────┬────────┘
              │
              ▼
      Speed Calibration
       (Isotonic Model)
              │
              ▼
      ┌────────────────┐
      │ 5-State EKF    │
      │ State Propagation│
      └───────┬────────┘
              │
              ▼
     OSM Road-Heading
        Constraint
              │
              ▼
      Estimated Vehicle
          Trajectory
```

---

## Core Architecture

GHOST uses a five-state Extended Kalman Filter:

$$\mathbf{X} = [x, y, v, \theta, b_{\text{gyro}}]^T$$

Where:

| State | Description |
| :--- | :--- |
| **x** | Local metric Easting position |
| **y** | Local metric Northing position |
| **v** | Forward vehicle velocity |
| **θ** | Vehicle heading angle |
| **b_gyro** | Gyroscope yaw-rate bias |

During a GNSS outage, the EKF continuously propagates the vehicle state using inertial measurements and the learned speed estimate. An OSM-based road-heading constraint provides additional geometric information about the expected vehicle heading.

---

## Machine Learning Component

### 1D-CNN Speed Estimator
The ML component estimates vehicle speed from onboard inertial measurements. The model learns the relationship between IMU-derived temporal patterns and vehicle velocity. Instead of depending on an OBD-II speedometer feed, GHOST estimates speed directly from the available sensor stream.

### Speed Calibration
The raw CNN exhibited systematic speed-regime bias, particularly during higher-speed operation. GHOST therefore applies leakage-free Isotonic Regression calibration to the CNN output. The calibration model is trained using clean training data only and is kept strictly separate from the GNSS outage evaluation interval.

---

## Development & Experiment Progression

GHOST was developed through multiple experimental phases:

| Phase / Experiment | Approach | Final Drift | Status |
| :--- | :--- | :---: | :---: |
| **Phase 4** | Initial dead-reckoning pipeline | 69.73% | Baseline |
| **Phase 5.2** | Improved preprocessing / fusion | 53.76% | Intermediate |
| **Phase 6** | 1D-CNN + EKF + OSM constraint | 18.73% | Uncalibrated |
| **Experiment 1** | Leakage-free linear speed calibration | 4.54% | Calibrated |
| **Experiment 3** | **Isotonic speed calibration** | **2.40%** | **Best Result** |
| **Experiment 4A** | Soft regime-gated calibration | 5.60% | Variant |

```text
Improvement Progression:
69.73% (Raw DR) ──► 53.76% ──► 18.73% (Raw CNN) ──► 4.54% (Linear) ──► 2.40% (Isotonic)
```

The final demonstrated configuration is:
**1D-CNN Speed Estimator + Leakage-Free Isotonic Speed Calibration + Frozen 5-State EKF + OSM Road-Heading Constraint**

---

## Final Demonstrated Result

GHOST was evaluated during a 60-second complete GNSS outage:

| Metric | Value |
| :--- | :--- |
| **GNSS outage duration** | 60 s |
| **Ground-truth outage distance** | 1162.5 m |
| **Estimated integrated distance** | 1128.1 m |
| **Final position error** | 27.95 m |
| **Positional drift** | **2.40%** |
| **SIH benchmark limit** | **≤ 10.00%** |
| **Benchmark status** | **PASS** |

### Drift Calculation
$$\text{Positional Drift} = \frac{\text{Final Position Error}}{\text{Distance Traveled}} \times 100 = \frac{27.95\text{ m}}{1162.50\text{ m}} \times 100 \approx \mathbf{2.40\%}$$

### Benchmark Margin
$$\text{Margin} = 10.00\% - 2.40\% = \mathbf{7.60\text{ percentage points}}$$

The demonstrated drift is **76% below** the maximum allowable 10% drift limit for the evaluated outage.

---

## Position Error Decomposition

The final position error decomposes into along-track and cross-track components:

```text
Along-track error  : -1.16 m
Cross-track error  : -27.93 m
Final position error:  27.95 m
```

**Verification:**
$$\sqrt{(-1.16)^2 + (-27.93)^2} = \sqrt{1.3456 + 780.0849} = \sqrt{781.4305} \approx \mathbf{27.95\text{ m}}$$

---

## Validation & Leakage Audit

The final model evaluation was audited to ensure that the GNSS outage interval was not used during calibration fitting.

### Leakage Protection
* Evaluated Outage Window: **30.0 s → 89.9 s**
* Outage samples were excluded from calibration fitting.
* Isotonic calibration was fitted using clean training data only.

### Final Audit Status

| Audit Check | Status |
| :--- | :---: |
| **Leakage audit** | PASS |
| **Parameter consistency** | PASS |
| **Mathematical drift verification** | PASS |
| **Isotonic monotonicity** | PASS |
| **Artifact verification** | PASS |
| **SIH benchmark** | PASS |
| **Technical readiness** | PASS |

---

## Model Comparison

| Configuration | Final Error | Drift | Status |
| :--- | :---: | :---: | :---: |
| **Phase 6 Raw CNN** | 217.67 m | 18.72% | FAIL |
| **Experiment 1 Linear** | 52.80 m | 4.54% | PASS |
| **Experiment 4A Soft Gate** | 65.09 m | 5.60% | PASS |
| **Experiment 3 Isotonic** | **27.95 m** | **2.40%** | **PASS** |

*Note: The 2.40% result is the demonstrated result on the evaluated outage trajectory.*

---

## Dataset

The project uses vehicle trajectory and inertial sensor data for model development and evaluation.

* **Primary Sequence (`Vfa01`)**: 10 Hz sampling · ~19.14 minutes · 11,486 records
* **Backup Sequence (`Vta2`)**: 10 Hz sampling · ~18.32 minutes · 10,991 records

---

## Data Processing Pipeline

```text
Raw Sensor Logs
      │
      ▼
Data Loading
      │
      ▼
Timestamp Synchronization
      │
      ▼
Cleaning & Validation
      │
      ▼
Feature Engineering
      │
      ▼
CNN Training Data
      │
      ▼
Speed Estimation
      │
      ▼
Calibration
      │
      ▼
EKF Fusion
      │
      ▼
Trajectory Evaluation
```

---

## GNSS Outage Strategy

```text
GNSS AVAILABLE                    GNSS UNAVAILABLE
──────────────                    ────────────────
  GNSS + IMU                           IMU Sensor
      │                                    │
      ▼                                    ▼
  EKF Update                           1D-CNN Speed
      │                                    │
      ▼                                    ▼
Vehicle State                      Speed Calibration
                                           │
                                           ▼
                                    EKF Propagation
                                           │
                                           ▼
                                 OSM Heading Constraint
                                           │
                                           ▼
                                   Estimated Position
```

---

## Interactive 3D Prototype

GHOST includes an interactive web-based visualization demonstrating system behavior in real time:
* Vehicle movement & 3D GLTF models
* Live GNSS availability / blackout state indicators
* Dual-car split view (Raw Baseline vs GHOST AI Tracker)
* Project Mode Technical Dashboard

```text
visualization_3d/
├── src/            # React 3D components & Project Mode Dashboard
├── public/models/  # Optimized 3D vehicle models (phase4_car.glb, phase6_car.glb)
└── package.json    # React, Three.js, R3F, Drei dependencies
```

---

## Repository Structure

```text
GHOST/
├── 3d_cars/           # Raw 3D vehicle assets
├── data/              # Maps, processed CSV datasets, raw logs
├── dead_reckoning/    # Pure inertial integration modules
├── fusion/            # EKF state estimation & sensor fusion
├── map_matching/      # OSM road network constraints
├── models/            # Trained PyTorch 1D-CNN model weights & scalers
├── notebooks/         # Exploratory data analysis notebooks
├── preprocessing/     # Data cleaning, CNN diagnosis & calibration scripts
├── results/           # Full experiment plots, diagnostic reports & evaluation logs
├── visualization/     # Python Plotly & Matplotlib plotting utilities
├── visualization_3d/  # Interactive React/Three.js 3D web application & dashboard
├── app.py             # Main entry point application
└── README.md          # Project documentation
```

---

## Technology Stack

| Layer | Technology |
| :--- | :--- |
| **Language** | Python 3.10+ |
| **Machine Learning** | PyTorch / 1D-CNN |
| **Calibration** | Scikit-learn (Isotonic Regression) |
| **Sensor Fusion** | 5-State Extended Kalman Filter (EKF) |
| **Mapping** | OpenStreetMap (OSM / XML) |
| **Data Processing** | Pandas / NumPy / SciPy |
| **Visualization** | Plotly / Matplotlib |
| **3D Web Prototype** | React 18 / Three.js / React Three Fiber / Drei / Vite |

---

## Running the 3D Prototype

1. Navigate to the visualization directory:
   ```bash
   cd visualization_3d
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the local Vite development server:
   ```bash
   npm run dev
   ```

4. Open the local browser URL provided by Vite (e.g., `http://localhost:3000`).

---

## Project Status

| Metric / Check | Value / Status |
| :--- | :--- |
| **Current Phase** | Phase 7 — Final Validation Complete |
| **Final Architecture** | 1D-CNN + Isotonic Calibration + 5-State EKF + OSM Constraint |
| **Demonstrated Drift** | **2.40%** |
| **SIH Requirement** | **≤ 10.00%** |
| **Benchmark Status** | **SIH BENCHMARK: PASS** |

---

## Limitations

The current results represent the best demonstrated performance on the evaluated GNSS outage trajectory:
* GHOST is designed as a dead-reckoning continuity system during temporary outages, not a full replacement for GNSS under all conditions.
* Further validation across additional vehicle types, road topographies, and sensor packages is recommended for production automotive deployment.

---

## Project Information

* **Project**: GHOST — GNSS-Free Hybrid Onboard Sensor Tracker
* **Domain**: AI / ML · Intelligent Transportation · Sensor Fusion · Navigation
* **SIH Problem Statement**: 26168
* **Organization**: ISRO / Department of Space
* **Final Demonstrated Drift**: **2.40%** (Benchmark: ≤10%)
* **Status**: **PASS**