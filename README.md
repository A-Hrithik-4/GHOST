# Project GhostTrack — AI-ML Intelligent Dead Reckoning System
> **SIH Problem Statement 26168 (ISRO)**

GhostTrack is an edge-deployable Intelligent Dead Reckoning (IDR) & GNSS Fusion Engine designed to maintain continuous, lane-level vehicle navigation during complete GNSS outages without OBD-II speedometer feeds.

## 📁 Repository Structure
```
GhostTrack/
├── data/
│   ├── raw/               # IO-VNBD raw dataset link
│   ├── processed/         # Synchronized 10Hz CSV files (ghosttrack_primary_Vfa01.csv)
│   └── maps/              # OSM road network data
├── preprocessing/         # Data loading, sync & cleaning scripts
├── models/                # PyTorch speed estimation neural networks
├── dead_reckoning/        # Inertial kinematics & dynamic speed integrators
├── map_matching/          # OSM segment snapping & Non-Holonomic Constraints
├── fusion/                # Adaptive EKF & outage handoff engine
├── visualization/         # UI rendering & trajectory plotters
├── results/               # Day-by-day benchmark metrics & plots
└── README.md
```

## 📊 Phase 1 Status
- **Primary Sequence:** `Vfa01` (11,486 rows, 19.14 mins, 10 Hz)
- **Backup Sequence:** `Vta2` (10,991 rows, 18.32 mins, 10 Hz)
- **Feasibility Spike:** Map-matching concept verified & proven.
- **Phase 1 Complete:** YES ✅
