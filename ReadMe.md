# Communication and Error Correction via Polarisation of Single Photons and Time Ordering
> A Master's Thesis Project by Shek Lun Leung, KTH Royal Institute of Technology (2024)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<!-- *Cover Image: "An oil painting constructed in DALL.E 2 showing a single photon with polarization being sent by Alice and being corrected by Bob by distant transmission."* -->

---

## Project Overview

This repository contains the Python code, simulations, and analysis for the Master's thesis, "Communication and Error Correction via Polarisation of Single Photons and Time Ordering". The research investigates the information-carrying capacity of single photons and proposes a novel protocol named **Beyond Pulse Position Modulation (BPPM)**.

The core of this work is a comparative analysis of BPPM against three other established communication protocols:
1.  **Pulse Position Modulation (PPM)**
2.  **On-Off Keying (OOK)**
3.  **A General Protocol** (n photons placed arbitrarily in M time bins)

The project evaluates these protocols based on various performance metrics, including information bits per symbol, bits per photon, and bits per time bin, with a special focus on mutual information under different channel noise conditions. The simulations model photon loss and addition errors and explore the trade-offs between error correction, energy efficiency, and spectral efficiency.

All simulations and visualizations were conducted using Python with the NumPy, Matplotlib, and SciPy libraries. The codebase is structured to be modular and reusable, with GPU acceleration support via PyTorch for computationally intensive calculations.

## Key Features

- **Implementation of Four Protocols:** Python-based models for BPPM, PPM, OOK, and a General combinatorial protocol.
- **Comprehensive Metrics Analysis:** Calculation of key performance indicators such as bits per photon (energy efficiency) and bits per time bin (spectral efficiency).
- **Error and Channel Modeling:** Simulation of photon loss and addition errors to evaluate protocol robustness.
- **Mutual Information Calculation:** In-depth analysis of channel capacity and information retention under noisy conditions.
<!-- - **Modular and Reusable Code:** A central `functions.py` library contains all core logic, making the analysis notebooks clean and easy to follow. -->
- **GPU Acceleration:** Support for Apple's Metal Performance Shaders (MPS) via PyTorch for accelerating large-scale array computations.


## Installation

To run the simulations on your local machine, follow these steps.

1.  **Clone the repository:**
    ```bash
    https://github.com/alanspace/Communication_and_Error_Correction_via_Polarisation_of_Single_Photons_and_Time_Ordering.git
    cd Communication_and_Error_Correction_via_Polarisation_of_Single_Photons_and_Time_Ordering
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python3 -m venv qkd
    source qkd/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is included with the necessary packages. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For Apple Silicon Macs, the latest PyTorch version will automatically include MPS support.*

## Usage

All the analysis and plot generation can be reproduced by running the cells in the Jupyter Notebooks from the directory `Simulation`.

1.  **Start Jupyter:**
    ```bash
    jupyter notebook
    ```
2.  **Open the jupyter notebooks :** Run the cells sequentially to perform the calculations and generate the figures presented in the thesis.
<!-- 3.  **Core Logic:** The notebook imports the custom library `functions.py` (as `fn`). All the complex calculations for each protocol are contained within this file. -->
<!-- 3.  **GPU Usage:** The `functions.py` file includes PyTorch-based functions (suffixed with `_pt`) that will automatically detect and use Apple's MPS backend if it is available on your machine. This significantly speeds up calculations involving large arrays. -->

## Key Results Summary

- **BPPM Performance:** The proposed BPPM protocol demonstrates superior mutual information and resilience at low error probabilities (e.g., P ≤ 0.01), making it highly effective for communication over noisy channels where error correction is critical.
- **Performance Trade-offs:** The analysis reveals a fundamental trade-off. While BPPM excels in noisy, low-power scenarios, its performance advantage diminishes for larger codes (higher `n`), which become more fragile.
- **Comparative Analysis:** The plots provide a clear visual comparison of the energy efficiency (bits/photon) and spectral efficiency (bits/time bin) across all four protocols, highlighting the optimal use case for each.

## Citing this Work

If you use the code or findings from this project in your research, please cite the original thesis:

```bibtex
@mastersthesis{leung2024,
  author       = {Shek Lun Leung},
  title        = {Communication and Error Correction via Polarisation of Single Photons and Time Ordering},
  school       = {KTH Royal Institute of Technology},
  year         = {2024},
  address      = {Stockholm, Sweden},
  month        = {June},
  department   = {Department of Applied Physics},
  degree       = {Master's thesis in Engineering Physics (Quantum Technology)}
}
```

License
This project is licensed under the MIT License. See the LICENSE.md file for details.

Acknowledgements
This work was performed as a Master's thesis project at the Department of Applied Physics, KTH Royal Institute of Technology, under the supervision of Dr. Jonas Almlöf, Dr. Richard Schatz, and Dr. Oskars Ozolins. Full acknowledgements can be found in the included pdf.