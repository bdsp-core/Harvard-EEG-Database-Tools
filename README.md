# Harvard-EEG-Database-Tools

This repository provides tools for reading and plotting data from the Harvard EEG Database. It also provides code for common signal processing and visualization operations. 

This repository contains helper code and examples for working with the **Harvard Electroencephalography Database (HEEDB)**.  
It includes utilities to:

- Read raw EEG recordings stored in **EDF** format  
- Parse structured metadata and **EEG reports** to extract labels  
- Compute basic **dataset statistics** (per-patient and per-recording)  



## 1. Reading EDF files with MNE

We use **[MNE-Python](https://mne.tools/stable/index.html)** as the main library for reading and working with EEG data in EDF format.

### What is MNE?

MNE is a Python toolkit for EEG/MEG analysis. In this project, we mainly use it for:

- Loading EEG recordings from **EDF** files
- Inspecting channel names and sampling frequency
- Renaming channels and standardizing montages (e.g., 10–20 system)
- Converting data to NumPy arrays for downstream machine learning
- Basic visualization (raw traces, PSD, etc.)

Official documentation:  
👉 https://mne.tools/stable/index.html

