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

The repository includes an example for reading edf file using MNE:

👉  **`read_data.ipynb`**

## 2. Extracting Labels from Clinical EEG Reports

This project extracts EEG labels (e.g., seizure, spike, burst suppression, epileptiform activity, slowing) directly from the **clinical EEG reports** associated with each EDF file. The process consists of two main steps:

---

### 2.1 Aligning EDF Recording Time With Report Time

Each HEEDB EEG has two relevant timestamps:

- **EDF start time** – stored in the EDF header  
- **Report timestamp** – stored in the clinical EEG report  

We align these two timestamps so that the clinical findings can be mapped onto the correct EEG files.

---

### 2.2 Extracting Labels Using a Medical LLM (Yes/No Q&A)

We extract EEG-level labels by asking a **medical large language model (Medical-LLaMA)** structured yes/no questions.  
Each question is phrased explicitly to ensure a deterministic answer.

Example prompts: Dose the patient have any seizure events noted? Answer Yes or No.

These questions are fed to a medical LLM, which returns a YES or NO answer for each label.

🔗 **Medical-LLaMA documentation / example interface**  
[https://medical-llama.hf.space ](https://huggingface.co/ContactDoctor/Bio-Medical-Llama-3-8B) 

---

### ⚠️ Privacy and IRB Reminder

For real HEEDB data, **always use a local LLM** (e.g., Llama-3-70B-Instruct, MedLlama locally, or an in-hospital model).  
Never send clinical reports or PHI to online endpoints.

---

### 2.3 Code Example: Extracting Seizure Labels

The repository includes an example script demonstrating the complete workflow:

👉 **`format_reports.py`**, **`process_reports_by_medicalllama.py`**

This script contains:

- Loading raw EEG reports  
- Asking structured yes/no questions  
- Parsing LLM answers  
- Attaching labels to the corresponding EDF recording  
- A full example for extracting seizure labels

### 2.4 Output: EEG Labels Extracted from Reports

All extracted EEG-level labels (e.g., seizure, spikes, burst suppression, slowing, epileptiform activity) have been aggregated and saved into:

👉 **`MGB_EEG_with_reports.csv`** **`BWH_EEG_with_reports.csv`** **`BCH_EEG_with_reports.csv`** **`BIDMC_EEG_with_reports.csv`**

This CSV serves as the main entry point for downstream modeling and statistical analysis.


## 3. Dataset Statistics

After extracting labels and metadata, we compute several descriptive statistics for the HEEDB dataset. These statistics help us understand population characteristics, label distributions, and overall dataset composition.

We provide ready-to-run notebooks for reproducing all results.

---

### 3.1 Demographic Statistics

We summarize core demographic information for the EEG cohort, including:

- **Number of unique patients**
- **Number of unique EEG**
- **Age distribution**
- **Sex distribution**
- **Race distribution**


These analyses are demonstrated in:

👉 **`statistics.ipynb`**

### 3.2 Label Distribution Statistics

We compute the distribution of key EEG labels extracted from clinical reports.  



We compute the distribution of medication extracted from clinical notes.  

This helps assess class balance, prevalence of clinically important patterns, and the diversity of EEG phenomena in the dataset.

All label distribution analyses are demonstrated in:

👉 **`medication_statistics.ipynb`**

### 3.3 ICD Distribution Statistics

We compute the distribution of ICD extracted from clinical notes.  

We mainly focus on ICD-10-CM for Neurology, and we have summarized it in the file:

👉 **`ICD-10-CM_for_Neurology.json`**

The extracted patient-level ICD codes, as well as the summary files, are in:

👉 **`HEEDB_ICD10_for_Neurology.xlsx`**  **`HEEDB_ICD10_for_Neurology_statistics.xlsx`**

All label distribution analyses are demonstrated in:

👉 **`ICD_statistics.ipynb`** 


### 3.4 Medication Distribution Statistics

We compute the distribution of medication extracted from clinical notes.  

All label distribution analyses are demonstrated in:

👉 **`medication_statistics.ipynb`** 


## 4. Citation

Please cite the HEEDB paper when using these tools or analyses:

Sun C., Jing J., Turley N., Alcott C., Kang W.-Y., Cole A. J., Goldenholz D. M., Lam A., Amorim E., Chu C., Cash S., Moura Junior V., Gupta A., Ghanta M., Nearing B., Nascimento F. A., Struck A., Kim J., Sartipi S., Tauton A.-M., Fernandes M., Sun H., Bayas G., Gallagher K., Wagenaar J. B., Sinha N., Lee-Messer C., Tsien Silvers C., Gunapati B., Rosand J., Peters J., Loddenkemper T., Lee J. W., Zafar S., Westover M. B. Harvard Electroencephalography Database: A comprehensive clinical electroencephalographic resource from four Boston hospitals. Epilepsia. First published June 4, 2025. https://doi.org/10.1111/epi.18487
