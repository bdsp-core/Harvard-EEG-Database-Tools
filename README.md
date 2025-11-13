# Harvard-EEG-Database-Tools

This repository provides tools for reading and plotting data from the Harvard EEG Database. It also provides code for common signal processing and visualization operations. 

## 1. Demographics Summary

This module summarizes patient- and study-level demographics from HEEDB.

**Main features:**

- Counts of unique patients and EEG studies  
- Age distribution (e.g., histogram or age groups)  
- Sex distribution (e.g., male / female / unknown)  
- Race / ethnicity breakdown


## 2. LLM-based Report Labeling

This module uses a Large Language Model (LLM) to extract key information from EEG reports and generate **report-level labels** for each EEG.

**Input:**

- Report TXT files

**Output:**

- A labeled CSV with one row per EEG, features are.:
  - normal  
  - abnormal  
  - bets  
  - wicket  
  - spindles  
  - vertex wave  
  - posts  
  - breach  
  - spikes  
  - seizure  
  - lpd  
  - gpd  
  - lrda  
  - grda  
  - bs  
  - foc slowing  
  - gen slowing  
  - pdr  
  - awake  
  - n1  
  - n2  
  - dravet  
  - eses  
  - fold  
  - jae  
  - jeavons  
  - jme  
  - k_complexes  
  - bects  
  - bipd  
  - cjd  
  - diffuse Beta  
  - low voltage  
  - ppr  
  - status  
  - sunflower  
  - uninterpretable  
  - wham  
  - angelman  
