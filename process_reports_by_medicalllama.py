#  sudo ~/miniconda3/envs/torchenv/bin/python data_processing/list_seizure_patients.py

import os
import glob
import pandas as pd
from format_reports import login_huggingface,load_llama_medical,ask_llama_medical,load_llm,format_reports_by_llm,load_llama_medical_cpu
from tqdm import tqdm
import logging
import torch
logging.basicConfig(level=logging.INFO)
mne_logger = logging.getLogger('mne')
mne_logger.setLevel(logging.WARNING)


DATA_ROOT_DIR='/data'

csv_file_path = os.path.join(DATA_ROOT_DIR, 'EEGs_And_Reports_20231024.csv')
eeg_dir=os.path.join(DATA_ROOT_DIR, 'eeg_edf/')
annotation_dir=os.path.join(DATA_ROOT_DIR, 'annotation_csv/')
reports_dir=os.path.join(DATA_ROOT_DIR, 'reports_txt/')

def ask_llm_about_seizure(raw_reports_txt_dir, type="EMU"):
    login_huggingface()
    pipeline = load_llama_medical()
    instruction_message = 'Dose the patient have any seizure events noted? Answer Yes or No.'
    instruction_message2 = 'Is there any organized or evolving patterns suggestive of seizures? Answer Yes or No.'

    data_dic = pd.read_csv(csv_file_path).astype(str)

    list_file_name=f"/data/{type}_seizure_patients.txt"
    if os.path.exists(list_file_name):
        # Delete the old file
        os.remove(list_file_name)
    with open(list_file_name, "a") as output_file:
        file_paths = glob.glob(os.path.join(raw_reports_txt_dir, '*.txt'))

        for raw_report_path in tqdm(file_paths, desc="Processing reports"):
            data_name = os.path.splitext(os.path.basename(raw_report_path))[0]
            BidsFolder, SessionID_new = data_name.split('_')
            ServiceName = data_dic[
                (data_dic['BidsFolder'] == BidsFolder) & (data_dic['SessionID_new'] == SessionID_new.split('-')[-1])][
                'ServiceName'].iloc[0]
            if ServiceName == type:
                with open(raw_report_path, 'r') as file:
                    report = file.read()
                    has_seizure = ask_llama_medical(report, pipeline, instruction_message=instruction_message)
                    has_seizure_signal=ask_llama_medical(report, pipeline, instruction_message=instruction_message2)
                    has_seizure = has_seizure.strip()
                    has_seizure_signal = has_seizure_signal.strip()
                    if (has_seizure == "Yes" or has_seizure == "yes") and (has_seizure_signal == "Yes" or has_seizure_signal == "yes") :
                        output_file.write(f"{data_name}\n")
                        print(f"A {type} seizure patient {data_name}")

                torch.cuda.empty_cache()

def double_check(type="LTM"):
    login_huggingface()
    pipeline = load_llama_medical()
    instruction_message = 'Dose the patient have seizures from the digital analysis? Answer Yes or No.'
    output_file=f"/data/{type}_seizure_patients_2.txt"

    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, "a") as output_file:
        with open(f"/data/{type}_seizure_patients.txt", "r") as f:
            index_file = f.readlines()
        index = [line.strip() for line in index_file]

        for report_index in tqdm(index, desc="Double check reports"):
            report_file=os.path.join(reports_dir,f"{report_index}.txt")
            with open(report_file, 'r') as file:
                report = file.read()
            has_seizure = ask_llama_medical(report, pipeline, instruction_message=instruction_message)
            has_seizure = has_seizure.strip()
            if (has_seizure == "Yes" or has_seizure == "yes"):
                output_file.write(f"{report_index}\n")
                print(f"An {type} seizure patient {report_index}")

                torch.cuda.empty_cache()

def format_file_path(input_file,output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            modified_line = line.replace('_', '/')
            outfile.write(modified_line)

def merge_list(list1_file,list2_file,output_file):

    with open(list1_file, "r") as f:
        index_file = f.readlines()
    list1 = [line.strip() for line in index_file]

    with open(list2_file, "r") as f:
        index_file = f.readlines()
    list2 = [line.strip() for line in index_file]

    merged_list = list(set(list1 + list2))
    merged_list.sort()

    if os.path.exists(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as file:
        for item in merged_list:
            file.write(item + '\n')

def combine():
    patient_list_dir = os.path.join(DATA_ROOT_DIR, 'patient_list/')
    os.makedirs(patient_list_dir,exist_ok=True)
    list_file_name="/data/patient_list/seizure_patients.txt"
    LTM_file_name = "/data/LTM_EMU_seizure_patients.txt"
    Routine_file_name = "/data/Routine_seizure_patients.txt"

    if os.path.exists(list_file_name):
        os.remove(list_file_name)

    with open(LTM_file_name, 'r') as f:
        LTM_list_content = f.readlines()
    LTM_list = [line.strip() for line in LTM_list_content]

    with open(Routine_file_name, 'r') as f:
        Routine_list_content = f.readlines()
    Routine_list = [line.strip() for line in Routine_list_content]


    normal_list=LTM_list+Routine_list

    with open(list_file_name, 'w') as file:
        for file_name in normal_list:
            file.write(f"{file_name}\n")
    print(f"List file {list_file_name} has been saved.")

def main(type="EMU"):
    ask_llm_about_seizure(raw_reports_txt_dir=reports_dir,type=type)
    double_check(type)

if __name__ == "__main__":
    # main(type="Routine")
    combine()

    # merge_list("/data/LTM_seizure_patients_2.txt","/data/EMU_seizure_patients.txt",output_file="/data/LTM_EMU_seizure_patients.txt")
    # format_file_path(input_file="/data/LTM_EMU_seizure_patients.txt", output_file="/data/LTM_EMU_seizure_patients_aws_path.txt")

    # login_huggingface()
    # instruction_message = 'Dose the patient have seizures from the digital analysis? Answer Yes or No.'
    # instruction_message2 = 'Dose the patient have any organized or evolving patterns suggestive of seizures? Answer Yes or No.'
    #
    # with open("/data/reports_txt/sub-S0002119156682_ses-1.txt", 'r') as file:
    #     report = file.read()
    # pipeline = load_llama_medical()
    # has_seizure = llama_medical_seizure(report,pipeline,instruction_message)
    # print(has_seizure.strip())
    # has_seizure2 = llama_medical_seizure(report, pipeline, instruction_message2)
    # print(has_seizure2.strip())


