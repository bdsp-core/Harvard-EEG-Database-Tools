import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import AutoModel,BitsAndBytesConfig
import transformers
from huggingface_hub import login
import re
import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize, word_tokenize

instructed_prompt = f"Summarize EEG report. It should include Monitoring Session, Patient Information, Medications and Treatments,\
     EEG Technical Information, EEG Description, Clinical Details, EEG Patterns, Digital Analysis, Summary Impression, and other information may be important. \
    The original report:"

format_prompt ='Extract and list every clinical information and the data details from the following EEG report in a structured format. The original report:'

correct_punctuation_prompt='Correct the punctuation and segmentations of the following EEG report without missing any words.'


def login_huggingface():
    huggingface_token = '' # use your token
    login(token=huggingface_token)

def remove_noise_text(text):
    # # Remove redundent patient info
    # cleaned_text = re.sub(r"patient name:.*?(?=\w)", "", input_text, flags=re.IGNORECASE)

    # cleaned_text = re.sub(r"patient's name:.*?(?=\w)", "", input_text, flags=re.IGNORECASE)

    # # Remove doctor info
    # cleaned_text = re.sub(r"fellow:.*?(?=\s{2})", "", cleaned_text, flags=re.IGNORECASE)

    # # cleaned_text = re.sub(r'i .*? have .*?methodology', 'methodology', cleaned_text, flags=re.IGNORECASE)

    # # Remove website
    # # cleaned_text = re.sub(r'website: .*', 'website: (Unknown website)', cleaned_text, flags=re.IGNORECASE)

    # # Replace date
    # cleaned_text = re.sub(r'\*+/\*+/\*+', '(Unknown date)', cleaned_text)
    # cleaned_text = re.sub(r'\*+/\*', '(Unknown date)', cleaned_text)

    # Remove redundent symbels
    text = re.sub(r'[?!\-]', '', text)

    # Remove asterisks
    text = text.replace('*', '')

    # Remove redundant spaces
    text = re.sub(r'\s{3,}', '  ', text)

    # Replace sequences of whitespace characters longer than 4 with a newline
    # text = re.sub(r'\s{4,}', '\n', text)

    # Strip leading and trailing spaces
    text = text.strip()
    return text

def clean_reports_by_type(text, type):
    text = remove_noise_text(text)

    if type=='LTM':
        add_text = f'this report is from the long-term EEG monitory\n'
    elif type=='Routine':
        add_text = f'this report is from the routine EEG monitory\n'
    elif type=='EMU':
        add_text = f'this report is from the epilepsy monitory unit\n'
    elif type=='Faulkner':
        add_text = f'this report is from the {type}\n'
    elif type=='OR':
        add_text = f'this report is from the {type}\n'
    elif type=='FISH':
        add_text = f'this report is from the {type}\n'
    else:
        return text
    return add_text+text

def correct_punctuation(text):
    sentences = sent_tokenize(text)
    corrected_sentences = []

    for sentence in sentences:
        # Check if the sentence ends with a punctuation mark
        if not re.search(r'[.!?]$', sentence):
            sentence += '.'  # Add a period if missing
        corrected_sentences.append(sentence)

    return ' '.join(corrected_sentences)

def load_llm(llm_name):
    # Determine the device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if llm_name == "gpt2":
        model_name = "gpt2"  # "gpt2-medium", "gpt2-large", or "gpt2-xl" for larger models
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

    elif llm_name == "llama2":
        model_id = "meta-llama/Llama-2-7b-chat-hf"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.use_default_system_prompt = False

        model = AutoModelForCausalLM.from_pretrained(model_id,
                                                     torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
        model.to(device)

    elif llm_name == "t5-medical":
        model_id = "Falconsai/medical_summarization"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.use_default_system_prompt = False

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                      torch_dtype=torch.float16 if device.type == "cuda" else torch.float32)
        model.to(device)


    else:
        raise ValueError("Invalid model name.")
    return tokenizer, model, device



def format_reports_by_llm(llm_name, tokenizer, model, device, report, max_length=2048,prompt_instruction=format_prompt):

    if llm_name == "gpt2":
        inputs = tokenizer(report, return_tensors="pt", max_length=max_length, truncation=True,
                           skip_special_tokens=True)
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=2048,
            temperature=0.3,
            top_p=1.0,
            do_sample=False,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    elif llm_name == "llama2":
        # prompt = instructed_prompt + report
        prompt = prompt_instruction + report

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate the response
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2)

        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    elif llm_name == "t5-medical":
        input_ids = tokenizer.encode(report, return_tensors="pt").to(device)

        # Generate the response
        with torch.no_grad():
            output = model.generate(input_ids, max_length=max_length, num_beams=4, no_repeat_ngram_size=2)

        # Decode the response
        response = tokenizer.decode(output[0], skip_special_tokens=True)

    else:
        raise ValueError("Invalid model name.")

    if prompt_instruction is not None:
        # Extract the corrected and segmented text from the output
        response = response.replace(prompt_instruction, "").strip()

    else: response = response.strip()

    return response


def llama_medical(report, type):

    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"

    pipeline = transformers.pipeline ("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                     device_map="auto", )

    if type=='LTM':
        source = 'long-term EEG monitory'
    elif type=='Routine':
        source = 'routine EEG monitory'
    elif type=='EMU':
        source = f'epilepsy monitory unit\n'
    else:
        source=''

    report = remove_noise_text(report)
    system_message="You are an expert trained on neurology and clinical domain!"
    instruction_message = f'Study the following {source} EEG report: {report}'

    system_messages = [{"role": "system", "content": system_message},
                {"role": "user",
                 "content": instruction_message},
                ]

    extract_info_messages =[
                    # 'Summarize the impression from this EEG report.',\
                    # 'Identify the EEG background details described in this report.',\
                    # 'Extract the descriptions of EEG signals from this report.',\
                    # 'Highlight any abnormalities mentioned in the EEG signals in this report.', \
                    'Identify and summarize each the events.', \
                    'Describe the EEG signals in each event. And no need to describe the background.',\
                    'Extract the time of each event. ', \
                    # 'What\'s the duration of each event?'
                            ]

    for i, extract_info_message in enumerate(extract_info_messages):
        extract_info_message=[{"role": "user",
                 "content": extract_info_message}]
        messages=system_messages+extract_info_message

        prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        outputs = pipeline(prompt, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6,
                           top_p=0.9, )
        print(outputs[0]["generated_text"][len(prompt):]+"\n")


def load_llama_medical():
    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"

    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                     device_map="auto", )
    return pipeline

def load_llama_medical_cpu():
    model_id = "ContactDoctor/Bio-Medical-Llama-3-8B"

    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16},
                                     device_map="cpu", )
    return pipeline

def llama_medical_impression(report,pipeline):
    report = remove_noise_text(report)

    system_message = "You are an expert trained on neurology and clinical domain!"
    instruction_message = f'Summarize the impression from this EEG report: {report}'

    messages = [{"role": "system", "content": system_message},
                       {"role": "user",
                        "content": instruction_message},
                       ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipeline(prompt, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.6,
                           top_p=0.9, )

    return outputs[0]["generated_text"][len(prompt):]


def ask_llama_medical(report, pipeline,instruction_message):
    report = remove_noise_text(report)

    system_message = "You are an expert trained on neurology and clinical domain!"
    instruction_message = f'{report} {instruction_message}'

    messages = [{"role": "system", "content": system_message},
                       {"role": "user",
                        "content": instruction_message},
                       ]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    terminators = [pipeline.tokenizer.eos_token_id, pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    outputs = pipeline(prompt, max_new_tokens=256, eos_token_id=terminators, do_sample=True, temperature=0.5,
                           top_p=0.9, )

    return outputs[0]["generated_text"][len(prompt):]

if __name__ == "__main__":

    report=" this is an abnormal eeg in awake, drowsy and asleep states: 1.multiple runs(up to 10-15 per hour, 10-45 second long) of generalized,  sharp/spike and wave discharges at 2-4 hz with admixed significant muscle  artifact. occasionally, these runs tend to have 1 second long lead in from  the right hemisphere. clinically, patient appeared to be still, staring,  at times some eyelid flutter, very slow flexion of the right arm at the  elbow with low amplitude tremulous movements of the right upper extremity.  at the end of the run patient was noted to take a deep sigh. these are  probably consistent with atypical absence seizures. 2. two brief tonic seizures 3. abundant, generalized spike/sharp and wave as described above(both  sporadic and in runs) without any clinical correlate. on an average, these  discharges occurred at least once every 15 second page during sleep. 4. occasional, generalized paroxysmal fast activity( 10-12 hz) noted during  sleep. 5. occasional, right temporal spikes which at times may occur in runs  lasting for 3-5 seconds   clinical indication: we conducted 24 hrs of eeg/cctv monitoring on this 35  year old woman with a history of tsc, refractory epilepsy, cognitive  impairment. the goal of monitoring was to determine the nature of these  spells and if epileptic, locate the focus (foci) of activity.   method: we performed continuous eeg/cctv monitoring from **/**/** until  **/**/**. standard international **-** system electrode placement was used  including bilateral anterior temporal electrodes.  standard referential  and bipolar montages were used for eeg review, as well as specifically  reformatted arrays when necessary. a push-button was available for the  patient's family and/or staff to flag the occurrence of subjective or  objective behavioral abnormalities. continuous computerized spike and  seizure detection was performed throughout the recording period. in  addition, random samples of background recording from throughout the  monitoring session were manually reviewed. behavior was monitored by  continuous video recordings of patient activities and by maintenance of a  log of activities by patient's family and the nursing staff. eeg was  reviewed on a daily basis and interim reports were generated as required.  ekg samples were reviewed daily.   background: background consists of generalized, 30-50 uv, polymorphic  theta activity at 5-7 hz with some ap organization. reactivity to external  stimulus is present.  awake the background activity consisted of a rhythmic, well-formed  symmetric posterior dominant rhythm at 8-9 hz and **-** uv, with good  anterior-posterior organization and good reactivity. drowsy sections  showed slow roving eye movements, a decrement in the posterior dominant  rhythm and generalized background slowing, maximal fronto-centrally.  symmetric vertex sharp waves, central sleep spindles and k-complexes were  present during stage ii sleep.      interictal epileptiform abnormalities:  abundant, generalized spike/sharp and wave as described below(both  sporadic and in runs) without any clinical correlate. on an average, these  discharges occurred at least once every 15 second page during sleep.   -occasional, up to 160 uv, generalized paroxysmal fast activity( 10-12 hz)  was noted during sleep.       -occasional, up to 200 uv, right temporal spikes were noted at with  maximal amplitude at f8/t4/t2 which at times may occur in runs lasting for  3-5 seconds   events/seizures: multiple runs(up to 10-15 per hour, 10-45 second long) of up to 350 uv,  generalized, sharp/spike and wave discharges at 2-4 hz admixed significant  muscle artifact was noted. occasionally, these runs tend to have 1 second  long lead in from the right hemisphere. clinically, patient appeared to be  still, staring, at times some eyelid flutter, very slow flexion of the  right arm at the elbow with low amplitude tremulous movements of the right  upper extremity. at the end of the run patient was noted to take a deep  sigh. these are probably consistent with atypical absence seizures.       at 18:47:30: generalized up to 90 uv slow delta wave followed generalized  voltage attenuation with overriding faster sharply contoured beta activity  at 13-15 hz was noted lasting for 10 seconds. clinically, patient was  laying in bed with her head supported by a pillow, sudden myoclonic jerk  in the bilateral upper extremities with arms going up(right more than left  but could be because of her position in the bed) followed by stiffening of  the arms and head with low amplitude tremulous movements with arms raised  up. this is consistent with a tonic seizure.   at 20:45:15 patient was noted to have the same ictal pattern as above for  the tonic seizure. she was however off camera.     *------------------------------------------------------* ecg: no dysrhythmia *------------------------------------------------------* ***** ******, md **/**/** *** epilepsy fellow   i (********* a. ******, md phd) have  reviewed this eeg recording with the  fellow and edited this final report.   start date: */**/**** start time: 3:10 pm end date: */**/**** end time: 9:14 am *------------------------------------------------------*  "
    type = "EMU"

    llm_name = "llama-medical"

    if llm_name != "gpt2":
        login_huggingface()

    if  llm_name=="llama-medical":
        llama_medical(report,type)

    else:
        tokenizer, model, device = load_llm(llm_name=llm_name)

        formated_report = format_reports_by_llm(llm_name=llm_name,prompt_instruction=correct_punctuation_prompt, tokenizer=tokenizer, model=model, device=device, report=report,max_length=2048)
        print(formated_report)
        formated_report=remove_noise_text(formated_report)

