import numpy as np
from IPython.display import Audio, display
import librosa
import os
import wget
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pprint
import argparse
pp = pprint.PrettyPrinter(indent=4)
from omegaconf import OmegaConf
import shutil
import json
from nemo.collections.asr.parts.utils.decoder_timestamps_utils import ASRDecoderTimeStamps
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import os
from IPython.display import clear_output
import torch
import nemo.collections.asr as nemo_asr
import nemo
import os
import torch
import csv
from pydub import AudioSegment
from nemo.collections.asr.models import EncDecCTCModel
import re
import git



def split_json_file(input_path):
    # Ensure the output directory exists
    output_path = os.path.join(os.getcwd(), "split_manifest")
    lines_per_part = 1  # Set this to the number of lines you want per file

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Read all lines from the file
    with open(input_path, 'r') as file:
        lines = file.readlines()

    # Calculate the number of parts needed
    total_lines = len(lines)
    num_parts = (total_lines + lines_per_part - 1) // lines_per_part  # Ensure all lines are covered
    print(f"Total lines: {total_lines}, Lines per part: {lines_per_part}, Total parts: {num_parts}")

    # Split and write to new files
    for part in range(num_parts):
        start = part * lines_per_part
        end = min(start + lines_per_part, total_lines)  # Avoid going out of range
        part_file_path = os.path.join(output_path, f"part_{part + 1}.json")

        # Write the current part to its file
        with open(part_file_path, 'w') as part_file:
            for line in lines[start:end]:
                part_file.write(line)
        print(f"Part {part + 1} written to {part_file_path}")

def crop_audio(input_wav, start_ms, end_ms):
    audio = AudioSegment.from_wav(input_wav)
    audio = audio.set_frame_rate(16000)
    cropped_audio = audio[start_ms*1000:end_ms*1000]
    cropped_audio.export("temp_wav_output/croped_file.wav", format="wav")

def transcribe_audio(checkpoint_path, data_dir, output_csv='transcriptions.csv', batch_size=4):
    # Restore the ASR model from the checkpoint
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(checkpoint_path)

    # List all .wav files in the directory
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

    # Prepare the list of audio paths
    audio_paths = [os.path.join(data_dir, wav) for wav in wav_files]

    # Transcribe the audio files in batches
    transcriptions = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i + batch_size]
        transcripts = asr_model.transcribe(audio=batch_paths, batch_size=len(batch_paths))
        transcriptions.extend(transcripts)
    print(transcriptions)
    # Prepare data for CSV
    csv_data = []
    for wav, transcript in zip(wav_files, transcriptions):
        audio_name = os.path.splitext(wav)[0]
        csv_data.append([audio_name, transcript])

    # Write to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['audio', 'transcript'])
        writer.writerows(csv_data)

    print(f"Transcriptions saved to {output_csv}")

def time_to_seconds(minutes, seconds):
    return int(minutes) * 60 + float(seconds)


# Function to process a single file and convert it to JSON
def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Regular expression to parse the data
    pattern = re.compile(r'\[(\d{2}):(\d{2}\.\d{2}) - (\d{2}):(\d{2}\.\d{2})\] (speaker_\d+): (.+)')

    # Parse the input data and convert to JSON format
    segments = []
    for match in pattern.finditer(data):
        start_minutes, start_seconds, end_minutes, end_seconds, speaker, text = match.groups()
        start_time = time_to_seconds(start_minutes, start_seconds)
        end_time = time_to_seconds(end_minutes, end_seconds)
        segments.append({
            "start": start_time,
            "end": end_time,
            "speaker": speaker,
            "text": text
        })

    # Write the JSON data to a file
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=4)


def longFile(model_path):
    data_dir = os.getcwd()
    DOMAIN_TYPE = "telephonic"  # Can be meeting or telephonic based on domain type of the audio file
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

    if not os.path.exists(os.path.join(data_dir, CONFIG_FILE_NAME)):
        CONFIG = wget.download(CONFIG_URL, data_dir)
    else:
        CONFIG = os.path.join(data_dir, CONFIG_FILE_NAME)

    cfg = OmegaConf.load(CONFIG)
    print(OmegaConf.to_yaml(cfg))

    output_path = os.path.join(data_dir, 'split_manifest')
    lines_per_part = 1  # Set this to the number of lines you want per file

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    pretrained_speaker_model = 'titanet_large'
    cfg.diarizer.out_dir = data_dir  # Directory to store intermediate files and prediction outputs
    cfg.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
    cfg.diarizer.clustering.parameters.oracle_num_speakers = False
    cfg.batch_size = 1
    cfg.diarizer.msdd_model.parameters.infer_batch_size = 1
    cfg.diarizer.asr.parameters.asr_batch_size = 1
    # Using Neural VAD and Conformer ASR
    cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    cfg.diarizer.asr.model_path = "/kaggle/input/the-best-results/results/Some name of our experiment/checkpoints/conformer.nemo"
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.asr.parameters.asr_based_vad = False
    cfg.diarizer.ignore_overlap = False

    too_big = []
    for manifest_file in os.listdir(output_path):
        file_path = os.path.join(output_path,manifest_file)
        with open(file_path, 'r') as file:
            dur = json.load(file)['duration']
        if dur > 200:
            too_big.append(file_path)
            continue
        cfg.diarizer.manifest_filepath = file_path
        asr_decoder_ts = ASRDecoderTimeStamps(cfg.diarizer)
        asr_model = asr_decoder_ts.set_asr_model()
        word_hyp, word_ts_hyp = asr_decoder_ts.run_ASR(asr_model)
        asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
        asr_diar_offline.word_ts_anchor_offset = asr_decoder_ts.word_ts_anchor_offset
        diar_hyp, diar_score = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
        trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)
        clear_output()

    # Define the data directory
    repo_url = "https://github.com/motawie0/NeMo.git"
    # Directory where you want to clone the repository

    exit_code = os.system(f"git clone {repo_url}")
    # Create necessary directories
    os.makedirs(os.path.join(data_dir, 'temp_wav_output'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'temp_wav'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'long_audio_json'), exist_ok=True)

    # Restore the ASR model
    asr_model = EncDecCTCModel.restore_from(model_path)

    for file in too_big:
        output_inference_path = os.path.join(data_dir, 'output_inference')
        # Ensure the output_inference_path exists and clear its contents
        if os.path.exists(output_inference_path):
            for root, dirs, files in os.walk(output_inference_path):
                for f in files:
                    os.unlink(os.path.join(root, f))
        else:
            os.makedirs(output_inference_path, exist_ok=True)

        with open(file, 'r') as file:
            audio_path = json.load(file)['audio_filepath']

        meta = {
            'audio_filepath': audio_path,
            'offset': 0,
            'duration': None,
            'label': 'infer',
            'text': '-',
            'num_speakers': None,
            'rttm_filepath': None,
            'uem_filepath': None
        }

        with open('input_manifest.json', 'w') as fp:
            json.dump(meta, fp)
            fp.write('\n')

        # Run the diarization inference
        cfg.diarizer.out_dir = output_inference_path
        cfg.diarizer.manifest_filepath = os.path.join(data_dir,'input_manifest.json')

        OmegaConf.save(cfg, "/kaggle/working/diar_infer_telephonic.yaml")

        os.system(
            f'HYDRA_FULL_ERROR=1 python {data_dir}/NeMo/examples/speaker_tasks/diarization/neural_diarizer/multiscale_diar_decoder_infer.py --config-path {data_dir} --config-name diar_infer_telephonic.yaml')
        rttm_preds_path = os.path.join(output_inference_path, 'pred_rttms')
        rttm_file_path = os.listdir(rttm_preds_path)
        data = []

        with open(os.path.join(rttm_preds_path, rttm_file_path[0]), 'r') as file_pred_rttm:
            for line in file_pred_rttm:
                parts = line.strip().split()
                if parts[0] == "SPEAKER":
                    # Extract the start time and duration
                    start_time = float(parts[3])
                    end = start_time + float(parts[4])
                    speaker = parts[7]
                    crop_audio(audio_path, start_time, end)
                    transcripts = \
                    asr_model.transcribe(audio=os.path.join(data_dir, 'temp_wav_output/croped_file.wav'), batch_size=1)[0]
                    data.append((start_time, end, speaker, transcripts))

        segments = [{"start": start, "end": end, "speaker": speaker, "text": text} for start, end, speaker, text in data]
        audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
        output_json_path = os.path.join(data_dir, 'long_audio_json', f'{audio_filename}.json')

        print(f"dumping in {output_json_path}")

        # Write the JSON data to a file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)
    # Paths to the input and output directories
    input_directory = os.path.join(data_dir, 'pred_rttms')
    output_directory = os.path.join(data_dir, 'small_audio_json')

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Process each .txt file in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_directory, filename)
            output_file_name = os.path.splitext(filename)[0] + '.json'
            output_file_path = os.path.join(output_directory, output_file_name)

            # Process the file and convert to JSON
            process_file(input_file_path, output_file_path)

            print(f"Processed {input_file_path} -> {output_file_path}")



# Function to convert time in "MM:SS.SS" format to seconds



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSON file into multiple parts and infrence the model")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .nemo file")
    args = parser.parse_args()

    split_json_file(args.input_path)
    longFile(args.model_path)