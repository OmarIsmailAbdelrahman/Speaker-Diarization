import os
import json
import librosa
import argparse


def create_manifest(audio_dir, data_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Open the output file
    with open(os.path.join(data_dir, 'input_manifest.json'), 'w') as fp:
        # Iterate over each audio file in the directory
        for audio_filename in os.listdir(audio_dir):
            if audio_filename.endswith('.wav'):  # or any other audio file extension
                audio_filepath = os.path.join(audio_dir, audio_filename)

                # Get duration of the audio file using librosa
                duration = librosa.get_duration(filename=audio_filepath)

                meta = {
                    'audio_filepath': audio_filepath,
                    'offset': 0,
                    'duration': duration,
                    'label': 'infer',
                    'text': '-',
                    'num_speakers': None,
                    'rttm_filepath': None,
                    'uem_filepath': None
                }
                # Write each metadata object as a separate line in the JSON file
                fp.write(json.dumps(meta) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a manifest JSON file for a speaker diarization dataset.")
    parser.add_argument('audio_dir', type=str, help='Directory containing the audio files.')
    parser.add_argument('data_dir', type=str, help='Directory to save the output manifest file.')

    args = parser.parse_args()

    create_manifest(args.audio_dir, args.data_dir)
