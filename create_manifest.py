import os
import json
import librosa
import argparse


def create_manifest(audio_dir):
    # Create the output directory if it doesn't exist

    # Open the output file
    with open('input_manifest.json', 'w') as fp:
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
    parser.add_argument('--audio_dir', type=str, required=True, help='Directory containing the audio files.')

    args = parser.parse_args()

    create_manifest(args.audio_dir)
