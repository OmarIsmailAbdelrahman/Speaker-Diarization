import logging
import traceback
import diart.operators as dops
import rich
import rx.operators as ops
from diart import SpeakerDiarization, SpeakerDiarizationConfig
from diart.sources import MicrophoneAudioSource
import diart.models as m
import torch
from huggingface_hub import login
import os
import sys
import numpy as np
from pyannote.core import Segment
from contextlib import contextmanager
import numpy as np
from pyannote.core import Annotation, SlidingWindowFeature, SlidingWindow
import traceback
import rich
import rx.operators as ops
import diart.operators as dops
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import nemo.collections.asr as nemo_asr
import sounddevice as sd
import argparse

login(token='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')


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class ChadNemo:
    def __init__(self, model_name="Some name of our experiment.nemo"):
        self.model = nemo_asr.models.EncDecCTCModel.restore_from(model_name)
        self._buffer = ""
        self.model.eval()
        self.model.cuda()
        self.model.preprocessor.featurizer.pad_to = 0
        self.model.preprocessor.featurizer.dither = 0.0

    def transcribe(self, waveform):
        waveform = np.array(waveform.data.astype("float32"))
        speech = torch.from_numpy(waveform.reshape(1, -1))
        length = torch.from_numpy(np.array([speech.shape[1]]))
        logits, logits_len, greedy_predictions = self.model.forward(
            input_signal=speech.float().to(self.model.device),
            input_signal_length=length.to(self.model.device)
        )

        transcription, _ = self.model.decoding.ctc_decoder_predictions_tensor(
            decoder_outputs=greedy_predictions,
            decoder_lengths=logits_len,
            return_hypotheses=False,
        )
        return {"text": transcription[0]}

    def identify_speakers(self, transcription, diarization, time_shift):
        speaker_captions = []
        segments = transcription.split('.')
        for segment in segments:
            if not segment.strip():
                continue

            start = time_shift
            end = time_shift + len(segment.split())
            dia = diarization.crop(Segment(start, end))

            speakers = dia.labels()
            num_speakers = len(speakers)
            if num_speakers == 0:
                caption = (-1, segment)
            elif num_speakers == 1:
                spk_id = int(speakers[0].split("speaker")[1])
                caption = (spk_id, segment)
            else:
                max_speaker = int(np.argmax([
                    dia.label_duration(spk) for spk in speakers
                ]))
                caption = (max_speaker, segment)
            speaker_captions.append(caption)

        return speaker_captions

    def __call__(self, diarization, waveform):
        transcription = self.transcribe(waveform)
        self._buffer += transcription["text"]
        time_shift = waveform.sliding_window.start
        speaker_transcriptions = self.identify_speakers(transcription["text"], diarization, time_shift)
        return speaker_transcriptions


def concat(chunks, collar=0.05):
    first_annotation = chunks[0][0]
    first_waveform = chunks[0][1]
    annotation = Annotation(uri=first_annotation.uri)
    data = []
    for ann, wav in chunks:
        annotation.update(ann)
        data.append(wav.data)
    annotation = annotation.support(collar)
    window = SlidingWindow(
        first_waveform.sliding_window.duration,
        first_waveform.sliding_window.step,
        first_waveform.sliding_window.start,
    )
    data = np.concatenate(data, axis=0)
    return annotation, SlidingWindowFeature(data, window)


def colorize_transcription(transcription):
    colors = 2 * [
        "\033[91m",  # bright_red
        "\033[94m",  # bright_blue
        "\033[92m",  # bright_green
        "\033[38;5;214m",  # orange3
        "\033[38;5;198m",  # deep_pink1
        "\033[93m",  # yellow2
        "\033[95m",  # magenta
        "\033[96m",  # cyan
        "\033[35m",  # bright_magenta
        "\033[34m",  # dodger_blue2
    ]
    result = []
    reset_color = "\033[0m"
    for speaker, text in transcription:
        if speaker == -1:
            result.append(text)
        else:
            result.append(f"{colors[speaker]}{text}{reset_color}")
    return "\n".join(result)


def main(model_name, mic_device):
    config = SpeakerDiarizationConfig(
        duration=5,
        step=0.5,
        latency="min",
        tau_active=0.5,
        rho_update=0.1,
        delta_new=0.57,
        segmentation=m.SegmentationModel.from_pretrained('pyannote/segmentation-3.0'),
        embedding=m.EmbeddingModel.from_pretrained('nvidia/speakerverification_en_titanet_large'),
        device=torch.device("cuda")
    )

    dia = SpeakerDiarization(config)
    source = MicrophoneAudioSource(config.step, device=mic_device)

    asr = ChadNemo(model_name)
    transcription_duration = 2
    batch_size = int(transcription_duration // config.step)

    source.stream.pipe(
        dops.rearrange_audio_stream(
            config.duration, config.step, config.sample_rate
        ),
        ops.buffer_with_count(count=batch_size),
        ops.map(dia),
        ops.map(concat),
        ops.filter(lambda ann_wav: ann_wav[0].get_timeline().duration() > 0),
        ops.starmap(asr),
        ops.map(colorize_transcription),
    ).subscribe(
        on_next=rich.print,
        on_error=lambda _: traceback.print_exc()
    )

    print("Listening...")
    source.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time speaker diarization and transcription.")
    parser.add_argument('--model_name', type=str, required=True, help='Name of the NeMo ASR model.')
    parser.add_argument('--mic_device', type=int, required=True, help='Microphone device ID.')

    args = parser.parse_args()
    main(args.model_name, args.mic_device)
