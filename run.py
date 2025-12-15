import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutomaticSpeechRecognitionPipeline,
)
import argparse
import subprocess
import tempfile
import os
import soundfile as sf
import numpy as np


def transcribe_audio(file_name):
    # 1. Load audio
    # Use ffmpeg to convert any audio/video file to standardized format
    print(f"Processing audio from: {file_name}")

    # Create a temporary audio file
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio_path = temp_audio.name
    temp_audio.close()

    # Extract/convert audio using ffmpeg (works for both audio and video files)
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            file_name,
            "-vn",  # No video
            "-acodec",
            "pcm_s16le",  # PCM 16-bit
            "-ar",
            "16000",  # 16kHz sample rate
            "-ac",
            "1",  # Mono
            "-y",  # Overwrite output file
            temp_audio_path,
        ],
        check=True,
        capture_output=True,
    )

    try:
        # Load audio with soundfile (avoids torchcodec dependency)
        waveform, sample_rate = sf.read(temp_audio_path, dtype="float32")

        # 2. Preprocess
        # Ensure waveform is numpy array
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform)

        # Since ffmpeg already outputs 16kHz mono, no resampling needed
        assert sample_rate == 16000, f"Expected 16kHz, got {sample_rate}Hz"

        # 3. Load Model
        processor = WhisperProcessor.from_pretrained(
            pretrained_model_name_or_path="/opt/hf_models/Breeze-ASR-25"
        )
        model = (
            WhisperForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path="/opt/hf_models/Breeze-ASR-25"
            )
            .to("cuda")
            .eval()
        )

        # 4. Build Pipeline
        asr_pipeline = AutomaticSpeechRecognitionPipeline(
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=0,
        )

        # 5. Inference
        output = asr_pipeline(waveform, return_timestamps=True)
        print("Result:", output["text"])
    finally:
        # Clean up temporary audio file
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)


# Set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio file using Whisper."
    )
    parser.add_argument(
        "--file_name", type=str, required=True, help="Path to the input audio file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments from the command line
    args = parse_args()

    # Call the transcription function with the provided file_name
    transcribe_audio(args.file_name)
