# Breeze ASR 25

## Prerequisites

- ffmpeg
- python 3.12
- rocm drivers
- uv (Python package manager)

**Note:** torchcodec is not supported on rocm, so we need to use ffmpeg to convert the audio file to wav format.

## Setup

### 1. Clone the repository with submodules

```bash
git clone --recursive https://github.com/YOUR_USERNAME/Breeze-ASR-25.git
cd Breeze-ASR-25
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Install dependencies

Using uv (recommended):

```bash
uv sync
```

Or using pip:

```bash
pip install -r pyproject.toml
```

### 3. Download the Breeze-ASR-25 model

The model needs to be downloaded and placed in `/opt/hf_models/Breeze-ASR-25`. You can download it from Hugging Face:

```bash
# Using huggingface-cli
huggingface-cli download MediaTek-Research/Breeze-7B-32k-Instruct-v1_0 --local-dir /opt/hf_models/Breeze-ASR-25
```

Or modify the model path in `run.py` to point to your preferred location.

### 4. Verify ROCm installation

Check if ROCm is properly configured:

```bash
python check_rocm.py
```

This should display your GPU information.

## Usage

### Transcribe an audio or video file

```bash
python run.py --file_name <path_to_audio_or_video_file>
```

Example:

```bash
python run.py --file_name videoplayback.mp4
```

The script will:

1. Extract audio from the input file (supports both audio and video formats)
2. Convert it to 16kHz mono WAV format using ffmpeg
3. Process it through the Breeze-ASR-25 Whisper model
4. Output the transcribed text

## Technical Notes

### Audio Processing Pipeline

1. **Input**: Any audio/video format supported by ffmpeg
2. **Conversion**: ffmpeg converts to PCM 16-bit, 16kHz, mono WAV
3. **Loading**: soundfile library loads the audio as float32 numpy array
4. **Inference**: Whisper model processes and transcribes
5. **Cleanup**: Temporary files are automatically removed

### Model Details

- **Base Model**: Whisper (fine-tuned for Traditional Chinese)
- **Sample Rate**: 16kHz
- **Channels**: Mono
- **Device**: CUDA (ROCm for AMD GPUs)

### ROCm Compatibility

This project is designed to work with AMD GPUs using ROCm. The main consideration is that `torchcodec` doesn't support ROCm, so we use ffmpeg + soundfile for audio loading instead.
