# WhisperPlay

WhisperPlay is a web application that provides transcription services for audio files. It allows users to upload audio files, transcribe them with Whisper, and manage the resulting transcripts.

Users can play the audio file and navigate the transcript by sentences.

It is a great tool for learning foreign languages.

## Features
- **Reading by Sentence**: Navigate the transcript by sentences and listen to the audio with exact timestamps located with transcript.
- **Transcription**: Automatically transcribes uploaded audio files using a transcription service.
- **Audio Upload**: Supports multiple audio formats including MP3, WAV, M4A, and OGG.
- **User Authentication**: Secure login system to manage user sessions.
- **Job Management**: View, update, and delete transcription jobs.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Interface

### Index Page

![WhisperPlay Screenshot](https://github.com/reeyarn/WhisperPlay/blob/main/screenshots/index.png)

### Transcript Reading Page
![WhisperPlay Screenshot](https://github.com/reeyarn/WhisperPlay/blob/main/screenshots/transcript.png)


## Installation



### Prerequisites

- Python 3.8 or higher
- Flask
- NLTK
- Bcrypt
- mlx_whisper for MacOS
- fast_whisper for Linux with CUDA
- Other dependencies listed in `requirements.txt`

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/reeyarn/WhisperPlay.git
   cd WhisperPlay
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data**:
   ```python
   import nltk
   nltk.download('punkt')
   ```

5. **Run the application**:
   ```bash
   python main.py
   ```

6. **Access the application**:
   Open your web browser and go to `http://localhost:5001`.
   Or you can change the port in `main.py` to your desired port.

## Usage

1. **Upload Audio**: Navigate to the upload page and select an audio file to upload.
2. **Transcribe**: The application will automatically transcribe the uploaded audio.
3. **Manage Jobs**: View the status of transcription jobs, update titles, or delete jobs as needed.
4. **Listen to the Transcript**: Navigate the transcript by sentences and listen to the audio with exact timestamps located with transcript.

## Configuration

- **Environment Variables**: Set `CUDA_VISIBLE_DEVICES` to specify which GPU to use.
- **Application Settings**: Modify `app.config` in `main.py` to change upload and transcript directories, and other settings.

## Preparing MLX Whisper Model

```git clone https://github.com/ml-explore/mlx-examples.git
python mlx-examples/whisper/convert.py --torch-name-or-path primeline/whisper-large-v3-german --mlx-path mlx_models/whisper-large-v3-german
```
## Acknowledgments

I would like to express our gratitude to the following projects for their contributions and inspiration:


- Cursor, Anthropic, devv.ai, and all the other AI code editors and services that help me to build this app.
- [whisper_mlx @ ml-explore](https://github.com/ml-explore/): Apple's MLX for MacOS to run OpenAI's Whisper.
- [fast_whisper](https://github.com/SYSTRAN/faster-whisper/): A project to run efficient transcription processing with CUDA.
- HuggingFace community.


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [reeyarn@gmail.com](mailto:reeyarn@gmail.com).
