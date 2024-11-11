# WhisperPlay

WhisperPlay is a web application that provides transcription services for audio files. It allows users to upload audio files, transcribe them with Whisper, and manage the resulting transcripts.

Users can play the audio file and navigate the transcript by sentences.

It is a great tool for learning foreign languages.

## Features

- **Audio Upload**: Supports multiple audio formats including MP3, WAV, M4A, and OGG.
- **Transcription**: Automatically transcribes uploaded audio files using a transcription service.
- **User Authentication**: Secure login system to manage user sessions.
- **Job Management**: View, update, and delete transcription jobs.
- **Logging**: Comprehensive logging for debugging and monitoring.

## Installation

### Prerequisites

- Python 3.8 or higher
- Flask
- NLTK
- Bcrypt
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
   python app.py
   ```

6. **Access the application**:
   Open your web browser and go to `http://localhost:5001`.

## Usage

1. **Upload Audio**: Navigate to the upload page and select an audio file to upload.
2. **Transcribe**: The application will automatically transcribe the uploaded audio.
3. **Manage Jobs**: View the status of transcription jobs, update titles, or delete jobs as needed.

## Configuration

- **Environment Variables**: Set `CUDA_VISIBLE_DEVICES` to specify which GPU to use.
- **Application Settings**: Modify `app.config` in `app.py` to change upload and transcript directories, and other settings.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact [reeyarn@gmail.com](mailto:reeyarn@gmail.com).
