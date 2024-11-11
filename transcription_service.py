import os
import json
from pathlib import Path
import platform
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize

from typing import Dict, List, Any
from copy import deepcopy
import threading


def reorganize_whisper_output(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Reorganizes Whisper output by combining text into proper sentences while preserving word timestamps.
    
    Args:
        result: Original Whisper transcription result with word timestamps
        
    Returns:
        Dict with reorganized segments based on proper sentence boundaries
    """
    # Download necessary NLTK data (run once)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    # Create a new result dictionary
    
    result["segments"] = [
            {
                'id': idx,
                'start': seg['start'],
                'end': seg['end'],
                'text': seg['text'].strip(),
                'words': [
                    {
                        'start': word['start'],
                        'end': word['end'],
                        'word': word['word']
                    }
                    for word in seg.get('words', [])
                ]
            }
            for idx, seg in enumerate(result['segments'])
            if seg['text'].strip()
        ]

    nltk_result = deepcopy(result)
    nltk_result["segments"] = []
    
    # Combine all words with their timestamps
    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            all_words.extend(segment["words"])
    
    # Combine all text
    full_text = " ".join(word["word"].strip() for word in all_words)
    
    # Use NLTK to split into proper sentences
    sentences = nltk.sent_tokenize(full_text)
    
    current_word_idx = 0
    
    # Process each sentence
    for sentence_idx, sentence in enumerate(sentences):
        sentence_words = sentence.split()
        segment_words = []
        
        # Create new segment for the sentence
        new_segment = {
            "id": sentence_idx,
            "text": sentence,
            "words": []
        }
        
        # Match words and their timestamps
        while len(segment_words) < len(sentence_words) and current_word_idx < len(all_words):
            current_word = all_words[current_word_idx]
            segment_words.append(current_word)
            new_segment["words"].append(current_word)
            current_word_idx += 1
            
        # Set start and end times for the segment
        if new_segment["words"]:
            new_segment["start"] = new_segment["words"][0]["start"]
            new_segment["end"] = new_segment["words"][-1]["end"]
            
        nltk_result["segments"].append(new_segment)

    for seg in range(1,len(nltk_result["segments"])-1):
        previous_end = nltk_result["segments"][seg-1]["end"]
        next_start = nltk_result["segments"][seg+1]["start"]
        if nltk_result["segments"][seg]["end"] < next_start:
            nltk_result["segments"][seg]["end"] = next_start
        #if nltk_result["segments"][seg]["start"] > previous_end:
        #    nltk_result["segments"][seg]["start"] = previous_end
        #nltk_result["segments"][seg]["start"] = round(nltk_result["segments"][seg]["start"], 0)
        #nltk_result["segments"][seg]["end"] = round(nltk_result["segments"][seg]["end"], 0)+1
    return nltk_result


class TranscriptionService:

    _instance = None
    _lock = threading.Lock()    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TranscriptionService, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance    
    def __init__(self):
        if self._initialized:
            return
            
        self.is_mac = platform.system() == 'Darwin' and platform.processor() == 'arm'
        
        if self.is_mac:
            import mlx_lm
            import mlx_whisper
            self.mlx_whisper = mlx_whisper
        else:
            # Import and initialize CUDA-related libraries
            import torch
            torch.cuda.empty_cache()  # Clear CUDA cache
            
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float32",
                cpu_threads=4,
                num_workers=1
            )
            
        self._initialized = True
    def _ensure_model_loaded(self):
        """Ensure the model is loaded in the current thread context"""
        if not self.is_mac and self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float32",
                cpu_threads=4,
                num_workers=1
            )
    def transcribe(self, file_path: str) -> dict:
        """
        Transcribe audio file using either MLX (Mac) or faster-whisper (CUDA)
        """
        try:
            if self.is_mac:
                # MLX transcription for Mac
                result = self.mlx_whisper.transcribe(
                    file_path,
                    path_or_hf_repo="./mlx_models/whisper-large-v3-german",
                    word_timestamps=True
                )
                result = reorganize_whisper_output(result)  
            else:
                # faster-whisper transcription for CUDA

                segments, info = self.model.transcribe(
                    file_path,
                    word_timestamps=True,
                    vad_filter=True,
                    language="de",
                    vad_parameters=dict(
                        min_silence_duration_ms=500,
                        speech_pad_ms=400
                    )
                )
                
                # Convert faster-whisper format to match MLX format
                result = {
                    "segments": []
                }
                
                for segment in segments:
                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text,
                        "words": []
                    }
                    
                    # Add word-level information if available
                    if segment.words:
                        for word in segment.words:
                            segment_dict["words"].append({
                                "start": word.start,
                                "end": word.end,
                                "word": word.word
                            })
                    
                    result["segments"].append(segment_dict)

            return result

        except Exception as e:
            print(f"Transcription error: {str(e)}")
            raise

    def get_model_info(self) -> dict:
        """
        Return information about the current transcription model
        """
        if self.is_mac:
            return {
                "platform": "MLX (Apple Silicon)",
                "model": "whisper-large-v3-german",
                "device": "Apple Silicon"
            }
        else:
            return {
                "platform": "faster-whisper (CUDA)",
                "model": "large-v3",
                "device": "CUDA GPU"
            }


if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Transcribe audio files')
    parser.add_argument('in_audio_file_path', help='Path to the audio file to transcribe')
    parser.add_argument('out_json_file_path', help='Path to the output JSON file')
    
    # Parse arguments
    args = parser.parse_args()
    print(args.in_audio_file_path)
    print(args.out_json_file_path)
    service = TranscriptionService()

    print(service.get_model_info())
    result = service.transcribe(args.in_audio_file_path)

    with open(args.out_json_file_path, 'w') as f:
        json.dump(result, f, indent=2)
