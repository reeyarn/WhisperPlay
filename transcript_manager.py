import os
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class TranscriptManager:
    def __init__(self, uploads_dir: str = "uploads", transcripts_dir: str = "transcripts", current_user: str = None):
        """
        Initialize the transcript manager with directory paths.
        
        Args:
            uploads_dir: Directory where audio files are uploaded
            transcripts_dir: Directory where transcript JSON files are stored
        """
        self.set_dirs(uploads_dir, transcripts_dir)
        self.set_current_user(current_user)
    def set_dirs(self, uploads_dir: str, transcripts_dir: str):
        self.uploads_dir = Path(uploads_dir)
        self.transcripts_dir = Path(transcripts_dir)
        # Create directories if they don't exist
        self.uploads_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
    def set_current_user(self, user: str):
        self.current_user = user

    def save_transcript(self, audio_filename: str, transcript_data: dict) -> str:
        """
        Save transcript data to a JSON file named after the audio file.
        
        Args:
            audio_filename: Name of the original audio file
            transcript_ Dictionary containing the transcript data
            
        Returns:
            Path to the saved transcript file
        """
        # Create transcript filename from audio filename
        transcript_filename = Path(audio_filename).stem + ".json"
        transcript_path = self.transcripts_dir / transcript_filename
        self.transcripts_dir.mkdir(exist_ok=True)
        # Save transcript data
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
        return str(transcript_path)

    def get_transcript(self, audio_filename: str) -> Optional[dict]:
        """
        Retrieve transcript data for a specific audio file.
        
        Args:
            audio_filename: Name of the audio file
            
        Returns:
            Dictionary containing transcript data if found, None otherwise
        """
        transcript_filename = Path(audio_filename).stem + ".json"
        transcript_path = self.transcripts_dir / transcript_filename
        
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None

    def get_jobs_status(self) -> List[Dict[str, any]]:
        jobs = []
        audio_files = [f for f in self.uploads_dir.glob('*')
                    if f.suffix.lower() in ('.mp3', '.wav', '.m4a', '.ogg')]

        # Create directories if they don't exist
        self.uploads_dir.mkdir(exist_ok=True)
        self.transcripts_dir.mkdir(exist_ok=True)
        
        for audio_file in audio_files:
            transcript_path = self.transcripts_dir / f"{audio_file.stem}.json"
            upload_time = audio_file.stat().st_mtime
            formatted_time = datetime.fromtimestamp(upload_time).strftime('%Y-%m-%d %H:%M:%S')
            
            job_status = {
                "filename": audio_file.name,
                "user": self.current_user,
                "upload_time": formatted_time,  # Already formatted time
                "has_transcript": transcript_path.exists(),
                "transcript_path": str(transcript_path) if transcript_path.exists() else None,
                "title": ""  # Default empty title
            }
            
            if job_status["has_transcript"]:
                transcript_stat = transcript_path.stat()
                job_status["transcript_time"] = transcript_stat.st_mtime
            # Get title from transcript file if it exists
            if transcript_path.exists():
                try:
                    with open(transcript_path, 'r') as f:
                        transcript = json.load(f)
                        job_status["title"] = transcript.get('title', '')
                        if transcript.get('deleted', False):
                            continue                        
                except:
                    pass
            
                        
            jobs.append(job_status)
        
        jobs.sort(key=lambda x: x["upload_time"], reverse=True)
        return jobs
    def delete_job(self, filename: str) -> bool:
        """
        Delete both the audio file and its transcript if they exist.
        Args:
            filename: Name of the audio file
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Get the paths
            audio_path = self.uploads_dir / filename
            #transcript_path = self.transcripts_dir / f"{Path(filename).stem}.json"
            
            # Delete audio file if it exists
            if audio_path.exists():
                audio_path.unlink()
            
            # Delete transcript if it exists
            #if transcript_path.exists():
            #    transcript_path.unlink()
            
            return True
        except Exception as e:
            
            print(f"Error deleting job: {str(e)}")
            return False
    def update_job_title(self, filename: str, title: str) -> bool:
        """
        Update the title of a job in its transcript file
        """
        try:
            transcript_path = self.transcripts_dir / f"{Path(filename).stem}.json"
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    transcript = json.load(f)
                
                # Add or update the title
                transcript['title'] = title
                
                with open(transcript_path, 'w') as f:
                    json.dump(transcript, f, indent=2)
                return True
            return False
        except Exception as e:
            print(f"Error updating title: {str(e)}")
            return False
    def mark_job_deleted(self, filename: str) -> bool:
        """
        Mark a job as deleted in its transcript file
        """
        try:
            transcript_path = self.transcripts_dir / f"{Path(filename).stem}.json"
            if transcript_path.exists():
                with open(transcript_path, 'r') as f:
                    transcript = json.load(f)
                
                # Mark as deleted
                transcript['deleted'] = True
                
                with open(transcript_path, 'w') as f:
                    json.dump(transcript, f, indent=2)
                return True
            else:
                return self.delete_job(filename)
        except Exception as e:
            print(f"Error marking job as deleted: {str(e)}")
            return False        