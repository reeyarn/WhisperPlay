

from typing import Dict, List, Any
from copy import deepcopy
import nltk
#nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datetime import datetime
import json
from flask import send_from_directory
import bcrypt

import logging


import subprocess
from flask import Flask, render_template, jsonify, request, Response, abort, session
import secrets
from werkzeug.utils import secure_filename
import os
from transcript_manager import TranscriptManager
#import mlx_whisper
from copy import deepcopy

# Configure CUDA to use the first GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Initialize transcription service
transcription_service = None

def get_transcription_service():
    global transcription_service
    if transcription_service is None:
        #transcription_service = TranscriptionService()
        transcription_service = None
    return transcription_service

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TRANSCRIPT_FOLDER'] = 'transcripts'
transcript_manager = TranscriptManager(uploads_dir=app.config['UPLOAD_FOLDER'], transcripts_dir=app.config['TRANSCRIPT_FOLDER'])
app.secret_key = secrets.token_hex(16)  # Required for session management

app.config['users'] = {
    "podcast": bcrypt.hashpw('addyourpassword'.encode('utf-8'), bcrypt.gensalt(10)),
    }
app.config["current_user"] = None

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Create a file handler to log to a file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the file handler to the app's logger
app.logger.addHandler(file_handler)



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
                        'word': word['word'] if "word" in word.keys() else word['text']
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

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'mp3', 'wav', 'm4a', 'ogg'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this custom filter
@app.template_filter('datetime')
def format_datetime(timestamp_str: str):
    """Convert a timestamp to a formatted date string
    timestamp_str = "2024-11-10 18:57:49"
    """
    try:
        # Convert timestamp to float in case it's a string
        datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        return timestamp_str
    except ValueError:
        try:
            # If that fails, try to parse as timestamp float
            timestamp_float = float(timestamp_str)
            return datetime.fromtimestamp(timestamp_float).strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            app.logger.error(f"Invalid date format: {timestamp_str}")
            return timestamp_str    
@app.route('/')
def index():
    """Render the main page with the list of jobs."""
    jobs = transcript_manager.get_jobs_status()
    return render_template('index.html', jobs=jobs)

@app.route('/api/jobs')
def get_jobs():
    """API endpoint to get the current status of all jobs."""
    jobs = transcript_manager.get_jobs_status()
    return jsonify(jobs)



@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'audio' not in request.files:
        return jsonify({
            'success': False,
            'message': 'No file part'
        }), 400
    
    file = request.files['audio']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'message': 'No selected file'
        }), 400
    
    # Check file size (32MB limit)
    if request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            'success': False,
            'message': 'File size exceeds 256MB limit'
        }), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(transcript_manager.uploads_dir, filename)
        
        # Save the file
        file.save(file_path)
        
        # Start transcription automatically
        try:
            
            return jsonify({
                'success': True,
                'message': 'File uploaded  successfully',
                'filename': filename,
                
            })
        except Exception as e:
            app.logger.error(f'Transcription failed: {str(e)}')
            return jsonify({
                'success': False,
                'message': f'Transcription failed: {str(e)}'
            }), 500
    
    return jsonify({
        'success': False,
        'message': 'File type not allowed'
    }), 400

from transcription_service import TranscriptionService

# Initialize the transcription service

transcription_service = TranscriptionService()

# Update your transcribe route to use the new service
@app.route('/api/transcribe/<filename>', methods=['POST'])
def transcribe_audio(filename):
    """Transcribe an audio file"""
    if not session.get('authenticated', False):
        return jsonify({
            'success': False,
            'message': 'Authentication required'
        }), 401

    try:
        app.logger.info(f'Transcribing file: {filename}')
        file_path = os.path.join(transcript_manager.uploads_dir, filename)
        # Get transcription using the appropriate service
        transcript_path = os.path.join(transcript_manager.transcripts_dir, 
                                    f"{os.path.splitext(filename)[0]}.json")
            
        app.logger.info(f'File path: {file_path}')

        
        #service = get_transcription_service()
        #result = service.transcribe(file_path)
        # Run transcription service as a subprocess
        app.logger.info(f'Running transcription service. python3 transcription_srvice.py "{file_path}" "{transcript_path}"')
        process = subprocess.run(
            ['python3', 'transcription_service.py', file_path, transcript_path],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            app.logger.error(f'Transcription failed in subprocess: {process.stderr}')
            return jsonify({
            'success': False,
                'message': f'Transcription failed in subprocess: {process.stderr}'
            }), 500        
        
        return jsonify({
            'success': True,
            'message': f'Transcription completed successfully. Json saved to: {transcript_path}'
        })        
        
        
        #with open(transcript_path, 'w') as f:
        #    json.dump(result, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': 'Transcription completed successfully'
        })

    except Exception as e:
        import traceback
        traceback.print_exc()  # Print full stack trace
        return jsonify({
            'success': False,
            'message': f'Transcription failed: {str(e)}'
        }), 500        

# Add a route to get model info
@app.route('/api/model-info')
def get_model_info():
    """Get information about the current transcription model"""
    #service = get_transcription_service()
    service = None
    return jsonify(service.get_model_info())



@app.route('/api/transcript/<filename>')
def get_transcript(filename):
    """API endpoint to get a specific transcript."""
    transcript = transcript_manager.get_transcript(filename)
    if transcript:
        return jsonify(transcript)
    return jsonify({"error": "Transcript not found"}), 404

@app.route('/resource/<path:path>')
def serve_resource(path):
    """Serve static resources from the resource directory"""
    return send_from_directory('resource', path)

@app.route('/transcript_anonymous/<filename>')
def view_transcript_anonymous(filename):
    """
    Route to view a transcript.
    Returns a rendered template with the transcript content.
    """
    transcript = transcript_manager.get_transcript(filename)
    try:
        transcript = reorganize_whisper_output(transcript)
    except Exception as e:
        app.logger.error(f'Error reorganizing transcript: {str(e)}')
        #transcript = transcript
    
    return render_template('transcript.html', 
                         filename=filename, 
                         transcript=transcript)

@app.route('/transcript/<user>/<filename>')
def view_user_transcript(user, filename):
    """
    Route to view a transcript.
    Returns a rendered template with the transcript content.
    """
    transcript = transcript_manager.get_transcript(filename)
    try:
        transcript = reorganize_whisper_output(transcript)
    except Exception as e:
        app.logger.error(f'Error reorganizing transcript: {str(e)}')
        #transcript = transcript
    
    return render_template('transcript.html', 
                         filename=filename, 
                         transcript=transcript, user=user)


# ... keep your other existing routes ...

@app.route('/uploads_old/<filename>')
def serve_audio_old(filename):
    """
    Route to serve audio files from the uploads directory.
    """
    app.logger.info(f'Serving audio file: {filename} from {transcript_manager.uploads_dir}')
    return send_from_directory(transcript_manager.uploads_dir, filename)

@app.route('/uploads_anonymous/<filename>')
def serve_audio_anonymous(filename):
    """
    Route to serve audio files from the uploads directory.
    """
    app.logger.info(f'Attempting to serve audio file: {filename}')
    app.logger.info(f'Current user: {app.config["current_user"]}')
    app.logger.info(f'Upload directory: {transcript_manager.uploads_dir}')
    
    if not os.path.exists(os.path.join(transcript_manager.uploads_dir, filename)):
        app.logger.error(f'File not found: {filename}')
        abort(404)
        
    return send_from_directory(transcript_manager.uploads_dir, filename)

@app.route('/uploads/<user>/<filename>')
def serve_audio_user(user, filename):
    """
    Route to serve audio files from the uploads directory.
    """
    app.logger.info(f'Attempting to serve user {user} audio file: {filename}')
    app.logger.info(f'Current user: {app.config["current_user"]}')
    app.logger.info(f'Upload directory: {transcript_manager.uploads_dir}')
    
    if not os.path.exists(os.path.join(transcript_manager.uploads_dir, filename)):
        app.logger.error(f'File not found: {filename}')
        abort(404)
        
    return send_from_directory(os.path.join(transcript_manager.uploads_dir), filename)



# Add this new route to your app.py
@app.route('/api/jobs/<filename>', methods=['DELETE'])
def delete_job(filename):
    """
    Delete a job and its associated files (audio and transcript).
    """
    try:
        success = transcript_manager.mark_job_deleted(filename)
        if success:
            return jsonify({
                'success': True,
                'message': 'Job deleted successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to delete job'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# Authentication constants

@app.route('/api/login', methods=['POST'])
def login():
    """Handle login requests"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    if username in app.config['users']:
        if bcrypt.checkpw(password.encode('utf-8'), app.config['users'][username]):
            session['authenticated'] = True
            app.config["current_user"] = username   
            transcript_manager.set_current_user(username)
            transcript_manager.set_dirs( os.path.join(app.config['UPLOAD_FOLDER'], app.config["current_user"]), os.path.join(app.config['TRANSCRIPT_FOLDER'], app.config["current_user"]))
        #transcript_manager.uploads_dir = os.path.join(app.config['UPLOAD_FOLDER'], app.config["current_user"])
        #transcript_manager.transcripts_dir = os.path.join(app.config['TRANSCRIPT_FOLDER'], app.config["current_user"])
            return jsonify({
                'success': True,
                'message': 'Login successful'
            })
    
    return jsonify({
        'success': False,
        'message': 'Invalid credentials'
    }), 401

@app.route('/api/check-auth')
def check_auth():
    """Check if user is authenticated"""
    if session.get('authenticated', False):
        return jsonify({
            'authenticated': True,
            "user": app.config["current_user"] 
        })
    else:
        return jsonify({
            'authenticated': False,
        })

@app.route('/api/logout')
def logout():
    """Handle logout requests"""
    session.pop('authenticated', None)
    app.config["current_user"] = None
    transcript_manager.set_current_user(None)
    #transcript_manager.uploads_dir = app.config['UPLOAD_FOLDER']
    #transcript_manager.transcripts_dir = app.config['TRANSCRIPT_FOLDER']
    transcript_manager.set_dirs(app.config['UPLOAD_FOLDER'], app.config['TRANSCRIPT_FOLDER'] )
    return jsonify({
        'success': True,
        'message': 'Logged out successfully'
    })
@app.route('/api/jobs/<filename>/title', methods=['POST'])
def update_job_title(filename):
    """Update the title of a job"""
    if not session.get('authenticated', False):
        return jsonify({
            'success': False,
            'message': 'Authentication required'
        }), 401
    
    data = request.json
    title = data.get('title', '').strip()
    
    if not title:
        return jsonify({
            'success': False,
            'message': 'Title cannot be empty'
        }), 400
    app.logger.info(f'Updating title for {filename} to {title}')
    success = transcript_manager.update_job_title(filename, title)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Title updated successfully'
        })
    
    return jsonify({
        'success': False,
        'message': 'Failed to update title'
    }), 500

@app.route('/api/jobs/<filename>/delete', methods=['POST'])
def mark_job_deleted(filename):
    """Mark a job as deleted"""
    if not session.get('authenticated', False):
        return jsonify({
            'success': False,
            'message': 'Authentication required'
        }), 401
    
    success = transcript_manager.mark_job_deleted(filename)
    
    if success:
        return jsonify({
            'success': True,
            'message': 'Job marked as deleted successfully'
        })
    
    return jsonify({
        'success': False,
        'message': 'Failed to mark job as deleted'
    }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)