# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
from settings.config import load_settings # Import the function

current_settings = load_settings()

parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=current_settings.get("chunk_size"),
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default=current_settings.get("model_path"),
    required=False
)
parser.add_argument(
    "--model_path2",
    help="The path of a second specific model to load",
    type=str,
    default=current_settings.get("model_path2"),
    required=False
)
parser.add_argument(
    "--model_path3",
    help="The path of a third specific model to load",
    type=str,
    default=current_settings.get("model_path3"),
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default=current_settings.get("inference_framework"),
    required=False
)
parser.add_argument(
    "--silence_threshold",
    help="RMS audio level below which audio is considered silent",
    type=int,
    default=current_settings.get("silence_threshold"), # Adjust this based on your microphone and environment
    required=False
)
parser.add_argument(
    "--silence_duration_seconds",
    help="Duration of silence in seconds to stop recording after speech has started",
    type=float,
    default=current_settings.get("silence_duration_seconds"),
    required=False
)
parser.add_argument(
    "--no_speech_timeout_seconds",
    help="Duration in seconds to wait for speech after wakeword detection before timing out",
    type=float,
    default=current_settings.get("no_speech_timeout_seconds"),
    required=False
)

args=parser.parse_args()

# Ensure chunk_size has a valid integer value
if args.chunk_size is None:
    print("Warning: chunk_size not found in settings or command line. Using default value of 1280.")
    args.chunk_size = 1280 # Default chunk size if not provided

# Get microphone stream
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
user_models = []
if args.model_path and args.model_path.strip() != "":
    user_models.append(args.model_path)
if args.model_path2 and args.model_path2.strip() != "":
    user_models.append(args.model_path2)
if args.model_path3 and args.model_path3.strip() != "":
    user_models.append(args.model_path3)

if user_models:
    owwModel = Model(wakeword_models=user_models, inference_framework=args.inference_framework)
else:
    # Pass empty lists for wakeword_models and wakeword_model_names
    # to work around an issue in openwakeword where it might try to zip None values
    # when default models are intended. This should allow it to proceed and load defaults.
    owwModel = Model(wakeword_models=[], wakeword_model_names=[], inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    print("Starting service...")
    print("\n\n")
    print("Listening for wakewords")

    is_recording = False
    recorded_frames = []
    consecutive_silent_chunks = 0
    speech_has_started = False

    # Calculate how many consecutive silent chunks are needed for different timeouts
    SILENCE_CHUNKS_FOR_SPEECH_END = int(args.silence_duration_seconds * RATE / CHUNK)
    NO_SPEECH_TIMEOUT_CHUNKS = int(args.no_speech_timeout_seconds * RATE / CHUNK)


    while True:
        audio_data = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

        if is_recording:
            recorded_frames.append(audio_data)
            
            rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))

            if not speech_has_started:
                if rms >= args.silence_threshold:
                    speech_has_started = True
                    print("Speech started.") 
                    consecutive_silent_chunks = 0 
                else: 
                    consecutive_silent_chunks += 1
                    if consecutive_silent_chunks >= NO_SPEECH_TIMEOUT_CHUNKS:
                        print(f"\nNo speech detected for {args.no_speech_timeout_seconds}s after wakeword, stopping recording.")
                        if recorded_frames:
                            total_duration_seconds = len(recorded_frames) * CHUNK / RATE
                            print(f"Recorded {total_duration_seconds:.2f} seconds of audio (mostly silence).")
                        else:
                            print("No audio recorded.")
                        
                        is_recording = False
                        recorded_frames = []
                        consecutive_silent_chunks = 0
                        speech_has_started = False 
                        owwModel.reset() 
                        print("\nListening for wakewords")
            else: 
                if rms < args.silence_threshold:
                    consecutive_silent_chunks += 1
                    if consecutive_silent_chunks >= SILENCE_CHUNKS_FOR_SPEECH_END:
                        print(f"\nSilence detected for {args.silence_duration_seconds}s after speech, stopping recording.")
                        total_duration_seconds = len(recorded_frames) * CHUNK / RATE
                        print(f"Recorded {total_duration_seconds:.2f} seconds of audio.")
                        
                        is_recording = False
                        recorded_frames = []
                        consecutive_silent_chunks = 0
                        speech_has_started = False 
                        owwModel.reset() 
                        print("\nListening for wakewords.")
                else: 
                    consecutive_silent_chunks = 0
        
        else: 
            prediction = owwModel.predict(audio_data)

            for mdl in owwModel.prediction_buffer.keys():
                scores = list(owwModel.prediction_buffer[mdl])
                if scores and scores[-1] > 0.5: 
                    model_name_parts = mdl.split("\\")[-1].split("/")[-1] 
                
                    model_name = model_name_parts.split(".")[0]
                    print(f"\nWakeword Detected! - {model_name}")
                    print("Starting recording (waiting for speech)...")
                    
                    is_recording = True
                    recorded_frames = [] 
                    speech_has_started = False
                    consecutive_silent_chunks = 0
                    
                    break 
     
            if is_recording:
                continue