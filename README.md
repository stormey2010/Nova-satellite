# OpenWakeWord Listener

This project is designed to listen for a custom wake word using the OpenWakeWord library. Once the wake word is detected, it continues to listen for audio until a stop command is given. The recorded audio is then sent to a specified server for transcription.

## Project Structure

```
openwakeword-listener
├── src
│   ├── main.py          # Entry point of the application
│   ├── wakeword
│   │   └── detector.py  # Wake word detection logic
│   ├── audio
│   │   ├── recorder.py   # Audio recording functionality
│   │   └── sender.py     # Sends recorded audio for transcription
│   └── config.py        # Configuration settings
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd openwakeword-listener
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Modify the `config.py` file to set your custom wake word and the server URL for transcription.

2. Run the application:
   ```
   python src/main.py
   ```

3. Once the application is running, it will listen for the specified wake word. Upon detection, it will start recording audio.

4. To stop the recording, send a stop command (this can be implemented as a keyboard interrupt or a specific command).

5. The recorded audio will be sent to the configured server for transcription.

## Custom Wake Word

You can customize the wake word by changing the value in `config.py`. Ensure that the wake word model is properly set up and accessible by the application.

## Contributing

Feel free to submit issues or pull requests for improvements or bug fixes.