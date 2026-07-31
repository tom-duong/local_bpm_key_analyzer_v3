# Local BPM \& Key Analyzer v3

This version detects:

* BPM
* musical key, such as C major or A minor
* tonic/root/home note, such as C or A
* confidence estimates

It also includes a real Stop button. The backend runs each analysis as a job,
and cancelling it terminates the active FFmpeg process.

## Efficient handling of long media

The app does not download the complete song or video. It uses yt-dlp to resolve
the audio stream and FFmpeg to decode up to four short 30-second sections.

A one-hour video therefore does not require downloading one hour of audio.

## Requirements
* Python
* FFmpeg

