# Local BPM \& Key Analyzer v3

This version detects:

* BPM
* confidence estimates

## Efficient handling of long media

The app does not download the complete song or video. It uses yt-dlp to resolve
the audio stream and FFmpeg to decode up to four short 30-second sections.

A one-hour video therefore does not require downloading one hour of audio.

## Requirements
* Python
* FFmpeg

## Run directly on your PC/laptop

Install the exe file, no requirements at all. From youtube URL to BPM in less than 15 seconds.
