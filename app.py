from __future__ import annotations

import ipaddress
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
import webbrowser

import librosa
import numpy as np
from flask import Flask, jsonify, render_template, request
from yt_dlp import YoutubeDL


app = Flask(__name__)

# Fast BPM-only configuration
SAMPLE_RATE = 22050
SEGMENT_SECONDS = 15
MAX_SEGMENTS = 1
REQUEST_TIMEOUT_SECONDS = 30
FFMPEG_TIMEOUT_SECONDS = 45
ALLOWED_SCHEMES = {"http", "https"}


@dataclass
class AnalysisJob:
    status: str = "queued"
    progress: int = 0
    message: str = "Queued"
    result: dict | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    process: subprocess.Popen | None = None
    created_at: float = field(default_factory=time.time)


jobs: dict[str, AnalysisJob] = {}
jobs_lock = threading.Lock()

def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:5000")

def browser_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


def validate_public_url(url: str) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError("Only http:// and https:// URLs are supported.")

    if not parsed.hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    try:
        addresses = socket.getaddrinfo(
            parsed.hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
        )
    except socket.gaierror as exc:
        raise ValueError("Unable to resolve the URL hostname.") from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])

        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            raise ValueError("Local and private network URLs are not allowed.")


def extract_media(url: str) -> tuple[str, dict]:
    validate_public_url(url)

    options = {
        "format": "bestaudio/best",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "http_headers": browser_headers(),
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": REQUEST_TIMEOUT_SECONDS,
    }

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            raise ValueError("No media information was returned.")

        stream_url = info.get("url")

        if not stream_url:
            formats = info.get("formats") or []
            audio_formats = [
                item
                for item in formats
                if item.get("url") and item.get("vcodec") == "none"
            ]

            if not audio_formats:
                raise ValueError("No audio stream was found.")

            audio_formats.sort(
                key=lambda item: item.get("abr") or 0,
                reverse=True,
            )
            stream_url = audio_formats[0]["url"]

        metadata = {
            "source_type": (
                info.get("extractor_key")
                or info.get("extractor")
                or "Supported website"
            ),
            "website": (
                info.get("webpage_url_domain")
                or urlparse(url).hostname
            ),
            "title": info.get("title"),
            "uploader": (
                info.get("uploader")
                or info.get("channel")
                or info.get("creator")
                or info.get("artist")
            ),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "codec": info.get("acodec"),
            "bitrate_kbps": info.get("abr"),
            "http_headers": (
                info.get("http_headers")
                or browser_headers()
            ),
        }

        return stream_url, metadata

    except Exception as extractor_error:
        # Direct media URLs may not require a website-specific extractor.
        return url, {
            "source_type": "Direct media URL",
            "website": urlparse(url).hostname,
            "title": Path(urlparse(url).path).name or None,
            "uploader": None,
            "duration": None,
            "thumbnail": None,
            "codec": None,
            "bitrate_kbps": None,
            "http_headers": browser_headers(),
            "extractor_warning": str(extractor_error),
        }


def choose_sample_starts(duration: float | None) -> list[float]:
    """
    Analyze one short section.

    For longer media, start after the intro.
    """
    if not duration:
        return [0.0]

    if duration <= SEGMENT_SECONDS:
        return [0.0]

    safe_end = max(0.0, duration - SEGMENT_SECONDS)

    if duration > 45:
        return [min(20.0, safe_end)]

    return [min(5.0, safe_end)]


def header_string(headers: dict[str, str]) -> str:
    return "".join(
        f"{key}: {value}\r\n"
        for key, value in headers.items()
    )


def read_audio_segment(
    stream_url: str,
    start_seconds: float,
    duration_seconds: int,
    headers: dict[str, str],
    job: AnalysisJob,
) -> np.ndarray:
    """
    Decode a short remote audio section directly into memory.

    No full media file is downloaded or saved.
    """
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",

        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", "5",

        "-headers", header_string(headers),

        # Fast seek for remote streams
        "-ss", f"{start_seconds:.3f}",
        "-i", stream_url,

        "-t", str(duration_seconds),
        "-vn",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "pipe:1",
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install FFmpeg and make sure "
            "the ffmpeg command is available in PATH."
        ) from exc

    job.process = process
    started_at = time.time()

    try:
        while True:
            if job.cancel_event.is_set():
                process.terminate()

                try:
                    process.communicate(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

                raise InterruptedError("Analysis stopped.")

            if time.time() - started_at > FFMPEG_TIMEOUT_SECONDS:
                process.kill()
                _, stderr = process.communicate()

                error_text = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                message = "FFmpeg took too long to read the stream."

                if error_text:
                    message += f" {error_text}"

                raise TimeoutError(message)

            try:
                stdout, stderr = process.communicate(timeout=0.25)
                break
            except subprocess.TimeoutExpired:
                continue

    finally:
        job.process = None

    if process.returncode != 0:
        error_text = stderr.decode(
            "utf-8",
            errors="replace",
        ).strip()

        if not error_text:
            error_text = (
                f"FFmpeg exited with code {process.returncode}. "
                "The stream may have expired or rejected the connection."
            )

        raise ValueError(
            f"FFmpeg could not read this media section: {error_text}"
        )

    audio = np.frombuffer(stdout, dtype=np.float32)

    minimum_samples = SAMPLE_RATE * 3

    if audio.size < minimum_samples:
        extracted_seconds = audio.size / SAMPLE_RATE

        raise ValueError(
            "The extracted audio was too short "
            f"({extracted_seconds:.1f} seconds)."
        )

    return audio


def normalize_bpm(bpm: float) -> float:
    """
    Reduce common half-tempo and double-tempo mistakes.
    """
    while bpm < 70:
        bpm *= 2

    while bpm > 200:
        bpm /= 2

    return bpm


def estimate_segment_bpm(audio: np.ndarray) -> float | None:
    """
    Estimate BPM from one audio section.
    """
    onset_envelope = librosa.onset.onset_strength(
        y=audio,
        sr=SAMPLE_RATE,
        aggregate=np.median,
    )

    tempo = librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=SAMPLE_RATE,
        aggregate=np.median,
    )

    bpm = float(np.asarray(tempo).reshape(-1)[0])

    if not np.isfinite(bpm) or bpm <= 0:
        return None

    return normalize_bpm(bpm)


def run_analysis(job_id: str, url: str) -> None:
    with jobs_lock:
        job = jobs[job_id]

    try:
        job.status = "extracting"
        job.message = "Resolving media stream…"
        job.progress = 5

        stream_url, metadata = extract_media(url)

        if job.cancel_event.is_set():
            raise InterruptedError("Analysis stopped.")

        starts = choose_sample_starts(
            metadata.get("duration")
        )[:MAX_SEGMENTS]

        bpm_values: list[float] = []
        segment_results: list[dict] = []

        for index, start in enumerate(starts):
            if job.cancel_event.is_set():
                raise InterruptedError("Analysis stopped.")

            job.status = "analyzing"
            job.message = (
                f"Reading audio section {index + 1} "
                f"of {len(starts)}…"
            )
            job.progress = 20

            audio = read_audio_segment(
                stream_url=stream_url,
                start_seconds=start,
                duration_seconds=SEGMENT_SECONDS,
                headers=(
                    metadata.get("http_headers")
                    or browser_headers()
                ),
                job=job,
            )

            if job.cancel_event.is_set():
                raise InterruptedError("Analysis stopped.")

            job.message = "Detecting BPM…"
            job.progress = 70

            bpm = estimate_segment_bpm(audio)

            if bpm is None:
                continue

            bpm_values.append(bpm)

            segment_results.append({
                "start_seconds": round(start, 1),
                "bpm": round(bpm, 2),
            })

        if not bpm_values:
            raise ValueError(
                "A reliable BPM could not be detected."
            )

        bpm = float(np.median(bpm_values))
        bpm_deviation = float(np.std(bpm_values))

        if len(bpm_values) == 1:
            bpm_confidence = 90.0
        else:
            bpm_confidence = max(
                0.0,
                min(100.0, 100.0 - bpm_deviation * 6.0),
            )

        result = {
            "bpm": round(bpm),
            "precise_bpm": round(bpm, 2),
            "bpm_confidence_percent": round(
                bpm_confidence,
                1,
            ),
            "segments_analyzed": len(segment_results),
            "segment_results": segment_results,
            **{
                key: value
                for key, value in metadata.items()
                if key != "http_headers"
            },
        }

        job.result = result
        job.status = "complete"
        job.progress = 100
        job.message = "Analysis complete."

    except InterruptedError:
        job.status = "cancelled"
        job.message = "Analysis stopped."
        job.progress = 0

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        job.message = "Analysis failed."

    finally:
        job.process = None


def cleanup_old_jobs() -> None:
    cutoff = time.time() - 3600

    with jobs_lock:
        expired = [
            job_id
            for job_id, job in jobs.items()
            if (
                job.created_at < cutoff
                and job.status
                in {"complete", "error", "cancelled"}
            )
        ]

        for job_id in expired:
            jobs.pop(job_id, None)


@app.get("/")
def index():
    cleanup_old_jobs()
    return render_template("index.html")


@app.post("/api/jobs")
def create_job():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url", "")).strip()

    if not url:
        return jsonify({
            "error": "Enter a media or direct audio URL."
        }), 400

    job_id = uuid.uuid4().hex
    job = AnalysisJob()

    with jobs_lock:
        jobs[job_id] = job

    thread = threading.Thread(
        target=run_analysis,
        args=(job_id, url),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id}), 202


@app.get("/api/jobs/<job_id>")
def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({
            "error": "Analysis job not found."
        }), 404

    return jsonify({
        "status": job.status,
        "progress": job.progress,
        "message": job.message,
        "result": job.result,
        "error": job.error,
    })


@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)

    if not job:
        return jsonify({
            "error": "Analysis job not found."
        }), 404

    job.cancel_event.set()

    process = job.process

    if process and process.poll() is None:
        try:
            process.terminate()
        except OSError:
            pass

    return jsonify({"status": "cancelling"})


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 10000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )