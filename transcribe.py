#!/usr/bin/env python3
"""
Real-time audio transcription using Whisper.

Captures system audio via BlackHole and writes transcript incrementally to file.
Perfect for transcribing meetings, calls, or any system audio.

Usage:
    uv run transcribe                    # Silent transcription
    uv run transcribe --listen           # Also hear the audio
    uv run transcribe --model small      # Better accuracy
    uv run transcribe -o meeting.txt     # Custom output file
"""

import argparse
import datetime
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd


class LiveTranscriber:
    def __init__(
        self,
        input_device: str = "BlackHole 2ch",
        output_device: str = None,
        output_file: str = None,
        model_size: str = "base",
        chunk_seconds: float = 10.0,
        listen: bool = False,
    ):
        self.sample_rate = 16000  # Whisper expects 16kHz
        self.input_channels = 2  # BlackHole is stereo
        self.chunk_samples = int(chunk_seconds * self.sample_rate)

        # Find devices
        self.input_idx = self._find_device(input_device, "input")
        self.output_idx = self._find_device(output_device, "output") if output_device else None
        self.listen = listen

        # Output file
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"transcript_{timestamp}.txt"
        self.output_path = Path(output_file)

        # Buffers
        self.audio_buffer = []
        self.buffer_samples = 0
        self.lock = threading.Lock()
        self.running = False

        # Load Whisper model
        self._load_model(model_size)

    def _find_device(self, name: str, kind: str) -> int:
        """Find audio device by name."""
        if name is None:
            return None

        devices = sd.query_devices()
        name_lower = name.lower()

        for i, dev in enumerate(devices):
            if name_lower in dev["name"].lower():
                if kind == "input" and dev["max_input_channels"] > 0:
                    return i
                elif kind == "output" and dev["max_output_channels"] > 0:
                    return i

        print(f"\n❌ Could not find {kind} device: '{name}'")
        print(f"\nAvailable {kind} devices:")
        for i, dev in enumerate(devices):
            if kind == "input" and dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']}")
            elif kind == "output" and dev["max_output_channels"] > 0:
                print(f"  [{i}] {dev['name']}")
        sys.exit(1)

    def _load_model(self, model_size: str):
        """Load Whisper model."""
        print(f"🔄 Loading Whisper model ({model_size})...")

        from faster_whisper import WhisperModel

        # Determine best compute type for the platform
        import torch
        if torch.backends.mps.is_available():
            # Apple Silicon - use CPU with int8 (MPS not fully supported by CTranslate2)
            device = "cpu"
            compute_type = "int8"
            print("   Using CPU with int8 (optimized for Apple Silicon)")
        elif torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
            print("   Using NVIDIA GPU")
        else:
            device = "cpu"
            compute_type = "int8"
            print("   Using CPU")

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("✅ Model loaded!\n")

    def _transcribe_chunk(self, audio: np.ndarray) -> str:
        """Transcribe an audio chunk."""
        # Ensure float32
        audio = audio.astype(np.float32)

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True,  # Filter out silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
            ),
        )

        texts = []
        for segment in segments:
            text = segment.text.strip()
            if text:
                texts.append(text)

        return " ".join(texts)

    def _processor_thread(self):
        """Background thread that processes audio chunks."""
        while self.running:
            audio_chunk = None

            with self.lock:
                if self.buffer_samples >= self.chunk_samples:
                    # Combine buffered audio
                    combined = np.concatenate(self.audio_buffer)
                    audio_chunk = combined[: self.chunk_samples]

                    # Keep overflow for next chunk
                    overflow = self.buffer_samples - self.chunk_samples
                    if overflow > 0:
                        self.audio_buffer = [combined[self.chunk_samples :]]
                        self.buffer_samples = overflow
                    else:
                        self.audio_buffer = []
                        self.buffer_samples = 0

            if audio_chunk is not None:
                # Transcribe
                text = self._transcribe_chunk(audio_chunk)

                if text.strip():
                    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                    line = f"[{timestamp}] {text}\n"

                    # Print to console
                    print(f"\r{' ' * 40}\r📝 {text[:80]}{'...' if len(text) > 80 else ''}")

                    # Append to file
                    with open(self.output_path, "a", encoding="utf-8") as f:
                        f.write(line)
            else:
                time.sleep(0.05)

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        """Called for each audio block."""
        if status:
            print(f"\rAudio status: {status}", end="")

        # Convert stereo to mono and add to buffer
        mono = indata.mean(axis=1).astype(np.float32)

        with self.lock:
            self.audio_buffer.append(mono)
            self.buffer_samples += len(mono)

        # Passthrough audio if listening
        if self.listen:
            outdata[:] = indata
        else:
            outdata[:] = 0

    def run(self):
        """Main loop."""
        input_name = sd.query_devices(self.input_idx)["name"]
        output_name = (
            sd.query_devices(self.output_idx)["name"]
            if self.output_idx is not None
            else sd.query_devices(sd.default.device[1])["name"]
        )

        print("=" * 55)
        print("🎙️  LIVE TRANSCRIBER")
        print("=" * 55)
        print(f"  Input:      {input_name}")
        print(f"  Output:     {self.output_path}")
        print(f"  Listen:     {'Yes' if self.listen else 'No (silent)'}")
        print(f"  Chunk size: {self.chunk_samples / self.sample_rate:.0f}s")
        print("=" * 55)
        print("\n⏳ Listening... Press Ctrl+C to stop\n")

        # Initialize output file with header
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(f"# Transcript\n")
            f.write(f"# Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"# Source: {input_name}\n\n")

        self.running = True

        # Start processor thread
        processor = threading.Thread(target=self._processor_thread, daemon=True)
        processor.start()

        # Start audio stream
        try:
            with sd.Stream(
                device=(self.input_idx, self.output_idx),
                samplerate=self.sample_rate,
                channels=self.input_channels,
                callback=self._audio_callback,
                blocksize=1024,
            ):
                while True:
                    with self.lock:
                        buffered_sec = self.buffer_samples / self.sample_rate

                    # Show status (will be overwritten by transcription output)
                    print(f"\r  ⏺ Recording... ({buffered_sec:.1f}s buffered)", end="", flush=True)
                    time.sleep(0.5)

        except KeyboardInterrupt:
            self.running = False
            print(f"\n\n{'=' * 55}")
            print(f"✅ Done! Transcript saved to: {self.output_path}")
            print(f"{'=' * 55}\n")


def list_devices():
    """List available audio devices."""
    devices = sd.query_devices()
    print("\n📢 Available Audio Devices:\n")

    print("INPUT DEVICES:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"  [{i}] {dev['name']} ({dev['max_input_channels']}ch)")

    print("\nOUTPUT DEVICES:")
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            marker = " ← default" if i == sd.default.device[1] else ""
            print(f"  [{i}] {dev['name']} ({dev['max_output_channels']}ch){marker}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Live audio transcription with Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run transcribe                      # Basic silent transcription
  uv run transcribe --listen             # Also hear the audio
  uv run transcribe -m small             # Better accuracy (slower)
  uv run transcribe -m tiny              # Fastest (less accurate)
  uv run transcribe -o meeting.txt       # Custom output file
  uv run transcribe --chunk 15           # 15-second chunks

Model sizes (speed vs accuracy tradeoff):
  tiny   - Fastest, least accurate (~1GB RAM)
  base   - Good balance (default) (~1GB RAM)
  small  - Better accuracy (~2GB RAM)
  medium - High accuracy (~5GB RAM)
  large-v2 - Best accuracy (~10GB RAM)
        """,
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--input", "-i", default="BlackHole 2ch", help="Input device (default: BlackHole 2ch)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output transcript file (default: transcript_TIMESTAMP.txt)"
    )
    parser.add_argument(
        "--model", "-m", default="base",
        choices=["tiny", "base", "small", "medium", "large-v2"],
        help="Whisper model size (default: base)"
    )
    parser.add_argument(
        "--chunk", "-c", type=float, default=10.0,
        help="Chunk duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--listen", action="store_true",
        help="Also play audio through speakers (default: silent)"
    )

    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        transcriber = LiveTranscriber(
            input_device=args.input,
            output_file=args.output,
            model_size=args.model,
            chunk_seconds=args.chunk,
            listen=args.listen,
        )
        transcriber.run()
    except KeyboardInterrupt:
        print("\n\n👋 Stopped")


if __name__ == "__main__":
    main()
