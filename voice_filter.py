#!/usr/bin/env python3
"""
Real-time Voice Isolator for macOS

Captures system audio via BlackHole, removes music/background,
and outputs only vocals to your speakers.

Usage:
    uv run voice-filter

Requirements:
    - BlackHole 2ch installed (brew install blackhole-2ch)
    - Set BlackHole 2ch as your system audio output
    - Run this script to hear isolated vocals through your speakers
"""

import argparse
import sys
import time
import threading
from collections import deque

import numpy as np
import sounddevice as sd
import torch


class VoiceIsolator:
    def __init__(
        self,
        input_device: str = "BlackHole 2ch",
        output_device: str = None,
        chunk_seconds: float = 4.0,
        sample_rate: int = 44100,
    ):
        self.sample_rate = sample_rate
        self.channels = 2
        self.chunk_samples = int(chunk_seconds * sample_rate)
        self.block_size = 1024

        # Find audio devices
        self.input_idx = self._find_device(input_device, "input")
        self.output_idx = self._find_device(output_device, "output") if output_device else None

        # Buffers
        self.input_buffer = []
        self.input_samples = 0
        self.output_queue = deque()
        self.output_buffer = np.zeros((0, self.channels), dtype=np.float32)
        self.lock = threading.Lock()

        # State
        self.running = False
        self.processing = False

        # Load the model
        self._load_model()

    def _find_device(self, name: str, kind: str) -> int:
        """Find audio device by name."""
        if name is None:
            return None

        devices = sd.query_devices()
        name_lower = name.lower()

        for i, dev in enumerate(devices):
            if name_lower in dev["name"].lower():
                if kind == "input" and dev["max_input_channels"] >= self.channels:
                    return i
                elif kind == "output" and dev["max_output_channels"] >= self.channels:
                    return i

        # List available devices if not found
        print(f"\n❌ Could not find {kind} device: '{name}'")
        print(f"\nAvailable {kind} devices:")
        for i, dev in enumerate(devices):
            if kind == "input" and dev["max_input_channels"] > 0:
                print(f"  [{i}] {dev['name']}")
            elif kind == "output" and dev["max_output_channels"] > 0:
                print(f"  [{i}] {dev['name']}")
        sys.exit(1)

    def _load_model(self):
        """Load Demucs model for voice separation."""
        print("🔄 Loading voice separation model...")

        # Import here to show loading message first
        from demucs.pretrained import get_model
        from demucs.apply import apply_model

        self.apply_model = apply_model

        # Load the hybrid transformer model (best quality)
        self.model = get_model("htdemucs")
        self.model.eval()

        # Select best available device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("✅ Using Apple Silicon GPU (Metal)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("✅ Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("⚠️  Using CPU (slower)")

        self.model.to(self.device)
        print("✅ Model loaded!\n")

    def _process_audio(self, audio: np.ndarray) -> np.ndarray:
        """Run audio through Demucs to extract vocals."""
        # Convert to tensor: (samples, channels) -> (batch, channels, samples)
        audio_tensor = torch.tensor(audio.T, dtype=torch.float32, device=self.device)
        audio_tensor = audio_tensor.unsqueeze(0)

        # Normalize
        peak = audio_tensor.abs().max()
        if peak > 0:
            audio_tensor = audio_tensor / peak

        # Run model
        with torch.no_grad():
            sources = self.apply_model(
                self.model,
                audio_tensor,
                device=self.device,
                progress=False,
            )

        # htdemucs outputs: [drums, bass, other, vocals]
        # Index 3 = vocals
        vocals = sources[0, 3]

        # Restore original scale
        if peak > 0:
            vocals = vocals * peak

        # Convert back: (channels, samples) -> (samples, channels)
        return vocals.cpu().numpy().T

    def _processor_thread(self):
        """Background thread that processes audio chunks."""
        while self.running:
            audio_to_process = None

            with self.lock:
                if self.input_samples >= self.chunk_samples:
                    # Concatenate buffered chunks
                    audio_to_process = np.concatenate(self.input_buffer, axis=0)
                    audio_to_process = audio_to_process[: self.chunk_samples]

                    # Keep overflow for next chunk
                    overflow = self.input_samples - self.chunk_samples
                    if overflow > 0:
                        self.input_buffer = [audio_to_process[self.chunk_samples :]]
                        self.input_samples = overflow
                    else:
                        self.input_buffer = []
                        self.input_samples = 0

            if audio_to_process is not None:
                self.processing = True
                processed = self._process_audio(audio_to_process)
                with self.lock:
                    self.output_queue.append(processed)
                self.processing = False
            else:
                time.sleep(0.01)

    def _input_callback(self, indata, frames, time_info, status):
        """Called when audio is received from BlackHole."""
        if status:
            print(f"Input status: {status}")

        with self.lock:
            self.input_buffer.append(indata.copy())
            self.input_samples += frames

    def _output_callback(self, outdata, frames, time_info, status):
        """Called when audio needs to be sent to speakers."""
        if status:
            print(f"Output status: {status}")

        with self.lock:
            # Refill output buffer from queue if needed
            while len(self.output_buffer) < frames and self.output_queue:
                chunk = self.output_queue.popleft()
                self.output_buffer = np.concatenate([self.output_buffer, chunk], axis=0)

            # Output available audio
            available = len(self.output_buffer)
            if available >= frames:
                outdata[:] = self.output_buffer[:frames]
                self.output_buffer = self.output_buffer[frames:]
            elif available > 0:
                outdata[:available] = self.output_buffer
                outdata[available:] = 0
                self.output_buffer = np.zeros((0, self.channels), dtype=np.float32)
            else:
                outdata[:] = 0

    def run(self):
        """Main loop - run until Ctrl+C."""
        input_name = sd.query_devices(self.input_idx)["name"]
        output_name = (
            sd.query_devices(self.output_idx)["name"]
            if self.output_idx
            else sd.query_devices(sd.default.device[1])["name"]
        )

        print("=" * 50)
        print("🎤 VOICE ISOLATOR")
        print("=" * 50)
        print(f"  Input:   {input_name}")
        print(f"  Output:  {output_name}")
        print(f"  Latency: ~{self.chunk_samples / self.sample_rate:.1f} seconds")
        print("=" * 50)
        print("\n⏳ Buffering... (play some audio)")
        print("   Press Ctrl+C to stop\n")

        self.running = True

        # Start processor thread
        processor = threading.Thread(target=self._processor_thread, daemon=True)
        processor.start()

        # Start audio streams
        input_stream = sd.InputStream(
            device=self.input_idx,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._input_callback,
        )

        output_stream = sd.OutputStream(
            device=self.output_idx,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            callback=self._output_callback,
        )

        started_output = False

        try:
            with input_stream, output_stream:
                while True:
                    # Status display
                    with self.lock:
                        buffered = self.input_samples / self.sample_rate
                        queued = sum(len(q) for q in self.output_queue) / self.sample_rate
                        out_buf = len(self.output_buffer) / self.sample_rate

                    if not started_output and queued > 0:
                        print("🎵 Processing audio...\n")
                        started_output = True

                    status = "🔄 Processing" if self.processing else "⏳ Waiting"
                    print(
                        f"\r  {status} | "
                        f"In: {buffered:.1f}s | "
                        f"Ready: {queued + out_buf:.1f}s   ",
                        end="",
                        flush=True,
                    )
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\n👋 Stopped")
            self.running = False


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
        description="Real-time voice isolation - removes music, keeps vocals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run voice-filter                    # Use defaults (BlackHole → default output)
  uv run voice-filter --list             # List available audio devices
  uv run voice-filter -o "MacBook Pro"   # Specify output device
  uv run voice-filter --latency 3        # Lower latency (may affect quality)
        """,
    )

    parser.add_argument(
        "--list", "-l", action="store_true", help="List available audio devices"
    )
    parser.add_argument(
        "--input", "-i", default="BlackHole 2ch", help="Input device name (default: BlackHole 2ch)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Output device name (default: system default)"
    )
    parser.add_argument(
        "--latency", "-t", type=float, default=4.0, help="Chunk size in seconds (default: 4.0)"
    )

    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    try:
        isolator = VoiceIsolator(
            input_device=args.input,
            output_device=args.output,
            chunk_seconds=args.latency,
        )
        isolator.run()
    except KeyboardInterrupt:
        print("\n👋 Stopped")


if __name__ == "__main__":
    main()



