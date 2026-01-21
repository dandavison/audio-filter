# Audio Filter Tools 🎤

Two real-time audio processing tools for macOS:

1. **Voice Isolator** - Removes music, keeps vocals (for workout videos)
2. **Live Transcriber** - Real-time speech-to-text (for meetings/calls)

## How It Works

```
YouTube/Video Audio → BlackHole → Voice Isolator → Your Speakers
                                        ↓
                              (ML removes music,
                               keeps voice only)
```

## Quick Start

### 1. Install BlackHole (virtual audio driver)

```bash
brew install blackhole-2ch
```

### 2. Install dependencies

```bash
cd /Users/dan/src/audio-filter
uv sync
```

### 3. Configure Audio Routing

1. Open **System Settings → Sound → Output**
2. Select **"BlackHole 2ch"** as your output device
3. Your Mac audio now routes through BlackHole (you won't hear it yet!)

### 4. Run the Voice Isolator

```bash
uv run voice-filter
```

Now play a video - after ~4 seconds of buffering, you'll hear just the vocals!

## Usage

```bash
# Basic usage
uv run voice-filter

# List available audio devices
uv run voice-filter --list

# Specify output device
uv run voice-filter --output "MacBook Pro Speakers"

# Lower latency (may reduce quality)
uv run voice-filter --latency 3
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-l, --list` | List audio devices | - |
| `-i, --input` | Input device | BlackHole 2ch |
| `-o, --output` | Output device | System default |
| `-t, --latency` | Chunk size (seconds) | 4.0 |

## Tips

### For workout videos:
1. Start the video paused
2. Run `uv run voice-filter`
3. Wait for "Processing audio..." message
4. Play the video - you'll hear the instructor ~4 seconds delayed
5. Play your own music through a separate device/speaker, or use the "Multi-Output Device" trick below

### Create a Multi-Output Device (hear both):
If you want to hear the isolated voice AND play your own music:

1. Open **Audio MIDI Setup** (search in Spotlight)
2. Click **+** → **Create Multi-Output Device**
3. Check both **BlackHole 2ch** and your speakers
4. Set this as your system output
5. Play your music from a different app/device

### Reduce latency:
- Use `--latency 2` or `--latency 3` for faster response
- Trade-off: shorter chunks may have more artifacts

## Technical Details

- Uses **Demucs** (Meta's hybrid transformer model) for source separation
- Runs on Apple Silicon GPU (Metal) when available
- Processes audio in chunks for real-time streaming
- ~4 second latency by default for best quality

## Troubleshooting

**"Could not find input device: BlackHole 2ch"**
- Make sure BlackHole is installed: `brew install blackhole-2ch`
- Restart your Mac if just installed

**No audio output**
- Check that your output device is correct: `uv run voice-filter --list`
- Make sure BlackHole is set as system output in Sound settings

**Audio is choppy**
- Try increasing latency: `--latency 5`
- Close other CPU-intensive apps

**Model loading is slow**
- First run downloads the model (~200MB)
- Subsequent runs are faster

---

# Live Transcriber 🎙️

Real-time audio transcription using Whisper. Captures system audio and writes transcript to a file as it goes.

## How It Works

```
Zoom/Meeting Audio → BlackHole → Transcriber → transcript.txt
                                      ↓
                              (optional: speakers)
```

## Usage

```bash
# Basic - silent transcription (won't hear audio)
uv run transcribe

# Also hear the audio while transcribing
uv run transcribe --listen

# Better accuracy (uses more RAM, slightly slower)
uv run transcribe --model small

# Custom output file
uv run transcribe -o allhands_meeting.txt

# Faster updates (shorter chunks)
uv run transcribe --chunk 5
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-l, --list` | List audio devices | - |
| `-i, --input` | Input device | BlackHole 2ch |
| `-o, --output` | Output file | transcript_TIMESTAMP.txt |
| `-m, --model` | Whisper model size | base |
| `-c, --chunk` | Chunk duration (seconds) | 10 |
| `--listen` | Also play audio through speakers | off |

## Model Sizes

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fastest | Basic | ~1GB |
| base | Fast | Good | ~1GB |
| small | Medium | Better | ~2GB |
| medium | Slow | High | ~5GB |
| large-v2 | Slowest | Best | ~10GB |

## For Zoom Meetings

1. Set **BlackHole 2ch** as system output before joining
2. Run `uv run transcribe`
3. Join your meeting - transcript builds automatically
4. Press Ctrl+C when done - transcript is saved to file

### Want to hear the meeting too?

Option A: Use `--listen` flag (you'll hear through default speakers)
```bash
uv run transcribe --listen
```

Option B: Create a Multi-Output Device in Audio MIDI Setup:
1. Open **Audio MIDI Setup** (Spotlight search)
2. Click **+** → **Create Multi-Output Device**
3. Check **BlackHole 2ch** + your speakers
4. Use this as system output

