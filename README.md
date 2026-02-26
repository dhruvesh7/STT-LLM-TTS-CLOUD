# 🎙️ Voice Assistant — STT → LLM → TTS Pipeline Using Cloud 

A real-time voice assistant that chains **OpenAI Whisper** (speech-to-text), **GPT-4o** (language model), and **OpenAI TTS** (text-to-speech) into a seamless conversational loop — with built-in per-turn and session latency tracking.

---

## ✨ Features

- 🎤 **Push-to-talk recording** via your microphone
- 🔍 **Whisper STT** — accurate speech transcription
- 🤖 **GPT-4o** — multi-turn conversational memory
- 🔊 **OpenAI TTS** — natural-sounding voice responses
- ⏱️ **Latency tracking** — per-stage and session average breakdowns
- 🔁 **Conversation history** — context preserved across turns

---

## 📋 Requirements

- Python 3.8+
- An [OpenAI API key](https://platform.openai.com/api-keys)
- A working microphone

---

## 🚀 Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/your-username/voice-assistant.git
cd voice-assistant
```

### 2. Install dependencies

```bash
pip install openai sounddevice soundfile numpy
```

> **Note for Linux users:** You may also need `portaudio`:
> ```bash
> sudo apt-get install portaudio19-dev
> ```

### 3. Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

Or create a `.env` file and load it before running.

### 4. Run the assistant

```bash
python voice_assistant.py
```

---

## 🕹️ How to Use

1. Press **Enter** to start recording
2. Speak your message
3. Press **Enter** again to stop
4. Wait for the assistant to transcribe, think, and speak back
5. Press **Ctrl+C** to quit and see session latency averages

---

## ⚙️ Configuration

All settings are at the top of `voice_assistant.py`:

| Variable | Default | Description |
|---|---|---|
| `STT_MODEL` | `whisper-1` | Whisper model for transcription |
| `LLM_MODEL` | `gpt-4o` | GPT model for responses |
| `TTS_MODEL` | `tts-1` | TTS model (`tts-1` or `tts-1-hd`) |
| `TTS_VOICE` | `alloy` | Voice: `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer` |
| `SAMPLE_RATE` | `16000` | Audio recording sample rate (Hz) |
| `CHANNELS` | `1` | Mono (`1`) or stereo (`2`) |
| `SYSTEM_PROMPT` | *see file* | Personality/instructions for the assistant |

---

## 📊 Latency Output

After each turn, a breakdown is printed:

```
┌─ Latency Breakdown ──────────────────────┐
│  🔍 STT  (Whisper)     420 ms  █████
│  🤖 LLM  (GPT-4o)     890 ms  ███████████
│  🔊 TTS  (Speech)      310 ms  ████
│  ──────────────────────────────────────
│  ⏱  Total E2E         1620 ms
└──────────────────────────────────────────┘
```

On exit (Ctrl+C), session averages across all turns are displayed.

---

## 🗂️ Project Structure

```
voice_assistant.py   # Main script (single-file)
README.md
```

---

## 🔒 Security Note

Never commit your API key to source control. Use environment variables or a secrets manager. The default fallback key in the source file should be replaced or removed before sharing.

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `openai` | STT, LLM, and TTS API calls |
| `sounddevice` | Microphone recording and audio playback |
| `soundfile` | WAV encoding/decoding |
| `numpy` | Audio buffer handling |

---

## 🛣️ Roadmap / Ideas

- [ ] Streaming TTS for lower perceived latency
- [ ] Wake word detection (always-on mode)
- [ ] `.env` file support
- [ ] Configurable via CLI arguments
- [ ] Swap in local models (Whisper.cpp, Ollama, Piper TTS)

---

## 📄 License

MIT
