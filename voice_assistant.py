"""
Voice Assistant: STT → LLM → TTS Pipeline  (with latency tracking)
Uses OpenAI Whisper (STT), GPT-4o (LLM), and TTS API

Requirements:
    pip install openai sounddevice soundfile numpy

Usage:
    python voice_assistant.py
    Press Enter to start recording, Enter again to stop.
"""

import os
import io
import time
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

STT_MODEL   = "whisper-1"
LLM_MODEL   = "gpt-4o"
TTS_MODEL   = "tts-1"
TTS_VOICE   = "alloy"          # alloy | echo | fable | onyx | nova | shimmer
SAMPLE_RATE = 16000
CHANNELS    = 1

SYSTEM_PROMPT = "You are a helpful voice assistant. Keep responses concise and conversational."

# ── Client ────────────────────────────────────────────────────────────────────

client = OpenAI(api_key=OPENAI_API_KEY)
conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ── Latency tracker ───────────────────────────────────────────────────────────

class LatencyTracker:
    def __init__(self):
        self._starts: dict = {}
        self.stages:  dict = {}
        self.session_totals: dict = {}

    def reset(self):
        self._starts = {}
        self.stages  = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = time.perf_counter() - self._starts[stage]
        self.stages[stage] = elapsed
        self.session_totals.setdefault(stage, []).append(elapsed)
        return elapsed

    def report(self):
        total = sum(self.stages.values())
        order  = ["stt", "llm", "tts"]
        labels = {"stt": "🔍 STT  (Whisper)",
                  "llm": "🤖 LLM  (GPT-4o) ",
                  "tts": "🔊 TTS  (Speech) "}
        print("\n┌─ Latency Breakdown ──────────────────────┐")
        for key in order:
            ms  = self.stages.get(key, 0) * 1000
            bar = "█" * max(1, int(ms / 80))
            print(f"│  {labels[key]}  {ms:6.0f} ms  {bar}")
        print(f"│  {'─'*38}")
        print(f"│  ⏱  Total E2E              {total*1000:6.0f} ms")
        print("└──────────────────────────────────────────┘")

    def session_summary(self):
        if not self.session_totals:
            return
        print("\n╔═ Session Average Latencies ══════════════╗")
        order  = ["stt", "llm", "tts"]
        labels = {"stt": "STT", "llm": "LLM", "tts": "TTS"}
        grand  = 0.0
        for key in order:
            vals = self.session_totals.get(key, [])
            if vals:
                avg = sum(vals) / len(vals)
                grand += avg
                n = len(vals)
                print(f"║  {labels[key]}  avg: {avg*1000:6.0f} ms  ({n} turn{'s' if n>1 else ''})")
        print(f"║  E2E avg:  {grand*1000:6.0f} ms")
        print("╚══════════════════════════════════════════╝")


tracker = LatencyTracker()

# ── Step 1: Record audio ──────────────────────────────────────────────────────

def record_audio() -> np.ndarray:
    print("\n🎤 Recording… (press Enter to stop)")
    frames = []
    stop_event = threading.Event()

    def _record():
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            while not stop_event.is_set():
                data, _ = stream.read(1024)
                frames.append(data.copy())

    thread = threading.Thread(target=_record, daemon=True)
    thread.start()
    input()
    stop_event.set()
    thread.join()

    audio = np.concatenate(frames, axis=0)
    print(f"   Recorded {len(audio)/SAMPLE_RATE:.1f}s of audio")
    return audio


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


# ── Step 2: STT ───────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> str:
    print("🔍 Transcribing…", end="", flush=True)
    tracker.start("stt")

    audio_file = io.BytesIO(audio_to_wav_bytes(audio))
    audio_file.name = "audio.wav"

    transcript = client.audio.transcriptions.create(
        model=STT_MODEL,
        file=audio_file,
        response_format="text",
    )
    elapsed = tracker.stop("stt")
    text = transcript.strip()

    print(f" {elapsed*1000:.0f}ms")
    print(f"   You said: \"{text}\"")
    return text


# ── Step 3: LLM ───────────────────────────────────────────────────────────────

def chat(user_text: str) -> str:
    print("🤖 Thinking…", end="", flush=True)
    tracker.start("llm")

    conversation_history.append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=conversation_history,
        temperature=0.7,
    )
    reply = response.choices[0].message.content.strip()
    conversation_history.append({"role": "assistant", "content": reply})

    elapsed = tracker.stop("llm")
    print(f" {elapsed*1000:.0f}ms")
    print(f"   Assistant: \"{reply}\"")
    return reply


# ── Step 4: TTS ───────────────────────────────────────────────────────────────

def speak(text: str):
    print("🔊 Generating speech…", end="", flush=True)
    tracker.start("tts")

    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    audio_bytes = response.content
    elapsed = tracker.stop("tts")
    print(f" {elapsed*1000:.0f}ms")

    buf = io.BytesIO(audio_bytes)
    data, sample_rate = sf.read(buf, dtype="float32")
    sd.play(data, samplerate=sample_rate)
    sd.wait()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Voice Assistant  |  STT → LLM → TTS")
    print("  Ctrl+C to quit")
    print("=" * 55)

    turn = 0
    while True:
        try:
            input("\nPress Enter to start speaking…")
            turn += 1
            print(f"\n─── Turn {turn} ───────────────────────────────────")

            tracker.reset()

            audio     = record_audio()
            user_text = transcribe(audio)

            if not user_text:
                print("⚠️  No speech detected. Try again.")
                continue

            reply = chat(user_text)
            speak(reply)

            tracker.report()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            tracker.session_summary()
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
