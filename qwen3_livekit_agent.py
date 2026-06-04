"""
Qwen3-Omni — Native Audio-to-Audio LiveKit Agent
=================================================
Replaces the GLM-4-Voice pipeline with Qwen3-Omni (transformers backend).

Pipeline per turn:
  1. Silero VAD detects end-of-speech, returns buffered audio frames.
  2. Frames are assembled into a float32 WAV written to a temp file.
  3. Qwen3-Omni (Thinker+Talker) processes the audio in one forward pass
     and returns interleaved text + a 24 kHz audio tensor.
  4. The audio tensor is resampled to 48 kHz and published back to the
     LiveKit room as 10 ms PCM frames.

Prerequisites
-------------
- CUDA GPU(s) with ~80 GB VRAM total in BF16 (or enough for device_map="auto")
- transformers >= 5.2.0, qwen-omni-utils, soundfile, torchaudio
  Optional: flash-attn (reduces VRAM ~10 %)
- Model weights downloaded locally to MODEL_PATH:
    huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --local-dir ./Qwen3-Omni-30B-A3B-Instruct
- .env with LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
- LiveKit server running (docker compose up -d  or  livekit-server --dev)

Run
---
    python qwen3_livekit_agent.py start
"""

import asyncio
import io
import os
import tempfile
import threading
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from dotenv import load_dotenv
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from livekit import rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.vad import VADEventType
from livekit.plugins import silero
from logger import get_logger

load_dotenv()

logger = get_logger("qwen3-livekit")

# ── Configuration ─────────────────────────────────────────────────────────────
_ROOT      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_ROOT, "Qwen3-Omni-30B-A3B-Instruct")

SPEAKER_VOICE     = "Chelsie"   # Ethan | Chelsie | Aiden
MAX_HISTORY_TURNS = 5           # user+assistant pairs kept in context

SR_IN    = 16_000   # Silero VAD / LiveKit input sample rate
SR_OUT   = 24_000   # Qwen3-Omni talker outputs 24 kHz
SR_LK    = 48_000   # LiveKit standard sample rate
FRAME_MS = 10       # publish audio in 10 ms frames

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. "
    "Interact with users using short, brief, straightforward language, maintaining a natural tone. "
    "Never use formal phrasing, mechanical expressions, bullet points, or overly structured language. "
    "Your output must consist only of the spoken content you want the user to hear. "
    "Do not include any descriptions of actions, emotions, sounds, or voice changes. "
    "Do not use asterisks, brackets, parentheses, or any other symbols to indicate tone or actions. "
    "Keep replies concise and conversational, as if talking face-to-face. "
    "Always respond in English."
)


# ── Model container ───────────────────────────────────────────────────────────
class QwenModels:
    """Qwen3-Omni model + processor — loaded once per worker, shared across sessions."""

    def __init__(self):
        logger.info("Loading Qwen3-Omni model (device_map=auto) …")
        try:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
        except Exception:
            # Fallback if flash-attn is not installed
            logger.warning("flash_attention_2 unavailable — loading with default attn")
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                dtype="auto",
                device_map="auto",
            )

        logger.info("Loading Qwen3-Omni processor …")
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
        torch.cuda.empty_cache()
        logger.info("QwenModels ready ✓")


# ── Per-session conversation state ────────────────────────────────────────────
class QwenSession:
    """One instance per LiveKit room; tracks multi-turn conversation history."""

    def __init__(self, models: QwenModels):
        self.m       = models
        self.history: list[dict] = []   # alternating user/assistant text messages

    def _build_messages(self, audio_path: str) -> list[dict]:
        """Assemble [system] + text history + current audio turn."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            }
        ]
        messages.extend(self.history)
        messages.append({
            "role": "user",
            "content": [{"type": "audio", "audio": audio_path}],
        })
        return messages

    def _update_history(self, response_text: str) -> None:
        """Record the just-completed turn in the text-only history."""
        self.history.append({"role": "user",      "content": "[voice input]"})
        self.history.append({"role": "assistant",  "content": response_text})
        # Keep at most MAX_HISTORY_TURNS user+assistant pairs
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def run_generation(
        self,
        audio_path: str,
        out_q: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Blocking Qwen3-Omni inference on `audio_path`.
        Puts one float32 numpy array (SR_OUT Hz) onto out_q, then None as sentinel.
        Runs in a daemon thread so it does not block the event loop.
        """
        m = self.m

        messages  = self._build_messages(audio_path)
        text_prompt = m.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
        inputs = m.processor(
            text=text_prompt,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(m.model.device).to(m.model.dtype)

        try:
            text_ids, audio_tensor = m.model.generate(
                **inputs,
                speaker=SPEAKER_VOICE,
                thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=2048,
                thinker_do_sample=True,
                thinker_temperature=0.7,
                thinker_top_p=0.8,
                thinker_top_k=20,
                use_audio_in_video=False,
            )
        except Exception as exc:
            logger.error("Qwen3-Omni generation failed: %s", exc)
            asyncio.run_coroutine_threadsafe(out_q.put(None), loop)
            return

        response_text = m.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        logger.info("Assistant text: %s", response_text[:300])
        self._update_history(response_text)

        if audio_tensor is not None:
            # float32 waveform at SR_OUT Hz
            chunk = audio_tensor.reshape(-1).float().cpu().numpy()
            asyncio.run_coroutine_threadsafe(out_q.put(chunk), loop)

        asyncio.run_coroutine_threadsafe(out_q.put(None), loop)   # sentinel


# ── Audio helpers ─────────────────────────────────────────────────────────────
_resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _resampler(src: int, dst: int) -> torchaudio.transforms.Resample:
    key = (src, dst)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(src, dst)
    return _resamplers[key]


def _frames_to_wav_bytes(frames: list[rtc.AudioFrame], sr: int) -> bytes:
    """Concatenate int16 LiveKit frames into a WAV byte buffer."""
    pcm = np.concatenate(
        [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    ).astype(np.float32) / 32_768.0
    buf = io.BytesIO()
    sf.write(buf, pcm, samplerate=sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


async def stream_to_room(
    source: rtc.AudioSource,
    audio_np: np.ndarray,
    src_sr: int = SR_OUT,
) -> None:
    """Resample float32 audio to SR_LK and publish as 10 ms LiveKit frames."""
    wf  = torch.from_numpy(audio_np).float().unsqueeze(0)
    wf  = _resampler(src_sr, SR_LK)(wf)
    pcm = (wf.squeeze().numpy() * 32_767).clip(-32_768, 32_767).astype(np.int16)

    frame_n = int(SR_LK * FRAME_MS / 1000)   # 480 samples @ 48 kHz
    for i in range(0, len(pcm), frame_n):
        chunk = pcm[i : i + frame_n]
        if len(chunk) < frame_n:
            chunk = np.pad(chunk, (0, frame_n - len(chunk)))
        await source.capture_frame(
            rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=SR_LK,
                num_channels=1,
                samples_per_channel=frame_n,
            )
        )


# ── LiveKit agent ─────────────────────────────────────────────────────────────
async def _set_agent_state(room: rtc.Room, state: str) -> None:
    """Update the lk.agent.state attribute so frontend hooks can track state."""
    try:
        await room.local_participant.set_attributes({"lk.agent.state": state})
        logger.debug("Agent state → %s", state)
    except Exception as exc:
        logger.debug("set_attributes unsupported: %s", exc)


def prewarm(proc) -> None:
    # Only load the CPU-only VAD here so the prewarm process stays lightweight.
    # QwenModels (~80 GB BF16) loads inside entrypoint() to avoid double-loading
    # when the pool spawns a refill process.
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info("Qwen3-Omni agent connected → room: %s", ctx.room.name)
    await _set_agent_state(ctx.room, "initializing")

    vad = ctx.proc.userdata["vad"]

    logger.info("Loading Qwen3-Omni models for session …")
    loop   = asyncio.get_running_loop()
    models = await loop.run_in_executor(None, QwenModels)

    session = QwenSession(models)

    # ── Outgoing audio track ──────────────────────────────────────────────────
    out_src   = rtc.AudioSource(SR_LK, 1)
    out_track = rtc.LocalAudioTrack.create_audio_track("qwen3-voice", out_src)
    await ctx.room.local_participant.publish_track(
        out_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("Agent audio track published")
    await _set_agent_state(ctx.room, "listening")

    # Serialize inference so two VAD events don't overlap
    inference_lock = asyncio.Lock()

    # ── Per-track VAD → inference pipeline ───────────────────────────────────
    async def handle_track(track: rtc.Track) -> None:
        audio_stream = rtc.AudioStream(track, sample_rate=SR_IN, num_channels=1)
        vad_stream   = vad.stream()
        speech_buf: list[rtc.AudioFrame] = []
        speaking     = False

        async def _push_to_vad() -> None:
            async for ev in audio_stream:
                vad_stream.push_frame(ev.frame)

        async def _vad_loop() -> None:
            nonlocal speaking, speech_buf
            async for ev in vad_stream:
                if ev.type == VADEventType.START_OF_SPEECH:
                    speaking   = True
                    speech_buf = []
                elif ev.type == VADEventType.INFERENCE_DONE and speaking:
                    speech_buf.extend(ev.frames)
                elif ev.type == VADEventType.END_OF_SPEECH:
                    speaking   = False
                    frames     = speech_buf[:]
                    speech_buf = []
                    if frames:
                        asyncio.create_task(_respond(frames))

        async def _respond(frames: list[rtc.AudioFrame]) -> None:
            """Full Qwen3-Omni pipeline for one user turn, serialized by inference_lock."""
            turn_start = time.perf_counter()

            async with inference_lock:
                loop = asyncio.get_running_loop()

                # Persist speech frames to a temp WAV so process_mm_info can read it
                wav_bytes = _frames_to_wav_bytes(frames, SR_IN)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(wav_bytes)
                    audio_path = tmp.name

                duration_s = sum(
                    len(f.data) // 2 for f in frames
                ) / SR_IN
                logger.info(
                    "Running Qwen3-Omni inference (%.1f s of audio) …", duration_s
                )
                await _set_agent_state(ctx.room, "thinking")

                out_q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
                llm_start = time.perf_counter()

                t = threading.Thread(
                    target=session.run_generation,
                    args=(audio_path, out_q, loop),
                    daemon=True,
                )
                t.start()

                first_chunk = True
                while True:
                    chunk = await out_q.get()
                    if chunk is None:
                        break
                    if first_chunk:
                        now         = time.perf_counter()
                        ttfb        = now - llm_start
                        e2e_latency = now - turn_start
                        logger.info(
                            "⚡ Metrics — TTFB: %.3fs | E2E latency: %.3fs",
                            ttfb, e2e_latency,
                        )
                        await _set_agent_state(ctx.room, "speaking")
                        first_chunk = False
                    await stream_to_room(out_src, chunk)

                t.join()
                total_time = time.perf_counter() - llm_start
                logger.info("Turn complete. Total generation time: %.3fs", total_time)

                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

                torch.cuda.empty_cache()
                await _set_agent_state(ctx.room, "listening")

        await asyncio.gather(_push_to_vad(), _vad_loop())

    # ── Subscribe to audio tracks ─────────────────────────────────────────────
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_args) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Audio track subscribed — starting Qwen3-Omni pipeline")
            asyncio.ensure_future(handle_track(track))

    # Catch tracks already published before we connected
    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(handle_track(pub.track))

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        num_idle_processes=1,
    ))
