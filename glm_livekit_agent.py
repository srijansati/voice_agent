"""
GLM-4-Voice 9B INT4 — Native Audio-to-Audio LiveKit Agent
==========================================================
Bypasses the STT → LLM → TTS chain entirely.

Pipeline per turn:
  1. Silero VAD detects end-of-speech, returns buffered audio frames.
  2. WhisperVQ encoder converts raw audio → discrete speech tokens.
  3. Tokens are wrapped in a GLM prompt and sent to model_server.py.
  4. model_server streams back interleaved text + audio token IDs.
  5. Audio token blocks are decoded by Flow + HiFT into 22050 Hz audio.
  6. Audio is resampled to 48 kHz and published back to the LiveKit room.

Prerequisites
-------------
- CUDA GPU (INT4 model requires it)
- model_server.py must already be running:
    python model_server.py --port 10000 --model-path ./glm-4-voice-9b-int4
- .env with LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET
- LiveKit server running (docker compose up -d  or  livekit-server --dev)

Run
---
    python glm_livekit_agent.py start
"""

import asyncio
import json
import logging
import os
import sys
import threading
import uuid

import numpy as np
import requests
import torch
import torchaudio
from dotenv import load_dotenv
from transformers import AutoTokenizer, WhisperFeatureExtractor

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.vad import VADEventType
from livekit.plugins import silero

# cosyvoice and Matcha-TTS live alongside this file
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "cosyvoice"))
sys.path.insert(0, os.path.join(_ROOT, "third_party", "Matcha-TTS"))

from flow_inference import AudioDecoder  # noqa: E402 (path must be inserted first)
from speech_tokenizer.modeling_whisper import WhisperVQEncoder  # noqa: E402
from speech_tokenizer.utils import extract_speech_token  # noqa: E402

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("glm-livekit")

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_PATH       = os.path.join(_ROOT, "glm-4-voice-9b-int4")
TOKENIZER_PATH   = os.path.join(_ROOT, "glm-4-voice-tokenizer")
FLOW_CONFIG      = os.path.join(_ROOT, "glm-4-voice-decoder", "config.yaml")
FLOW_CKPT        = os.path.join(_ROOT, "glm-4-voice-decoder", "flow.pt")
HIFT_CKPT        = os.path.join(_ROOT, "glm-4-voice-decoder", "hift.pt")
MODEL_SERVER_URL = "http://localhost:10000/generate_stream"
DEVICE           = "cuda"

SR_IN    = 16_000   # WhisperVQ encoder expects 16 kHz
SR_OUT   = 22_050   # Flow model outputs 22.05 kHz
SR_LK    = 48_000   # LiveKit standard sample rate
FRAME_MS = 10       # publish audio in 10 ms frames

SYSTEM_PROMPT = (
    "User will provide you with a speech instruction. Do it step by step. "
    "First, think about the instruction and respond in a interleaved manner, "
    "with 13 text token followed by 26 audio tokens."
)

# Audio token decode block sizes (matching web_demo.py)
BLOCK_INIT = 10
BLOCK_GROW = 20


# ── Model container ──────────────────────────────────────────────────────────
class GLMModels:
    """All GLM-4-Voice components — loaded once in prewarm(), shared per worker."""

    def __init__(self):
        logger.info("Loading GLM tokenizer …")
        self.tokenizer    = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.audio_offset = self.tokenizer.convert_tokens_to_ids("<|audio_0|>")
        self.end_token_id = self.tokenizer.convert_tokens_to_ids("<|user|>")

        logger.info("Loading WhisperVQ speech tokenizer …")
        self.whisper        = WhisperVQEncoder.from_pretrained(TOKENIZER_PATH).eval().to(DEVICE)
        self.feat_extractor = WhisperFeatureExtractor.from_pretrained(TOKENIZER_PATH)

        logger.info("Loading Flow + HiFT audio decoder …")
        self.decoder = AudioDecoder(
            config_path=FLOW_CONFIG,
            flow_ckpt_path=FLOW_CKPT,
            hift_ckpt_path=HIFT_CKPT,
            device=DEVICE,
        )
        torch.cuda.empty_cache()
        logger.info("GLMModels ready ✓")


# ── Per-session conversation state ────────────────────────────────────────────
class GLMSession:
    """One instance per LiveKit room session; tracks conversation history."""

    def __init__(self, models: GLMModels):
        self.m       = models
        self.history = ""  # accumulated prompt string (system + all turns)

    # ── Step 1: audio → speech token string ──────────────────────────────────
    def encode_audio(self, waveform: torch.Tensor, sr: int) -> str:
        """
        Convert a [1, N] float32 waveform at `sr` Hz into the GLM speech token
        string '<|begin_of_audio|><|audio_X|>...<|end_of_audio|>'.
        Returns "" when the encoder produces no tokens (silence / too short).
        Called via run_in_executor — must be thread-safe.
        """
        tokens = extract_speech_token(
            self.m.whisper,
            self.m.feat_extractor,
            [(waveform, sr)],   # extract_speech_token accepts (tensor, sr) tuples
        )[0]
        if not tokens:
            return ""
        body = "".join(f"<|audio_{t}|>" for t in tokens)
        return f"<|begin_of_audio|>{body}<|end_of_audio|>"

    # ── Step 2: build full prompt ─────────────────────────────────────────────
    def build_prompt(self, speech_token_str: str) -> str:
        h = self.history.strip()
        if "<|system|>" not in h:
            h += f"<|system|>\n{SYSTEM_PROMPT}"
        h += f"<|user|>\n{speech_token_str}<|assistant|>streaming_transcription\n"
        return h

    # ── Step 3: GLM inference (blocking — run inside a daemon thread) ─────────
    def run_generation(
        self,
        prompt: str,
        out_q: asyncio.Queue,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        POST to model_server, decode the streaming token response, and put
        decoded float32 audio chunks (SR_OUT = 22050 Hz) onto `out_q`.
        Puts None as sentinel when done.  Runs in a daemon thread.
        """
        m   = self.m
        dev = DEVICE

        # ── HTTP request ──────────────────────────────────────────────────────
        try:
            resp = requests.post(
                MODEL_SERVER_URL,
                data=json.dumps({
                    "prompt":         prompt,
                    "temperature":    0.2,
                    "top_p":          0.8,
                    "max_new_tokens": 2000,
                }),
                stream=True,
                timeout=120,
            )
        except requests.ConnectionError:
            logger.error(
                "Cannot reach model_server at %s — start it with:\n"
                "  python model_server.py --port 10000",
                MODEL_SERVER_URL,
            )
            asyncio.run_coroutine_threadsafe(out_q.put(None), loop)
            return

        # ── Streaming token loop ──────────────────────────────────────────────
        text_tokens:     list[int] = []
        audio_tokens:    list[int] = []
        complete_tokens: list[int] = []

        # Context tensors that grow as we produce audio
        feat_ctx = torch.zeros(1, 0, 80,            device=dev)  # mel prompt features
        tok_ctx  = torch.zeros(1, 0, dtype=torch.int64, device=dev)  # audio token context
        mels: list[torch.Tensor] = []
        prev_mel  = None
        uid       = str(uuid.uuid4())
        finalize  = False
        block_sz  = BLOCK_INIT

        for raw in resp.iter_lines():
            if not raw:
                continue
            data = json.loads(raw)
            if data.get("error_code", 0) != 0:
                logger.error("model_server returned error: %s", data)
                break
            token_id: int = data["token_id"]

            if token_id == m.end_token_id:
                finalize = True

            # ── Decode a block of audio tokens ────────────────────────────────
            if len(audio_tokens) >= block_sz or (finalize and audio_tokens):
                block_sz = BLOCK_GROW
                tts_tok  = torch.tensor(audio_tokens, device=dev).unsqueeze(0)

                if prev_mel is not None:
                    # Use all previously decoded mel frames as context for Flow
                    feat_ctx = torch.cat(mels, dim=-1).transpose(1, 2)

                tts_speech, tts_mel = m.decoder.token2wav(
                    tts_tok,
                    uuid=uid,
                    prompt_token=tok_ctx,
                    prompt_feat=feat_ctx,
                    finalize=finalize,
                )
                prev_mel  = tts_mel
                mels.append(tts_mel)
                tok_ctx   = torch.cat((tok_ctx, tts_tok), dim=-1)
                audio_tokens = []

                chunk = tts_speech.squeeze().cpu().numpy()  # float32, SR_OUT Hz
                asyncio.run_coroutine_threadsafe(out_q.put(chunk), loop)

            if not finalize:
                complete_tokens.append(token_id)
                if token_id >= m.audio_offset:
                    audio_tokens.append(token_id - m.audio_offset)
                else:
                    text_tokens.append(token_id)

        # ── Persist conversation history for next turn ────────────────────────
        completion   = m.tokenizer.decode(complete_tokens, spaces_between_special_tokens=False)
        self.history = prompt + completion

        asyncio.run_coroutine_threadsafe(out_q.put(None), loop)  # sentinel


# ── LiveKit audio utilities ──────────────────────────────────────────────────
_resamplers: dict[tuple[int, int], torchaudio.transforms.Resample] = {}


def _resampler(src: int, dst: int) -> torchaudio.transforms.Resample:
    key = (src, dst)
    if key not in _resamplers:
        _resamplers[key] = torchaudio.transforms.Resample(src, dst)
    return _resamplers[key]


async def stream_to_room(
    source: rtc.AudioSource,
    audio_np: np.ndarray,
    src_sr: int = SR_OUT,
) -> None:
    """Resample float32 audio to SR_LK and push it as 10 ms LiveKit frames."""
    wf  = torch.from_numpy(audio_np).float().unsqueeze(0)
    wf  = _resampler(src_sr, SR_LK)(wf)
    pcm = (wf.squeeze().numpy() * 32_767).clip(-32_768, 32_767).astype(np.int16)

    frame_n = int(SR_LK * FRAME_MS / 1000)   # 480 samples per 10 ms at 48 kHz
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
def prewarm(proc) -> None:
    # Only load the tiny CPU-only VAD here.  GLMModels (~4 GB GPU) must NOT be
    # loaded during prewarm: when a job is assigned the pool immediately spawns
    # a refill process that also calls prewarm(), so two model copies would try
    # to load simultaneously and OOM.  GPU models load inside entrypoint instead.
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()
    logger.info("GLM agent connected → room: %s", ctx.room.name)

    vad = ctx.proc.userdata["vad"]

    # Load GPU models now, inside the job process.  max_concurrent_jobs=1
    # ensures only one session runs at a time, so this never races with another
    # entrypoint loading the same models.
    logger.info("Loading GLM models for session …")
    loop   = asyncio.get_running_loop()
    models = await loop.run_in_executor(None, GLMModels)

    session = GLMSession(models)

    # ── Outgoing audio track (agent voice) ───────────────────────────────────
    out_src   = rtc.AudioSource(SR_LK, 1)
    out_track = rtc.LocalAudioTrack.create_audio_track("glm-voice", out_src)
    await ctx.room.local_participant.publish_track(
        out_track,
        rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
    )
    logger.info("Agent audio track published")

    # One lock so we never run two GLM inferences concurrently
    inference_lock = asyncio.Lock()

    # ── Per-track audio pipeline ──────────────────────────────────────────────
    async def handle_track(track: rtc.Track) -> None:
        """VAD-driven loop: detect speech → run GLM → stream audio back."""
        audio_stream = rtc.AudioStream(track, sample_rate=SR_IN, num_channels=1)
        vad_stream   = vad.stream()
        speech_buf: list[rtc.AudioFrame] = []
        speaking     = False

        # Forward every audio frame into the VAD stream
        async def _push_to_vad() -> None:
            async for ev in audio_stream:
                vad_stream.push_frame(ev.frame)

        # Consume VAD events and trigger inference on END_OF_SPEECH
        async def _vad_loop() -> None:
            nonlocal speaking, speech_buf
            async for ev in vad_stream:
                if ev.type == VADEventType.START_OF_SPEECH:
                    speaking   = True
                    speech_buf = []

                elif ev.type == VADEventType.INFERENCE_DONE and speaking:
                    # Accumulate frames while user is talking
                    speech_buf.extend(ev.frames)

                elif ev.type == VADEventType.END_OF_SPEECH:
                    speaking    = False
                    frames      = speech_buf[:]
                    speech_buf  = []
                    if frames:
                        asyncio.create_task(_respond(frames))

        async def _respond(frames: list[rtc.AudioFrame]) -> None:
            """Full GLM pipeline for one user turn, serialized by inference_lock."""
            async with inference_lock:
                loop = asyncio.get_running_loop()

                # Assemble float32 waveform [1, N] at SR_IN
                raw = np.concatenate(
                    [np.frombuffer(f.data, dtype=np.int16) for f in frames]
                ).astype(np.float32) / 32_768.0
                waveform = torch.from_numpy(raw).unsqueeze(0)   # [1, N]

                # Speech tokenization runs on GPU — offload to thread pool
                speech_str: str = await loop.run_in_executor(
                    None, session.encode_audio, waveform, SR_IN
                )
                if not speech_str:
                    logger.warning("No speech tokens extracted — skipping turn")
                    return

                prompt = session.build_prompt(speech_str)
                logger.info(
                    "Running GLM inference (%d speech frames, %.1f s) …",
                    len(frames),
                    len(raw) / SR_IN,
                )

                # GLM generation runs in a daemon thread; audio chunks flow
                # back via an asyncio.Queue so we can push them without blocking.
                out_q: asyncio.Queue[np.ndarray | None] = asyncio.Queue()
                t = threading.Thread(
                    target=session.run_generation,
                    args=(prompt, out_q, loop),
                    daemon=True,
                )
                t.start()

                while True:
                    chunk = await out_q.get()
                    if chunk is None:
                        break
                    await stream_to_room(out_src, chunk)

                t.join()
                logger.info("Turn complete")

        await asyncio.gather(_push_to_vad(), _vad_loop())

    # ── Subscribe to audio tracks ─────────────────────────────────────────────
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, *_args) -> None:
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info("Audio track subscribed — starting GLM pipeline")
            asyncio.ensure_future(handle_track(track))

    # Handle tracks that were already published before we connected
    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.track and pub.track.kind == rtc.TrackKind.KIND_AUDIO:
                asyncio.ensure_future(handle_track(pub.track))

    await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    cli.run_app(WorkerOptions(
        entrypoint_fnc=entrypoint,
        prewarm_fnc=prewarm,
        # Keep one idle process ready so sessions start without process-spawn
        # delay.  The idle process holds no GPU (prewarm only loads VAD).
        num_idle_processes=1,
        # Only one active session at a time.  A second job request is queued
        # by LiveKit until the current session ends and GPU memory is freed.
        max_concurrent_jobs=1,
    ))
