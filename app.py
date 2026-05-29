import logging
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    room_io,
    TurnHandlingOptions,
    EndpointingOptions,
    InterruptionOptions,
    PreemptiveGenerationOptions
)
from livekit.plugins import openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from custom_agent.wifi_agent import WifiTroubleshootingAgent
# Load the keys from the .env file
load_dotenv()

# Configure logging to see what the agent is doing in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-agent")

def prewarm(proc):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    """
    The main coordinator function. LiveKit triggers this when a user 
    requests an agent session or dials in.
    """
    logger.info(f"Connecting to room: {ctx.room.name}")

    await ctx.connect()
    
    # Configure the session manager with Local Voice Activity Detection (VAD)
    # This prevents the agent from talking over the user.
    # -------------------------------------------------------------------------
    # PATH A: OpenAI Realtime API — single model hop, lowest latency (~300-500ms)
    # Uncomment this block and comment out PATH B to use it.
    # -------------------------------------------------------------------------
    # realtime_model = openai.realtime.RealtimeModel()
    # session = AgentSession(
    #     vad=ctx.proc.userdata["vad"],
    #     llm=realtime_model,
    # )
    # await session.start(
    #     room=ctx.room,
    #     agent=WifiTroubleshootingAgent(),
    #     room_options=room_io.RoomOptions(
    #         audio_input=room_io.AudioInputOptions(),
    #     ),
    # )

    # -------------------------------------------------------------------------
    # PATH B: Optimized STT→LLM→TTS pipeline (~400-700ms)
    # Key changes from original:
    #   - min_delay 0.3→0.05: was adding 300ms of silence-wait before STT
    #   - max_delay 1.0→0.4: reduce worst-case endpointing wait
    #   - MultilingualModel turn detector: ends turns on semantics, not just silence
    #   - use_tts_aligned_transcript removed: saves a processing step
    # To get further below this floor: swap STT→Deepgram, TTS→Cartesia (see README)
    # -------------------------------------------------------------------------
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=openai.STT(model="whisper-1", use_realtime=True),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=openai.TTS(model="gpt-4o-mini-tts", speed=1.5),
        turn_detection=MultilingualModel(),
        turn_handling=TurnHandlingOptions(
            endpointing=EndpointingOptions(
                min_delay=0.05,
                max_delay=0.4,
            ),
            interruption=InterruptionOptions(
                enabled=False
            ),
            preemptive_generation=PreemptiveGenerationOptions(
                enabled=True,
                preemptive_tts=True
            )
        ),
    )

    await session.start(
        room=ctx.room,
        agent=Agent(instructions= "You are an AI assistant, answer the user in 1-2 lines"),
    )

    logger.info("Agent ready - listening for speech...")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )