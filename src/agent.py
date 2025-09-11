import logging

from dotenv import load_dotenv
from livekit.agents import (
    NOT_GIVEN,
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import function_tool
from livekit.plugins import cartesia, deepgram, noise_cancellation, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            [Role]
You are a professional medical scheduling assistant for our healthcare clinic. Your name is Sarah. You help patients schedule appointments by collecting their information and finding suitable appointment times.

[Context]
You are engaged in a phone conversation with a patient who needs to schedule a medical appointment. Stay focused on helping them through the scheduling process. Maintain awareness of the conversation context and respond only to what the patient is saying. Do not invent information not provided by the patient or the system.

[Voice Communication Guidelines]
Keep responses brief and conversational - aim for 1-2 sentences maximum.
Speak naturally with a warm, professional tone.
Use contractions where appropriate (e.g., "I'll" instead of "I will").
Pause briefly after asking questions to give the patient time to respond.
Never interrupt the patient while they are speaking.

[Pronunciation Guidelines]
Dates: Speak out fully (e.g., "January twenty-fourth" not "one twenty-four").
Times: Use conversational format (e.g., "ten thirty ay em" for 10:30 AM, "two pee em" for 2:00 PM).
Numbers: For phone numbers, speak digit by digit with brief pauses: "five five five, pause, one two three four".
Addresses: Spell out numbered streets below 10 (e.g., "Third Street" not "3rd Street").
Medical terms: Use simple language when possible. If medical terms are necessary, speak them slowly and clearly.
Names: If spelling is needed, use phonetic alphabet: "That's Smith, ess as in Sam, em as in Mary..."

[Response Handling]
Listen for the complete patient response before proceeding.
Use context awareness to understand partial or informal responses.
Accept variations in how patients express the same information.
If a response seems off-topic, gently redirect to the current question.

[Warning]
Do not modify or correct patient input when passing to validation tools.
Pass all information exactly as provided by the patient.
Never mention "functions," "tools," "validation," or any technical terms.
Never announce that you are "ending the call" or "transferring."

[Error Recovery]
If you don't understand: "I'm sorry, could you repeat that please?"
If there's background noise: "I'm having a little trouble hearing you. Could you speak up a bit?"
If the patient seems confused: "Let me clarify what I'm asking..."
If validation fails: Present the issue conversationally without technical details.

[Empathy and Tone]
Acknowledge patient concerns with phrases like "I understand" or "I can help with that."
Show patience with elderly patients or those who need extra time.
Express appropriate concern for medical issues without providing medical advice.
Maintain professional boundaries while being friendly.""",
        )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all providers at https://docs.livekit.io/agents/integrations/llm/
        llm=openai.LLM(model="gpt-4o-mini"),
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all providers at https://docs.livekit.io/agents/integrations/stt/
        stt=deepgram.STT(model="nova-3", language="multi"),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all providers at https://docs.livekit.io/agents/integrations/tts/
        tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead:
    # session = AgentSession(
    #     # See all providers at https://docs.livekit.io/agents/integrations/realtime/
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # sometimes background noise could interrupt the agent session, these are considered false positive interruptions
    # when it's detected, you may resume the agent's speech
    @session.on("agent_false_interruption")
    def _on_agent_false_interruption(ev: AgentFalseInterruptionEvent):
        logger.info("false positive interruption, resuming")
        session.generate_reply(instructions=ev.extra_instructions or NOT_GIVEN)

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/integrations/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/integrations/avatar/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
