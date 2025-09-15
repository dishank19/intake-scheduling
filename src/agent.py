import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from langfuse import get_client, Langfuse, observe
from livekit import api
from livekit.agents import (
    Agent,
    AgentFalseInterruptionEvent,
    AgentSession,
    cli,
    get_job_context,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    metrics,
    NOT_GIVEN,
    RoomInputOptions,
    RunContext,
    WorkerOptions,
)
from livekit.agents.llm import function_tool
from livekit.plugins import assemblyai, inworld, noise_cancellation, openai, silero, elevenlabs
from tasks.appointment_scheduling_task import AppointmentSchedulingTask
from tasks.patient_intake_task import PatientIntakeTask

logger = logging.getLogger("agent")

load_dotenv(".env.local")

langfuse_client = Langfuse(
  secret_key="sk-lf-9fa49c27-4d28-4d5a-a277-0e2819b8a0a9",
  public_key="pk-lf-e39b133d-aee4-4da4-a485-13b93868d90e",
  host="https://us.cloud.langfuse.com"
)

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
            [Role]
You are a professional medical scheduling assistant for Bay Area Health. Your name is Sarah. You help patients schedule appointments by collecting their information and finding suitable appointment times.

[Context]
You are engaged in a phone conversation with a patient who needs to schedule a medical appointment. Stay focused on helping them through the scheduling process. Maintain awareness of the conversation context and respond only to what the patient is saying. Do not invent information not provided by the patient or the system.

[Voice Communication Guidelines]
Keep responses natural and conversational.
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

[Call Termination]
When the caller indicates they are done (e.g., goodbye, thank you), invoke end_call. Always finish any current speech before hangup.

[Error Recovery]
If you don't understand: "I'm sorry, could you repeat that please?"
If there's background noise: "I'm having a little trouble hearing you. Could you speak up a bit?"
If the patient seems confused: "Let me clarify what I'm asking..."
If validation fails: Present the issue conversationally without technical details.

[Empathy and Tone]
Acknowledge patient concerns with phrases like "I understand" or "I can help with that."
Show patience with elderly patients or those who need extra time.
Express appropriate concern for medical issues without providing medical advice.
Maintain professional boundaries while being friendly.
""",
        )

    async def on_enter(self) -> None:
        # Warm, brief intro and small talk before starting intake
        await self.session.generate_reply(
            instructions=(
                "Hi there! I'm Sarah, a virtual intake assistant with Bay Area Health. It's nice to meet you. How can I help you today?"
            )
        )

        intake_task = PatientIntakeTask(
            chat_ctx=self.chat_ctx,
            llm=self.session.llm,
            stt=self.session.stt,
            tts=self.session.tts,
            vad=self.session.vad,
        )
        patient_info = await intake_task

        # Preserve structured state across tasks

        self.session.userdata["patient_info"] = patient_info


        scheduling_task = AppointmentSchedulingTask(
            patient_info=patient_info,
            chat_ctx=self.chat_ctx,
            llm=self.session.llm,
            stt=self.session.stt,
            tts=self.session.tts,
            vad=self.session.vad,
        )
        final_info = await scheduling_task

        await self.session.generate_reply(
            instructions=(
                "Thank you for scheduling with us. "
                f"Your appointment with {final_info.appointment_doctor} is set for {final_info.appointment_time}. "
                "Confirmation emails have been sent. Is there anything else I can help you with today?"
            )
        )


    @function_tool
    @observe(name="tool.end_call")
    async def end_call(self, ctx: RunContext):
        """Called when the user wants to end the call"""
        # let the agent finish speaking
        await ctx.wait_for_playout()
        await hangup_call()



async def hangup_call():
    ctx = get_job_context()
    if ctx is None:
        # Not running in a job context
        return
    await ctx.api.room.delete_room(
        api.DeleteRoomRequest(
            room=ctx.room.name,
        )
    )


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


@observe(name="voice-session")
async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Ensure patient records storage exists
    try:
        if not os.path.exists("patient_records.json"):
            with open("patient_records.json", "w") as f:
                json.dump([], f)
    except Exception:
        logger.exception("Failed to initialize patient_records.json")

    # Set up a voice AI pipeline using OpenAI, Cartesia, Deepgram, and the LiveKit turn detector
    session = AgentSession(
        llm=openai.LLM(model="gpt-4o-mini"),
        stt = assemblyai.STT(
        end_of_turn_confidence_threshold=0.7,
        min_end_of_turn_silence_when_confident=160,
        max_turn_silence=2400,
        ),
        vad=silero.VAD.load(),
        # tts=cartesia.TTS(voice="6f84f4b8-58a2-430c-8c79-688dad597532"),
        # tts=elevenlabs.TTS(model="eleven_v3", voice_id= "Z3R5wn05IrDiVCyEkUrK"),
        tts=inworld.TTS(model="inworld-tts-1", voice="Ashley"),
        # tts=elevenlabs.TTS(model="eleven_v3", voice_id= "Z3R5wn05IrDiVCyEkUrK"),
        # turn_detection=MultilingualModel(),
        turn_detection='stt',
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # Langfuse v3: root observation via decorator; capture current trace id
    langfuse_client = get_client()
    trace_id = langfuse_client.get_current_trace_id()
    lf_data = {
        "client": langfuse_client,
        "trace_id": trace_id,
        "ttft_values": [],
        "ttfb_values": [],
    }
    session.userdata = {"langfuse": lf_data}

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

        # Forward key metrics to Langfuse safely (if supported)
        lf = session.userdata.get("langfuse") or {}
        client = lf.get("client")
        trace_id = lf.get("trace_id")
        if not (client and trace_id):
            return
        create_score = getattr(getattr(client, "score", None), "create", None)
        if create_score is None:
            return
        m = ev.metrics
        if isinstance(m, metrics.LLMMetrics):
            ttft_val = float(m.ttft)
            ttft_list = lf.get("ttft_values")
            if isinstance(ttft_list, list):
                ttft_list.append(ttft_val)
            create_score(name="llm.ttft_s", value=ttft_val, trace_id=trace_id)
            create_score(name="llm.tokens_in", value=float(m.input_tokens), trace_id=trace_id)
            create_score(name="llm.tokens_out", value=float(m.output_tokens), trace_id=trace_id)
            create_score(name="llm.tps", value=float(m.tokens_per_second), trace_id=trace_id)
        elif isinstance(m, metrics.TTSMetrics):
            ttfb_val = float(m.ttfb)
            ttfb_list = lf.get("ttfb_values")
            if isinstance(ttfb_list, list):
                ttfb_list.append(ttfb_val)
            create_score(name="tts.ttfb_s", value=ttfb_val, trace_id=trace_id)
            create_score(name="tts.audio_s", value=float(m.audio_duration), trace_id=trace_id)
        elif isinstance(m, metrics.STTMetrics):
            create_score(name="stt.audio_s", value=float(m.audio_duration), trace_id=trace_id)
        elif isinstance(m, metrics.EOUMetrics):
            create_score(name="eou.delay_s", value=float(m.end_of_utterance_delay), trace_id=trace_id)
            create_score(name="stt.final_delay_s", value=float(m.transcription_delay), trace_id=trace_id)

    # (Optional) If you later need to attach transcripts/audio, prefer decorators or explicit calls,
    # avoid async callbacks here to keep `.on()` usage synchronous-only.

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    @session.on("close")
    def _on_close(_):
        async def do_close():
            lf = session.userdata.get("langfuse") or {}
            client = lf.get("client")
            trace_id = lf.get("trace_id")
            ttft_values = lf.get("ttft_values") or []
            ttfb_values = lf.get("ttfb_values") or []
            if client and trace_id:
                create_score = getattr(getattr(client, "score", None), "create", None)
                if create_score is not None:
                    if ttft_values:
                        create_score(name="llm.ttft_mean_s", value=sum(ttft_values) / len(ttft_values), trace_id=trace_id)
                    if ttfb_values:
                        create_score(name="tts.ttfb_mean_s", value=sum(ttfb_values) / len(ttfb_values), trace_id=trace_id)
                flush = getattr(client, "flush", None)
                if callable(flush):
                    flush()

        session.userdata["_close_task"] = asyncio.create_task(do_close())

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
            noise_cancellation=noise_cancellation.BVCTelephony(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
