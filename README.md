# Assort Health – Voice AI Intake & Scheduling Agent (LiveKit)

Deployment status
- Deployed on LiveKit Cloud
- Phone number: +1 408 549 7080
- The number can be called anytime to reach the agent

This repository contains a production-ready voice AI agent built on LiveKit Agents for Python. It answers inbound phone calls, performs a structured medical intake, validates addresses against OpenStreetMap (Nominatim), offers appointment options, books a slot, and sends confirmation emails. It is designed to satisfy the Assort Health take‑home requirements end‑to‑end.

## What it does (requirements mapping)

- Patient Intake (voice):
  - Collects: name, date of birth, insurance payer and ID, referral details, chief complaint, address, phone, optional email
  - Confirms each item in natural language before persisting
  - Address validation: calls OpenStreetMap Nominatim and returns a suggested normalized address for confirmation (fast fallbacks; always proposes a suggestion)
- Appointment Scheduling:
  - Presents small sets (1–2 at a time) of realistic sample slots across multiple providers and days
  - Books selected slot, then sends confirmation emails via SendGrid
- Telephony:
  - Inbound calling via LiveKit SIP; dispatch rules included
  - Mid‑call termination tool (`end_call`) and LiveKit‑recommended hangup flow (room delete after playout)

## Tech stack

- LiveKit Agents (Python): session, tools, workflows, telephony
- STT: AssemblyAI; LLM: OpenAI; TTS: Inworld (easily swappable)
- Address validation: OpenStreetMap Nominatim API
- Email: SendGrid

## Repository layout

- `src/agent.py` – Agent orchestration, intro/small talk, task chaining, end‑call tool (LiveKit hangup)
- `src/tasks/patient_intake_task.py` – Intake workflow, per‑field confirmation, address/phone/DOB validation tools
- `src/tasks/appointment_scheduling_task.py` – Appointment options and booking; synchronous SendGrid confirmation before task completes
- `dispatch-rule.json` – Example LiveKit dispatch rule for inbound telephony
- `inbound-trunk.json` – Example SIP trunk descriptor (reference)

## Environment variables

Create `.env.local` and set:

- LIVEKIT_URL
- LIVEKIT_API_KEY
- LIVEKIT_API_SECRET
- OPENAI_API_KEY
- ASSEMBLYAI_API_KEY
- SENDGRID_API_KEY

Optional (if you swap providers):
- CARTESIA_API_KEY, ELEVENLABS_API_KEY, etc.

## Setup

```bash
cd /Users/dishankj/Workspace/assort_test
uv sync
uv run python src/agent.py download-files
```

## Running

### Console test (no phone)
```bash
uv run python src/agent.py console
```

### Telephony (inbound)
1) Ensure agent runs with a name for explicit dispatch (configured in `src/agent.py` entrypoint via LiveKit CLI app). Start the worker in dev:
```bash
uv run python src/agent.py dev
```
2) Create a dispatch rule that routes inbound calls to a room and dispatches this agent (example in `dispatch-rule.json`).
```bash
lk sip dispatch create dispatch-rule.json
```
3) Point your SIP provider number at LiveKit. Call the number to reach the agent.

LiveKit hangup pattern is implemented per the docs (finish playout then delete the room). Reference: https://docs.livekit.io/agents/start/telephony/#hangup

## How it meets the assignment

- Collects and confirms: name, DOB, insurance payer/ID, referral and physician, chief complaint, address (validated), phone, optional email
- Offers multiple realistic appointment options by provider and time; books chosen slot
- Sends confirmation emails (SendGrid) after booking completes
- Works over a phone number via SIP into LiveKit

## Notes on address validation

- Uses Nominatim with fast query fallbacks (full, reduced) and short timeouts
- Always returns a `suggested_address`; agent reads back and asks for confirmation
- If suggestion is declined, agent re-collects missing parts and revalidates

## Commands cheat‑sheet

```bash
# Install deps
uv sync

# Download agent model assets (VAD, turn detector, etc.)
uv run python src/agent.py download-files

# Run locally with console UI
uv run python src/agent.py console

# Run for telephony/dev
uv run python src/agent.py dev

# Create dispatch rule (after configuring SIP trunk in LiveKit Cloud)
lk sip dispatch create dispatch-rule.json
```


