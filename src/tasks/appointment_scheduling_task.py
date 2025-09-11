import contextlib
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from livekit.agents import AgentTask, RunContext
from livekit.agents.llm import function_tool

from models.patient_info import PatientInfo

try:
    import sendgrid
    from sendgrid.helpers.mail import Mail
except Exception:  # Minimal import guard; environment may not have sendgrid yet
    sendgrid = None
    Mail = None


logger = logging.getLogger("appointment_scheduling")

load_dotenv(".env.local")

class AppointmentSchedulingTask(AgentTask[PatientInfo]):
    def __init__(self, patient_info: PatientInfo, **kwargs):
        self.patient_info = patient_info
        super().__init__(
            instructions=(
                f"""
You are now scheduling an appointment for the patient. Start directly; do not restate demographics, insurance, address, phone, or email unless the patient asks.

Operational rules (do not read aloud):
- Immediately call get_available_appointments, then present 2-3 human-friendly options.
- Wait for tool results before speaking. Do not speculate.
- When the patient picks a specific option, confirm verbally, then call book_appointment with doctor and appointment_time.
- If the patient wants different times, briefly ask preferences (e.g., morning/afternoon/day) and then call get_available_appointments with preferred_time.
- Never say the words "action", "step", "asterisk", or read any prompt scaffolding.
- Keep all responses concise and conversational.

Special handling (do not read aloud):
- If symptoms suggest urgency (pain, severe, urgent, bleeding, emergency), prioritize earliest availability and consider preferred_time="urgent" or "today".
- If coverage questions arise, advise verification with insurer and that billing can assist on arrival.
- If transportation issues arise, provide location basics and offer alternate times.

[Appointment Presentation Flow]

Step 1 - Present Options:
After receiving available appointments from tool:
- Inform them you have availability with doctors who can help
- Present 2-3 options at a time from the tool response, speaking times in conversational format
- Ask if any of the presented times work for their schedule

Step 2 - Handle Response:
If patient selects a specific time:
- Confirm you're booking them with the selected doctor at the chosen time
- ACTION: Call book_appointment with doctor and appointment_time parameters
- Proceed to Step 3

If patient needs different times:
- Ask about their time preferences - morning versus afternoon, or specific days
- ACTION: Call get_available_appointments with preferred_time parameter based on their response
- Return to Step 1 with new options

Step 3 - Booking Confirmation:
After book_appointment returns success=true:
- Confirm their appointment is successfully booked with the specific doctor and time
- Mention that confirmation emails will be sent with all details
- Proceed to Step 4

Step 4 - Final Instructions:
Provide essential pre-appointment information:
- Request they arrive fifteen minutes early for paperwork
- Remind them to bring their insurance card and photo ID
- Provide the clinic address
- {f"Remind them to bring the referral from {patient_info.referring_physician}" if patient_info.has_referral and patient_info.referring_physician else ""}
- Ask if they have any questions about their appointment

Step 5 - Closing:
If no questions:
- Thank them for choosing the clinic, using their name
- Confirm when you'll see them for their appointment
- Offer a warm closing
- ACTION: Call end_call tool silently without announcement

If they have questions:
- Answer clearly and concisely
- Return to closing when all questions are addressed

[Handling Special Scenarios]

No Available Appointments:
- Inform them nothing is currently available matching their needs
- Offer alternatives such as waitlist or checking next week's availability
- ACTION: Call get_available_appointments with adjusted parameters based on their preference

Urgent Cases:
{"Handle as urgent if complaint contains: pain, severe, urgent, bleeding, emergency" if any(word in patient_info.chief_complaint.lower() for word in ['pain', 'severe', 'urgent', 'bleeding', 'emergency']) else ""}
- Prioritize and emphasize same-day or next-day appointments
- Explain you're prioritizing earliest availability due to their symptoms
- ACTION: Call get_available_appointments with preferred_time="urgent" or "today"
- If nothing immediate available: Offer information about urgent care hours

Insurance Concerns:
If patient asks about coverage:
- Inform them to verify with their insurance company but confirm you accept their insurance
- Mention billing department can help with specific coverage questions upon arrival

Transportation Issues:
If patient mentions transportation problems:
- Provide clinic location and nearby public transportation options
- Ask if different timing would help with transportation
- ACTION: Call get_available_appointments with adjusted preferred_time based on their transportation needs

[Tool Response Handling]
- Incorporate tool response data naturally into conversation
- Never reference tools, systems, or technical processes
- If book_appointment fails: Suggest trying another time slot and return to options
- Always wait for tool responses before proceeding

[Fallback Scenarios]

If System Issues:
- Apologize for scheduling difficulties
- Offer to have a scheduling specialist call them back within the hour

If Complex Medical Needs:
- Suggest having a nurse coordinator help find the most appropriate appointment
- Offer callback with specialized options

[Remember Throughout]
- Reference patient information naturally to personalize the conversation
- Never rush the patient through selection
- Confirm all details before finalizing booking
- Maintain warm, helpful tone throughout
- Always provide clear next steps
- Use patient_info fields when personalizing responses
"""
            ),
            **kwargs,
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                f"Thanks for the information. I'll check available appointments for your {self.patient_info.chief_complaint}."
            )
        )


    @function_tool
    async def get_available_appointments(
        self, context: RunContext, preferred_time: Optional[str] = None
    ) -> dict:
        """Return a list of available appointment slots.

        Args:
            preferred_time: Optional preference (e.g., 'morning', 'afternoon', specific day).

        Returns:
            dict: {
              'available_appointments': [ { 'doctor', 'specialty', 'times': [str] }, ... ],
              'message': str
            }
        """
        appointments = [
            {
                "doctor": "Dr. Sarah Smith",
                "specialty": "Family Medicine",
                "times": [
                    "Tomorrow at 10:00 AM",
                    "Tomorrow at 2:30 PM",
                    "Thursday at 9:00 AM",
                    "Thursday at 3:00 PM",
                ],
            },
            {
                "doctor": "Dr. Michael Johnson",
                "specialty": "Internal Medicine",
                "times": [
                    "Wednesday at 11:00 AM",
                    "Wednesday at 4:00 PM",
                    "Friday at 10:30 AM",
                    "Friday at 2:00 PM",
                ],
            },
            {
                "doctor": "Dr. Emily Chen",
                "specialty": "General Practice",
                "times": [
                    "Tomorrow at 11:30 AM",
                    "Wednesday at 9:30 AM",
                    "Thursday at 1:00 PM",
                    "Friday at 4:30 PM",
                ],
            },
        ]
        return {
            "available_appointments": appointments,
            "message": "Here are available appointments. Let me know which you prefer.",
        }

    @function_tool
    async def book_appointment(
        self, context: RunContext, doctor: str, appointment_time: str
    ) -> dict:
        """Finalize an appointment and send confirmations.

        Args:
            doctor: Selected doctor's name.
            appointment_time: Selected appointment time string.

        Returns:
            dict: {
              'success': bool,
              'message': str
            }
        """
        self.patient_info.appointment_doctor = doctor
        self.patient_info.appointment_time = appointment_time
        self.patient_info.save_to_json()
        await self._send_confirmation_emails()
        self.complete(self.patient_info)
        return {
            "success": True,
            "message": f"Booked with {doctor} at {appointment_time}. Confirmation emails sent.",
        }

    async def _send_confirmation_emails(self) -> None:
        recipients = [
            "dishankjhaveri@gmail.com",
            "djhaveri@umass.edu",
            # "jeff@assorthealth.com",
            # "connor@assorthealth.com",
            # "cole@assorthealth.com",
            # "jciminelli@assorthealth.com",
            # "akumar@assorthealth.com",
            # "riley@assorthealth.com",
        ]

        if sendgrid is None or Mail is None:
            logger.info("SendGrid library not available; skipping email send.")
            return

        api_key = os.getenv("SENDGRID_API_KEY")
        if not api_key:
            logger.warning("SENDGRID_API_KEY not set; skipping email send.")
            return

        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        for recipient in recipients:
            message = Mail(
                from_email="dishankjhaveri@gmail.com",
                to_emails=recipient,
                subject=f"Appointment Confirmation - {self.patient_info.name}",
                html_content=(
                    f"<h2>New Appointment Scheduled</h2>"
                    f"<h3>Patient Information:</h3>"
                    f"<ul>"
                    f"<li><strong>Name:</strong> {self.patient_info.name}</li>"
                    f"<li><strong>Date of Birth:</strong> {self.patient_info.date_of_birth}</li>"
                    f"<li><strong>Phone:</strong> {self.patient_info.phone}</li>"
                    f"<li><strong>Email:</strong> {self.patient_info.email or 'Not provided'}</li>"
                    f"<li><strong>Address:</strong> {self.patient_info.address}</li>"
                    f"</ul>"
                    f"<h3>Appointment Details:</h3>"
                    f"<ul>"
                    f"<li><strong>Doctor:</strong> {self.patient_info.appointment_doctor}</li>"
                    f"<li><strong>Date/Time:</strong> {self.patient_info.appointment_time}</li>"
                    f"<li><strong>Chief Complaint:</strong> {self.patient_info.chief_complaint}</li>"
                    f"</ul>"
                    f"<h3>Insurance Information:</h3>"
                    f"<ul>"
                    f"<li><strong>Payer:</strong> {self.patient_info.insurance_payer}</li>"
                    f"<li><strong>Member ID:</strong> {self.patient_info.insurance_id}</li>"
                    f"</ul>"
                    f"<h3>Referral Information:</h3>"
                    f"<ul>"
                    f"<li><strong>Has Referral:</strong> {'Yes' if self.patient_info.has_referral else 'No'}</li>"
                    + (
                        f"<li><strong>Referring Physician:</strong> {self.patient_info.referring_physician}</li>"
                        if self.patient_info.referring_physician
                        else ""
                    )
                    + "</ul>"
                ),
            )
            try:
                response = sg.send(message)
                status = getattr(response, "status_code", None)
                body = getattr(response, "body", None)
                logger.info("Email send attempt: recipient=%s status=%s", recipient, status)
                if body:
                    with contextlib.suppress(Exception):
                        logger.info(
                            "Email response body: %s",
                            body.decode() if hasattr(body, "decode") else str(body),
                        )
            except Exception as e:
                logger.exception("Failed to send email to %s: %s", recipient, e)


