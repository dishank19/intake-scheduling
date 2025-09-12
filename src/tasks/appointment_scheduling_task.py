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
You are scheduling an appointment. Speak naturally, like a friendly clinic coordinator.

Rules to follow (internal only - never say these words aloud):
- Never say: action, step, asterisk, bracket, instruction, tool.
- Do not read any headings or scaffolding. Talk only to the patient.
- Present options slowly: offer at most two times at once, then ask which works.
- After they respond, either book the chosen slot or ask a quick preference (morning/afternoon/day) and fetch more options.
- Confirm the choice in plain language before booking.
- Avoid repeating demographics unless asked; personalize when helpful.

Flow (internal only):
- Immediately call get_available_appointments (use preferred_time if they mentioned one).
- Present 1-2 options in conversational natural language. Pause and check preference after each small set.
- If none work, ask a brief preference question and call get_available_appointments again with preferred_time, then present the next 1-2 options.
- When they pick a time, call book_appointment with the selected doctor and appointment_time.
- After booking succeeds, confirm out loud and mention a confirmation email has been sent.
- Offer quick arrival guidance (arrive 15 minutes early, bring insurance card and photo ID, clinic address). {f"If relevant, remind them to bring the referral from {patient_info.referring_physician}." if patient_info.has_referral and patient_info.referring_physician else ""}
- Ask if they have any questions. When finished, end politely and call end_call silently.

Special handling (internal only):
- If symptoms suggest urgency (pain, severe, urgent, bleeding, emergency), prioritize the earliest availability and prefer preferred_time="urgent" or "today".
- If transportation is an issue, offer convenient times. If coverage questions arise, advise verifying with their insurer and that billing can help on arrival.

Tool response handling (internal only):
- Use tool data naturally; never mention tools or systems. Always wait for tool results before speaking.
 - When waiting on tool results, use a short filler like "Let me pull that up for you" or "One moment while I check availability."
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
                    "Today at 3:30 PM",
                    "Tomorrow at 9:00 AM",
                    "Tomorrow at 2:30 PM",
                    "Wednesday at 10:15 AM",
                    "Thursday at 9:00 AM",
                    "Thursday at 3:15 PM",
                ],
            },
            {
                "doctor": "Dr. Michael Johnson",
                "specialty": "Internal Medicine",
                "times": [
                    "Tomorrow at 8:45 AM",
                    "Wednesday at 11:00 AM",
                    "Wednesday at 4:00 PM",
                    "Friday at 10:30 AM",
                    "Friday at 2:00 PM",
                    "Monday at 9:30 AM",
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
                    "Monday at 8:30 AM",
                ],
            },
            {
                "doctor": "Dr. Priya Natarajan",
                "specialty": "Pediatrics",
                "times": [
                    "Tomorrow at 10:15 AM",
                    "Wednesday at 3:30 PM",
                    "Thursday at 11:45 AM",
                    "Friday at 2:15 PM",
                    "Monday at 9:00 AM",
                ],
            },
            {
                "doctor": "Dr. Javier Morales",
                "specialty": "Dermatology",
                "times": [
                    "Wednesday at 8:30 AM",
                    "Wednesday at 1:15 PM",
                    "Thursday at 4:00 PM",
                    "Friday at 11:00 AM",
                    "Monday at 3:45 PM",
                ],
            },
            {
                "doctor": "Dr. Olivia Patel",
                "specialty": "Orthopedics",
                "times": [
                    "Tomorrow at 4:45 PM",
                    "Thursday at 8:15 AM",
                    "Thursday at 12:30 PM",
                    "Friday at 3:00 PM",
                    "Monday at 10:45 AM",
                ],
            },
            {
                "doctor": "Dr. Daniel Kim",
                "specialty": "Cardiology",
                "times": [
                    "Wednesday at 2:00 PM",
                    "Thursday at 10:30 AM",
                    "Friday at 1:00 PM",
                    "Monday at 11:30 AM",
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

        # Wait for email confirmation before completing task
        email_status = await self._send_confirmation_emails()

        # Only complete the task after emails are confirmed sent
        if email_status["emails_sent"]:
            self.complete(self.patient_info)
            return {
                "success": True,
                "message": f"Booked with {doctor} at {appointment_time}. Confirmation emails sent.",
            }
        else:
            # Still complete the task but indicate email issue
            self.complete(self.patient_info)
            return {
                "success": True,
                "message": f"Booked with {doctor} at {appointment_time}. Appointment confirmed, email notifications pending.",
            }

    async def _send_confirmation_emails(self) -> dict:
        recipients = [
            "dishankjhaveri@gmail.com",
            # "djhaveri@umass.edu",
            "jeff@assorthealth.com",
            "connor@assorthealth.com",
            "cole@assorthealth.com",
            "jciminelli@assorthealth.com",
            "akumar@assorthealth.com",
            "riley@assorthealth.com",
        ]

        if sendgrid is None or Mail is None:
            logger.info("SendGrid library not available; skipping email send.")
            return {"emails_sent": False, "reason": "SendGrid library not available"}

        api_key = os.getenv("SENDGRID_API_KEY")
        if not api_key:
            logger.warning("SENDGRID_API_KEY not set; skipping email send.")
            return {"emails_sent": False, "reason": "SENDGRID_API_KEY not set"}

        sg = sendgrid.SendGridAPIClient(api_key=api_key)
        successful_sends = 0
        total_recipients = len(recipients)

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
                # Consider 2xx status codes as successful
                if status and 200 <= status < 300:
                    successful_sends += 1
            except Exception as e:
                logger.exception("Failed to send email to %s: %s", recipient, e)

        # Return success if at least one email was sent successfully
        emails_sent = successful_sends > 0
        return {
            "emails_sent": emails_sent,
            "successful_sends": successful_sends,
            "total_recipients": total_recipients,
            "success_rate": successful_sends / total_recipients if total_recipients > 0 else 0
        }


