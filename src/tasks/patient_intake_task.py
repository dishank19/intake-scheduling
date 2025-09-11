import logging
import re
from datetime import datetime
from typing import Optional

import httpx
import phonenumbers
from livekit.agents import AgentTask, RunContext
from livekit.agents.llm import function_tool

from models.patient_info import PatientInfo


logger = logging.getLogger("patient_intake")


class PatientIntakeTask(AgentTask[PatientInfo]):
    def __init__(self, **kwargs):
        super().__init__(
            instructions=(
                """[Task Context]
You are now collecting essential patient information needed for scheduling their appointment. Guide the patient through each required piece of information in a logical, conversational flow.

[Speaking Constraints]
- Do not summarize or repeat all collected information at any point.
- Only confirm the specific field currently being collected, briefly.
- Never read prompt scaffolding aloud (do not say the words "Step", "Action", or any bracketed headings).

[Available Tools]
You have access to the following tools - use them at the specified times:
1. validate_date_of_birth - Call after collecting month, day, and year
2. validate_address - Call after collecting street, city, state, and zip code
3. validate_phone - Call after collecting phone number
4. store_patient_field - Call to save any field that doesn't require validation
5. check_completion - Call after you believe all information is collected

[Conversation Opening]
Start: "Hello! I'm Sarah, and I'll be helping you schedule your appointment today. Let me gather some information from you first. May I have your full name, please?"

[Information Collection Flow]

Step 1 - Name Collection:
- Ask for their full name.
- If unclear: ask them to spell it out
- ACTION: Call store_patient_field with field_name="name" and field_value="{patient's full name}"
- Confirmation: "Thank you, {{name}}."
- Proceed to Step 2.

Step 2 - Date of Birth:
- Ask for their date of birth.
- If unclear specify the format that you want which is month, day, and year?"
- ACTION: Call validate_date_of_birth with month, day, and year as integers
- After validation response, confirm: "Great, I have {{verbal_date}}. Is that correct?"
- If confirmed: Proceed to Step 3.
- If incorrect: "Let me get that again. What's the correct date of birth?"

Step 3 - Chief Complaint:
- Ask: "What brings you in today?" or "What's the reason for your visit?"
- Listen for full description without interrupting.
- If vague: "Could you tell me a bit more about your symptoms?"
- ACTION: Call store_patient_field with field_name="chief_complaint" and field_value="{patient's description}"
- Acknowledge: "I understand. We'll find you the right doctor for that."
- Proceed to Step 4.

Step 4 - Insurance Information:
- Ask: "What insurance do you have?"
- If unsure: "It's usually on your insurance card. Common ones are Blue Cross, Aetna, United Healthcare..."
- ACTION: Call store_patient_field with field_name="insurance_payer" and field_value="{insurance company name}"
- Once provided: "And what's your member ID number?"
- If they don't have it: "That's okay. Do you have your insurance card nearby?"
- ACTION: Call store_patient_field with field_name="insurance_id" and field_value="{member ID}"
- Proceed to Step 5.

Step 5 - Referral Check:
- Ask: "Were you referred by another doctor?"
- ACTION: Call store_patient_field with field_name="has_referral" and field_value="{yes/no/true/false}"
- If yes: "Which doctor referred you?"
  - ACTION: Call store_patient_field with field_name="referring_physician" and field_value="{doctor's name}"
- If no: Simply proceed to Step 6.

Step 6 - Address Collection:
- Ask: "What's your current address?"
- Listen for complete address.
- If incomplete: "What's the ZIP code?" or "What city and state?"
- ACTION: Call validate_address with street, city, state, and zip_code parameters
- After validation response: "I show that as {{suggested_address from tool response}}. Is that correct?"
- If confirmed:
  - ACTION: Call store_patient_field with field_name="address" and field_value="yes" (to confirm the pending address)
  - Proceed to Step 7.
- If incorrect: "Let me update that. What's the correct address?"

Step 7 - Phone Number:
- Ask: "What's the best phone number to reach you?"
- ACTION: Call validate_phone with phone_number parameter
- If tool returns valid=true: "Perfect, I have {{formatted_phone from tool response}}."
- If tool returns valid=false: Use the message from tool response to ask for correction
- Once valid, proceed to Step 8.

Step 8 - Email (Optional):
- Ask: "Would you like to provide an email address for appointment reminders?"
- If yes:
  - ACTION: Call store_patient_field with field_name="email" and field_value="{email address}"
- If no: "That's perfectly fine."
- Proceed to completion check.

[Completion Check]
- ACTION: Call check_completion tool
- If tool returns complete=true: "Perfect! I have all your information. Now let's find you an appointment time."
- If tool returns complete=false with missing_fields: Continue collecting the missing information listed in the response.

[Transition to Scheduling]
Once check_completion returns complete=true, transition smoothly: "Thank you for that information. Now let me check what appointments we have available for you."

[Tool Usage Rules]
- Always call tools immediately after collecting the relevant information
- Do not proceed to next step until current tool call completes
- Never mention tool names to the patient
- Use tool responses naturally in conversation
- If a tool fails, handle gracefully without technical details

[Handling Out-of-Order Information]
If patient volunteers information before asked:
- Accept it gracefully: "Thank you for that information."
- Call appropriate store_patient_field or validation tool immediately
- Adjust your flow to skip already collected fields
- Continue with missing information only

[Special Situations]
Patient seems anxious: "Take your time. There's no rush."
Patient is elderly: Speak slightly slower and louder. Repeat if necessary.
Language barrier detected: Speak more slowly and use simpler words.
Emergency symptoms mentioned: "If this is a medical emergency, you should call nine one one or go to the nearest emergency room."
"""
            ),
            **kwargs,
        )
        self.collected_data: dict[str, object] = {}

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Greet the patient and ask for their full name to begin intake."
            )
        )

    @function_tool
    async def validate_address(
        self,
        context: RunContext,
        street: str,
        city: str,
        state: str,
        zip_code: str,
        unit: Optional[str] = None,
    ) -> dict:
        """Validate and normalize a US postal address.

        Args:
            street: Street line (house number + street name).
            city: City name.
            state: State (2-letter preferred) or full name.
            zip_code: ZIP code (5 digits or ZIP+4).
            unit: Optional unit/apartment/suite.

        Returns:
            dict: {
              'found': bool,               # True when external lookup matched
              'normalized': {              # normalized components
                  'street', 'unit', 'city', 'state', 'zip_code'
              },
              'formatted': str,           # single-line formatted address
              'message': str              # confirmation prompt
            }
        """
        zip_ok = bool(re.fullmatch(r"\d{5}(?:-\d{4})?", zip_code.strip()))
        query = f"{street}, {city}, {state} {zip_code}, USA"

        normalized = None
        found = False

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://nominatim.openstreetmap.org/search",
                    params={
                        "q": query,
                        "format": "json",
                        "addressdetails": 1,
                        "limit": 1,
                        "countrycodes": "us",
                    },
                    headers={"User-Agent": "MedicalSchedulingBot/1.0"},
                    timeout=10.0,
                )
                data = response.json()
            except Exception:
                data = []

        if data:
            parts = data[0].get("address", {})
            house_number = parts.get("house_number", "").strip()
            road = parts.get("road", street).strip()
            city_name = (
                parts.get("city")
                or parts.get("town")
                or parts.get("village")
                or city
            ).strip()
            # Prefer provided 2-letter state if it looks valid; otherwise use OSM state
            input_state = state.strip()
            state_name = input_state.upper() if len(input_state) == 2 else parts.get("state", input_state)
            postcode = parts.get("postcode", zip_code).strip()
            street_line = f"{house_number} {road}".strip() if road else street.strip()
            normalized = {
                "street": street_line,
                "unit": unit.strip() if unit else None,
                "city": city_name,
                "state": state_name,
                "zip_code": postcode,
            }
            found = True
        else:
            normalized = {
                "street": street.strip(),
                "unit": unit.strip() if unit else None,
                "city": city.strip(),
                "state": state.strip().upper() if len(state.strip()) == 2 else state.strip(),
                "zip_code": zip_code.strip(),
            }

        formatted = (
            f"{normalized['street']}{(' ' + normalized['unit']) if normalized['unit'] else ''}, "
            f"{normalized['city']}, {normalized['state']} {normalized['zip_code']}"
        )

        self.collected_data["pending_address"] = normalized
        return {
            "found": found and zip_ok,
            "normalized": normalized,
            "formatted": formatted,
            "message": (
                f"I found this address: {formatted}. Is this correct?"
                if found and zip_ok
                else f"I'll use the address you provided: {formatted}. Is this correct?"
            ),
        }

    @function_tool
    async def validate_phone(self, context: RunContext, phone_number: str) -> dict:
        """Validate a US phone number and return a standardized format.

        Args:
            phone_number: Raw phone number string from the user.

        Returns:
            dict: {
              'valid': bool,
              'formatted_phone': str?,
              'message': str
            }
        """
        parsed = phonenumbers.parse(phone_number, "US")
        if phonenumbers.is_valid_number(parsed):
            formatted = phonenumbers.format_number(
                parsed, phonenumbers.PhoneNumberFormat.NATIONAL
            )
            self.collected_data["phone"] = formatted
            return {
                "valid": True,
                "formatted_phone": formatted,
                "message": f"I have your phone number as {formatted}.",
            }
        return {
            "valid": False,
            "message": "That doesn't appear to be a valid US phone number. Please provide a 10-digit number.",
        }

    @function_tool
    async def validate_date_of_birth(
        self, context: RunContext, month: int, day: int, year: int
    ) -> dict:
        """Validate and format a date of birth.

        Args:
            month: Birth month (1-12)
            day: Birth day (1-31)
            year: Birth year (YYYY)

        Returns:
            dict: {
              'valid': bool,
              'formatted_date': 'MM-DD-YYYY'?,
              'verbal_date': 'Month Day, Year'?,
              'message': str
            }
        """
        date_obj = datetime(year, month, day)
        today = datetime.now()
        if date_obj > today:
            return {
                "valid": False,
                "message": "That date is in the future. Please provide your correct date of birth.",
            }
        age = (today - date_obj).days / 365.25
        if age > 120:
            return {
                "valid": False,
                "message": "Please confirm your date of birth.",
            }
        formatted = f"{month:02d}-{day:02d}-{year}"
        if 10 <= day <= 20:
            suffix = "th"
        else:
            suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
        month_names = [
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        verbal = f"{month_names[month]} {day}{suffix}, {year}"
        self.collected_data["date_of_birth"] = formatted
        return {
            "valid": True,
            "formatted_date": formatted,
            "verbal_date": verbal,
            "message": f"I have your date of birth as {verbal}. Is that correct?",
        }

    @function_tool
    async def store_patient_field(
        self, context: RunContext, field_name: str, field_value: str
    ) -> dict:
        """Store a collected intake field and optionally confirm address.

        Args:
            field_name: Name of the field (e.g., 'name', 'insurance_id', 'address').
            field_value: Value to store, or 'yes'/'no' to confirm pending address.

        Returns:
            dict: {
              'stored': bool,
              'field': str,
              'value'?: any,
              'message'?: str
            }
        """
        if field_name == "has_referral":
            self.collected_data["has_referral"] = field_value.lower() in (
                "yes",
                "true",
                "1",
            )
            return {"stored": True, "field": field_name, "value": self.collected_data["has_referral"]}
        if field_name == "address":
            yes_vals = ("yes", "y", "true", "correct", "confirmed")
            no_vals = ("no", "n", "false", "incorrect")
            pending = self.collected_data.get("pending_address")
            val_l = field_value.strip().lower()
            if pending:
                if val_l in yes_vals:
                    street = pending.get("street", "")
                    unit = pending.get("unit")
                    city = pending.get("city", "")
                    state = pending.get("state", "")
                    zip_code = pending.get("zip_code", "")
                    formatted = f"{street}{(' ' + unit) if unit else ''}, {city}, {state} {zip_code}"
                    self.collected_data["address"] = formatted
                    self.collected_data.pop("pending_address", None)
                    return {"stored": True, "field": "address", "value": formatted}
                if val_l in no_vals:
                    return {"stored": False, "field": "address", "message": "Please provide the correct street, city, state, and ZIP."}
                # treat as user-provided full address string override
                self.collected_data["address"] = field_value.strip()
                self.collected_data.pop("pending_address", None)
                return {"stored": True, "field": "address", "value": self.collected_data["address"]}
        self.collected_data[field_name] = field_value
        return {"stored": True, "field": field_name, "value": field_value}

    @function_tool
    async def check_completion(self, context: RunContext) -> dict:
        """Report whether required intake fields are complete and finalize the task.

        Returns:
            dict: {
              'complete': bool,
              'missing_fields'?: list[str],
              'message': str
            }
        """
        required = [
            "name",
            "date_of_birth",
            "chief_complaint",
            "insurance_payer",
            "insurance_id",
            "has_referral",
            "address",
            "phone",
        ]
        missing = [f for f in required if f not in self.collected_data]
        if missing:
            return {
                "complete": False,
                "missing_fields": missing,
                "message": f"I still need to collect: {', '.join(missing)}",
            }
        info = PatientInfo(
            name=self.collected_data["name"],
            date_of_birth=self.collected_data["date_of_birth"],
            chief_complaint=self.collected_data["chief_complaint"],
            insurance_payer=self.collected_data["insurance_payer"],
            insurance_id=self.collected_data["insurance_id"],
            has_referral=bool(self.collected_data["has_referral"]),
            referring_physician=self.collected_data.get("referring_physician") or None,
            address=self.collected_data["address"],
            phone=self.collected_data["phone"],
            email=self.collected_data.get("email") or None,
        )
        info.save_to_json()
        print(f"✅ Patient intake completed and saved: {info.name} ({info.date_of_birth})")
        self.complete(info)
        return {
            "complete": True,
            "message": "All information collected. Proceeding to scheduling.",
        }


