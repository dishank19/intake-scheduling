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
You are now collecting essential patient information needed for scheduling their appointment. Guide the patient through each required piece of information in a natural, conversational flow. Confirm every item with the patient before saving it.

[Speaking Constraints]
- Do not summarize or read scaffolding. Never say the words "Step", "Action", or any bracketed headings.
- Confirm each field with the patient before saving it.

[Available Tools]
You have access to the following tools - use them at the specified times:
1. validate_date_of_birth - Call after collecting month, day, and year
2. validate_address - Call after collecting street, city, state, and zip code
3. validate_phone - Call after collecting phone number
4. store_patient_field - Persist a field value after verbal confirmation
5. check_completion - Call after you believe all information is collected

[Conversation Opening]
Start with a brief, friendly exchange. Introduce yourself as Sarah, a virtual intake assistant with Bay Area Health. Ask how you can help today. After a short exchange, begin intake by requesting the patient's full name.

[Information Collection Flow]

Step 1 - Name Collection:
- Collect the full name; request spelling if unclear.
- ACTION: Call store_patient_field with field_name=name and field_value
- Confirm the captured name with the patient in natural language. Dont make conversations robotic
- ACTION: Call confirm_field with field_name=name and confirmed=true/false
- Proceed to Step 2.

Step 2 - Date of Birth:
- Collect date of birth; specify the format (month, day, year) when needed.
- Use a brief filler before validation, then ACTION: Call validate_date_of_birth with month, day, and year as integers.
- Confirm the verbalized date with the patient.
- If confirmed:
  - ACTION: Call store_patient_field with field_name=date_of_birth and field_value=formatted_date
  - ACTION: Call confirm_field with field_name=date_of_birth and confirmed=true
- If not confirmed: ask again and revalidate.

Step 3 - Chief Complaint:
- Elicit the reason for the visit and listen fully.
- If vague, request a brief elaboration.
- ACTION: Call store_patient_field with field_name=chief_complaint and field_value
- Confirm the captured reason with the patient (concise restatement).
- ACTION: Call confirm_field with field_name=chief_complaint and confirmed=true/false
- Provide a brief acknowledgment and proceed.

Step 4 - Insurance Information:
- Collect insurance payer; provide gentle examples if needed.
- ACTION: Call store_patient_field with field_name=insurance_payer and field_value
- Confirm the payer with the patient.
- ACTION: Call confirm_field with field_name=insurance_payer and confirmed=true/false
- Collect member ID (or note if unavailable); suggest referencing the insurance card when needed.
- ACTION: Call store_patient_field with field_name=insurance_id and field_value
- Confirm the member ID with the patient.
- ACTION: Call confirm_field with field_name=insurance_id and confirmed=true/false
- Proceed to Step 5.

Step 5 - Referral Check:
- Determine referral status.
- ACTION: Call store_patient_field with field_name=has_referral and field_value
- Confirm referral status with the patient.
- ACTION: Call confirm_field with field_name=has_referral and confirmed=true/false
- If yes, capture the referring physician name.
  - ACTION: Call store_patient_field with field_name=referring_physician and field_value
  - Confirm the physician name with the patient.
  - ACTION: Call confirm_field with field_name=referring_physician and confirmed=true/false
- If no: proceed.

Step 6 - Address Collection:
- Collect the full address; if incomplete, request missing parts (street, ZIP, city/state).
- Use a brief filler before validation. ACTION: Call validate_address with street, city, state, and zip_code parameters.
- Read back the normalized address for confirmation.
- If confirmed:
  - ACTION: Call store_patient_field with field_name=address and field_value=yes (confirms pending address)
  - Proceed to Step 7.
- If not confirmed: request corrections and revalidate.

Step 7 - Phone Number:
- Collect the best contact number.
- Use a brief filler before validation. ACTION: Call validate_phone with phone_number parameter.
- If valid, confirm the formatted number with the patient.
- If valid: ACTION: Call store_patient_field with field_name=phone and field_value=formatted_phone; ACTION: Call confirm_field with field_name=phone and confirmed=true
- If not valid: request correction and revalidate.
- Proceed to Step 8.

Step 8 - Email (Optional):
- Offer to capture an email address for reminders.
- If provided:
  - ACTION: Call store_patient_field with field_name=email and field_value
  - Confirm the email with the patient.
  - ACTION: Call confirm_field with field_name=email and confirmed=true/false
- If declined: proceed.
- Proceed to completion check.

[Completion Check]
- ACTION: Call check_completion tool
- If tool returns complete=true: "Perfect! I have all your information. Now let's find you an appointment time."
- If tool returns complete=false with missing_fields: Continue collecting the missing information listed in the response.

[Transition to Scheduling]
Once check_completion returns complete=true, transition smoothly: "Thank you for that information. Now let me check what appointments we have available for you."

[Tool Usage Rules]
- Before calling a tool, speak a brief filler like "One moment while I check that."
- Always call tools immediately after collecting the relevant information
- Stage values first (store_patient_field), confirm verbally with the patient, then commit (confirm_field)
- Do not proceed to next step until current tool call completes
- Never mention tool names to the patient
- Use tool responses naturally in conversation

[Handling Out-of-Order Information]
If patient volunteers information before asked:
- Accept it gracefully: "Thank you for that information."
- Call appropriate store_patient_field or validation tool immediately, then confirm and use confirm_field
- Adjust your flow to skip already collected fields
- Continue with missing information only

[Pronunciation Guidelines]
Dates: Speak out fully (e.g., "January twenty-fourth" not "one twenty-four").
Times: Use conversational format (e.g., "ten thirty ay em" for 10:30 AM, "two pee em" for 2:00 PM).
Numbers: For phone numbers, speak digit by digit with brief pauses: "five five five, pause, one two three four".
Addresses: Spell out numbered streets below 10 (e.g., "Third Street" not "3rd Street").
Medical terms: Use simple language when possible. If medical terms are necessary, speak them slowly and clearly.

[Response Handling]
Listen for the complete patient response before proceeding. Use context awareness to understand partial or informal responses. Accept variations in how patients express the same information. If a response seems off-topic, gently redirect to the current question.

[Error Recovery]
If you don't understand: "I'm sorry, could you repeat that please?"
If there's background noise: "I'm having a little trouble hearing you. Could you speak up a bit?"
If the patient seems confused: "Let me clarify what I'm asking..."

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
        """Validate and normalize a US postal address and return a suggested address.

        Args:
            street: Street line (house number + street name).
            city: City name.
            state: State (2-letter preferred) or full name.
            zip_code: ZIP code (5 digits or ZIP+4).
            unit: Optional unit/apartment/suite.

        Returns:
            dict: {
              'found': bool,                  # True when external lookup matched
              'normalized': {                 # normalized components
                  'street', 'unit', 'city', 'state', 'zip_code'
              },
              'suggested_address': str,       # suggested single-line address to read back
              'message': str                   # confirmation prompt using suggested address
            }
        """
        zip_ok = bool(re.fullmatch(r"\d{5}(?:-\d{4})?", zip_code.strip()))
        queries = [
            f"{street}, {city}, {state} {zip_code}, USA",
            f"{street}, {city}, {state}, USA",
            f"{city}, {state} {zip_code}, USA",
        ]

        result_parts = None
        async with httpx.AsyncClient() as client:
            for q in queries:
                try:
                    response = await client.get(
                        "https://nominatim.openstreetmap.org/search",
                        params={
                            "q": q,
                            "format": "json",
                            "addressdetails": 1,
                            "limit": 1,
                            "countrycodes": "us",
                        },
                        headers={"User-Agent": "MedicalSchedulingBot/1.0"},
                        timeout=4.0,
                    )
                    data = response.json()
                except Exception:
                    data = []
                if data:
                    result_parts = data[0].get("address", {})
                    break

        if result_parts:
            house_number = result_parts.get("house_number", "").strip()
            road = result_parts.get("road", street).strip()
            city_name = (
                result_parts.get("city")
                or result_parts.get("town")
                or result_parts.get("village")
                or city
            ).strip()
            input_state = state.strip()
            state_name = input_state.upper() if len(input_state) == 2 else result_parts.get("state", input_state)
            postcode = result_parts.get("postcode", zip_code).strip()
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
            found = False

        suggested = (
            f"{normalized['street']}{(' ' + normalized['unit']) if normalized['unit'] else ''}, "
            f"{normalized['city']}, {normalized['state']} {normalized['zip_code']}"
        )

        self.collected_data["pending_address"] = normalized
        return {
            "found": found and zip_ok,
            "normalized": normalized,
            "suggested_address": suggested,
            "message": (
                f"Please confirm the address: {suggested}. Is this correct?"
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
            return {
                "valid": True,
                "formatted_phone": formatted,
                "message": f"I have your phone number as {formatted}. Is that correct?",
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


