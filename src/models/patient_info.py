from dataclasses import dataclass, asdict
from typing import Optional
import json
from datetime import datetime


@dataclass
class PatientInfo:
    name: str
    date_of_birth: str
    chief_complaint: str

    insurance_payer: str
    insurance_id: str

    has_referral: bool
    referring_physician: Optional[str] = None

    address: str = ""
    phone: str = ""
    email: Optional[str] = None

    appointment_time: Optional[str] = None
    appointment_doctor: Optional[str] = None

    def save_to_json(self, filepath: str = "patient_records.json") -> None:
        existing_data = []
        try:
            with open(filepath, "r") as f:
                content = f.read()
                if content:
                    existing_data = json.loads(content)
        except FileNotFoundError:
            existing_data = []

        record = asdict(self)
        record["timestamp"] = datetime.now().isoformat()
        existing_data.append(record)

        with open(filepath, "w") as f:
            json.dump(existing_data, f, indent=2)


