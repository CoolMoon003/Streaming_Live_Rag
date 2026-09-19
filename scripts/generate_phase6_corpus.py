from pathlib import Path
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "phase6_enterprise"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


DOCUMENTS = {
    "travel_policy.md": """# Travel Policy

## 1. Domestic Travel
Employees travelling within the country may claim eligible transportation and accommodation expenses when the trip is approved for business purposes. Domestic travel normally requires manager approval before booking.

## 2. International Travel
International business travel requires prior approval from the employee's department manager and the finance travel desk. Employees must obtain the required travel authorization before purchasing international tickets.

## 3. Air Travel
Employees should use economy-class airfare for standard business travel. Premium cabin travel requires documented exceptional circumstances and additional approval from a department head.

## 4. Hotel Accommodation
Business travellers may claim reasonable hotel accommodation supported by an itemized hotel invoice. Accommodation above the standard company limit requires advance approval.

## 5. Late Bookings
Employees should book flights and hotels at least seven days before departure. A late booking should include a business justification explaining why advance booking was not possible.

## 6. Travel Changes and Cancellations
Cancellation charges may be reimbursed when the cancellation was required for business reasons or caused by an approved schedule change. Personal cancellation charges are normally not reimbursable.
""",

    "expense_reimbursement_policy.md": """# Expense and Reimbursement Policy

## 1. Eligible Expenses
Employees may claim reasonable business expenses including transportation, accommodation, meals, and approved business supplies. Expenses must be incurred for legitimate company business.

## 2. International Expenses
International business expenses must include appropriate supporting documentation. Hotel claims should include the hotel invoice and proof of payment. Foreign-currency claims must identify the transaction currency.

## 3. Submission Deadline
Expense claims should normally be submitted within thirty days of the expense date. Late submissions require an explanation and may require additional approval.

## 4. Receipts
Receipts or equivalent proof of purchase are required for reimbursable expenses above the applicable receipt threshold. Missing receipts should be accompanied by a declaration explaining the circumstances.

## 5. Meals
Business meal reimbursement requires the date, location, business purpose, and names or roles of attendees when applicable. Personal meals without a business purpose are not reimbursable.

## 6. Transportation
Taxi, public transport, mileage, and other approved transportation expenses may be reimbursed when connected to authorized business activity. Claims should identify the trip date and destination.

## 7. Reimbursement Processing
Approved expense claims are processed through the finance reimbursement workflow. Employees should provide accurate bank and employee information to avoid payment delays.
""",

    "workshop_event_policy.md": """# Workshop and Event Policy

## 1. Workshop Approval
Internal workshops require approval from the responsible department before venue or service commitments are made. External events may require additional procurement review.

## 2. Venue Cancellation
Workshop venue cancellation terms depend on the supplier agreement. Cancellation charges may be accepted when the event was approved and the cancellation was caused by a documented business requirement.

## 3. Catering
Catering arrangements should reflect the expected attendance and event duration. Approved catering may include meals, refreshments, and dietary alternatives for participants.

## 4. Vendor Selection
Event organizers should obtain quotations when required by procurement rules. Vendor selection should consider cost, service requirements, and contractual terms.

## 5. Participant Records
The event organizer should maintain the participant list, event agenda, venue information, and relevant supplier records.

## 6. Event Expenses
Approved workshop expenses may include venue rental, catering, equipment, printing, and other directly related business costs.
""",

    "leave_policy.md": """# Leave Policy

## 1. Annual Leave
Employees may request annual leave through the employee leave system. Leave should normally be requested in advance so that teams can plan staffing.

## 2. Sick Leave
Employees who are unable to work due to illness should notify their manager according to departmental procedures. Supporting documentation may be required under applicable company rules.

## 3. Emergency Leave
Emergency leave may be requested when unexpected personal circumstances prevent an employee from attending work. The employee should notify the manager as soon as reasonably possible.

## 4. Leave Approval
Leave requests are reviewed by the employee's reporting manager. Extended leave may require additional departmental approval.

## 5. Leave Records
Employees are responsible for checking their leave balances and ensuring submitted leave dates are accurate.
""",

    "remote_work_policy.md": """# Remote Work Policy

## 1. Remote Work Eligibility
Eligible employees may work remotely when their role and team responsibilities permit remote work. Eligibility is subject to company and department requirements.

## 2. Remote Work Approval
Regular remote work arrangements require manager approval. Temporary remote work may be approved for specific business or personal circumstances.

## 3. Working Hours
Employees working remotely must remain available during agreed working hours and attend required meetings.

## 4. Equipment
Company equipment issued for remote work remains company property. Employees are responsible for reasonable care and secure handling of equipment.

## 5. Information Security
Confidential company information must only be accessed through approved systems and secure network connections while working remotely.
""",

    "procurement_policy.md": """# Procurement Policy

## 1. Purchase Requests
Employees must submit a purchase request before committing company funds for goods or services unless an approved exception applies.

## 2. Approval Thresholds
Purchases above the applicable departmental threshold require additional approval. The approval level depends on the total expected purchase value.

## 3. Vendor Quotations
Procurement may require multiple vendor quotations for competitive purchases. The number of quotations depends on the purchase category and value.

## 4. Purchase Orders
A purchase order should be issued before a supplier begins work when the procurement process requires a purchase order.

## 5. Emergency Procurement
Emergency procurement may be used when an urgent business requirement prevents the normal purchasing timeline. The reason for the emergency should be documented.

## 6. Invoice Processing
Supplier invoices should reference the applicable purchase order or approved procurement request. Finance may reject invoices that lack required supporting information.
""",

    "it_asset_policy.md": """# IT Asset Policy

## 1. Asset Assignment
Company laptops, monitors, phones, and other IT equipment are assigned to employees for authorized business use.

## 2. Asset Registration
Issued equipment must be recorded in the company's asset management system with the appropriate asset identifier.

## 3. Acceptable Use
Company IT assets should primarily be used for legitimate business purposes and must comply with information security requirements.

## 4. Loss or Theft
Lost or stolen company equipment must be reported promptly to the IT service desk and the employee's manager.

## 5. Asset Return
Employees must return assigned equipment when requested, when transferring roles where required, or when leaving the company.

## 6. Damaged Equipment
Accidental damage should be reported to IT with a description of the incident so that repair or replacement can be assessed.
""",

    "information_security_policy.md": """# Information Security Policy

## 1. Passwords
Employees must use strong passwords and must not share authentication credentials with other people.

## 2. Multi-Factor Authentication
Systems supporting sensitive business information should use approved multi-factor authentication mechanisms where available.

## 3. Access Control
Employees should receive only the access required for their role. Access rights should be reviewed when responsibilities change.

## 4. Phishing
Suspicious emails, links, attachments, and login requests should be reported through the company's security reporting process.

## 5. Confidential Information
Confidential information must only be shared with authorized recipients and through approved communication channels.

## 6. Security Incidents
Suspected security incidents should be reported promptly to the security or IT response team so that containment and investigation can begin.
""",

    "data_privacy_policy.md": """# Data Privacy Policy

## 1. Personal Data
Personal data must be collected and processed only for legitimate and authorized business purposes.

## 2. Data Minimization
Teams should collect only the personal information necessary for the stated business requirement.

## 3. Data Access
Access to personal data must be limited to employees and systems that require the information for authorized purposes.

## 4. Data Retention
Personal data should not be retained longer than required by the applicable business, legal, or regulatory requirement.

## 5. Data Sharing
Sharing personal data with external parties requires appropriate authorization and contractual or legal safeguards where applicable.

## 6. Privacy Incidents
Suspected unauthorized disclosure or loss of personal data should be reported through the privacy incident process.
""",

    "employee_onboarding_policy.md": """# Employee Onboarding Policy

## 1. Joining Documentation
New employees must provide required joining documentation before or during onboarding according to HR instructions.

## 2. Identity Verification
Identity and employment information may be verified as part of the onboarding process.

## 3. IT Account Creation
IT accounts should be requested through the approved onboarding workflow. Access is provisioned according to the employee's role.

## 4. Equipment Allocation
Eligible employees may receive company equipment such as laptops and access devices after the required onboarding requests are completed.

## 5. Orientation
New employees should complete required orientation activities covering company processes, security, and workplace expectations.

## 6. Probation Records
Probation-related records should be maintained by the appropriate HR and management teams.
""",

    "training_certification_policy.md": """# Training and Certification Policy

## 1. Training Requests
Employees may request business-relevant training through the approved learning process.

## 2. Manager Approval
External training or paid certification programs normally require manager approval before registration.

## 3. Certification Reimbursement
Approved certification fees may be reimbursed when the certification is relevant to the employee's role and the required supporting documents are provided.

## 4. Training Attendance
Employees attending company-sponsored training should complete the required sessions and participation records.

## 5. Learning Records
Completed training and certifications should be recorded in the company's learning management system where applicable.

## 6. Training Cancellation
Cancellation charges for approved training may be reimbursed when the cancellation resulted from a documented business requirement.
""",

    "business_meeting_policy.md": """# Business Meeting Policy

## 1. Meeting Expenses
Reasonable expenses directly related to approved business meetings may be reimbursed according to the expense policy.

## 2. External Attendees
Meetings involving external participants should have a documented business purpose and appropriate organizer approval.

## 3. Business Meals
Business meal expenses should include the meeting date, location, purpose, and relevant participant information.

## 4. Meeting Travel
Travel required for an approved business meeting must follow the applicable travel authorization process.

## 5. Virtual Meetings
Employees should use approved company conferencing tools for meetings involving confidential business information.

## 6. Meeting Records
Important business decisions and agreed actions should be documented by the meeting organizer.
""",

    "relocation_policy.md": """# Relocation Policy

## 1. Relocation Eligibility
Employees may qualify for relocation support when an approved company transfer requires a change of primary work location.

## 2. Relocation Approval
Relocation support requires prior approval from the relevant manager and HR team before eligible expenses are incurred.

## 3. Moving Expenses
Eligible relocation expenses may include approved transportation, packing, and shipment costs within applicable limits.

## 4. Temporary Accommodation
Approved temporary accommodation may be reimbursed for eligible relocation periods subject to company limits.

## 5. Documentation
Relocation claims must include receipts and other supporting documents required by HR or finance.

## 6. Repayment Conditions
Certain relocation benefits may be subject to repayment conditions when an employee leaves the company within an applicable period.
""",

    "office_facilities_policy.md": """# Office Facilities Policy

## 1. Workplace Access
Employees must use their assigned access credentials when entering controlled office areas.

## 2. Meeting Rooms
Meeting rooms should be reserved through the approved booking system and released when no longer required.

## 3. Visitor Access
Visitors should be registered according to office security procedures and may require host confirmation.

## 4. Facilities Requests
Maintenance, furniture, electrical, and workplace service requests should be submitted through the facilities service channel.

## 5. Office Equipment
Shared office equipment should be used responsibly and reported when damaged or unavailable.

## 6. Emergency Procedures
Employees should follow posted emergency procedures and instructions from authorized facilities or safety personnel.
""",

    "incident_reporting_policy.md": """# Incident Reporting Policy

## 1. Reportable Incidents
Employees should report workplace, security, data, safety, or operational incidents through the applicable reporting process.

## 2. Immediate Escalation
Incidents presenting an immediate safety or security risk should be escalated without waiting for the normal reporting cycle.

## 3. Security Incidents
Suspected unauthorized access, credential compromise, malware, or data exposure should be reported to the security team.

## 4. Workplace Incidents
Workplace accidents or safety concerns should be reported to the responsible facilities or safety contact.

## 5. Incident Details
Reports should include the date, time, location, people involved where appropriate, and a concise description of what occurred.

## 6. Follow-up
Relevant teams may investigate incidents, request additional information, and document corrective actions.
""",
}


EVALUATION_QUERIES = [
    {
        "id": "P6-Q001",
        "query": "What approval is required for international business travel?",
        "expected_chunks": ["DOC_TRAVEL_POLICY_C002"],
        "intent": "international travel approval",
    },
    {
        "id": "P6-Q002",
        "query": "What documents are needed for an international hotel reimbursement claim?",
        "expected_chunks": ["DOC_EXPENSE_REIMBURSEMENT_POLICY_C002"],
        "intent": "international reimbursement documentation",
    },
    {
        "id": "P6-Q003",
        "query": "How long do I have to submit an expense claim?",
        "expected_chunks": ["DOC_EXPENSE_REIMBURSEMENT_POLICY_C003"],
        "intent": "expense submission deadline",
    },
    {
        "id": "P6-Q004",
        "query": "What are the cancellation terms for a workshop venue?",
        "expected_chunks": ["DOC_WORKSHOP_EVENT_POLICY_C002"],
        "intent": "workshop cancellation",
    },
    {
        "id": "P6-Q005",
        "query": "What catering arrangements are allowed for workshops?",
        "expected_chunks": ["DOC_WORKSHOP_EVENT_POLICY_C003"],
        "intent": "workshop catering",
    },
    {
        "id": "P6-Q006",
        "query": "What rules apply when an employee travels within the country?",
        "expected_chunks": ["DOC_TRAVEL_POLICY_C001"],
        "intent": "domestic travel",
    },
    {
        "id": "P6-Q007",
        "query": "What approval is required for external training or certification?",
        "expected_chunks": ["DOC_TRAINING_CERTIFICATION_POLICY_C002"],
        "intent": "training approval",
    },
    {
        "id": "P6-Q008",
        "query": "What proof is needed to claim certification reimbursement?",
        "expected_chunks": ["DOC_TRAINING_CERTIFICATION_POLICY_C003"],
        "intent": "certification reimbursement",
    },
    {
        "id": "P6-Q009",
        "query": "What should I do if my company laptop is stolen?",
        "expected_chunks": ["DOC_IT_ASSET_POLICY_C004"],
        "intent": "lost or stolen IT asset",
    },
    {
        "id": "P6-Q010",
        "query": "What should employees do about suspicious phishing emails?",
        "expected_chunks": ["DOC_INFORMATION_SECURITY_POLICY_C004"],
        "intent": "phishing",
    },
    {
        "id": "P6-Q011",
        "query": "How should personal data be handled and retained?",
        "expected_chunks": [
            "DOC_DATA_PRIVACY_POLICY_C001",
            "DOC_DATA_PRIVACY_POLICY_C004",
        ],
        "intent": "personal data retention",
    },
    {
        "id": "P6-Q012",
        "query": "What is required before buying goods for the company?",
        "expected_chunks": ["DOC_PROCUREMENT_POLICY_C001"],
        "intent": "purchase request",
    },
    {
        "id": "P6-Q013",
        "query": "Who approves regular remote work?",
        "expected_chunks": ["DOC_REMOTE_WORK_POLICY_C002"],
        "intent": "remote work approval",
    },
    {
        "id": "P6-Q014",
        "query": "What documents are required for relocation claims?",
        "expected_chunks": ["DOC_RELOCATION_POLICY_C005"],
        "intent": "relocation documentation",
    },
    {
        "id": "P6-Q015",
        "query": "How should a workplace incident be reported?",
        "expected_chunks": ["DOC_INCIDENT_REPORTING_POLICY_C001"],
        "intent": "incident reporting",
    },
    {
        "id": "P6-Q016",
        "query": "What approval is needed for international travel and what documentation is needed for reimbursement?",
        "expected_chunks": [
            "DOC_TRAVEL_POLICY_C002",
            "DOC_EXPENSE_REIMBURSEMENT_POLICY_C002",
        ],
        "intents": [
            "international travel approval",
            "international reimbursement documentation",
        ],
    },
    {
        "id": "P6-Q017",
        "query": "What approval is needed for a workshop and what catering arrangements are allowed?",
        "expected_chunks": [
            "DOC_WORKSHOP_EVENT_POLICY_C001",
            "DOC_WORKSHOP_EVENT_POLICY_C003",
        ],
        "intents": [
            "workshop approval",
            "workshop catering",
        ],
    },
    {
        "id": "P6-Q018",
        "query": "What happens if company equipment is lost and how should the incident be reported?",
        "expected_chunks": [
            "DOC_IT_ASSET_POLICY_C004",
            "DOC_INCIDENT_REPORTING_POLICY_C006",
        ],
        "intents": [
            "lost company equipment",
            "incident follow-up",
        ],
    },
]


def make_chunk_id(filename: str, section_number: int) -> str:
    stem = Path(filename).stem.upper()
    stem = re.sub(r"[^A-Z0-9]+", "_", stem).strip("_")
    return f"DOC_{stem}_C{section_number:03d}"


def build_processed_chunks():
    chunks = []

    for filename, document in DOCUMENTS.items():
        lines = document.splitlines()

        current_title = ""
        current_section = None
        current_body = []

        sections = []

        for line in lines:
            stripped = line.strip()

            if stripped.startswith("# ") and not stripped.startswith("## "):
                current_title = stripped[2:].strip()

            elif stripped.startswith("## "):
                if current_section is not None:
                    sections.append((current_section, current_body))

                current_section = stripped[3:].strip()
                current_body = []

            elif stripped:
                current_body.append(stripped)

        if current_section is not None:
            sections.append((current_section, current_body))

        for index, (section, body_lines) in enumerate(sections, start=1):
            text = " ".join(body_lines).strip()

            chunks.append(
                {
                    "chunk_id": make_chunk_id(filename, index),
                    "document_id": f"DOC_{Path(filename).stem.upper()}",
                    "source": filename,
                    "section": section,
                    "text": text,
                }
            )

    return chunks


def main():
    print("=" * 60)
    print("PHASE 6 CORPUS GENERATOR")
    print("=" * 60)

    # Write Markdown source documents.
    for filename, content in DOCUMENTS.items():
        path = RAW_DIR / filename
        path.write_text(content.strip() + "\n", encoding="utf-8")

    chunks = build_processed_chunks()

    phase6_chunks_path = PROCESSED_DIR / "phase6_chunks.jsonl"

    with phase6_chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    eval_path = PROCESSED_DIR / "phase6_eval_queries.json"

    eval_path.write_text(
        json.dumps(EVALUATION_QUERIES, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDocuments created : {len(DOCUMENTS)}")
    print(f"Chunks created    : {len(chunks)}")
    print(f"Evaluation queries: {len(EVALUATION_QUERIES)}")

    print("\nRaw corpus:")
    print(f"  {RAW_DIR}")

    print("\nProcessed corpus:")
    print(f"  {phase6_chunks_path}")

    print("\nEvaluation queries:")
    print(f"  {eval_path}")

    print("\nFirst 10 chunk IDs:")
    for chunk in chunks[:10]:
        print(f"  {chunk['chunk_id']}")

    print("\nPHASE 6 CORPUS GENERATION: PASS")


if __name__ == "__main__":
    main()