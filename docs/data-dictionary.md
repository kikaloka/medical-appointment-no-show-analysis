# Data dictionary

Exact CSV headers are retained below. Definitions use source descriptions and field labels; unresolved coding is explicitly qualified. One row is an appointment. No example patient rows are published.

| Source field | Meaning and handling |
|---|---|
| PatientId | Patient identifier, read as text; locally groups repeated appointments. Never exported. |
| AppointmentID | Appointment identifier, read as text and checked for duplicates. Never exported. |
| Gender | Recorded F/M category; does not establish gender identity or a complete taxonomy. Not analyzed here. |
| ScheduledDay | When the appointment was booked, per source description; parsed as timestamp. |
| AppointmentDay | Recorded appointment date; midnight/date-like in this snapshot. Exact local-time semantics unverified. |
| Age | Recorded years; zero retained, negative age excluded, high ages retained with sensitivity check. |
| Neighbourhood | Hospital location/neighborhood, per source description. No small-area results published. |
| Scholarship | Bolsa Família welfare-program indicator, per source description; not an academic scholarship. Not analyzed. |
| Hipertension | Source's misspelled hypertension indicator; 0/1 observed. Diagnostic definition unverified; not analyzed. |
| Diabetes | Recorded condition indicator; 0/1 observed. Diagnostic definition unverified; not analyzed. |
| Alcoholism | Source-labeled condition indicator; 0/1 observed. Definition unverified; not analyzed. |
| Handcap | Disability-related source code; values 0–4, not assumed binary. Count/severity meaning unresolved; not analyzed. |
| SMS_received | Recorded SMS status, 0/1. Delivery, timing, message content and assignment are not verified. |
| No-show | **Yes means missed; No means attended.** Locally encoded as missed = 1/0. |

Derived fields: lead_days is the difference between normalized appointment and booking calendar dates; lead_group/age_group are documented descriptive bands; missed is the outcome. Identifiers and patient-level derived values stay in memory.

For your first learning exercise, inspect shape, types and missingness locally, explain identifier versus measurement, and propose a question with an outcome and denominator before plotting.

