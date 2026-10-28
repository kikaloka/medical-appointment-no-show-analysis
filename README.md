# Appointment attendance analysis

Descriptive analysis of missed medical appointments in Brazil.

- 110,521 appointments analyzed; 20.19% were missed.
- Missed rates were 4.65% for same-day bookings and 33.00% for bookings 31+ days ahead.
- Lead-time-standardized missed rates were 26.88% with SMS recorded and 30.85% without. These observational comparisons do not establish causation.

[Notebook](notebooks/appointment_attendance.ipynb) · [HTML](results/notebook-report.html) · [Results](results/summary.json) · [Data dictionary](docs/data-dictionary.md)

Run from the project directory in a Python 3.13 virtual environment:

```text
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_notebook.py
python scripts/validate.py
```

Data are downloaded locally and checksum-verified. Raw records are excluded because redistribution rights were not established. [Sources and methods provenance](docs/sources.md).

