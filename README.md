# Office Cybersecurity Anomaly Detection

A small synthetic anomaly-detection project for office cybersecurity events.

## Quick start

From the project folder:

```bash
./venv/Scripts/python build_dataset.py
./venv/Scripts/python main.py
```

Optional checks:

```bash
./venv/Scripts/python validate.py
```

Run the dashboard:

```bash
./venv/Scripts/python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## What it does

This project generates office activity data and detects suspicious behavior such as:

- brute force attacks
- credential stuffing
- lateral movement
- impossible travel
- device spoofing
- privilege escalation
- low-and-slow exfiltration

## Project structure

- `build_dataset.py` - builds the synthetic office event dataset
- `main.py` - full training and evaluation pipeline
- `validate.py` - data and label validation utilities
- `app.py` - lightweight Flask dashboard
- `generator/` - synthetic dataset and attack generation logic
- `detector/` - feature engineering, scaling, sequence generation, and LSTM model
- `data/` - train/test event files and baselines

## Required environment

Use the project virtual environment instead of the system Python:

```bash
./venv/Scripts/python
```

## Current status

The core model pipeline is verified and runs successfully end-to-end.
The dashboard is also live and reads the generated dataset for a security summary view.

## Notes

- Use the local venv for all commands.
- Do not use system Python for this project.
- The ML pipeline is the main verified workflow; the web dashboard is layered on top of it.
