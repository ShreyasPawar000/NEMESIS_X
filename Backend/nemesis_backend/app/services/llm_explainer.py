import os
import json
from google import genai
from app.services.evidence_extractor import extract_evidence

MODEL_NAME = "gemini-3.5-flash-lite"


def generate_explanation(alert, frames):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("Gemini API key not configured.")
        return None

    evidence = extract_evidence(alert, frames)

    prompt = f"""
You are NEMESIS-X, an intelligent video investigation AI.

Generate a professional, detailed INCIDENT REPORT using
ONLY the supplied tracking evidence.

REPORT FORMAT:

# NEMESIS-X INCIDENT REPORT

## 1. Incident Summary
Summarize the detected anomalous pattern.

## 2. Incident Details
Include video ID, anomaly score, start time and end time.
Clarify that the score is not a probability.

## 3. Object and Track Analysis
Describe observed object classes and involved track IDs.

## 4. Temporal and Movement Analysis
Explain observed movement, displacement, direction
and changes over time where evidence is available.

## 5. Evidence Analysis
Mention supporting frame IDs and timestamps.

## 6. AI Assessment
Explain what the evidence suggests.
Clearly distinguish observations from hypotheses.

## 7. Recommended Actions
Suggest reasonable investigation steps.
Do not claim criminal activity or intent.

IMPORTANT:
- Never invent events or activities.
- Never assume suspicious intent.
- Do not invent missing measurements.
- Explicitly mention insufficient evidence.
- Use professional investigation-report language.
- Make the report detailed but concise.

TRACKING EVIDENCE:
{json.dumps(evidence, indent=2)}

ALERT INFORMATION:
{json.dumps(alert, indent=2, default=str)}

Return the complete report in Markdown.
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text.strip() if response.text else None

    except Exception as error:
        print("Gemini report generation failed:", str(error))
        return None