import os
import json
import time
from datetime import date
from google import genai
from google.genai import types

CURRENT_DATE = date.today()
CURRENT_YEAR = CURRENT_DATE.year
CURRENT_MONTH = CURRENT_DATE.strftime("%B %Y")

def extract_features_from_cv(pdf_path: str, jd_path: str) -> dict:
    """Extract the five resume features from a CV PDF using Google Gemini."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Set it before calling extract_features_from_cv().")

    client = genai.Client(api_key=api_key)

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    response_schema = {
        "type": "object",
        "properties": {
            "years_experience":   {"type": "integer"},
            "skills_match_score": {"type": "number"},
            "education_level": {
                "type": "string",
                "enum": ["High School", "Bachelors", "Masters", "PhD"]
            },
            "project_count":  {"type": "integer"},
            "resume_length":  {"type": "integer"},
            "job_requirements": {
                "type": "object",
                "properties": {
                    "min_years_experience": {"type": "integer", "nullable": True},
                    "min_education_level": {
                        "type": "string",
                        "enum": ["High School", "Bachelors", "Masters", "PhD"],
                        "nullable": True,
                    },
                    "min_project_count": {"type": "integer", "nullable": True},
                },
                "required": ["min_years_experience", "min_education_level", "min_project_count"],
            },
        },
        "required": [
            "years_experience", "skills_match_score",
            "education_level", "project_count", "resume_length", "job_requirements"
        ],
    }

    prompt = f"""Analyze this CV and job description carefully. Extract two things:

=== PART 1: CV FEATURES ===

1. years_experience (integer):
   - Sum ALL non-overlapping professional/internship work experience durations.
   - Today's date is {CURRENT_MONTH}. Treat "Present", "Current", or "Now" as {CURRENT_YEAR}.
   - Parse ranges precisely: "Jan 2023 – Present" = 3 years, "2020 – 2022" = 2 years, "Sep 2021 – Mar 2023" = 1.5 → round to 2.
   - Internships and part-time roles count. Freelance and self-employment count.
   - Overlap periods (concurrent roles) are counted only once.
   - If no work experience is found, return 0.

2. skills_match_score (float between 0 and 100):
   - Identify every technical skill, tool, language, framework, methodology, and certification in the CV.
   - Compare them against the requirements listed in the JOB DESCRIPTION below.
   - Return the percentage of JD requirements the candidate satisfies.
   - Example: JD has 10 distinct requirements, candidate matches 7 → return 70.0

3. education_level (string — EXACTLY one of: "High School", "Bachelors", "Masters", "PhD"):
   - Return the candidate's HIGHEST fully completed degree only.
   - A degree in progress (e.g. "currently studying", "enrolled", "expected 2026") does NOT count as completed.
   - If no completed degree is found, return "High School".

4. project_count (integer):
   - Count ALL distinct projects: professional, academic, personal, open-source.
   - Do not count the same project twice even if mentioned in multiple sections.
   - If none are found, return 0.

5. resume_length (integer):
   - Count the total number of words in the entire CV text (all sections combined).

=== PART 2: JOB REQUIREMENTS ===

Extract the minimum thresholds the job description requires or implies. These are used to give the candidate actionable feedback specific to this role.

job_requirements object:
- min_years_experience (integer or null):
  - Only extract if the JD uses explicit language like "X+ years", "at least X years", "minimum X years".
  - Internship / entry-level / fresh-graduate / no-experience-required → 0.
  - Words like "ideal for", "preferred", "nice to have" do NOT count as requirements → return null.
  - Return null if experience is not mentioned at all.

- min_education_level (string or null — one of: "High School", "Bachelors", "Masters", "PhD"):
  - Only extract if the JD explicitly states a degree is REQUIRED (e.g. "Bachelor's degree required", "must have a degree").
  - "Ideal for a student", "preferred", "a plus", or describing the target audience does NOT count → return null.
  - Return null if education is not explicitly required.

- min_project_count (integer or null):
  - Minimum number of projects explicitly required.
  - Return null if not mentioned (most JDs don't specify this).

--- JOB DESCRIPTION ---
{jd_text}
--- END JOB DESCRIPTION ---

Return a JSON object with these keys: years_experience, skills_match_score, education_level, project_count, resume_length, job_requirements."""

    print(f"Uploading {pdf_path} to Gemini Files API...")
    sample_file = client.files.upload(file=pdf_path)

    models_to_try = [
        ("gemini-2.5-flash", 3, 15),
        ("gemini-1.5-flash", 2, 10),
    ]

    for model, max_retries, wait_sec in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                print(f"Extracting features with {model} (attempt {attempt}/{max_retries})...")
                response = client.models.generate_content(
                    model=model,
                    contents=[sample_file, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.1,
                    ),
                )
                payload = json.loads(response.text)
                if not isinstance(payload, dict):
                    raise RuntimeError("Unexpected Gemini response format.")
                return payload
            except json.JSONDecodeError as e:
                raise RuntimeError("Gemini returned invalid JSON.") from e
            except Exception as e:
                err = str(e)
                if "503" in err or "UNAVAILABLE" in err:
                    if attempt < max_retries:
                        print(f"  Server busy, retrying in {wait_sec}s…")
                        time.sleep(wait_sec)
                    else:
                        print(f"  {model} still unavailable, trying next model…")
                elif "429" in err or "RESOURCE_EXHAUSTED" in err:
                    print(f"  {model} quota exhausted, trying next model…")
                    break
                else:
                    raise

    raise RuntimeError("All Gemini models are currently unavailable. Please try again later.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CV features using Gemini")
    parser.add_argument("pdf_path", help="Path to the CV PDF file")
    parser.add_argument("jd_path", help="Path to the job description .txt file")
    args = parser.parse_args()

    for path in [args.pdf_path, args.jd_path]:
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            exit(1)

    features = extract_features_from_cv(args.pdf_path, args.jd_path)
    print("\n--- Extracted Features ---")
    print(json.dumps(features, indent=2))