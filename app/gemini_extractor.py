import os
import json
from google import genai
from google.genai import types

def extract_features_from_cv(pdf_path: str) -> dict:
    \"\"\"
    Uses Google Gemini API to extract structured features from a CV PDF.
    
    Required features for the ML/DL models:
    - years_experience (int)
    - skills_match_score (float, 0-100)
    - education_level (str: High School, Bachelors, Masters, PhD)
    - project_count (int)
    - resume_length (int, number of words)
    - github_activity (int, number of commits/contributions if mentioned, else 0)
    \"\"\"
    
    # Initialize the client. Assumes GEMINI_API_KEY is set in the environment.
    client = genai.Client()
    
    # Define the JSON schema we want Gemini to return
    # This ensures the output matches our model's exact input format
    response_schema = {
        "type": "object",
        "properties": {
            "years_experience": {"type": "integer"},
            "skills_match_score": {"type": "number"},
            "education_level": {
                "type": "string", 
                "enum": ["High School", "Bachelors", "Masters", "PhD"]
            },
            "project_count": {"type": "integer"},
            "resume_length": {"type": "integer"},
            "github_activity": {"type": "integer"}
        },
        "required": ["years_experience", "skills_match_score", "education_level", "project_count", "resume_length", "github_activity"]
    }
    
    print(f"Uploading {pdf_path} to Gemini...")
    # Upload the PDF file
    sample_file = client.files.upload(file=pdf_path)
    
    print("Extracting features...")
    # Call the model to extract the structured data
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=[
            sample_file,
            "Analyze this CV and extract the following features strictly adhering to the JSON schema."
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=response_schema,
            temperature=0.1,
        ),
    )
    
    # Parse the response
    extracted_data = json.loads(response.text)
    return extracted_data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract CV features using Gemini")
    parser.add_argument("pdf_path", help="Path to the CV PDF file")
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: File {args.pdf_path} not found.")
        exit(1)
        
    try:
        features = extract_features_from_cv(args.pdf_path)
        print("\\n--- Extracted Features ---")
        print(json.dumps(features, indent=2))
        
        print("\\nThese features can now be mapped/scaled and passed to the MVP model (e.g. Logistic Regression or Medium NN).")
    except Exception as e:
        print(f"Error during extraction: {e}")
