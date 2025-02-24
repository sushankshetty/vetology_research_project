import pandas as pd
import openai
import re
import spacy
import logging

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Set OpenAI API key
openai.api_key = "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"

# Function to clean and preprocess text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").replace("_x000D_", " ").strip().lower()
    return re.sub(r'\s+', ' ', text)  # Remove extra spaces

# Function to extract measurements
def extract_measurements(text):
    pattern = r"\b\d+(\.\d+)?\s?(cm|mm|ml|%|kg|g|mg|m)\b"
    return ", ".join(match[0] + " " + match[1] for match in re.findall(pattern, text))

# Function to extract medical conditions using NLP
def extract_conditions(text, condition_list):
    doc = nlp(text)
    found_conditions = [ent.text for ent in doc.ents if ent.text in condition_list]
    return ", ".join(set(found_conditions))

# Function to mask personal information
def mask_personal_info(text):
    doc = nlp(text)
    masked_text = text
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "DATE", "GPE", "ORG", "LOC", "FAC", "NORP"]:
            masked_text = masked_text.replace(ent.text, "[REDACTED]")
    return masked_text


# Function to process text using GPT
def analyze_text_with_llm(column_data, classifiers):
    masked_data = mask_personal_info(column_data)
    structured_data = {
        "clean_text": masked_data,
        "measurements": extract_measurements(masked_data),
        "conditions": extract_conditions(masked_data, classifiers)
    }
    prompt = f"""
    You are a veterinary radiology assistant analyzing canine thoracic radiographs. 
    Your task is to carefully evaluate radiology reports and classify each medical condition with precision.
    
    Below are the conditions to classify and their detailed explanations:
    
    - **Gastritis**: Inflammation of the stomach lining, often associated with vomiting, loss of appetite, and nausea. Look for mentions of "gastric wall thickening," "mucosal irritation," or "stomach inflammation."
    - **Ascites**: Accumulation of fluid in the abdominal cavity. Indications include "abdominal effusion," "fluid distention," or "free peritoneal fluid."
    - **Colitis**: Inflammation of the colon, often linked to diarrhea or discomfort. Terms like "colon wall thickening," "inflammatory bowel disease," or "mucosal irregularities" suggest this condition.
    - **Liver Mass**: Any abnormal growth or lesion in the liver. Look for "hepatic mass," "nodular lesions," "hepatic tumor," or "focal liver enlargement."
    - **Pancreatitis**: Inflammation of the pancreas, often causing digestive issues. Keywords include "pancreatic swelling," "peri-pancreatic fat stranding," or "pancreatic echogenicity changes."
    - **Microhepatia**: A condition where the liver is smaller than normal. Signs include "reduced liver volume," "small liver," or "diminished hepatic silhouette."
    - **Small Intestinal Obstruction**: A blockage in the intestines preventing normal movement of contents. Look for "dilated small intestines," "intestinal gas accumulation," or "mechanical obstruction."
    - **Splenic Mass**: An abnormal growth in the spleen, potentially cancerous. Mentions of "splenic nodules," "enlarged spleen with mass effect," or "heterogeneous splenic texture" indicate this.
    - **Splenomegaly**: Enlargement of the spleen, which may indicate various underlying conditions. Terms like "diffusely enlarged spleen," "prominent splenic size," or "splenic hyperplasia" suggest this diagnosis.
    - **Hepatomegaly**: Enlargement of the liver, often related to liver disease. Look for "generalized liver enlargement," "increased hepatic dimensions," or "diffuse hepatic swelling."
    
    Your task is to classify each condition as:
    - "Abnormal" if the condition is explicitly present.
    - "Normal" if absent, ruled out, or not mentioned.
    - "Unknown" if the report lacks enough details to classify with certainty, the context does not match any classifier, or the data is too ambiguous to decide.
    
    Output in the format:
    gastritis: Abnormal, ascites: Normal, colitis: Unknown, liver_mass: Abnormal, pancreatitis: Unknown, etc.
    
    Radiology Report:
    {structured_data['clean_text']}
    Measurements Extracted:
    {structured_data['measurements']}
    Conditions Mentioned:
    {structured_data['conditions']}
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert veterinary AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )
    return eval(response['choices'][0]['message']['content'])


'''# Process Excel file
def process_excel(file_path, column1, column2, classifiers):
    df = pd.read_excel(file_path)
    df.fillna({col: "" if df[col].dtype == "object" else 0 for col in df.columns}, inplace=True)
    
    for classifier in classifiers:
        if classifier not in df.columns:
            df[classifier] = None
    
    for index, row in df.iterrows():
        text_data = clean_text(row[column1]) + " " + clean_text(row[column2])
        try:
            analysis_results = analyze_text_with_llm(text_data, classifiers)
            for classifier, status in analysis_results.items():
                df.at[index, classifier] = status
        except Exception as e:
            for classifier in classifiers:
                df.at[index, classifier] = f"Error: {str(e)}"
    
    return df
'''

def process_excel(file_path, column1, column2, classifiers):
    df = pd.read_excel(file_path)

    #  Convert classifier columns to object type before assignment
    for classifier in classifiers:
        if classifier not in df.columns:
            df[classifier] = None  # Create missing columns
        df[classifier] = df[classifier].astype(object)  # Ensure they can store strings

    df.fillna("", inplace=True)  # Fill NaN with empty strings

    for index, row in df.iterrows():
        text_data = clean_text(row[column1]) + " " + clean_text(row[column2])
        try:
            analysis_results = analyze_text_with_llm(text_data, classifiers)

            if "error_code" in analysis_results:
                error_message = f"Error {analysis_results['error_code']}: {analysis_results['error_message']}"
                for classifier in classifiers:
                    df.at[index, classifier] = str(error_message)  # Ensure string type
            else:
                for classifier, status in analysis_results.items():
                    df.at[index, classifier] = str(status)  #  Convert to string before storing

        except Exception as e:
            error_message = f"Error UNKNOWN: {str(e)}"
            logging.error(error_message)
            for classifier in classifiers:
                df.at[index, classifier] = str(error_message)  #  Explicit conversion to string

    return df


# Main execution
if __name__ == "__main__":
    file_path = "C:\\Users\\hp\\Desktop\\VetologyPresentation\\Project Files\\canine_abdomen_scoring.xlsx"
    column1 = "Findings (original radiologist report)"
    column2 = "Conclusions (original radiologist report)"
    classifiers = ["gastritis", "ascites", "colitis", "liver_mass", "pancreatitis", 
                   "microhepatia", "small_intestinal_obstruction", "splenic_mass", 
                   "splenomegaly", "hepatomegaly"]
    
    updated_df = process_excel(file_path, column1, column2, classifiers)
    output_file = "classified_results_with_nlp.xlsx"
    updated_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")