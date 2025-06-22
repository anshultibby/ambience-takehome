# ICD-10 Code Assistant

A Streamlit web application that analyzes medical transcripts and suggests appropriate ICD-10 codes using AI-powered analysis.

## Features

- **AI-Powered Analysis**: Uses OpenAI's GPT-4 to analyze medical transcripts
- **ICD-10 Code Lookup**: Integrates with ICD-10 alphabetical index and tabular list
- **Interactive Interface**: Easy-to-use web interface built with Streamlit
- **Detailed Reasoning**: Provides reasoning for each suggested code
- **Processing Log**: Shows detailed steps of the analysis process
- **Export Results**: Download results as JSON

## Prerequisites

1. **OpenAI API Key**: You'll need an OpenAI API key to use the service
2. **ICD-10 XML Files**: The following files must be in the project directory:
   - `icd10cm_index_2025.xml` (ICD-10 Alphabetical Index)
   - `icd10cm_tabular_2025.xml` (ICD-10 Tabular List)

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure XML files are present**:
   - Place `icd10cm_index_2025.xml` and `icd10cm_tabular_2025.xml` in the project directory
   - The app will show file status in the sidebar

## Usage

1. **Start the application**:
   ```bash
   streamlit run streamlit_icd_app.py
   ```

2. **Configure the app**:
   - Enter your OpenAI API key in the sidebar
   - Adjust max processing steps if needed (default: 20)

3. **Analyze a transcript**:
   - Paste a medical transcript in the text area
   - Or click "Load Sample Transcript" to use the example
   - Click "🔍 Analyze Transcript" to start processing

4. **Review results**:
   - ICD-10 codes will appear in the right column
   - Each code includes reasoning and alternate codes
   - View the processing log to see detailed steps

5. **Export results**:
   - Click "📥 Export Results as JSON" to download the analysis

## Sample Transcript

The app includes a sample transcript for testing:

```
Patient is a 65-year-old male with a history of type 2 diabetes mellitus. 
He presents today with complaints of severe leg pain and swelling in both lower extremities. 
His A1c is 12, indicating poor glycemic control. He is currently on insulin therapy.
Patient also reports high triglycerides and has been diagnosed with hypertension.
BMI is 39, indicating class 2 obesity. There are concerns about diabetic neuropathy 
and potential peripheral artery disease affecting blood supply to the legs.
Patient has poor wound healing and kidney function needs monitoring.
```

## How It Works

1. **AI Analysis**: The system uses a specialized prompt to guide GPT-4 through ICD-10 coding
2. **Tool Calls**: The AI makes function calls to:
   - `lookup_alphabetical_index()` - Search for conditions
   - `lookup_tabular_list()` - Get detailed code information
   - `create_icd10_suspect_entry()` - Create code entries with reasoning
3. **Iterative Process**: The AI continues until all relevant codes are identified
4. **Results**: Final results include codes, reasoning, and alternate options

## Project Structure

```
├── streamlit_icd_app.py           # Main Streamlit application
├── openai_tools_converter.py     # Tool calling and context management
├── simple_xml_lookup.py          # XML parsing functions
├── descriptions.py               # ICD-10 guidelines and prompts
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── icd10cm_index_2025.xml       # ICD-10 Alphabetical Index (required)
└── icd10cm_tabular_2025.xml     # ICD-10 Tabular List (required)
```

## Configuration

- **Model**: Uses GPT-4 (gpt-4o) by default
- **Max Steps**: Configurable via sidebar (5-50 steps)
- **XML Files**: Must be in project directory with exact filenames

## Troubleshooting

1. **Missing XML Files**: Ensure both XML files are in the project directory
2. **API Key Issues**: Verify your OpenAI API key is valid and has sufficient credits
3. **Processing Errors**: Check the processing log for detailed error information
4. **No Results**: Try adjusting the transcript or increasing max processing steps

## Security Notes

- API keys are entered as password fields and not stored
- All processing happens locally except for OpenAI API calls
- No transcript data is permanently stored

## License

This project is for educational and research purposes. Ensure compliance with ICD-10 licensing terms for commercial use. 