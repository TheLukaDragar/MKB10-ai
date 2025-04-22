# MKB-10 Medical Diagnosis Classification Pipeline

This project implements an automated pipeline for classifying medical diagnoses using MKB-10 (ICD-10) codes, powered by Large Language Models (LLMs). The system takes medical diagnosis text as input and outputs relevant MKB-10 codes with their descriptions in both Slovenian and English.

## 🌟 Features

- Automated medical text classification using LLM
- Hierarchical code matching system
- Support for both Slovenian and English descriptions
- Detailed reasoning for each code assignment
- JSON output for easy integration
- Colorized console output for better readability

## 🔧 Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. Ensure you have the following data files in your project directory:
- `sklopi_slo_df.csv` - Slovenian MKB-10 category mappings
- `mkb_slo_df_eng.csv` - Combined Slovenian-English MKB-10 codes

4. Configure your LLM endpoint in `bot.py`:
```python
base_url = "http://localhost:8001/"  # Update with your LLM endpoint
```

## 🚀 How It Works

The pipeline follows these steps:

1. **Initial Analysis**
   - Takes a medical diagnosis text as input
   - Uses LLM to analyze and identify relevant MKB-10 categories
   - Saves initial reasoning to `reasoning.txt`

2. **Category Matching**
   - Matches identified categories against the MKB-10 database
   - Performs both exact and hierarchical matching
   - Saves initial matches to `mkb10_matches.json`

3. **Code Refinement**
   - For each category group:
     - Performs detailed analysis of potential codes
     - Uses LLM to reason about specific code applicability
     - Validates and filters codes based on context
   - Saves final codes to `mkb10_final_codes.json`

## 📁 File Structure

```
.
├── bot.py              # Main pipeline implementation
├── prompts.py         # LLM prompt templates
├── sklopi_slo_df.csv  # Slovenian category mappings
├── mkb_slo_df_eng.csv # Combined SLO-ENG mappings
├── reasoning.txt      # Generated reasoning output
├── mkb10_matches.json # Initial code matches
└── mkb10_final_codes.json # Final code assignments
```

## 🔍 Output Format

The final output (`mkb10_final_codes.json`) follows this structure:

```json
{
  "final_codes": [
    {
      "code": "S60.8",
      "rationale": "Reasoning for code assignment",
      "slo_description": "Slovenski opis",
      "eng_description": "English description",
      "category_group": "S00-S09"
    }
  ]
}
```

## 💻 Usage Example

```python
# Example diagnosis text
diagnosis = """
Anamneza: Včeraj je panel s kolesa in se udaril po desni dlani...
"""

# Run the worker
python worker.py

# Run the pipeline
python bot.py
```

## 🔄 Pipeline Steps in Detail

1. **Text Analysis**
   - The system first analyzes the input diagnosis text using an LLM
   - Identifies relevant MKB-10 category ranges
   - Generates detailed reasoning for category selection

2. **Category Matching**
   - Performs multi-level matching:
     - Exact code matches
     - Hierarchical category matches
     - Parent-child relationship validation

3. **Code Refinement**
   - Each potential code undergoes:
     - Contextual validation
     - Description matching
     - Reasoning generation
     - Final filtering

4. **Output Generation**
   - Generates structured JSON output
   - Includes both Slovenian and English descriptions
   - Provides reasoning for each code assignment
   - Groups codes by category

## 🛠 Customization

You can customize the pipeline by:

1. Modifying prompt templates in `prompts.py`
2. Adjusting the LLM parameters in `bot.py`
3. Adding additional validation rules
4. Customizing output formats

## 📝 Logging

The system includes comprehensive logging:
- Colorized console output for different stages
- Detailed LLM interaction logging
- Error and warning messages
- Processing status updates

## ⚠️ Error Handling

The pipeline includes robust error handling for:
- Invalid MKB-10 codes
- LLM response parsing
- File I/O operations
- JSON validation
- Category matching errors

## 🤝 Contributing

Feel free to contribute to this project by:
1. Opening issues for bugs or feature requests
2. Submitting pull requests with improvements
3. Enhancing documentation
4. Adding test cases

## 📄 License

[Add your license information here] 