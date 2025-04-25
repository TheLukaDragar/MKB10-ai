summarize_and_select_categories_prompt = """
<instructions>
You are a medical coding expert for the Slovenian version of MKB-10-AM (ICD-10-AM) coding system.
Your task is to identify and select the most appropriate diagnostic code ranges from the provided available_categories list that match the patient's diagnosis and circumstances.

KEY CODING PRINCIPLES:
1. ONLY select code ranges that appear in the provided available_categories list
2. Follow the hierarchical structure: primary condition first, followed by external causes and contributing factors
3. Multiple codes may be necessary to fully represent the case

SYSTEMATIC ANALYSIS PROCESS:
1. First, carefully examine all provided available_categories
2. Next, thoroughly analyze the diagnosis text to identify:
   - Primary injuries/conditions (noting exact anatomical locations)
   - Secondary injuries/conditions
   - Mechanism of injury/illness (how it happened)
   - Activity during occurrence (what patient was doing)
   - Place of occurrence (where it happened)
   - Status of patient (e.g., driver, passenger, pedestrian)
   - Complicating factors (e.g., infection, foreign body)

3. FIRST STEP - COMPREHENSIVE DIAGNOSIS SUMMARY:
   Begin by writing a detailed summary of the diagnosis in English, including:
   - All primary and secondary conditions with specific anatomical locations
   - Causal mechanisms and circumstances
   - Patient status and activities at time of injury/illness
   - Any complicating factors
   Use this summary as the basis for your code selection decisions.
   
4. For each identified element, select the most specific matching code:
   - For injuries: Select codes that exactly match the anatomical location AND type of injury
   - For external causes: Select codes that match both the mechanism AND circumstances
   - For activities: Select codes that specify what the patient was doing
   - For places: Select codes indicating where the incident occurred

CODE SELECTION REQUIREMENTS:
1. Always include code ranges for ALL anatomical locations mentioned in diagnosis
2. Always include at least one external cause code when injury is present
3. Include activity and place codes when information is available
4. For multiple injuries to the same body region, select all applicable codes

OUTPUT FORMAT:
- List all selected code ranges in proper sequence (primary condition first)
- Include the Slovenian description for each code
- For each code, provide specific rationale for selection based on text evidence
- Highlight any uncertainties or areas where more specific information would help
</instructions>

<input>
<diagnosis>
{diagnosis}
</diagnosis>

<available_categories>
{available_categories}
</available_categories>
</input>
"""

extract_categories_prompt = """
<instructions>
Extract ALL MKB10-AM/ICD-10-AM version 11th edition categories mentioned in the format XX-XX or XX (like A00-A09, Z80-Z99, etc.) from the above response that are marked as APPLICABLE, RELEVANT, or indicated as a diagnosis. Include both primary and secondary categories.

Key points:
1. Include ALL relevant categories without prioritization
2. Do not filter out any categories based on their importance level
3. Include both primary and secondary diagnoses
4. DO NOT include any codes marked as "NOT APPLICABLE" or "NOT RELEVANT"

Return the extracted applicable codes in JSON format.
</instructions>

<input>
{text}
</input>

<output_format>
{{
    "applicable_categories": [
        "I70-I79",
        "J20-J22",
        "A00-A09",
    ]
}}
</output_format>

</output>
"""

extract_specific_codes_prompt = """
<instructions>
Select applicable codes FROM THE PROVIDED CODES ONLY.

Rules for code selection:
1. ONLY use codes from the provided matched_codes list
2. Do not suggest codes from other categories
3. Match the exact anatomical location and injury type
4. Use the most specific code available from the provided options
5. Consider laterality when available in the codes
6. If there are multiple codes list them all
7. Include at least 5 codes, ordered by relevance
8. Do not include ranges like U50–U72 or similar

The output must follow the exact JSON format specified.
</instructions>

<input>
<diagnosis>
{diagnosis}
</diagnosis>

<matched_codes>
{matched_codes}
</matched_codes>
</input>

<output_format>
{{
    "final_codes": [
        {{
            "code": "S40.81",  // Must be one of the provided codes
            "rationale": "Matches exactly with provided code list and describes left shoulder abrasion"
        }}
    ]
}}
</output_format>

<example>
Diagnosis excerpt: "Abrasion of left shoulder"
Available codes: ["S40.81", "S40.82", "S40.9"]
Correct: S40.81 (matches exact location and laterality)
Wrong: S40.9 (too general when more specific code is available)
</example>
</output>
"""

reason_specific_codes_prompt = """
<instructions>
You are a medical coding expert specializing in MKB-10-AM (ICD-10-AM) coding system. Your task is to analyze the provided diagnosis and select the most appropriate codes FROM THE PROVIDED CATEGORY ONLY.

Key Principles:
1. ONLY consider codes that are provided in the matched_codes list
2. Do not suggest codes from other categories or ranges
3. For each injury that matches this category's anatomical region:
   - Use the most specific code available from the provided options
   - Consider laterality (left/sin. or right/dex.) when available
   - Match the exact type of injury described
4. ALWAYS include at least 5 codes, even if some are less directly relevant
5. Order all codes by relevance, with most relevant codes first

First, analyze:
0. Write a comprehensive summary of the diagnosis in English, including primary and secondary conditions, anatomical locations, and relevant circumstances.
1. Which injuries from the diagnosis fit THIS specific category
2. Which of the provided codes most precisely match these injuries
3. Whether laterality information can be captured with the available codes
</instructions>

<input>
<diagnosis>
{diagnosis}
</original_diagnosis>

<matched_codes>
{matched_codes}
</matched_codes>
</input>

<output_format>
Provide your analysis in a structured format:

DIAGNOSIS SUMMARY:
[Comprehensive summary of the diagnosis in English]

CATEGORY ANALYSIS:
[Which injuries from the diagnosis belong to this category's anatomical region]

AVAILABLE CODES ASSESSMENT:
[Analysis of how the provided codes match these specific injuries]

RECOMMENDED MATCHES:
[Specific matches between injuries and the available codes, with reasoning, ordered by relevance. List at least 5 codes with the following format for each:]

1. CODE: [most relevant code]
   RELEVANCE: [Primary - direct match to main condition]
   RATIONALE: [Specific reasoning for selection]

2. CODE: [second most relevant code]
   RELEVANCE: [Secondary - related to condition]
   RATIONALE: [Specific reasoning for selection]

... and so on for at least 5 codes total
</output_format>
</output>
"""
