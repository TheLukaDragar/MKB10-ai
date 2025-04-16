import os
from typing import List
import openai
import pandas as pd
from prompts import *
import logging
from datetime import datetime
from collections import defaultdict
from colorama import Fore, Style
import json
import dotenv

dotenv.load_dotenv()

# Configure logging with a more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

model_name = "nemotron" 
base_url = "http://localhost:8001/"
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=base_url.rstrip("/") + "/v1"
)


#load the available categories
available_categories = open("/Users/carbs/mkb102/sklopi_slo_df.csv", "r").read()


#call the llm to get selected categ

diagnosis = "Anamneza Včeraj je panel s kolesa in se udaril po desni dlani, levi rami, levi nadlahti, levi podlahti in levi dlani ter levem kolenu. Vročine in mrzlice ni imel. Antitetanična zaščita obstaja. Status ob sprejemu Vidne številne odrgnine v prelu desne dlani in po vseh prstih te roke. Največja rana v predelu desnega zapestja, okolica je blago pordela. Gibljvost v zapestju je popolnoma ohranjena. Brez NC izpadov. Na levi rami vidna odrgnina, prav tako tudi odrgnine brez znakov vnetja v področju leve nadlahti, leve podlahti in leve dlani. Dve večji odrgnini v predelu levega kolena. Levo koleno je blago otečeno. Ballottement negativen. Gibljivost v kolenu 0/90. Iztipam sklepno špranjo kolena, ki palpatorno ni občutljiva. Lachman in predalčni fenomen enaka v primerjavi z nepoškodovanim kolenom. Kolateralni ligamenti delujejo čvrsti. MCL nekoliko boleč na nateg in palpatorno. Diagnostični postopki RTG desno zapestje: brez prepričljivih znakov sveže poškodbe skeleta desna dlan: brez prepričljivih znakov sveže poškodbe skeleta levo koleno: brez prepričljivih znakov sveže poškodbe skeleta."

# First LLM

def llm_call(prompt, model_name="nemotron",**kwargs):
    # Log start of call with clear separator
    logger.info("="*80)
    logger.info(f"🤖 Starting LLM call with model: {model_name}")
    logger.info("-"*40 + " PROMPT " + "-"*40)
    logger.info(f"\033[94m{prompt}\033[0m")  # Blue color for prompt
    logger.info("-"*80)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4000,
        stream=True,
    
        **kwargs
    )
    
    logger.info("-"*40 + " RESPONSE " + "-"*39)
    full_response = ""
    start_time = datetime.now()
    
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            full_response += content
            # Print chunks in green with no newline
            print(f"\033[32m{content}\033[0m", end="", flush=True)
    
    # Calculate and log duration
    duration = (datetime.now() - start_time).total_seconds()
    print()  # Add newline after streaming response
    logger.info("-"*80)
    logger.info(f"✅ LLM call completed in {duration:.2f} seconds")
    logger.info("="*80)
    
    return full_response


if os.path.exists("reasoning.txt") and False:
    with open("reasoning.txt", "r") as f:
        reasoning = f.read()
else:
    #call the llm to get reasoning
    reasoning = llm_call(summarize_and_select_categories_prompt.format(diagnosis=diagnosis,available_categories=available_categories))

    #save to file
    with open("reasoning.txt", "w") as f:
        f.write(reasoning)


# Define Pydantic model for structured output
class MKB10Response(openai.BaseModel):
    applicable_categories: List[str]

class SpecificCode(openai.BaseModel):
    code: str
    rationale: str
    slo_description: str = ""
    eng_description: str = ""
    
    @property
    def category_prefix(self) -> str:
        return self.code[:1] if self.code else ""
    
    @classmethod
    def with_prefix(cls, prefix: str):
        """Create a new model class with code validation for specific prefix"""
        namespace = {
            '__annotations__': {
                'code': str,
                'rationale': str
            },
            'model_config': {
                'str_strip_whitespace': True
            }
        }
        
        def code_validator(v: str) -> str:
            if not v.startswith(prefix):
                raise ValueError(f"Code must start with {prefix}")
            return v
            
        namespace['_validate_code'] = classmethod(lambda cls, v: code_validator(v))
        
        return type(f'SpecificCode_{prefix}', (cls,), namespace)

class SpecificCodesResponse(openai.BaseModel):
    final_codes: List[SpecificCode]
    
    @classmethod
    def for_category(cls, category: str):
        """Create a response model for a specific category prefix"""
        prefix = category[0]  # Take first letter of category range (e.g., 'S' from 'S00-S09')
        specific_code_model = SpecificCode.with_prefix(prefix)
        
        namespace = {
            '__annotations__': {
                'final_codes': List[specific_code_model]
            }
        }
        
        return type(f'SpecificCodesResponse_{prefix}', (cls,), namespace)

extracted_categories = llm_call(extract_categories_prompt.format(text=reasoning),extra_body={"guided_json": MKB10Response.model_json_schema()},temperature=0.0)

def clean_code(code: str) -> str:
    """Clean MKB-10 code by removing dashes and standardizing format"""
    return code.replace('-', '').replace('–', '')

def find_matching_codes(categories: List[str], available_categories_df: pd.DataFrame) -> List[dict]:
    results = []
    
    # Clean all codes in the DataFrame once
    available_categories_df['clean_code'] = available_categories_df['SKLOP'].apply(clean_code)
    
    for category in categories:
        category = clean_code(category)
        
        # Find exact matches
        matches = available_categories_df[available_categories_df['clean_code'] == category]
        
        if not matches.empty:
            for _, row in matches.iterrows():
                results.append({
                    'code': row['SKLOP'],
                    'description': row['SLOVENSKI NAZIV'],
                    'matched_query': category
                })

        else:
            print(f"No matches found for {category}")
    
    return results

def find_matching_codes_hierarchical(categories: List[str], available_categories_df: pd.DataFrame) -> List[dict]:
    results = []
    
    # Clean all codes in the DataFrame once
    available_categories_df['clean_code'] = available_categories_df['KODA'].apply(clean_code)
    
    for category in categories:
        category = clean_code(category)
        found_match = False
        
        # First try exact match (safety check)
        exact_matches = available_categories_df[available_categories_df['clean_code'] == category]
        if not exact_matches.empty:
            for _, row in exact_matches.iterrows():
                results.append({
                    'code': row['KODA'],
                    'description': row['SLOVENSKI NAZIV'],
                    'english_description': row['ENGLISH DESCRIPTION POSTPROCESED'],
                    'matched_query': category,
                    'match_type': 'exact'
                })
            found_match = True
            continue
            
        # If no exact match, try hierarchical matching
        if not found_match:
            # Try to match on each level
            for level in ['SKLOP / 3. NIVO', 'SKLOP / 2. NIVO', 'SKLOP / 1. NIVO']:
                if level in available_categories_df.columns:  # Check if column exists
                    matches = available_categories_df[available_categories_df[level].apply(
                        lambda x: clean_code(str(x)) == category if pd.notna(x) else False
                    )]
                    
                    if not matches.empty:
                        for _, row in matches.iterrows():
                            results.append({
                                'code': row['KODA'],
                                'description': row['SLOVENSKI NAZIV'],
                                'english_description': row['ENGLISH DESCRIPTION POSTPROCESED'],
                                'matched_query': category,
                                'match_type': f'hierarchical_{level}'
                            })
                        found_match = True
                        break  # Stop after finding matches at the most specific level
        
        if not found_match:
            print(f"No matches found for {category}")
    
    return results

# Load both category files
df_slo = pd.read_csv("/Users/carbs/mkb102/sklopi_slo_df.csv")
df_eng = pd.read_csv("/Users/carbs/mkb102/mkb_slo_df_eng.csv")

# Parse the extracted categories from JSON string
categories = json.loads(extracted_categories)['applicable_categories']

# Get unique first letters from categories for filtering
letters = set(category[0] for category in categories)

# # Filter categories from df_slo that start with any of the letters
# mask = df_slo['SKLOP'].str[0].isin(letters)
# categories_containing_letters = df_slo[mask]['SKLOP'].values

# Find matching codes in both dataframes
matching_results_slo = find_matching_codes(categories, df_slo)
matching_results_hierarchical = find_matching_codes_hierarchical(categories, df_eng)

# Group results by query
def group_results_by_query(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[result['matched_query']].append(result)
    return dict(grouped)

# Group both result sets
grouped_slo = group_results_by_query(matching_results_slo)
grouped_hierarchical = group_results_by_query(matching_results_hierarchical)

# Create structured JSON output
json_output = {}
all_queries = sorted(set(list(grouped_slo.keys()) + list(grouped_hierarchical.keys())))

for query in all_queries:
    json_output[query] = {
        "original_matches": grouped_slo.get(query, []),
        "hierarchical_matches": grouped_hierarchical.get(query, [])
    }

# Save initial matches to JSON
output_filename = "mkb10_matches.json"
with open(output_filename, "w", encoding='utf-8') as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print(f"\n{Fore.CYAN}Initial matches saved to {output_filename}{Style.RESET_ALL}")

# Process each category group separately
all_final_codes = []

for query, matches in json_output.items():
    # Format matches for this category and create lookup dictionary
    category_codes = []
    descriptions_lookup = {}
    for match_type in ['original_matches', 'hierarchical_matches']:
        for match in matches[match_type]:
            code = match['code']
            descriptions_lookup[code] = {
                'slo_description': match['description'],
                'eng_description': match.get('english_description', '')
            }
            # Simplified data structure for LLM
            code_info = {
                'code': code,
                'slo_description': match['description'],
                'eng_description': match.get('english_description', '')
            }
            category_codes.append(code_info)
    
    if not category_codes:
        continue
        
    category_codes_text = json.dumps(category_codes, ensure_ascii=False, indent=2)
    
    print(f"\n{Fore.CYAN}Processing category group: {query}{Style.RESET_ALL}")
    
    # First LLM call - Reasoning step (unguided)
    print(f"{Fore.BLUE}Step 1: Analyzing codes...{Style.RESET_ALL}")
    reasoning_result = llm_call(
        reason_specific_codes_prompt.format(
            diagnosis=diagnosis,
            matched_codes=category_codes_text
        )
    )
    
    print(f"{Fore.BLUE}Step 2: Extracting specific codes...{Style.RESET_ALL}")
    # Second LLM call - Guided extraction with the reasoning context
    specific_codes = llm_call(
        extract_specific_codes_prompt.format(
            diagnosis=diagnosis,
            matched_codes=category_codes_text
        ) + f"\n\nPrevious analysis:\n{reasoning_result}",
        extra_body={"guided_json": SpecificCodesResponse.for_category(query).model_json_schema()},
        temperature=0.0
    )
    
    # Parse results and add descriptions from lookup
    try:
        category_final_codes = json.loads(specific_codes)
        if 'final_codes' in category_final_codes:
            valid_codes = []
            for code in category_final_codes['final_codes']:
                # Verify code belongs to current category
                if not code['code'].startswith(query[0]):
                    print(f"{Fore.YELLOW}Skipping code {code['code']} - does not belong to category {query}{Style.RESET_ALL}")
                    continue
                    
                # Only include codes that we have descriptions for
                if code['code'] in descriptions_lookup:
                    desc = descriptions_lookup[code['code']]
                    code['slo_description'] = desc['slo_description']
                    code['eng_description'] = desc['eng_description']
                    code['category_group'] = query
                    code['reasoning'] = reasoning_result  # Store the reasoning for reference
                    valid_codes.append(code)
                else:
                    print(f"{Fore.YELLOW}Skipping code {code['code']} - no matching description found{Style.RESET_ALL}")
            
            if valid_codes:
                all_final_codes.extend(valid_codes)
            else:
                print(f"{Fore.YELLOW}No valid codes found for category {query}{Style.RESET_ALL}")
    except json.JSONDecodeError as e:
        print(f"{Fore.RED}Error parsing results for category {query}: {e}{Style.RESET_ALL}")

# Prepare final output
final_output = {
    "final_codes": all_final_codes
}

# Save final codes to file
final_output_filename = "mkb10_final_codes.json"
with open(final_output_filename, "w", encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=2)

# Group codes by category for display
category_grouped_codes = defaultdict(list)
for code_info in final_output['final_codes']:
    category_grouped_codes[code_info['category_group']].append(code_info)

# Print organized results
print(f"\n{Fore.CYAN}Final Low-Level MKB-10 Codes by Category:{Style.RESET_ALL}")
print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

for category, codes in sorted(category_grouped_codes.items()):
    print(f"\n{Fore.YELLOW}Category {category}:{Style.RESET_ALL}")
    for code_info in codes:
        print(f"  → {Fore.GREEN}{code_info['code']}{Style.RESET_ALL}")
        print(f"    {Fore.CYAN}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
        print(f"    {Fore.CYAN}English:{Style.RESET_ALL} {code_info['eng_description']}")
        print(f"    {Fore.CYAN}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
        print()
    print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")











