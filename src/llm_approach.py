import os
from typing import List, Dict, Tuple
import openai
import pandas as pd
from prompts_top_n import *
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from colorama import Fore, Style
import json
import dotenv
import aiohttp
import asyncio
import random
from params import SECTIONS_FILE, WORKERS_FILE, ENGLISH_SECTIONS_FILE

dotenv.load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

base_url = "http://localhost:8001/"
client = openai.OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=base_url.rstrip("/") + "/v1"
)
available_categories = open(SECTIONS_FILE, "r").read()

WORKER_TIMEOUT = timedelta(minutes=1000)
MAX_RETRIES = 3

def get_available_workers():
    """
    Read and return list of available workers from the registration file
    """
    if not os.path.exists(WORKERS_FILE):
        logger.warning("No workers file found. Please start some workers first.")
        return []
    
    try:
        with open(WORKERS_FILE, 'r') as f:
            workers = json.load(f)
        
        # Filter out potentially dead workers
        current_time = datetime.now()
        active_workers = []
        for worker in workers:
            start_time = datetime.fromisoformat(worker['start_time'])
            if current_time - start_time <= WORKER_TIMEOUT:
                active_workers.append(worker)
        
        if not active_workers:
            logger.warning("No active workers found. Please start some workers.")
        
        return active_workers
    
    except Exception as e:
        logger.error(f"Error reading workers file: {e}")
        return []


async def try_worker(worker, request_data):
    """
    Try to make a request to a specific worker
    """
    async with aiohttp.ClientSession() as session:
        worker_url = f"http://{worker['host']}:{worker['port']}/process"
        try:
            async with session.post(worker_url, json=request_data) as response:
                if response.status == 200:
                    result = await response.json()
                    return True, result["result"]
                else:
                    logger.error(f"Worker {worker['worker_id']} returned status {response.status}")
                    return False, None
        except Exception as e:
            logger.error(f"Error calling worker {worker['worker_id']}: {e}")
            return False, None

async def _async_llm_call(prompt, model_name="nemotron", **kwargs):
    """
    Distributed LLM call that uses available workers
    """
    workers = get_available_workers()
    if not workers:
        raise RuntimeError("No available workers found. Please start some workers first.")
    
    # Make a copy of workers list to try different ones
    available_workers = workers.copy()
    retries = 0
    
    while available_workers and retries < MAX_RETRIES:
        # Randomly select a worker
        worker = random.choice(available_workers)
        available_workers.remove(worker)  # Remove this worker from retry pool
        
        # Prepare the request
        request_data = {
            "prompt": prompt,
            "model_name": model_name,
            "worker_id": worker['worker_id'],
            "extra_kwargs": kwargs
        }
        
        success, result = await try_worker(worker, request_data)
        if success:
            return result
        
        retries += 1
        if available_workers:
            logger.info(f"Retrying with another worker (attempt {retries}/{MAX_RETRIES})...")
    
    raise RuntimeError(f"All workers failed to process the request after {retries} attempts")

def llm_call(prompt, model_name="nemotron", **kwargs):
    """
    Synchronous wrapper for the async LLM call
    """
    return asyncio.run(_async_llm_call(prompt, model_name=model_name, **kwargs))

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
                    'level': row['RAVEN'],
                    'category': row['KATEGORIJA'],
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
                                'level': row['RAVEN'],
                                'category': row['KATEGORIJA'],
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


def group_results_by_query(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[result['matched_query']].append(result)
    return dict(grouped)


async def process_category_async(category: Dict, diagnosis: str, descriptions_lookup: Dict, num_recommendations: int = 5) -> Dict:
    """
    Process a single category asynchronously
    """
    query = category['matched_query']
    logger.info(f"{Fore.CYAN}Starting processing of category: {query}{Style.RESET_ALL}")
    
    try:
        # First get reasoning for this category
        logger.info(f"{Fore.BLUE}[{query}] Step 1: Analyzing codes...{Style.RESET_ALL}")
        reasoning_result = await _async_llm_call(
            reason_specific_codes_prompt.format(
                diagnosis=diagnosis,
                matched_codes=json.dumps(category['codes'], ensure_ascii=False, indent=2),
                num_recommendations=num_recommendations
            )
        )
        
        # Then get specific codes with the reasoning context
        logger.info(f"{Fore.BLUE}[{query}] Step 2: Extracting specific codes...{Style.RESET_ALL}")
        specific_codes = await _async_llm_call(
            extract_specific_codes_prompt.format(
                diagnosis=diagnosis,
                matched_codes=json.dumps(category['codes'], ensure_ascii=False, indent=2),
                num_recommendations=num_recommendations
            ) + f"\n\nPrevious analysis:\n{reasoning_result}",
            extra_body={"guided_json": SpecificCodesResponse.for_category(query).model_json_schema()},
            temperature=0.0
        )

        with open(f"debug/specific_codes_{query}.json", "w", encoding='utf-8') as f:
            try:
                parsed_data = json.loads(specific_codes)
                json.dump(parsed_data, f, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                f.write(specific_codes)
        
        try:
            category_final_codes = json.loads(specific_codes)
        except json.JSONDecodeError:
            logger.error(f"{Fore.RED}[{query}] Failed to parse specific_codes JSON for further processing{Style.RESET_ALL}")
            category_final_codes = {'final_codes': []} # Default to empty list to avoid crashing later

        if 'final_codes' in category_final_codes:
            valid_codes = []
            for code in category_final_codes['final_codes']:
                if code['code'].startswith(query[0]):
                    if code['code'] in descriptions_lookup:
                        desc = descriptions_lookup[code['code']]
                        code['slo_description'] = desc['slo_description']
                        code['eng_description'] = desc['eng_description']
                        code['category_group'] = query
                        code['reasoning'] = reasoning_result
                        valid_codes.append(code)
                        logger.info(f"{Fore.GREEN}[{query}] Added code: {code['code']}{Style.RESET_ALL}")
                    else:
                        logger.warning(f"{Fore.YELLOW}[{query}] Skipping code {code['code']} - no matching description found{Style.RESET_ALL}")
                else:
                    logger.warning(f"{Fore.YELLOW}[{query}] Skipping code {code['code']} - does not belong to category {query}{Style.RESET_ALL}")
            
            logger.info(f"{Fore.CYAN}Completed processing of category {query} with {len(valid_codes)} valid codes{Style.RESET_ALL}")
            return {
                'category': query,
                'codes': valid_codes,
                'reasoning': reasoning_result
            }
    except Exception as e:
        logger.error(f"{Fore.RED}Error processing category {query}: {e}{Style.RESET_ALL}")
        return {
            'category': query,
            'codes': [],
            'error': str(e)
        }

async def process_all_categories_async(categories: List[Dict], diagnosis: str, descriptions_lookup: Dict, num_recommendations: int = 5) -> List[Dict]:
    """
    Process all categories in parallel
    """
    logger.info(f"{Fore.CYAN}Starting parallel processing of {len(categories)} categories{Style.RESET_ALL}")
    tasks = [process_category_async(category, diagnosis, descriptions_lookup, num_recommendations) for category in categories]
    return await asyncio.gather(*tasks)

def process_categories_parallel(categories: List[Dict], diagnosis: str, descriptions_lookup: Dict, num_recommendations: int = 5) -> List[Dict]:
    """
    Synchronous wrapper for parallel category processing
    """
    return asyncio.run(process_all_categories_async(categories, diagnosis, descriptions_lookup, num_recommendations))

def reasoning_to_categories(diagnosis_text: str, available_categories: str, debug: bool = True) -> List[Dict]:
    """
    Extract the range of categories from the reasoning text
    """
    reasoning = llm_call(summarize_and_select_categories_prompt.format(diagnosis=diagnosis_text,available_categories=available_categories))
    if debug:
        with open("debug/reasoning.txt", "w") as f:
            f.write(reasoning)
    return reasoning


def extract_categories_ranges(reasoning_text: str, debug: bool = True) -> List[Dict]:
    """
    Extract the range of categories from the list of categories
    """
    extracted_categories = llm_call(extract_categories_prompt.format(text=reasoning_text),extra_body={"guided_json": MKB10Response.model_json_schema()},temperature=0.0)

    if debug:
        with open("debug/extracted_categories.txt", "w") as f:
            f.write(extracted_categories)
    return extracted_categories

def get_categories_from_json(json_text: str, debug: bool = True) -> List[Dict]:
    """
    Extract the range of categories from the json text
    """
    categories = json.loads(json_text)['applicable_categories']

    if debug:
        with open("debug/categories.txt", "w") as f:
            f.write(str(categories))

    return categories

def get_matching_codes(categories: List[Dict]) -> List[Dict]:
    """
    Get the matching codes for the categories
    """
    df_slo = pd.read_csv(SECTIONS_FILE)
    df_eng = pd.read_csv(ENGLISH_SECTIONS_FILE)

    matching_results_slo = find_matching_codes(categories, df_slo)
    matching_results_hierarchical = find_matching_codes_hierarchical(categories, df_eng)
    return matching_results_slo, matching_results_hierarchical

def get_all_queries(matching_results_slo: List[Dict], matching_results_hierarchical: List[Dict]) -> List[str]:
    """
    Get all queries from the matching results
    """
    grouped_slo = group_results_by_query(matching_results_slo)
    grouped_hierarchical = group_results_by_query(matching_results_hierarchical)
    all_queries = sorted(set(list(grouped_slo.keys()) + list(grouped_hierarchical.keys())))
    return grouped_slo, grouped_hierarchical, all_queries

def get_first_level_codes(all_queries: List[str], grouped_slo: Dict, grouped_hierarchical: Dict) -> List[Dict]:
    """
    Get the first level codes from the matching results
    """
    json_output = {}
    for query in all_queries:
        slo = grouped_slo.get(query, [])
        hier = grouped_hierarchical.get(query, [])
        hier = [code for code in hier if code['level'] == 3]

        json_output[query] = {
            "original_matches": slo,
            "hierarchical_matches": hier
        }

    output_filename = "debug/mkb10_matches.json"

    with open(output_filename, "w", encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"\n{Fore.CYAN}Initial matches saved to {output_filename}{Style.RESET_ALL}")

    all_categories, descriptions_lookup = analyse_first_level_codes(json_output)
    return all_categories, descriptions_lookup

def get_second_level_filtered_codes(all_queries: List[str], grouped_slo: Dict, grouped_hierarchical: Dict, codes: List[str]) -> List[Dict]:
    """
    Get the second level codes from the matching results
    """
    json_output = {}
    for query in all_queries:
        slo = grouped_slo.get(query, [])
        hier = grouped_hierarchical.get(query, [])
        hier = [code for code in hier if code['level'] >= 3 and code['category'] in codes]

        json_output[query] = {
            "original_matches": slo,
            "hierarchical_matches": hier
        }

    output_filename = "debug/mkb10_matches.json"

    with open(output_filename, "w", encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    print(f"\n{Fore.CYAN}Initial matches saved to {output_filename}{Style.RESET_ALL}")

    all_categories, descriptions_lookup = analyse_first_level_codes(json_output)
    return all_categories, descriptions_lookup


def analyse_first_level_codes(json_output: Dict) -> Tuple[List[Dict], Dict]: 
    all_categories = []
    descriptions_lookup = {}

    for query, matches in json_output.items():
        category_codes = []
        for match_type in ['original_matches', 'hierarchical_matches']:
            print(f"Processing {match_type} for query: {query}")
            for match in matches[match_type]:
                code = match['code']
                descriptions_lookup[code] = {
                    'slo_description': match['description'],
                    'eng_description': match.get('english_description', '')
                }
                code_info = {
                    'code': code,
                    'slo_description': match['description'],
                    'eng_description': match.get('english_description', ''),
                    'matched_query': query
                }
                category_codes.append(code_info)
        if category_codes:
            all_categories.append({'matched_query': query, 'codes': category_codes})

    logger.info(f"{Fore.CYAN}Starting parallel processing with {len(all_categories)} categories{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    return all_categories, descriptions_lookup

def extract_codes_from_results(results: List[Dict], debug: bool = True, file_name: str = "final_codes.json") -> List[Dict]:
    """
    Extract the codes from the results
    """
    all_final_codes = []
    for result in results:
        if 'error' not in result:
            all_final_codes.extend(result['codes'])

    if debug:
        final_output = {
            "final_codes": all_final_codes
        }

        with open(f"debug/{file_name}", "w", encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

    codes = [code_info['code'] for code_info in all_final_codes]
    print('codes', codes)

    logger.info(f"{Fore.GREEN}Successfully processed {len(all_final_codes)} codes across all categories{Style.RESET_ALL}")
    
    category_grouped_codes = defaultdict(list)
    for code_info in all_final_codes:
        category_grouped_codes[code_info['category_group']].append(code_info)


    print(f"\n{Fore.CYAN}Final Low-Level MKB-10 Codes by Category:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    return codes, category_grouped_codes

def get_categories_from_diagnosis(diagnosis: str, num_first_level_recommendations: int = 3, num_second_level_recommendations: int = 6) -> List[Dict]:
    """
    Get the categories from the diagnosis
    """

    reasoning = reasoning_to_categories(diagnosis, available_categories)
    
    extracted_categories = extract_categories_ranges(reasoning)

    categories = get_categories_from_json(extracted_categories)
    
    matching_results_slo, matching_results_hierarchical = get_matching_codes(categories)

    grouped_slo, grouped_hierarchical, all_queries = get_all_queries(matching_results_slo, matching_results_hierarchical)

    all_categories, descriptions_lookup = get_first_level_codes(all_queries, grouped_slo, grouped_hierarchical)

    first_level_results = process_categories_parallel(all_categories, diagnosis, descriptions_lookup, num_first_level_recommendations)

    first_level_codes, first_level_category_grouped_codes = extract_codes_from_results(first_level_results, file_name="first_level_codes.json")

    for category, codes in sorted(first_level_category_grouped_codes.items()):
        print(f"\n{Fore.YELLOW}Intermediate Category {category}:{Style.RESET_ALL}")
        for code_info in codes:
            print(f"  → {Fore.GREEN}{code_info['code']}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
            print(f"    {Fore.CYAN}English:{Style.RESET_ALL} {code_info['eng_description']}")
            print(f"    {Fore.CYAN}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

    second_level_categories, second_level_descriptions_lookup = get_second_level_filtered_codes(all_queries, grouped_slo, grouped_hierarchical, first_level_codes)

    second_level_results = process_categories_parallel(second_level_categories, diagnosis, second_level_descriptions_lookup, num_second_level_recommendations)
    _, second_level_category_grouped_codes = extract_codes_from_results(second_level_results, file_name="final_codes.json")

    results = []
    for category, codes in sorted(second_level_category_grouped_codes.items()):
        print(f"\n{Fore.GREEN}Final Category {category}:{Style.RESET_ALL}")
        for code_info in codes:
            print(f"  → {Fore.YELLOW}{code_info['code']}{Style.RESET_ALL}")
            print(f"    {Fore.LIGHTMAGENTA_EX}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
            print(f"    {Fore.LIGHTMAGENTA_EX}English:{Style.RESET_ALL} {code_info['eng_description']}")
            print(f"    {Fore.LIGHTMAGENTA_EX}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
            print()
            results.append({
                'category': category,
                'code': code_info['code'],
                'slo_description': code_info['slo_description'],
                'eng_description': code_info['eng_description'],
                'rationale': code_info['rationale']
            })
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

    return results


if __name__ == "__main__":
    with open("example_diagnosis.txt", "r") as f:
        diagnosis = f.read()

    categories = get_categories_from_diagnosis(diagnosis)
    # print(categories)
