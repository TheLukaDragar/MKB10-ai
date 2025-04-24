import os
from typing import List, Dict, Tuple
import openai
import pandas as pd
from prompts import *
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

import pandas as pd
from pylate import indexes, models, retrieve
from params import SECTIONS_FILE, ENGLISH_SECTIONS_FILE
from embedding_search import searcher

# # Initialize the ColBERT model
# model = models.ColBERT(
#     model_name_or_path="jinaai/jina-colbert-v2",
#     query_prefix="[QueryMarker]",
#     document_prefix="[DocumentMarker]",
#     attend_to_expansion_tokens=True,
#     trust_remote_code=True,
#     device="mps"
# )


# all_codes = pd.read_csv(ENGLISH_SECTIONS_FILE)

# documents = [f"{row['SLOVENSKI NAZIV']}" for _, row in all_codes.iterrows()]
# document_ids = [str(i) for i in range(len(documents))]
# codes = [row['KODA'] for _, row in all_codes.iterrows()]

# mapping = dict(zip(document_ids, codes))

# index = indexes.Voyager(
#     index_folder="pylate-index",
#     index_name="medical-classifications-with-llm",
#     #override=True,  # This will override any existing index
# )


# print("Encoding documents...")
# document_embeddings = model.encode(
# documents,
# batch_size=128,
# is_query=False,
# show_progress_bar=True,
# )

# print("Adding documents to index...")
# index.add_documents(
#     documents_ids=document_ids,
#     documents_embeddings=document_embeddings,
#     )


# retriever = retrieve.ColBERT(index=index)

# def get_relevant_docs(query, k=3):
#     """Retrieve the most relevant documents for a query."""
#     # Encode the query
#     query_embedding = model.encode(
#         [query],
#         batch_size=1,
#         is_query=True,
#         show_progress_bar=False,
#     )
    
#     results = retriever.retrieve(
#         queries_embeddings=query_embedding,
#         k=k,
#     )[0]  # Get first (and only) query results

#     return [(documents[int(res["id"])], codes[int(res["id"])], res["score"] ) for res in results]

# Example 

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


async def process_category_async(category: Dict, diagnosis: str, descriptions_lookup: Dict) -> Dict:
    """
    Process a single category asynchronously
    """
    query = category['matched_query']
    logger.info(f"{Fore.CYAN}Starting processing of category: {query}{Style.RESET_ALL}")
    
    try:
        # Get indices for valid codes in this category
        valid_indices = []
        df = pd.read_csv(ENGLISH_SECTIONS_FILE)
        codes_list = df['KODA'].tolist()
        for code_info in category['codes']:
            try:
                idx = codes_list.index(code_info['code'])
                valid_indices.append(idx)
            except ValueError:
                continue

        # First get embedding search results for this category
        logger.info(f"{Fore.BLUE}[{query}] Step 1: Running embedding search...{Style.RESET_ALL}")
        embedding_results = searcher.search_in_category(diagnosis, valid_indices, query)
        
        if embedding_results:
            logger.info(f"{Fore.GREEN}[{query}] Found {len(embedding_results)} relevant matches via embedding search{Style.RESET_ALL}")
            for doc, code, score in embedding_results:
                logger.info(f"{Fore.GREEN}[{query}] → {code} (score: {score:.2f}): {doc}{Style.RESET_ALL}")
        
        # Get reasoning for this category
        logger.info(f"{Fore.BLUE}[{query}] Step 2: Analyzing codes...{Style.RESET_ALL}")
        
        # Include embedding results in the reasoning prompt
        # embedding_context = ""
        # if embedding_results:
        #     embedding_context = "\n\nEmbedding search found these relevant matches:\n" + \
        #         "\n".join([f"- {code} ({score:.2f}): {doc}" for doc, code, score in embedding_results])
        
        reasoning_result = await _async_llm_call(
            reason_specific_codes_prompt.format(
                diagnosis=diagnosis,
                matched_codes=json.dumps(category['codes'], ensure_ascii=False, indent=2)
            ) #+ embedding_context
        )
        
        # Then get specific codes with the reasoning context
        logger.info(f"{Fore.BLUE}[{query}] Step 3: Extracting specific codes...{Style.RESET_ALL}")
        specific_codes = await _async_llm_call(
            extract_specific_codes_prompt.format(
                diagnosis=diagnosis,
                matched_codes=json.dumps(category['codes'], ensure_ascii=False, indent=2)
            ) + f"\n\nPrevious analysis:\n{reasoning_result}",
            extra_body={"guided_json": SpecificCodesResponse.for_category(query).model_json_schema()},
            temperature=0.0
        )
        
        # Parse results and add descriptions
        category_final_codes = json.loads(specific_codes)
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
                'reasoning': reasoning_result,
                'embedding_results': embedding_results
            }
    except Exception as e:
        logger.error(f"{Fore.RED}Error processing category {query}: {e}{Style.RESET_ALL}")
        return {
            'category': query,
            'codes': [],
            'error': str(e)
        }

async def process_all_categories_async(categories: List[Dict], diagnosis: str, descriptions_lookup: Dict) -> List[Dict]:
    """
    Process all categories in parallel
    """
    logger.info(f"{Fore.CYAN}Starting parallel processing of {len(categories)} categories{Style.RESET_ALL}")
    tasks = [process_category_async(category, diagnosis, descriptions_lookup) for category in categories]
    return await asyncio.gather(*tasks)

def process_categories_parallel(categories: List[Dict], diagnosis: str, descriptions_lookup: Dict) -> List[Dict]:
    """
    Synchronous wrapper for parallel category processing
    """
    return asyncio.run(process_all_categories_async(categories, diagnosis, descriptions_lookup))

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
    letters = set(category[0] for category in categories)

    if debug:
        with open("debug/categories.txt", "w") as f:
            f.write(str(categories))
        with open("debug/letters.txt", "w") as f:
            f.write(str(letters))

    return categories, letters

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

    print(f"\n{Fore.CYAN}Final MKB-10 Codes by Category with Embedding Search Results:{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    # Print embedding results and LLM results together for each category
    for result in results:
        if 'error' in result or not result.get('codes'):
            continue
            
        category = result['category']
        print(f"\n{Fore.YELLOW}Category {category}:{Style.RESET_ALL}")
        
        # Print embedding search results first
        if result.get('embedding_results'):
            print(f"\n{Fore.CYAN}Embedding Search Suggestions:{Style.RESET_ALL}")
            for doc, code, score in result['embedding_results']:
                print(f"  → {Fore.GREEN}{code}{Style.RESET_ALL} (score: {score:.2f})")
                print(f"    {doc}")
        
        # Print LLM-selected codes
        print(f"\n{Fore.CYAN}LLM Selected & Justified Codes:{Style.RESET_ALL}")
        for code_info in category_grouped_codes[category]:
            print(f"  → {Fore.GREEN}{code_info['code']}{Style.RESET_ALL}")
            print(f"    {Fore.MAGENTA}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
            if code_info['eng_description'] and code_info['eng_description'] != 'nan':
                print(f"    {Fore.MAGENTA}English:{Style.RESET_ALL} {code_info['eng_description']}")
            print(f"    {Fore.MAGENTA}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
            print()
        
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

    return codes, category_grouped_codes

if __name__ == "__main__":
    # Initialize the embedding searcher
    print(f"{Fore.CYAN}Initializing embedding search...{Style.RESET_ALL}")
    df_eng = pd.read_csv(ENGLISH_SECTIONS_FILE)
    searcher.initialize_with_dataframe(df_eng)
    print(f"{Fore.GREEN}Embedding search initialized{Style.RESET_ALL}")
    print(f"{Fore.BLUE}{'=' * 80}{Style.RESET_ALL}")

    diagnosis = '''Anamneza:
        Včeraj je padel s kolesa in se udaril po desni dlani, levi rami, levi nadlahti, levi podlahti in levi
        dlani ter levem kolenu. Vročine in mrzlice ni imel. Antitetanična zaščita obstaja.
        Status ob sprejemu:
        Vidne številne odrgnine v predelu desne dlani in po vseh prstih te roke. Največja rana v
        predelu desnega zapestja, okolica je blago pordela. Gibljvost v zapestju je popolnoma
        ohranjena. Brez NC izpadov.
        Na levi rami vidna odrgnina, prav tako tudi odrgnine brez znakov vnetja v področju leve
        nadlahti, leve podlahti in leve dlani.
        Dve večji odrgnini v predelu levega kolena. Levo koleno je blago otečeno. Ballottement
        negativen. Gibljivost v kolenu 0/90. Iztipam sklepno špranjo kolena, ki palpatorno ni občutljiva.
        Lachman in predalčni fenomen enaka v primerjavi z nepoškodovanim kolenom. Kolateralni
        ligamenti delujejo čvrsti. MCL nekoliko boleč na nateg in palpatorno.
        Diagnostični postopki:
        RTG
        desno zapestje: brez prepričljivih znakov sveže poškodbe skeleta
        desna dlan: brez prepričljivih znakov sveže poškodbe skeleta
        levo koleno: brez prepričljivih znakov sveže poškodbe skeleta
        Oskrba:
        Toaleta in preveza rane v ambulanti.
        V MOP:
        Toaleta odrgnin, Inadine na vnete odrgnine, Cuticell na ostale odrgnine.
        Preveza čez dva dni pri osebnem zdravniku, nato po njegovi presoji do zacelitve.
        Hlajenje z ledom preko tkanine večkrat dnevno, počitek, analgetiki.
        Dobi e-recept za Amoksiklav za 5 dni in Lekadol.
        '''

    # diagnosis = "Anamneza: Včeraj je padel s kolesa in se udaril po desni dlani, levi rami, levi nadlahti, levi podlahti in levi dlani ter levem kolenu. Vročine in mrzlice ni imel. Antitetanična zaščita obstaja. \nStatus ob sprejemu: Vidne številne odrgnine v prelu desne dlani in po vseh prstih te roke. Največja rana v predelu desnega zapestja, okolica je blago pordela. Gibljvost v zapestju je popolnoma ohranjena. Brez NC izpadov. Na levi rami vidna odrgnina, prav tako tudi odrgnine brez znakov vnetja v področju leve nadlahti, leve podlahti in leve dlani. Dve večji odrgnini v predelu levega kolena. Levo koleno je blago otečeno. Ballottement negativen. Gibljivost v kolenu 0/90. Iztipam sklepno špranjo kolena, ki palpatorno ni občutljiva. Lachman in predalčni fenomen enaka v primerjavi z nepoškodovanim kolenom. Kolateralni ligamenti delujejo čvrsti. MCL nekoliko boleč na nateg in palpatorno. Diagnostični postopki RTG desno zapestje: brez prepričljivih znakov sveže poškodbe skeleta desna dlan: brez prepričljivih znakov sveže poškodbe skeleta levo koleno: brez prepričljivih znakov sveže poškodbe skeleta."

    reasoning = reasoning_to_categories(diagnosis, available_categories)



    extracted_categories = extract_categories_ranges(reasoning)

    categories, letters = get_categories_from_json(extracted_categories)
    
    matching_results_slo, matching_results_hierarchical = get_matching_codes(categories)

    grouped_slo, grouped_hierarchical, all_queries = get_all_queries(matching_results_slo, matching_results_hierarchical)

    all_categories, descriptions_lookup = get_first_level_codes(all_queries, grouped_slo, grouped_hierarchical)

    first_level_results = process_categories_parallel(all_categories, diagnosis, descriptions_lookup)

    first_level_codes, first_level_category_grouped_codes = extract_codes_from_results(first_level_results, file_name="first_level_codes.json")


    for category, codes in sorted(first_level_category_grouped_codes.items()):
        print(f"\n{Fore.YELLOW}Intermediate Category {category}:{Style.RESET_ALL}")
        for code_info in codes:
            print(f"  → {Fore.GREEN}{code_info['code']}{Style.RESET_ALL}")
            print(f"    {Fore.CYAN}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
            print(f"    {Fore.CYAN}English:{Style.RESET_ALL} {code_info['eng_description']}")
            print(f"    {Fore.CYAN}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
            print()
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")

    second_level_categories, second_level_descriptions_lookup = get_second_level_filtered_codes(all_queries, grouped_slo, grouped_hierarchical, first_level_codes)

    second_level_results = process_categories_parallel(second_level_categories, diagnosis, second_level_descriptions_lookup)
    second_level_codes, second_level_category_grouped_codes = extract_codes_from_results(second_level_results, file_name="final_codes.json")

    for category, codes in sorted(second_level_category_grouped_codes.items()):
        print(f"\n{Fore.GREEN}Final Category {category}:{Style.RESET_ALL}")
        for code_info in codes:
            print(f"  → {Fore.YELLOW}{code_info['code']}{Style.RESET_ALL}")
            print(f"    {Fore.LIGHTMAGENTA_EX}Slovenski naziv:{Style.RESET_ALL} {code_info['slo_description']}")
            print(f"    {Fore.LIGHTMAGENTA_EX}English:{Style.RESET_ALL} {code_info['eng_description']}")
            print(f"    {Fore.LIGHTMAGENTA_EX}Rationale:{Style.RESET_ALL} {code_info['rationale']}")
            print()
        print(f"{Fore.BLUE}{'-' * 80}{Style.RESET_ALL}")
