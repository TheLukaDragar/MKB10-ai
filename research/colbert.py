import pandas as pd
from pylate import indexes, models, retrieve
from src.params import SECTIONS_FILE
# Initialize the ColBERT model
model = models.ColBERT(
    model_name_or_path="jinaai/jina-colbert-v2",
    query_prefix="[QueryMarker]",
    document_prefix="[DocumentMarker]",
    attend_to_expansion_tokens=True,
    trust_remote_code=True,
    device="mps"
)

sklopi_slo = pd.read_csv(SECTIONS_FILE)

documents = [f"{row['SKLOP']}: {row['SLOVENSKI NAZIV']}" for _, row in sklopi_slo.iterrows()]
document_ids = [str(i) for i in range(len(documents))]

index = indexes.Voyager(
    index_folder="pylate-index",
    index_name="medical-classifications",
    override=True,
)

print("Encoding documents...")
document_embeddings = model.encode(
    documents,
    batch_size=128,
    is_query=False,
    show_progress_bar=True,
)

print("Adding documents to index...")
index.add_documents(
    documents_ids=document_ids,
    documents_embeddings=document_embeddings,
)

retriever = retrieve.ColBERT(index=index)

def get_relevant_docs(query, k=3):
    """Retrieve the most relevant documents for a query."""
    # Encode the query
    query_embedding = model.encode(
        [query],
        batch_size=1,
        is_query=True,
        show_progress_bar=False,
    )
    
    # Retrieve top-k documents
    results = retriever.retrieve(
        queries_embeddings=query_embedding,
        k=k,
    )[0]  # Get first (and only) query results
    
    # Return documents with their scores
    return [(documents[int(res["id"])], res["score"]) for res in results]

# Example usage
if __name__ == "__main__":

    diagnosis = "Anamneza Včeraj je panel s kolesa in se udaril po desni dlani, levi rami, levi nadlahti, levi podlahti in levi dlani ter levem kolenu. Vročine in mrzlice ni imel. Antitetanična zaščita obstaja. Status ob sprejemu Vidne številne odrgnine v prelu desne dlani in po vseh prstih te roke. Največja rana v predelu desnega zapestja, okolica je blago pordela. Gibljvost v zapestju je popolnoma ohranjena. Brez NC izpadov. Na levi rami vidna odrgnina, prav tako tudi odrgnine brez znakov vnetja v področju leve nadlahti, leve podlahti in leve dlani. Dve večji odrgnini v predelu levega kolena. Levo koleno je blago otečeno. Ballottement negativen. Gibljivost v kolenu 0/90. Iztipam sklepno špranjo kolena, ki palpatorno ni občutljiva. Lachman in predalčni fenomen enaka v primerjavi z nepoškodovanim kolenom. Kolateralni ligamenti delujejo čvrsti. MCL nekoliko boleč na nateg in palpatorno. Diagnostični postopki RTG desno zapestje: brez prepričljivih znakov sveže poškodbe skeleta desna dlan: brez prepričljivih znakov sveže poškodbe skeleta levo koleno: brez prepričljivih znakov sveže poškodbe skeleta."
    diagnosis2 = """
Bolnik je imel vročinsko bolezen, generaliziran hud makularni izpuščaj in slabo počutje.
Zdravili so jo zaradi sindroma stafilokoknega toksičnega šoka z IV flukloksacilinom. 
Končna diagnoza: sindrom toksičnega šoka.
"""
    example_queries = [
       diagnosis,
       *[sentence.strip() for sentence in diagnosis2.split(".") if sentence.strip()]
    ]
    print("\nTesting retrieval:")
    for query in example_queries:
        print(f"\nQuery: {query}")
        results = get_relevant_docs(query,k=10)
        print("Top relevant documents:")
        for doc, score in results:
            print(f"Score {score:.2f}: {doc}")


#         S40.81 Excoratio omae et brachii sin.
# V18.0 Kolesar, poškodovan v transportni nezgodi brez trčenja, voznik, neprometna nezgoda
# S60.7 Excoriatio antebrachii et manus sin.
# Excoriatio manus dex.
# S80.0 Excoriatio genus sin.