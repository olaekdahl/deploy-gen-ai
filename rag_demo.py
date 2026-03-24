"""
RAG Demo: Parametric Knowledge vs Retrieved Knowledge
=====================================================

Demonstrates the difference between:
1. Knowledge stored in model weights (parametric)
2. Knowledge retrieved at runtime via RAG (non-parametric)

Domain: Chocolate cake recipe
Model: GPT-5.2 (via OpenAI API)
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"  # Use gpt-4o as proxy for GPT-5.2; replace with actual model name
TOP_K = 2  # Number of chunks to retrieve

# Initialize OpenAI client
client = OpenAI()  # Uses OPENAI_API_KEY environment variable


# ============================================================================
# DATA: VECTOR STORE DOCUMENTS
# ============================================================================

@dataclass
class Document:
    """A document chunk with metadata."""
    content: str
    metadata: dict
    embedding: np.ndarray = None


# The specific chocolate cake recipe (stored as complete chunks for reliable retrieval)
# NOTE: In production RAG, chunk size matters! Too small = incomplete context, too large = noise
CHOCOLATE_CAKE_RECIPE_CHUNKS = [
    {
        "chunk_id": 1,
        "content": """GRANDMA'S CHOCOLATE CAKE RECIPE (from 1952 Edition)

INGREDIENTS:
- 2 cups all-purpose flour
- 2 cups granulated sugar  
- 3/4 cup Dutch-process cocoa powder
- 2 tsp baking soda
- 1 tsp baking powder
- 1 tsp salt
- 2 eggs
- 1 cup buttermilk
- 1 cup hot black coffee (secret ingredient!)
- 1/2 cup vegetable oil

INSTRUCTIONS:
1. Preheat oven to 350°F (175°C). Grease and flour two 9-inch round cake pans.
2. In a large bowl, whisk together flour, sugar, Dutch-process cocoa, baking soda, baking powder, and salt.
3. Add eggs, buttermilk, hot black coffee, and vegetable oil. Beat with mixer for 2 minutes. Batter will be thin - this is normal!
4. Pour batter evenly into prepared pans.
5. Bake for 30-35 minutes until toothpick inserted in center comes out clean.
6. Cool in pans for 10 minutes, then remove to wire racks to cool completely.""",
        "metadata": {
            "recipe_id": "choc-cake-001",
            "source": "Grandma's Secret Recipes, 1952 Edition",
            "section": "cake"
        }
    },
    {
        "chunk_id": 2,
        "content": """GRANDMA'S CHOCOLATE FROSTING (pairs with 1952 Chocolate Cake)

FROSTING INGREDIENTS:
- 1 cup (2 sticks) softened butter
- 4 cups powdered sugar
- 1/2 cup Dutch-process cocoa powder
- 1/4 cup heavy cream
- 2 tsp vanilla extract

FROSTING INSTRUCTIONS:
1. Beat softened butter until creamy (about 2 minutes).
2. Gradually add powdered sugar and cocoa, alternating with heavy cream.
3. Add vanilla extract.
4. Beat on high for 3-4 minutes until light and fluffy.
5. Frost cooled cake layers, starting with a layer between the cakes, then covering top and sides.""",
        "metadata": {
            "recipe_id": "choc-cake-001",
            "source": "Grandma's Secret Recipes, 1952 Edition",
            "section": "frosting"
        }
    }
]

# Other documents (distractors and partially related content)
OTHER_DOCUMENTS = [
    {
        "content": "Vanilla sponge cake requires 3 eggs, 1 cup sugar, 1 cup flour, and 1 tsp vanilla extract. Beat eggs and sugar until pale and fluffy.",
        "metadata": {"recipe_id": "vanilla-sponge-001", "source": "Basic Baking Guide"}
    },
    {
        "content": "The history of chocolate dates back to ancient Mesoamerican civilizations. The Aztecs consumed chocolate as a bitter drink called xocolatl.",
        "metadata": {"type": "history", "source": "Food History Encyclopedia"}
    },
    {
        "content": "Proper oven calibration is essential for baking. An oven thermometer can help verify that your oven reaches the correct temperature.",
        "metadata": {"type": "baking-tips", "source": "Professional Baker's Handbook"}
    },
    {
        "content": "Cocoa powder comes in two varieties: natural and Dutch-process. Dutch-process cocoa is treated with alkali to neutralize acidity, resulting in a darker color and milder flavor.",
        "metadata": {"type": "ingredient-info", "source": "Ingredient Science Guide"}
    },
    {
        "content": "Carrot cake originated in Europe during the Middle Ages when sugar was expensive. Carrots provided natural sweetness to baked goods.",
        "metadata": {"recipe_id": "carrot-cake-history", "source": "Cake Origins"}
    },
    {
        "content": "To test if a cake is done, insert a toothpick into the center. If it comes out clean or with a few moist crumbs, the cake is ready.",
        "metadata": {"type": "baking-tips", "source": "Professional Baker's Handbook"}
    }
]


# ============================================================================
# COMPONENT 1: EMBEDDING FUNCTION
# ============================================================================

def embed(texts: List[str]) -> np.ndarray:
    """
    Generate embeddings for a list of texts using OpenAI's embedding model.
    
    Args:
        texts: List of strings to embed
        
    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)


# ============================================================================
# COMPONENT 2: VECTOR STORE
# ============================================================================

class SimpleVectorStore:
    """
    A minimal in-memory vector store for demonstration.
    In production, use Pinecone, Weaviate, ChromaDB, etc.
    """
    
    def __init__(self):
        self.documents: List[Document] = []
    
    def add_documents(self, documents: List[Document]):
        """Add documents with their embeddings to the store."""
        # Get embeddings for all documents at once
        texts = [doc.content for doc in documents]
        embeddings = embed(texts)
        
        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb
            self.documents.append(doc)
        
        print(f"Added {len(documents)} documents to vector store")
    
    def similarity_search(self, query: str, k: int = TOP_K) -> List[Tuple[Document, float]]:
        """
        Find the k most similar documents to the query.
        Uses cosine similarity.
        
        Returns:
            List of (document, similarity_score) tuples
        """
        # Embed the query
        query_embedding = embed([query])[0]
        
        # Calculate cosine similarity with all documents
        results = []
        for doc in self.documents:
            similarity = np.dot(query_embedding, doc.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc.embedding)
            )
            results.append((doc, similarity))
        
        # Sort by similarity (descending) and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


# ============================================================================
# COMPONENT 3: RETRIEVER
# ============================================================================

def retrieve(vector_store: SimpleVectorStore, query: str, k: int = TOP_K) -> List[Document]:
    """
    Retrieve the most relevant documents for a query.
    
    Args:
        vector_store: The vector store to search
        query: User's question
        k: Number of documents to retrieve
        
    Returns:
        List of relevant documents
    """
    print(f"\n{'='*60}")
    print("RETRIEVAL")
    print(f"{'='*60}")
    print(f"Query: {query}")
    print(f"Retrieving top {k} chunks...\n")
    
    results = vector_store.similarity_search(query, k)
    
    print("Retrieved chunks:")
    print("-" * 40)
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n[Chunk {i}] Similarity: {score:.4f}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.content[:100]}...")
    print("-" * 40)
    
    return [doc for doc, _ in results]


# ============================================================================
# COMPONENT 4: PROMPT ASSEMBLY
# ============================================================================

def build_prompt(query: str, context_docs: List[Document] = None) -> Tuple[str, str]:
    """
    Build the system and user prompts.
    
    Args:
        query: User's question
        context_docs: Retrieved documents (None for non-RAG)
        
    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    if context_docs:
        # RAG prompt: includes retrieved context
        context_text = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.content}"
            for doc in context_docs
        ])
        
        system_prompt = """You are a helpful cooking assistant. 

IMPORTANT INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the RETRIEVED CONTEXT below.
2. If the context contains a specific recipe, provide those exact steps.
3. Do not add steps or ingredients that are not in the retrieved context.
4. If the context doesn't contain enough information, say so.

RETRIEVED CONTEXT:
---
{context}
---

Use the above context to answer the user's question accurately.""".format(context=context_text)
        
        user_prompt = query
        
    else:
        # Non-RAG prompt: no context, relies on parametric knowledge
        system_prompt = """You are a helpful cooking assistant.
Answer the user's question based on your general knowledge.
If asked for a recipe, provide a reasonable general recipe."""
        
        user_prompt = query
    
    return system_prompt, user_prompt


# ============================================================================
# COMPONENT 5: GENERATION
# ============================================================================

def generate(system_prompt: str, user_prompt: str, show_prompt: bool = True) -> str:
    """
    Generate a response using the LLM.
    
    Args:
        system_prompt: Instructions for the model
        user_prompt: User's question
        show_prompt: Whether to print the full prompt
        
    Returns:
        Model's response text
    """
    if show_prompt:
        print(f"\n{'='*60}")
        print("PROMPT SENT TO MODEL")
        print(f"{'='*60}")
        print(f"\n[SYSTEM PROMPT]\n{system_prompt[:500]}...")
        print(f"\n[USER PROMPT]\n{user_prompt}")
        print(f"{'='*60}")
    
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,  # Lower temperature for more consistent output
        max_tokens=1000
    )
    
    return response.choices[0].message.content


# ============================================================================
# MAIN DEMO
# ============================================================================

def prepare_vector_store() -> SimpleVectorStore:
    """Initialize and populate the vector store."""
    print(f"\n{'='*60}")
    print("INITIALIZING VECTOR STORE")
    print(f"{'='*60}")
    
    store = SimpleVectorStore()
    
    # Create documents from the chocolate cake recipe (complete chunks)
    recipe_docs = [
        Document(content=chunk["content"], metadata=chunk["metadata"])
        for chunk in CHOCOLATE_CAKE_RECIPE_CHUNKS
    ]
    
    # Create documents from other content (distractors)
    other_docs = [
        Document(content=item["content"], metadata=item["metadata"])
        for item in OTHER_DOCUMENTS
    ]
    
    # Add all documents to the store
    all_docs = recipe_docs + other_docs
    store.add_documents(all_docs)
    
    print(f"\nVector store contents:")
    print(f"  - Chocolate cake recipe: {len(recipe_docs)} chunks (complete recipe)")
    print(f"  - Other documents: {len(other_docs)} chunks (distractors)")
    print(f"  - Total: {len(all_docs)} chunks")
    
    return store


def run_demo():
    """Run the full RAG demonstration."""
    print("\n" + "=" * 70)
    print("   RAG DEMO: Parametric Knowledge vs Retrieved Knowledge")
    print("=" * 70)
    
    # Step 1: Prepare vector store
    vector_store = prepare_vector_store()
    
    # The user's question
    query = "How do I bake a chocolate cake?"
    
    # =========================================================================
    # GENERATION A: Without RAG (Parametric Knowledge Only)
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("   GENERATION A: WITHOUT RAG (Parametric Knowledge)")
    print("=" * 70)
    print("\nThe model will answer using only its trained weights.")
    print("It has general baking knowledge but NOT this specific recipe.\n")
    
    system_prompt_no_rag, user_prompt = build_prompt(query, context_docs=None)
    response_no_rag = generate(system_prompt_no_rag, user_prompt)
    
    print(f"\n{'='*60}")
    print("MODEL OUTPUT (No RAG)")
    print(f"{'='*60}")
    print(response_no_rag)
    
    # =========================================================================
    # GENERATION B: With RAG (Retrieved Knowledge)
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("   GENERATION B: WITH RAG (Retrieved Knowledge)")
    print("=" * 70)
    print("\nThe model will answer using retrieved context from the vector store.")
    print("The specific recipe is injected at runtime, not from weights.\n")
    
    # Retrieve relevant documents
    retrieved_docs = retrieve(vector_store, query, k=TOP_K)
    
    # Build prompt with context
    system_prompt_rag, user_prompt = build_prompt(query, context_docs=retrieved_docs)
    response_rag = generate(system_prompt_rag, user_prompt)
    
    print(f"\n{'='*60}")
    print("MODEL OUTPUT (With RAG)")
    print(f"{'='*60}")
    print(response_rag)
    
    # =========================================================================
    # GENERATION C: WHY THE LLM MATTERS - Reasoning Over Retrieved Content
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("   GENERATION C: WHY THE LLM MATTERS")
    print("   (Reasoning, Synthesis, and Transformation)")
    print("=" * 70)
    print("""
A database lookup just returns raw text. The LLM adds value by:
  1. REASONING over the retrieved content
  2. SYNTHESIZING information from multiple chunks  
  3. TRANSFORMING the format to match user needs
  4. ANSWERING QUESTIONS that require understanding

Let's demonstrate with questions that require more than retrieval:
""")
    
    # Questions that demonstrate LLM reasoning over retrieved content
    reasoning_questions = [
        {
            "query": "I'm vegan - can I adapt this chocolate cake recipe? What substitutions would work?",
            "why": "Requires understanding ingredients AND applying external knowledge about vegan substitutions"
        },
        {
            "query": "I don't have buttermilk. What's a good substitute based on the other ingredients in this recipe?",
            "why": "Requires reading the recipe AND reasoning about ingredient chemistry"
        },
        {
            "query": "How long will this chocolate cake take from start to finish, including cooling time?",
            "why": "Requires extracting times from multiple steps AND calculating total"
        },
        {
            "query": "What's the most unusual ingredient in this recipe and why is it used?",
            "why": "Requires identifying 'unusual' (hot coffee) AND explaining its purpose"
        }
    ]
    
    for i, q in enumerate(reasoning_questions, 1):
        print(f"\n{'─'*60}")
        print(f"REASONING QUESTION {i}: {q['query']}")
        print(f"WHY LLM NEEDED: {q['why']}")
        print(f"{'─'*60}")
        
        # Retrieve (same chunks, different question)
        retrieved = retrieve(vector_store, q['query'], k=TOP_K)
        
        # Generate with RAG
        sys_prompt, usr_prompt = build_prompt(q['query'], context_docs=retrieved)
        response = generate(sys_prompt, usr_prompt, show_prompt=False)
        
        print(f"\nLLM RESPONSE:\n{response}")
    
    # =========================================================================
    # COMPARISON: Database vs RAG+LLM
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("   DATABASE LOOKUP vs RAG+LLM")
    print("=" * 70)
    print("""
┌────────────────────────────────────────────────────────────────────────┐
│                     PLAIN DATABASE LOOKUP                              │
├────────────────────────────────────────────────────────────────────────┤
│ Query: "chocolate cake recipe"                                         │
│ Result: Returns raw text blob of recipe                                │
│                                                                        │
│ Query: "Can I make it vegan?"                                          │
│ Result: ❌ No match found (or returns unrelated vegan recipes)         │
│                                                                        │
│ Query: "How long total?"                                               │
│ Result: ❌ Cannot calculate - just returns text containing times       │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                         RAG + LLM                                      │
├────────────────────────────────────────────────────────────────────────┤
│ Query: "chocolate cake recipe"                                         │
│ Result: ✅ Formatted recipe with clear steps                           │
│                                                                        │
│ Query: "Can I make it vegan?"                                          │
│ Result: ✅ Analyzes ingredients, suggests: butter→coconut oil,         │
│            eggs→flax eggs, buttermilk→oat milk+vinegar                 │
│                                                                        │
│ Query: "How long total?"                                               │
│ Result: ✅ Calculates: prep(15) + bake(35) + cool(30) + frost(10)      │
│            = ~90 minutes total                                         │
└────────────────────────────────────────────────────────────────────────┘

THE LLM PROVIDES:
  • Comprehension - Understands what the recipe means
  • Reasoning     - Can answer questions ABOUT the retrieved content  
  • Synthesis     - Combines info from multiple chunks
  • Transformation- Reformats for user needs (simplify, translate, etc.)
  • Conversation  - Handles follow-ups with context
""")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    print("\n\n" + "=" * 70)
    print("   ORIGINAL COMPARISON SUMMARY")
    print("=" * 70)
    
    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                        WITHOUT RAG                                  │
├─────────────────────────────────────────────────────────────────────┤
│ Source: Model's parametric knowledge (weights)                      │
│ Recipe: Generic chocolate cake (approximate)                        │
│ Specificity: General baking principles                              │
│ Grandma's 1952 recipe details: NOT PRESENT                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         WITH RAG                                    │
├─────────────────────────────────────────────────────────────────────┤
│ Source: Retrieved from vector store at runtime                      │
│ Recipe: Exact chocolate cake recipe (choc-cake-001)                 │
│ Specificity: Precise measurements (Dutch-process cocoa, hot coffee) │
│ Grandma's 1952 recipe details: PRESENT                              │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    # Check for specific markers from the retrieved recipe
    markers = ["Dutch-process", "hot black coffee", "buttermilk", "350°F", "1952"]
    print("Checking for specific recipe markers in outputs:")
    print("-" * 50)
    for marker in markers:
        in_no_rag = marker.lower() in response_no_rag.lower()
        in_rag = marker.lower() in response_rag.lower()
        print(f"  '{marker}': No-RAG={in_no_rag}, RAG={in_rag}")


# ============================================================================
# EDUCATIONAL NOTES
# ============================================================================

EDUCATIONAL_NOTES = """
================================================================================
                         EDUCATIONAL NOTES
================================================================================

1. WHERE DOES THE RECIPE KNOWLEDGE LIVE?
   ─────────────────────────────────────
   - WITHOUT RAG: The "knowledge" is distributed across billions of model 
     parameters (weights) learned during training. The model generates a 
     plausible recipe based on patterns it learned, but it's not retrieving
     a specific stored recipe.
   
   - WITH RAG: The specific recipe lives in the vector store (external 
     database). At runtime, relevant chunks are retrieved and injected into
     the prompt. The model's job is to synthesize and present this retrieved
     information, not to recall it from weights.

2. WHY ADDING MORE VECTORS DOESN'T MAKE THE MODEL "FORGET"
   ────────────────────────────────────────────────────────
   The vector store is completely separate from the model's weights.
   
   - Adding 1 million new recipes to the vector store: Model weights unchanged
   - Deleting all vectors: Model weights unchanged
   - The model never "learns" from vectors; it only reads them at inference time
   
   This is the key distinction between:
   - PARAMETRIC MEMORY: Encoded in weights, fixed after training
   - NON-PARAMETRIC MEMORY: External store, can be updated anytime

3. WHY RETRIEVAL QUALITY MATTERS MORE THAN VECTOR COUNT
   ─────────────────────────────────────────────────────
   If retrieval fails to find the chocolate cake recipe, the model cannot
   produce it—no matter how many other recipes are in the store.
   
   Critical factors:
   - Embedding quality: Do semantically similar texts have similar vectors?
   - Chunking strategy: Are recipe steps chunked logically?
   - Query formulation: Does the query match how content is written?
   - Top-k selection: Are we retrieving enough relevant chunks?
   
   A vector store with 10 perfectly-chunked recipes will outperform one with
   10,000 poorly-chunked recipes if retrieval accuracy is higher.

4. HOW THIS DIFFERS FROM TRAINING OR FINE-TUNING
   ──────────────────────────────────────────────
   
   ┌─────────────────┬────────────────────────────────────────────────────┐
   │ APPROACH        │ WHAT HAPPENS                                       │
   ├─────────────────┼────────────────────────────────────────────────────┤
   │ Training        │ Weights are adjusted on massive datasets.          │
   │                 │ Knowledge becomes "baked in" to parameters.        │
   │                 │ Expensive, requires significant compute.           │
   │                 │ Updates require retraining.                        │
   ├─────────────────┼────────────────────────────────────────────────────┤
   │ Fine-tuning     │ Weights are adjusted on smaller, specific data.    │
   │                 │ Modifies model behavior/knowledge.                 │
   │                 │ Still relatively expensive.                        │
   │                 │ Risk of catastrophic forgetting.                   │
   ├─────────────────┼────────────────────────────────────────────────────┤
   │ RAG             │ Weights are NOT modified.                          │
   │                 │ Knowledge is external and injected at runtime.     │
   │                 │ Cheap to update (just modify the vector store).    │
   │                 │ No forgetting—add/remove docs anytime.             │
   │                 │ Provides attribution (you know where info came     │
   │                 │ from).                                             │
   └─────────────────┴────────────────────────────────────────────────────┘

   RAG is ideal when:
   - Knowledge changes frequently (news, docs, inventory)
   - You need source attribution (legal, medical, compliance)
   - You can't afford to fine-tune
   - You want to control exactly what information the model can access

================================================================================
"""


def print_educational_notes():
    """Print the educational explanation."""
    print(EDUCATIONAL_NOTES)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "    RAG DEMONSTRATION: Parametric vs Retrieved Knowledge".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    print("\nThis demo requires an OpenAI API key.")
    print("Set it via: export OPENAI_API_KEY='your-key-here'\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set.")
        print("Please set it and run again.")
        exit(1)
    
    # Run the demonstration
    run_demo()
    
    # Print educational notes
    print_educational_notes()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
