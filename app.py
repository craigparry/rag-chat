import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# =====================================================================
# STEP 1: INITIALIZE THE LLM & EMBEDDING MODEL
# =====================================================================
print("Loading models...")
model_id = "google/gemma-4-12b-it"  # Using the instruction-tuned variant

# Configure 4-bit quantization to fit the 12B model comfortably on consumer GPUs
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load Gemma 4 Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

# Load a strong retrieval model from Hugging Face
embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# =====================================================================
# STEP 2: PREPARE AND CHUNK DOCUMENT DATA
# =====================================================================
# Sample internal data representing your knowledge base
documents = [
    "Gemma 4 12B features a massive 256K token context length and hybrid attention mechanisms.",
    "The hybrid attention structure in Gemma 4 interleaves local sliding window attention with global attention.",
    "Proportional RoPE (p-RoPE) is applied in Gemma 4's global layers to optimize memory for long contexts.",
    "Retrieval-Augmented Generation (RAG) minimizes LLM hallucinations by provisioning external facts during inference."
]

# Compute embeddings for our knowledge chunks
embeddings = embed_model.encode(documents)
dimension = embeddings.shape[1]

# Instantiate an in-memory FAISS Index for vector search
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype('float32'))

# =====================================================================
# STEP 3: RETRIEVAL AND QUERY EXECUTION
# =====================================================================
def run_gemma4_rag(user_query, top_k=2):
    print(f"\nUser Query: '{user_query}'")
    
    # Encode user query
    query_embedding = embed_model.encode([user_query])
    
    # Retrieve top match indexes from the FAISS database
    distances, indices = index.search(np.array(query_embedding).astype('float32'), top_k)
    
    # Compile retrieved data blocks into a unified context block
    retrieved_chunks = [documents[idx] for idx in indices[0] if idx != -1]
    context = "\n".join(retrieved_chunks)
    print(f"Retrieved Context:\n{context}")
    
    # Structure the message payload leveraging Gemma 4's native system prompt support
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the following context fragments to answer the user request precisely. If the answer is not present, state that you do not know."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"}
    ]
    
    # Formulate chat format strings using the built-in chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize input data strings and dispatch tensors directly to the target GPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Generate content utilizing strict token ceilings 
    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True
        )
        
    # Extract structural sequence fragments and decode into pure strings
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return response

# Execute the RAG test query
query = "How does Gemma 4 manage memory optimization for long input tokens?"
answer = run_gemma4_rag(query)
print(f"\nGemma 4 Response:\n{answer}")
