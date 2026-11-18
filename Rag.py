import fitz  
import requests
import torch
from PIL import Image
from IPython.display import display
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline

url = "https://arxiv.org/pdf/1706.03762"
pdf_filename = "text.pdf"

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    print(" PDF file downloaded successfully.")
    with open(pdf_filename, "wb") as f:
        f.write(response.content)
else:
    raise RuntimeError(" PDF download failed.")

doc = fitz.open(pdf_filename)
print(f" Converting all {len(doc)} pages into images ")

for page_num in range(len(doc)):
    page = doc[page_num]
    pix = page.get_pixmap(dpi=150)  
    pix.save(f"page_{page_num + 1}.png")


loader = PyPDFLoader(pdf_filename)
pages_docs = loader.load()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(pages_docs)

vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 3})


print("--- Loading local LLM model ---")
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    
    MODEL_ID,
    device_map="auto"
)

pipe = pipeline(
    
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    temperature=0.1,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id
)


llm = HuggingFacePipeline(pipeline=pipe)

user_query = "What is positional encoding?"


retrieved_docs = retriever.invoke(user_query)

if retrieved_docs:

    source_page_num = retrieved_docs[0].metadata.get("page", 0) + 1
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
else:
    source_page_num = 1
    context_text = ""

prompt = f"""
Context:
{context_text}

Question: {user_query}
Answer:
"""

response_answer = llm.invoke(prompt)

print("\n" + "="*40)
print(response_answer)
print(f"\n[Source Verified: Page {source_page_num}]")
print("="*40)

image_filename = f"page_{source_page_num}.png"


final_display_image = Image.open(image_filename)
display(final_display_image)
