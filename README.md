# RAG-PROJECT
A Retrieval-Augmented Generation (RAG) pipeline that answers questions from a PDF (the "Attention Is All You Need" paper) using a local LLM, and shows the exact page image the answer was retrieved from.

## ✨ Features
- Downloads a PDF automatically from arXiv
- Converts every PDF page into an image
- Splits text into chunks and stores embeddings in a Chroma vector database
- Retrieves the most relevant chunk for a question
- Runs a local LLM (TinyLlama), so no API key is needed
- Displays the source page image for answer verification

## 🛠️ Tech Stack
- **Python**
- **LangChain** (document loading, text splitting, vector store)
- **ChromaDB** (vector database)
- **Sentence Transformers**: `all-MiniLM-L6-v2` (embeddings)
- **Hugging Face Transformers**: `TinyLlama-1.1B-Chat` (LLM)
- **PyMuPDF** (PDF to image conversion)
- **Pillow** (image display)

## ⚙️ How It Works
1. Download the PDF and convert each page to a PNG image
2. Load the PDF text and split it into chunks (size 800, overlap 200)
3. Generate embeddings and store them in ChromaDB
4. For a user question, retrieve the top matching chunk
5. Generate an answer with the local LLM
6. Show the source page number and its image

## 🚀 Installation
```bash
git clone https://github.com/<rajesh-aiml>/.git
cd <Rag-project>
pip install -r requirements.txt
```

## ▶️ Usage
Open the notebook and run all cells:
```bash
googlecolab rag_project.ipynb
```
Change the question in this line:
```python
user_query = "What is positional encoding?"
```

## 📸 Example Output
<img width="1173" height="600" alt="output" src="https://github.com/user-attachments/assets/dfcd8ab8-742a-4a88-9051-0cb61a915fee" /> 
<img width="1082" height="340" alt="output2" src="https://github.com/user-attachments/assets/a6f7815d-60a3-4e07-8c41-7f8ac27f9d69" />
<img width="1141" height="677" alt="output3" src="https://github.com/user-attachments/assets/fdf423f2-fc2c-462b-8711-f858afcda46d" />




## ⚠️ Limitations
- TinyLlama is a small model, so answers can be less accurate
- Only the top 3 chunk is retrieved (`k=3`)
- Currently works with a single PDF


