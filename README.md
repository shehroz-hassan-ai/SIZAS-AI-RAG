# SIZAS AI

SIZAS AI is a Retrieval-Augmented Generation (RAG) assistant that answers questions from your PDF documents, complete with page-level citations and highlighted source sentences. It answers strictly from the uploaded document and clearly states when the answer is not found.

## Features

- Upload a PDF and ask questions about its content
- Page-level citations with every answer
- Highlighted source sentences for easy verification
- Document-grounded answers only, with a clear "not found" response when the information is missing

## Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **Vector Database:** Pinecone
- **LLM and Embeddings:** Configured via environment variables (see `.env.example`)

## Getting Started

### Prerequisites

- Python 3.14 or higher
- A Pinecone account and API key
- An API key for your chosen LLM provider

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shehroz-hassan-ai/SIZAS-AI-RAG.git
   cd SIZAS-AI-RAG
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   On macOS or Linux, use `source venv/bin/activate` instead.

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your API keys.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

**Shehroz Hassan**
GitHub: [@shehroz-hassan-ai](https://github.com/shehroz-hassan-ai)
