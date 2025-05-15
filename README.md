# AI TikTok Prototype

This repository contains the **AI TikTok Prototype**.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/AI_TikTok_prototype.git
   cd AI_TikTok_prototype
   ```

2. Create and activate virtual environment
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```

## Create Vector DB:

1. Ensure the `assets/letters` and `assets/books` folders contain the relevant files.
2. Run the script:
   ```bash
   python load_knowledge.py
   ```
3. The script will:
   - Load and process the PDFs.
   - Generate embeddings using the `HuggingFaceEmbeddings` model.
   - Save the FAISS vector database and metadata in the `vector_dbs` directory.

## Run the App

To start the application, run the following command:
```bash
streamlit run Homepage.py
```

- NB: ensure the `assets/video_data/videos` folder is populated with relevant videos.