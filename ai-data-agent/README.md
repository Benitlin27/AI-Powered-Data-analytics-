# 🤖 AI Data Analysis Agent

An end-to-end AI agent that ingests datasets, cleans data, runs EDA,
trains ML models, and explains everything in plain English using a local LLM (Ollama).

## Project Structure

```
ai-data-agent/
├── app.py                  ← Streamlit UI (entry point)
├── requirements.txt        ← Python dependencies
├── modules/
│   ├── ingestion.py        ← File parsing, schema detection
│   ├── cleaning.py         ← Null handling, encoding, outlier removal
│   ├── eda.py              ← Stats, correlations, distributions
│   ├── visualization.py    ← Auto-chart generation
│   ├── task_detector.py    ← Classification / regression / clustering
│   ├── model_trainer.py    ← Train, evaluate, pick best model
│   ├── aggregator.py       ← Bundle results for LLM
│   └── llm_client.py       ← Ollama integration + Q&A
```

## Setup & Run

### 1. Install Python 3.10+
Download from https://python.org/downloads

### 2. Install Ollama
Download from https://ollama.com/download
Then pull a model:
```bash
ollama pull llama3
```

### 3. Clone / open project in VS Code
Open the `ai-data-agent` folder in VS Code.

### 4. Create virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 5. Install dependencies
```bash
pip install -r requirements.txt
```

### 6. Start Ollama (in a separate terminal)
```bash
ollama serve
```

### 7. Run the app
```bash
streamlit run app.py
```

The app opens at http://localhost:8501

## Usage
1. Upload a CSV, Excel, or JSON file
2. Click through each stage: Clean → EDA → Charts → Train Models → AI Summary
3. Ask follow-up questions in the chat at the bottom
