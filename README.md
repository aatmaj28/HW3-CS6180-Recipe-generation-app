# 🤖🧑🏻‍🍳 ChefBot: AI Recipe Generation App

> **CS 6180 – Foundations of Generative AI | HW3**
> Northeastern University · Spring 2026

A Streamlit-powered conversational recipe generation app that uses a local LLM ([Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)) to create personalized recipes from user-supplied ingredients, with smart ingredient substitution, nutritional analysis, and YouTube video discovery.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🍳 **AI Recipe Generation** | Generate full recipes from a comma-separated list of ingredients using a local LLM |
| 🎛️ **Decoding Strategy Control** | Choose between Greedy Search, Sampling (Temperature / Top-K / Top-P), or Beam Search |
| 🎭 **Chef Persona Selector** | Get recipes narrated by Gordon Ramsay, Grandma, a Mad Scientist, a Pirate, or a Zen Monk |
| 📝 **Prompt Style Selector** | Pick from Structured, Concise, Detailed, or Creative (Unconventional) prompt formats |
| 🍽️ **Serving Size Scaler** | Automatically scales ingredient quantities for 2 / 4 / 6 / 8 servings |
| 🌍 **Cuisine Filter** | Filter results by cuisine: Asian, Indian, Middle Eastern, Mexican, Western, Mediterranean, African |
| 🔍 **Ingredient Substitution** | Find semantically similar alternative ingredients via **Annoy** (fast ANN) or **Direct Cosine Similarity** search using spaCy embeddings |
| 🥗 **Nutritional Analyzer** | Estimate macros (Calories, Protein, Carbs, Fat) and a health score for any generated recipe |
| 📺 **YouTube Video Embed** | Automatically searches and embeds a relevant how-to-cook video inside the app |
| 💾 **Download Recipe** | Save the generated recipe as a plain `.txt` file |

---

## 🛠️ Tech Stack

- **LLM:** `Qwen/Qwen2.5-1.5B-Instruct` (via 🤗 Transformers, runs locally)
- **NLP Embeddings:** spaCy `en_core_web_lg`
- **Approximate Nearest Neighbors:** Annoy
- **Frontend:** Streamlit
- **Data:** Preprocessed ingredient CSV (auto-downloaded from Google Drive)
- **Video Search:** `youtube-search` library

---

## 🚀 Setup & Installation

### Prerequisites
- Python **3.9** (recommended)
- ~6 GB free disk space (for the LLM weights and spaCy model)
- `conda` is strongly recommended to avoid compilation errors on Windows

### 1. Clone the Repository

```bash
git clone https://github.com/aatmaj28/HW3-CS6180-Recipe-generation-app.git
cd HW3-CS6180-Recipe-generation-app
```

### 2. Create Environment (Conda – Recommended)

```bash
conda create -n genai python=3.9 pytorch cpuonly spacy pandas numpy -c pytorch -c conda-forge -y
conda activate genai
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

### 5. Run the App

```bash
streamlit run Recipe_Bot.py
```

The app will open in your browser at `http://localhost:8501`.

> **Note:** On first launch, the app downloads the Qwen2.5-1.5B-Instruct model weights from Hugging Face (~3 GB) and the ingredient CSV from Google Drive. This only happens once; subsequent runs load everything from cache.

---

## 📦 Alternative: Pip-Only Installation (Windows)

If you don't have `conda`, install PyTorch manually first to avoid compilation errors with `annoy`:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

---

## 🗂️ Project Structure

```
HW3-CS6180-Recipe-generation-app/
├── Recipe_Bot.py                      # Main Streamlit application
├── requirements.txt                   # Python dependencies
├── processed_ingredients_with_id.csv  # Ingredient dataset (auto-downloaded if missing)
└── files/
    ├── HW3 - GenAI.pdf                # Assignment brief (detailed)
    ├── HW3___GenAI.pdf                # Assignment brief (summary)
    └── Hw_3_GenAI.ipynb               # Jupyter notebook (exploration / experiments)
```

---

## 🧠 How It Works

```
User Input (ingredients + options)
        │
        ▼
  Prompt Construction ──► Chef Persona + Prompt Style templating
        │
        ▼
  Qwen2.5-1.5B-Instruct  (local inference via Transformers)
        │
        ▼
  Generated Recipe Text
        │
   ┌────┴─────────────────────────┐
   │                              │
   ▼                              ▼
Nutritional Analyzer         Ingredient Substitution
(LLM macro estimation)       (spaCy + Annoy / Cosine)
   │
   ▼
YouTube Video Search & Embed
```

---

## ⚠️ Notes

- The model runs **entirely locally on CPU** by default. A CUDA-capable GPU will dramatically speed up generation.
- The first cold start can take several minutes while models are downloaded and cached.
- Beam Search is slower but produces more coherent recipes; Sampling with higher temperature yields more creative output.

---

## 📄 License

This project was created for academic purposes as part of CS 6180 at Northeastern University.
