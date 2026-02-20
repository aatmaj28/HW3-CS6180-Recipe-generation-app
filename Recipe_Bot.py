import streamlit as st
import pandas as pd
import spacy
import os
import gdown
from annoy import AnnoyIndex
import gdown
from annoy import AnnoyIndex
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from youtube_search import YoutubeSearch

# Load models
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        print("Downloading spaCy model 'en_core_web_lg'...")
        from spacy.cli import download
        download("en_core_web_lg")
        return spacy.load("en_core_web_lg")

nlp = load_spacy_model()

@st.cache_resource
def load_model():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_model()

# Download & Load Ingredient Data
GDRIVE_FILE_URL = "https://drive.google.com/uc?id=1-qf8ZIrBlsEixBJULmXyDJk4M4ktRurH"
CSV_FILE = "processed_ingredients_with_id.csv"

@st.cache_data
def load_ingredient_data():
    if not os.path.exists(CSV_FILE):  
        gdown.download(GDRIVE_FILE_URL, CSV_FILE, quiet=False)
    return pd.read_csv(CSV_FILE)["processed"].dropna().unique().tolist()

ingredient_list = load_ingredient_data()

# Compute Embeddings (Filter out zero vectors)
@st.cache_resource
def compute_embeddings():
    filtered_ingredients = []
    vectors = []

    for ing in ingredient_list:
        vec = nlp(ing.lower()).vector
        if np.any(vec):  # Exclude zero vectors
            filtered_ingredients.append(ing)
            vectors.append(vec)

    return np.array(vectors, dtype=np.float32), filtered_ingredients

ingredient_vectors, filtered_ingredient_list = compute_embeddings()

# Build Annoy Index (Fast Approximate Nearest Neighbors)
@st.cache_resource
def build_annoy_index():
    dim = ingredient_vectors.shape[1]
    index = AnnoyIndex(dim, metric="angular")  #  Uses angular distance (1 - cosine similarity)
    
    for i, vec in enumerate(ingredient_vectors):
        index.add_item(i, vec)
    
    index.build(50)  #  More trees = better accuracy
    return index
annoy_index = build_annoy_index()

#  Direct Cosine Similarity Search (Most Accurate)
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.any(vec1) and np.any(vec2) else 0

def direct_search_alternatives(ingredient):
    target_vec = nlp(ingredient.lower()).vector
    if not np.any(target_vec):
        return []

    scores = []
    for candidate, candidate_vec in zip(filtered_ingredient_list, ingredient_vectors):
        if candidate.lower() == ingredient.lower():
            continue
        score = cosine_similarity(target_vec, candidate_vec)
        scores.append((candidate, score))
    
    # Sort by similarity score in descending order
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 most similar ingredients
    return [item[0] for item in scores[:3]]

#  Annoy Search 
def annoy_search_alternatives(ingredient):
    vector = nlp(ingredient.lower()).vector
    if not np.any(vector):
        return []

    # Get nearest neighbors (fetch 4 to account for self-match)
    indices = annoy_index.get_nns_by_vector(vector, 4)
    
    candidates = []
    for i in indices:
        candidate = filtered_ingredient_list[i]
        if candidate.lower() != ingredient.lower():
            candidates.append(candidate)
            
    return candidates[:3] 

#  Generate Recipe
def generate_recipe(ingredients, cuisine, temperature, top_k, top_p, num_beams, do_sample, prompt_type, serving_size, chef_persona):
    
    # AI System Prompts (Task 2)
    prompts = {
        "Structured Format": (
            f"You are a professional chef assistant. Always respond with a recipe in this exact format using key ingredients {', '.join(ingredients.split(', '))} and {cuisine} style for {serving_size} people:\n"
            "1. TITLE: A creative name for the dish\n"
            "2. INGREDIENTS: A numbered list of all ingredients with exact measurements adjusted for {serving_size} servings\n"
            "3. INSTRUCTIONS: Step-by-step cooking instructions, numbered sequentially\n"
            "4. SERVING SUGGESTION: One sentence on how to serve the dish\n"
            "Always follow this structure precisely. Do not skip any section."
        ),
        "Concise (Brief)": (
            f"You are a chef assistant. Keep your recipe using {', '.join(ingredients.split(', '))} ({cuisine}) very short and concise for {serving_size} servings. "
            "Provide only the dish name, a brief ingredient list (no measurements), "
            "and instructions in 3-4 short steps maximum. No extra commentary or descriptions. "
            "Be as brief as possible while remaining useful."
        ),
        "Detailed (Elaborate)": (
            f"You are an expert chef and culinary instructor. Provide an extremely detailed recipe using {', '.join(ingredients.split(', '))} ({cuisine}) for {serving_size} servings. "
            "Include precise measurements, preparation techniques, cooking temperatures in both "
            "Fahrenheit and Celsius, timing for each step, chef's tips and tricks, possible "
            "variations, nutritional highlights, and a detailed plating/serving description. "
            "Explain the reasoning behind key cooking techniques. Be thorough and educational."
        ),
        "Creative (Unconventional)": (
            f"You are an avant-garde fusion chef known for surprising and unconventional dishes. "
            f"Create unexpected and creative recipes using {', '.join(ingredients.split(', '))} with a {cuisine} twist for {serving_size} people. "
            "Think molecular gastronomy, cross-cultural fusion, and surprising flavor combinations. "
            "Give the dish a unique, inventive name. Push culinary boundaries while keeping "
            "the dish actually edible and delicious. Surprise the user!"
        )
    }

    persona_prompts = {
        "Gordon Ramsay": "Roleplay: You are an angry, critical chef. style: SHOUTING, insults, perfectionism. Start response with 'LISTEN TO ME!'.",
        "Grandma": "Roleplay: You are a sweet grandmother. style: cozy, 'dearie', 'honey', old-fashioned advice. Start response with 'Oh, hello dearie!'.",
        "Mad Scientist": "Roleplay: You are a mad food scientist. style: chemical terms, maniacal laughter, lab safety. Start response with 'Welcome to the LAB!'.",
        "Pirate": "Roleplay: You are a pirate chef. style: pirate slang, ahoy, treasure. Start response with 'Ahoy matey!'.",
        "Zen Monk": "Roleplay: You are a monk. style: haikus, mindfulness, peace. Start response with 'Breathe in...'.",
        "Standard Assistant": "" 
    }

    base_prompt = prompts[prompt_type]
    chef_instruction = persona_prompts.get(chef_persona, "")
    
    input_text = f"{chef_instruction}\n\n{base_prompt}" if chef_persona != "Standard Assistant" else base_prompt 
    
    # Use chat template to format the prompt correctly for the model
    messages = [
        {"role": "user", "content": input_text}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)

    # Dynamic generation args
    gen_kwargs = {
        "max_new_tokens": 2048,  # Increased to prevent cutoff
        "num_return_sequences": 1,
        "repetition_penalty": 1.2,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "early_stopping": True if num_beams > 1 else False
    }

    if do_sample:
        gen_kwargs.update({
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        })

    outputs = model.generate(inputs, **gen_kwargs)
    # Decode only the new tokens (response)
    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()

    # Post-processing to remove script-like dialogue (Chef: ... Assistant: ...)
    if "Chef:" in response or "Assistant:" in response:
        # Try to find where the actual recipe starts
        match = re.search(r"(Title:|Ingredients:|1\.|Here is the recipe)", response, re.IGNORECASE)
        if match:
             response = response[match.start():] # Keep everything from the match onwards
    
    # Clean up formatting (remove asterisks and hash signs)
    response = response.replace("*", "").replace("#", "")

    return response

# Feature 1: Smart Nutritional Analyzer
def generate_nutrition(recipe_text):
    prompt = (
        "You are a nutritionist. Analyze the recipe below and provide estimated macros per serving.\n"
        "Do NOT repeat the recipe. Output ONLY the following data in this exact format:\n\n"
        "Calories: [Amount]\n"
        "Protein: [Amount]\n"
        "Carbs: [Amount]\n"
        "Fat: [Amount]\n"
        "Health Score: [1-10]/10\n"
        "Analysis: [One sentence summary]\n\n"
        f"RECIPE:\n{recipe_text[:1000]}" # Truncate to save context
    )
    
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    outputs = model.generate(inputs, max_new_tokens=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()

def parse_nutrition_data(text):
    try:
        data = {}
        # Regex to find values like "Calories: 500 kcal", "Calories: [500]", "Protein: 20g", "Protein: [20g]"
        # Patterns look for the key, optional brackets/spaces, then capture digits.
        patterns = {
            "Calories": r"Calories:.*?(\d+)",
            "Protein": r"Protein:.*?(\d+)",
            "Carbs": r"Carbs:.*?(\d+)",
            "Fat": r"Fat:.*?(\d+)",
            "Health Score": r"Health Score:.*?(\d+)",
            "Analysis": r"Analysis:\s*([\s\S]+)" # Capture everything after Analysis
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).strip()
                # Clean up trailing newlines or other keys
                val = re.split(r'\n|Calories:|Protein:|Carbs:|Fat:|Health Score:|Analysis:', val)[0].strip()
                data[key] = val
            else:
                data[key] = "N/A"
        return data
    except Exception:
        return None


#  Streamlit App UI
st.title("🤖🧑🏻‍🍳 ChefBot: AI Recipe Chatbot")

# Sidebar for Generation Parameters
with st.sidebar:
    st.header("🎛️ Text Generation Params")
    decoding_strategy = st.radio("Decoding Strategy", ["Greedy Search", "Sampling", "Beam Search"])

    # Defaults
    temperature = 1.0
    top_k = 50
    top_p = 0.95
    num_beams = 1
    do_sample = False

    if decoding_strategy == "Sampling":
        do_sample = True
        temperature = st.select_slider("Temperature", options=[0.5, 1.0, 2.0], value=1.0, help="Higher = Creative, Lower = Focused")
        top_k = st.select_slider("Top-K", options=[5, 50], value=50, help="Limits vocabulary to top K tokens")
        top_p = st.select_slider("Top-P", options=[0.7, 0.95], value=0.95, help="Nucleus sampling probability")
    
    elif decoding_strategy == "Beam Search":
        num_beams = st.select_slider("Num Beams", options=[1, 5], value=5, help="Number of beams for search")

    st.divider()
    st.header("🎭 System Prompt")
    
    # Feature 3: Chef's Voice Selector
    chef_persona = st.radio("Chef Persona 👨‍🍳", [
        "Standard Assistant", 
        "Gordon Ramsay", 
        "Grandma", 
        "Mad Scientist", 
        "Pirate", 
        "Zen Monk"
    ])
    
    prompt_type = st.selectbox("Select Prompt Style", [
        "Structured Format", 
        "Concise (Brief)", 
        "Detailed (Elaborate)",
        "Creative (Unconventional)"
    ])

# New Layout for Ingredients & Serving Size
col1, col2 = st.columns([3, 1])
with col1:
    ingredients = st.text_input("🥑🥦🥕 Ingredients (comma-separated):")
with col2:
    # Feature 2: Recipe Scaling Calculator
    serving_size = st.selectbox("Servings 🍽️", [2, 4, 6, 8], index=1)

cuisine = st.selectbox("Select a cuisine:", ["Any", "Asian", "Indian", "Middle Eastern", "Mexican",  "Western", "Mediterranean", "African"])

if st.button("Generate Recipe", use_container_width=True) and ingredients:
    with st.spinner(f"Cooking up a recipe in {chef_persona}'s kitchen... 🍳"):
        st.session_state["recipe"] = generate_recipe(ingredients, cuisine, temperature, top_k, top_p, num_beams, do_sample, prompt_type, serving_size, chef_persona)
        st.session_state["gen_params"] = {
            "Strategy": decoding_strategy,
            "Temp": temperature if do_sample else "N/A",
            "Top-K": top_k if do_sample else "N/A",
            "Top-P": top_p if do_sample else "N/A",
            "Beams": num_beams,
            "Prompt": prompt_type,
            "Persona": chef_persona,
            "Servings": serving_size
        }

if "recipe" in st.session_state:
    st.markdown("### 🍽️ Generated Recipe:")
    
    # Show active params
    if "gen_params" in st.session_state:
        params = st.session_state["gen_params"]
        st.info(f"Generated using: {params}")

    st.text_area("Recipe:", st.session_state["recipe"], height=400)

    st.markdown("---")
    # Feature 4: YouTube Search Link
    st.markdown("### 📺 Watch How It's Made")
    
    # Improved Search Query Logic
    recipe_lines = st.session_state["recipe"].split('\n')
    search_query = f"{cuisine} {ingredients.split(',')[0]} recipe" # Default fallback
    
    for line in recipe_lines[:3]: # Check first 3 lines for a title
        if "title" in line.lower() or "recipe" in line.lower() or "**" in line:
            clean_line = line.replace("TITLE:", "").replace("**", "").replace("#", "").strip()
            if len(clean_line) > 5:
                search_query = clean_line
                break

    # Embed First Video Found
    try:
        results = YoutubeSearch(search_query, max_results=1).to_dict()
        if results:
            video_url = "https://www.youtube.com" + results[0]['url_suffix']
            st.video(video_url)
        else:
            # Try a broader search if specific fails
            results = YoutubeSearch(f"{ingredients.split(',')[0]} recipe", max_results=1).to_dict()
            if results:
                video_url = "https://www.youtube.com" + results[0]['url_suffix']
                st.video(video_url)
            else:
                st.warning("No video found, but you can search manually:")
                youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
                st.markdown(f"👉 **[Click here to search on YouTube]({youtube_url})**")
    except Exception as e:
        # Fallback to link if API fails
        st.error(f"Error finding video: {e}")
        youtube_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        st.markdown(f"👉 **[Click here to find video tutorials on YouTube]({youtube_url})**")

    st.markdown("---")
    # Feature 1: Nutritional Analyzer Button & Display
    st.header("🥗 Analyze Nutrition")
    if st.button("🍎 Analyze Nutrition"):
        with st.spinner("Calculating macros... 🥗"):
            nutrition_text = generate_nutrition(st.session_state["recipe"])
            data = parse_nutrition_data(nutrition_text)
            
            if data and data["Health Score"] != "N/A":
                st.markdown("### 🥗 Nutritional Dashboard")
                
                # Determine Color for Health Score
                score = int(data["Health Score"])
                if score >= 8:
                    color = "#28a745" # Green
                    msg = "Excellent Choice! 🌿"
                elif score >= 5:
                    color = "#ffc107" # Yellow
                    msg = "Balanced Meal ⚖️"
                else:
                    color = "#dc3545" # Red
                    msg = "Indulgent Treat 🍔"

                # Health Score Card & Chart
                col_score, col_chart = st.columns([1, 2])
                
                with col_score:
                    st.markdown(
                        f"""
                        <div style="background-color: {color}; padding: 10px; border-radius: 10px; color: white; text-align: center; height: 100%;">
                            <h3 style="margin:0;">Health Score</h3>
                            <h1 style="margin:0; font-size: 3rem;">{score}/10</h1>
                            <p style="margin:0;">{msg}</p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col_chart:
                    # Create a simple bar chart data
                    chart_data = pd.DataFrame({
                        "Macro": ["Protein", "Carbs", "Fat"],
                        "Grams": [int(data['Protein']), int(data['Carbs']), int(data['Fat'])]
                    })
                    st.bar_chart(chart_data.set_index("Macro"), color="#4caf50", height=180)

                # Macro Columns - Ensure they are clearly visible
                st.markdown("---")
                cols = st.columns(4)
                cols[0].metric("🔥 Calories", f"{data['Calories']} kcal")
                cols[1].metric("🥩 Protein", f"{data['Protein']}g")
                cols[2].metric("🍞 Carbs", f"{data['Carbs']}g")
                cols[3].metric("🥑 Fat", f"{data['Fat']}g")
                
                st.info(f"**Analysis:** {data.get('Analysis', 'No analysis available')}")
            
            else:
                st.error("Could not parse nutrition data. Raw output:")
                st.write(nutrition_text)

    st.markdown("---")
    st.header("📂 Save Recipe")
    st.download_button(label="Download Recipe Text", 
                       data=st.session_state["recipe"], 
                       file_name="recipe.txt", 
                       mime="text/plain")

    #  Alternative Ingredient Section
    st.markdown("---")
    st.markdown("## 🔍 Find Alternative Ingredients")

    ingredient_to_replace = st.text_input("Enter an ingredient:")
    search_method = st.radio("Select Search Method:", ["Annoy (Fastest)", "Direct Search (Best Accuracy)"], index=0)

    if st.button("🔄 Find Alternatives", use_container_width=True) and ingredient_to_replace:
        search_methods = {
            "Annoy (Fastest)": annoy_search_alternatives,
            "Direct Search (Best Accuracy)": direct_search_alternatives
        }
        alternatives = search_methods[search_method](ingredient_to_replace)
        st.markdown(f"### 🌿 Alternatives for **{ingredient_to_replace.capitalize()}**:")
        st.markdown(f"➡️ {' ⟶ '.join(alternatives)}")
