# openAi.py - A "Human-in-the-Loop" Multi-Agent Content Pipeline

import os
import json
import re
import time
import frontmatter
from datetime import datetime
import google.generativeai as genai
# Note: huggingface_hub is required if you use Llama as the editor.
# Remove it if you use Gemini for everything.
from huggingface_hub import InferenceClient


# --- Import our settings from the config file ---
import config

# --- HELPER: Load prompt templates from their dedicated files ---
def load_prompt_template(filepath: str) -> str:
    """Loads a prompt template from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"🔥 CRITICAL ERROR: Prompt template file not found at '{filepath}'")
        exit()

# --- INITIALIZATION & CONSTANTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRITER_PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, config.PROMPT_TEMPLATE_FILE)
EDITOR_PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, config.EDITOR_PROMPT_TEMPLATE_FILE)
TOPICS_FILE_PATH = os.path.join(SCRIPT_DIR, config.TOPICS_FILE)
LINK_MAP_FILE_PATH = os.path.join(SCRIPT_DIR, config.LINK_MAP_FILE)

WRITER_PROMPT_TEMPLATE = load_prompt_template(WRITER_PROMPT_TEMPLATE_PATH)
EDITOR_PROMPT_TEMPLATE = load_prompt_template(EDITOR_PROMPT_TEMPLATE_PATH)

# Configure APIs
genai.configure(api_key=config.GEMINI_API_KEY)
# Initialize the HF client if you are using Llama as your editor
HF_CLIENT = InferenceClient(token=config.HF_API_TOKEN) 

# Create directories
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.DRAFTS_DIR, exist_ok=True)
# IMAGE_OUTPUT_DIR is no longer needed by the script but might be used by you manually
if not os.path.exists(config.IMAGE_OUTPUT_DIR):
    os.makedirs(config.IMAGE_OUTPUT_DIR)


# --- AGENT 1: The Writer (Gemini) ---
def generate_article_draft(article_data: dict) -> str:
    """Generates the initial raw article draft from Gemini."""
    print("✍️  Generating initial draft from Writer (Gemini)...")
    try:
        prompt = WRITER_PROMPT_TEMPLATE.format(topic=article_data["topic"])
        model = genai.GenerativeModel(config.ARTICLE_GENERATION_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini draft generation failed: {e}")
        return None

# --- AGENT 2: The Editor (Llama 3.1) ---
def refine_article(draft_content: str, article_data: dict) -> str:
    """Sends the draft to an Editor AI for refinement and image placeholder insertion."""
    print("🧐 Sending draft to Editor (Llama 3.1) for refinement...")
    try:
        system_prompt = EDITOR_PROMPT_TEMPLATE.format(topic=article_data["topic"])
        completion = HF_CLIENT.chat.completions.create(
            model=config.EDITOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": draft_content}
            ],
            max_tokens=4096
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"⚠️ Llama refinement failed: {e}")
        return None

# --- AGENT 3: The Producer (Python Script Functions) ---
def process_image_placeholders(refined_content: str, slug: str) -> (str, list):
    """
    Finds image placeholders and transforms them into a human-readable "To-Do" list
    and Hugo shortcode placeholders.
    Returns a tuple: (processed_content, image_todo_list)
    """
    print("🖼️  Processing dynamic image placeholders for manual insertion...")
    placeholder_pattern = re.compile(r'\[IMAGE\|([\w-]+)\|([^\]]+)\]')
    placeholders = placeholder_pattern.findall(refined_content)
    
    image_todo_list = []
    processed_content = refined_content

    if not placeholders:
        print("➡️ No dynamic image placeholders found.")
        return processed_content, image_todo_list

    for image_id, prompt in placeholders:
        image_filename = f"{slug}-{image_id}.jpg"
        hugo_path = f"/images/{image_filename}"
        caption = prompt.split(',')[0].strip()

        todo_item = {
            "filename": image_filename,
            "prompt": prompt.strip(),
            "hugo_path": hugo_path
        }
        image_todo_list.append(todo_item)
        
        shortcode = f'\n{{{{< figure src="{hugo_path}" caption="{caption}" >}}}}\n'
        placeholder_full_text = f'[IMAGE|{image_id}|{prompt}]'
        processed_content = processed_content.replace(placeholder_full_text, shortcode)
        
    return processed_content, image_todo_list

def save_article(content_body: str, article_data: dict, image_todo_list: list) -> str:
    """Constructs the final markdown file, including the image to-do list."""
    try:
        image_instructions = ""
        if image_todo_list:
            image_instructions += "\n\n<!--\n"
            image_instructions += "========================================\n"
            image_instructions += "🖼️ IMAGE TO-DO LIST (Manual Insertion)\n"
            image_instructions += "========================================\n"
            for item in image_todo_list:
                image_instructions += f"\n- FILENAME: {item['filename']}\n"
                image_instructions += f"  - PROMPT/IDEA: {item['prompt']}\n"
            image_instructions += "========================================\n"
            image_instructions += "-->\n"

        final_content = content_body + image_instructions
        
        post = frontmatter.Post(final_content)
        post.metadata = {
            "title": article_data["title"],
            "description": article_data["description"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "categories": article_data.get("categories", []),
            "tags": article_data.get("tags", [])
        }
        
        filename = f"{article_data['slug']}.md"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        return filepath
    except Exception as e:
        print(f"🔥 Error saving the final article: {e}")
        return None

# --- SEO & Linking Functions ---
def update_link_map(article_data: dict):
    """Adds the newly created article's info to the central link map."""
    link_map = []
    if os.path.exists(LINK_MAP_FILE_PATH):
        with open(LINK_MAP_FILE_PATH, "r", encoding="utf-8") as f:
            try:
                link_map = json.load(f)
            except json.JSONDecodeError: pass
    new_entry = {
        "slug": f"/{article_data['slug']}",
        "anchors": article_data.get("anchors", [])
    }
    link_map = [entry for entry in link_map if entry["slug"] != new_entry["slug"]]
    link_map.append(new_entry)
    with open(LINK_MAP_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(link_map, f, indent=2, ensure_ascii=False)

def apply_internal_links(filepath: str, current_slug: str):
    """Reads an article and injects internal links from the link map."""
    if not os.path.exists(LINK_MAP_FILE_PATH): return
    with open(LINK_MAP_FILE_PATH, "r", encoding="utf-8") as f:
        link_map = json.load(f)
    post = frontmatter.load(filepath)
    content = post.content
    links_added = 0
    max_links = 2
    for link_info in link_map:
        if link_info["slug"].strip('/') == current_slug: continue
        for anchor in link_info.get("anchors", []):
            if links_added >= max_links: break
            pattern = re.compile(r'(?<![\[(])\b' + re.escape(anchor) + r'\b(?![])])', re.IGNORECASE)
            new_content, count = pattern.subn(f'[{anchor}]({link_info["slug"]})', content, 1)
            if count > 0:
                content, links_added = new_content, links_added + 1
                break
        if links_added >= max_links: break
    if links_added > 0:
        post.content = content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        print(f"🔗 {links_added} internal link(s) applied.")

# --- MAIN WORKFLOW: The Orchestrator ---
def main():
    try:
        with open(TOPICS_FILE_PATH, "r", encoding="utf-8") as f:
            topics = json.load(f)
        if isinstance(topics, dict): topics = [topics]
    except Exception as e:
        print(f"⚠️ Could not read topics file: {e}"); return
    if not topics:
        print("🎉 No topics left in the queue."); return

    article_data = topics[0]
    print("----------------------------------------------------")
    print(f"🚀 Starting Human-in-the-Loop generation for: {article_data['title']}")

    # === Stage 1: Writer ===
    draft_content = generate_article_draft(article_data)
    if not draft_content: print("🔥 Initial draft generation failed. Aborting."); return
    try:
        draft_filename = f"{datetime.now().strftime('%Y-%m-%d')}-{article_data['slug']}.md"
        draft_filepath = os.path.join(config.DRAFTS_DIR, draft_filename)
        with open(draft_filepath, "w", encoding="utf-8") as f: f.write(draft_content)
        print(f"📝 Draft saved for review: {draft_filepath}")
    except Exception as e:
        print(f"⚠️ Could not save draft file: {e}")

    # === Stage 2: Editor ===
    refined_content_with_placeholders = refine_article(draft_content, article_data)
    if not refined_content_with_placeholders: print("🔥 Article refinement failed. Aborting."); return

    # === Stage 3: Producer ===
    final_article_body, image_todo_list = process_image_placeholders(refined_content_with_placeholders, article_data['slug'])

    hero_filename = f"{article_data['slug']}-hero.jpg"
    hero_hugo_path = f"/images/{hero_filename}"
    hero_prompt = article_data.get("image_prompt", f"A professional hero image for a blog post about {article_data['topic']}, photorealistic, detailed")
    image_todo_list.insert(0, {"filename": hero_filename, "prompt": hero_prompt, "hugo_path": hero_hugo_path})
    
    hero_shortcode = f'{{{{< figure src="{hero_hugo_path}" caption="Overview of {article_data["topic"]}" >}}}}\n\n'
    final_content_with_hero = hero_shortcode + final_article_body

    filepath = save_article(final_content_with_hero, article_data, image_todo_list)
    if not filepath: print("🔥 Saving the final article failed. Aborting."); return
    
    apply_internal_links(filepath, article_data['slug'])
    update_link_map(article_data)

    remaining_topics = topics[1:]
    with open(TOPICS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(remaining_topics, f, indent=2, ensure_ascii=False)

    print(f"✅✅✅ Article with image instructions generated: {filepath}")
    print(f"🔄 {len(remaining_topics)} topics left in queue.")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()