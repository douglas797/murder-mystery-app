#!/usr/bin/env python3
"""
Infinite Murder Mysteries - AI-Powered Fair-Play Text Detective Game
A complete Streamlit application for generating and playing coherent, solvable murder mysteries.
"""

import streamlit as st
import openai
import json
import re
from pathlib import Path
from datetime import datetime
import os

# =============================================================================
# CONFIG & HELPERS
# =============================================================================

st.set_page_config(
    page_title="Infinite Murder Mysteries | AI Detective",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROMPTS_DIR = Path(__file__).parent / "prompts"

def load_prompt(name: str) -> str:
    """Load a prompt template from the prompts/ directory."""
    path = PROMPTS_DIR / name
    if not path.exists():
        st.error(f"Prompt file missing: {name}")
        st.stop()
    return path.read_text(encoding="utf-8")

def get_openai_client():
    """Create OpenAI-compatible client.
    Supports Groq (free), Gemini (free), xAI, and OpenAI.
    """
    provider = st.session_state.get("provider", "Groq (Free)")
    api_key = st.session_state.get("api_key", "").strip()
    
    if not api_key:
        st.error("🔑 Please enter your API key in the sidebar first.")
        st.stop()
    
    if provider == "Groq (Free)":
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
    elif provider == "Gemini (Free)":
        # Google Gemini via OpenAI-compatible endpoint (completely free tier available)
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    elif provider == "xAI":
        return openai.OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1"
        )
    else:
        return openai.OpenAI(api_key=api_key)

def extract_json_from_text(text: str):
    """Robustly extract JSON from LLM response that may contain extra text or markdown."""
    text = text.strip()
    
    # 1. Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 2. Try to find ```json ... ``` block
    match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 3. Try to find the largest JSON-like object
    match = re.search(r'(\{.*\})', text, re.DOTALL)
    if match:
        candidate = match.group(1)
        # Clean common issues
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)  # trailing commas
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    
    return None

def initialize_session_state():
    """Set up all required session state variables."""
    defaults = {
        "provider": "Groq (Free)",
        "api_key": "",
        "model": "llama-3.3-70b-versatile",
        "mystery_bible": None,
        "game_active": False,
        "discovered_clues": set(),
        "visited_locations": set(),
        "interviewed_suspects": set(),
        "chat_history": [],
        "notebook": "",
        "last_judgment": None,
        "last_narrative_summary": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_game_state():
    """Reset only the per-case game state (keep API config)."""
    keys_to_reset = [
        "mystery_bible", "game_active", "discovered_clues", "visited_locations",
        "interviewed_suspects", "chat_history", "notebook", "last_judgment",
        "last_narrative_summary"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            if isinstance(st.session_state[key], (set, list, dict)):
                st.session_state[key] = type(st.session_state[key])()
            else:
                st.session_state[key] = None if key != "notebook" else ""
    st.session_state.game_active = False

# =============================================================================
# AI FUNCTIONS
# =============================================================================

def generate_mystery(custom_theme: str = "") -> dict | None:
    """Generate a brand new fair-play murder mystery using the AI."""
    client = get_openai_client()
    system_prompt = load_prompt("mystery_generator.txt")
    
    user_content = (
        "Generate a completely original, high-quality, fair-play murder mystery now.\n"
        f"Custom theme or setting preference: {custom_theme if custom_theme.strip() else 'Surprise me with something fresh and atmospheric. Vary the era and location creatively.'}"
    )
    
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("model", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.82,
            max_tokens=9000,
        )
        content = response.choices[0].message.content
        mystery = extract_json_from_text(content)
        
        if not mystery or "title" not in mystery or "solution" not in mystery:
            st.error("The AI returned an incomplete mystery. Please try generating again.")
            return None
        
        # Basic validation
        if not any(s.get("is_killer") for s in mystery.get("suspects", [])):
            st.error("Generated mystery has no killer marked. Regenerating...")
            return None
            
        return mystery
        
    except Exception as e:
        st.error(f"Error generating mystery: {str(e)}")
        return None

def gm_respond(player_action: str) -> tuple[str, dict]:
    """Send player action to the Game Master AI and return (narrative_text, state_update_dict)."""
    client = get_openai_client()
    gm_prompt = load_prompt("game_master.txt")
    
    bible = st.session_state.mystery_bible
    state = {
        "discovered_clue_ids": sorted(list(st.session_state.discovered_clues)),
        "visited_locations": sorted(list(st.session_state.visited_locations)),
        "interviewed_suspects": sorted(list(st.session_state.interviewed_suspects)),
        "notebook_excerpt": st.session_state.notebook[-600:] if st.session_state.notebook else "Empty so far.",
        "last_summary": st.session_state.get("last_narrative_summary", "")
    }
    
    user_context = f"""MYSTERY BIBLE (ABSOLUTE SECRET TRUTH - NEVER REVEAL DIRECTLY):
{json.dumps(bible, indent=2, ensure_ascii=False)}

CURRENT GAME STATE:
{json.dumps(state, indent=2)}

PLAYER ACTION / QUESTION / INVESTIGATION:
{player_action}

Follow your system instructions exactly. Provide immersive narrative first, then the state JSON block."""

    try:
        response = client.chat.completions.create(
            model=st.session_state.get("model", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": gm_prompt},
                {"role": "user", "content": user_context}
            ],
            temperature=0.68,
            max_tokens=2200,
        )
        full_response = response.choices[0].message.content
        
        # Split narrative and JSON state update
        state_update = {}
        narrative = full_response
        
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', full_response, re.DOTALL | re.IGNORECASE)
        if json_match:
            parsed = extract_json_from_text(json_match.group(1))
            if parsed:
                state_update = parsed
            narrative = full_response[:json_match.start()].strip()
        
        # Fallback: try to find any JSON at the end
        if not state_update:
            parsed = extract_json_from_text(full_response)
            if parsed and isinstance(parsed, dict) and "new_discovered_clue_ids" in parsed:
                state_update = parsed
                # Remove the JSON from narrative if it was appended raw
                narrative = re.sub(r'\n?\{.*"new_discovered_clue_ids".*\}\s*$', '', narrative, flags=re.DOTALL).strip()
        
        return narrative, state_update
        
    except Exception as e:
        return f"The detective pauses, looking thoughtful. (Technical issue contacting the Game Master: {str(e)[:100]})", {}

def evaluate_accusation(accusation: dict) -> dict:
    """Have the AI judge the player's formal accusation."""
    client = get_openai_client()
    judge_prompt = load_prompt("accusation_judge.txt")
    
    bible = st.session_state.mystery_bible
    
    context = f"""MYSTERY BIBLE (TRUTH):
{json.dumps(bible, indent=2, ensure_ascii=False)}

PLAYER'S FORMAL ACCUSATION:
{json.dumps(accusation, indent=2)}

Evaluate this accusation carefully according to your instructions. Output ONLY the JSON verdict."""

    try:
        response = client.chat.completions.create(
            model=st.session_state.get("model", "llama-3.3-70b-versatile"),
            messages=[
                {"role": "system", "content": judge_prompt},
                {"role": "user", "content": context}
            ],
            temperature=0.3,
            max_tokens=1200,
        )
        content = response.choices[0].message.content
        judgment = extract_json_from_text(content)
        
        if not judgment or "verdict" not in judgment:
            return {
                "verdict": "INCORRECT",
                "overall_score": 25,
                "feedback_narrative": "The case reviewer encountered an issue parsing the evaluation. Please try submitting your accusation again.",
                "reveal_solution": False
            }
        return judgment
        
    except Exception as e:
        return {
            "verdict": "INCORRECT",
            "overall_score": 0,
            "feedback_narrative": f"Technical error during judgment: {str(e)[:150]}",
            "reveal_solution": False
        }

def process_player_action(action_text: str):
    """Central function to handle any player action (chat or quick buttons)."""
    if not action_text or not st.session_state.game_active:
        return
    
    st.session_state.chat_history.append({
        "role": "player",
        "content": action_text,
        "timestamp": datetime.now().isoformat()
    })
    
    with st.spinner("🕵️ The Game Master considers your move..."):
        narrative, state_update = gm_respond(action_text)
    
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": narrative,
        "timestamp": datetime.now().isoformat()
    })
    
    # Apply state changes from GM
    if state_update:
        st.session_state.discovered_clues.update(state_update.get("new_discovered_clue_ids", []))
        st.session_state.visited_locations.update(state_update.get("new_visited_locations", []))
        st.session_state.interviewed_suspects.update(state_update.get("interviewed_suspects", []))
        if state_update.get("narrative_summary"):
            st.session_state.last_narrative_summary = state_update["narrative_summary"]
    
    st.rerun()

# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render the left configuration and new case sidebar."""
    with st.sidebar:
        st.header("⚙️ Detective Configuration")
        
        provider_label = st.selectbox(
            "AI Provider (Free options available)",
            [
                "Groq (Free) - Recommended (fastest)",
                "Gemini (Free) - Very generous limits",
                "xAI (Grok)",
                "OpenAI"
            ],
            index=0,
            help="Both Groq and Gemini are completely FREE. Groq is fastest. Gemini has very high free limits."
        )
        if "Groq" in provider_label:
            st.session_state.provider = "Groq (Free)"
        elif "Gemini" in provider_label:
            st.session_state.provider = "Gemini (Free)"
        elif "xAI" in provider_label:
            st.session_state.provider = "xAI"
        else:
            st.session_state.provider = "OpenAI"
        
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.get("api_key", ""),
            help="Groq: groq.com | Gemini: aistudio.google.com/app/apikey (both free, no card needed)"
        )
        st.session_state.api_key = api_key
        
        # Smart default model per provider
        if st.session_state.provider == "Groq (Free)":
            default_model = "llama-3.3-70b-versatile"
        elif st.session_state.provider == "Gemini (Free)":
            default_model = "gemini-1.5-flash"
        elif st.session_state.provider == "xAI":
            default_model = "grok-3"
        else:
            default_model = "gpt-4o-mini"
        
        model = st.text_input("Model Name", value=st.session_state.get("model", default_model))
        st.session_state.model = model
        
        st.divider()
        
        st.header("🎲 Start a New Case")
        
        custom_theme = st.text_input(
            "Custom Theme (optional)",
            placeholder="e.g. '1920s country house' or 'luxury cruise ship 2025' or 'Mars research colony'",
            help="Leave blank for a completely random atmospheric mystery."
        )
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🆕 Generate New Mystery", type="primary", use_container_width=True):
                if not st.session_state.api_key:
                    st.error("Enter your API key first!")
                else:
                    with st.spinner("Weaving a fiendishly clever and fair mystery... (20-45 seconds)"):
                        mystery = generate_mystery(custom_theme)
                    if mystery:
                        reset_game_state()
                        st.session_state.mystery_bible = mystery
                        st.session_state.game_active = True
                        st.session_state.chat_history = []
                        st.success(f"**Case File Opened:** {mystery['title']}")
                        st.rerun()
        
        with col_btn2:
            if st.button("🔄 Reset Everything", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                initialize_session_state()
                st.rerun()
        
        if st.session_state.get("mystery_bible"):
            st.divider()
            st.subheader("📁 Current Case")
            m = st.session_state.mystery_bible
            st.markdown(f"**{m['title']}**")
            st.caption(f"{m['setting']['location_name']} • {m['setting']['era']}")
            st.caption(f"Difficulty: {m.get('difficulty', 'Medium')}")
            
            if st.button("❌ Abandon This Case", use_container_width=True):
                reset_game_state()
                st.rerun()

def render_welcome_screen():
    """Show instructions when no active case."""
    st.title("🕵️ Infinite Murder Mysteries")
    st.markdown("### AI-Powered Fair-Play Text Detective Adventures")
    
    st.markdown("""
    Welcome, Detective. Every case you generate is **completely original**, **logically consistent**, and **fairly solvable** 
    using only the clues you discover. No plot holes. No impossible leaps. Just pure deductive satisfaction.
    """)
    
    with st.expander("How This Works (Read First)", expanded=True):
        st.markdown("""
        **1. Generate**  
        Click the button in the sidebar. The AI creates a full "Mystery Bible" (victim, 5-7 suspects, timeline, 10-15 clues, 
        complete solution with motive/method/opportunity, and red herrings). This stays hidden from you.
        
        **2. Investigate**  
        Use natural language in the chat or the quick-action buttons:
        - Search locations
        - Interview suspects
        - Examine evidence
        - Ask about timelines, relationships, inconsistencies
        
        The Game Master AI answers **perfectly consistently** with the hidden truth. It reveals clues only when your 
        actions logically uncover them.
        
        **3. Deduce**  
        Use the built-in notebook to track alibis, contradictions, and theories.
        
        **4. Accuse**  
        When ready, open the Accusation form, name the killer + method + motive + opportunity. 
        The AI judge gives detailed, fair feedback. Get it right and earn the full dramatic reveal.
        
        **Why it's solvable:** The generator is heavily prompted to create *fair-play* mysteries (Agatha Christie style). 
        The Game Master is instructed never to contradict the bible or give unearned information.
        """)
    
    st.info("👈 **Enter your FREE API key** (Groq or Gemini) in the sidebar, then click **Generate New Mystery**. Both are completely free with no credit card needed.")
    
    st.markdown("---")
    st.caption("Two completely free options: Groq (fastest) or Gemini (very high limits). Get keys at groq.com or aistudio.google.com/app/apikey — no card needed.")

def render_game_ui():
    """Main game interface when a mystery is active."""
    mystery = st.session_state.mystery_bible
    
    # Header
    st.title(f"🕵️ {mystery['title']}")
    
    # Progress bar
    total_clues = len(mystery.get("clues", []))
    found = len(st.session_state.discovered_clues)
    progress = found / total_clues if total_clues > 0 else 0
    
    col_prog1, col_prog2, col_prog3 = st.columns([3, 2, 2])
    with col_prog1:
        st.progress(progress, text=f"Evidence Found: {found} / {total_clues}")
    with col_prog2:
        interviewed = len(st.session_state.interviewed_suspects)
        total_sus = len(mystery.get("suspects", []))
        st.caption(f"**Suspects Interviewed:** {interviewed} / {total_sus}")
    with col_prog3:
        st.caption(f"**{mystery['setting']['location_name']}** • {mystery['setting']['era']}")
    
    # Main layout
    left_col, right_col = st.columns([2.6, 1.4], gap="medium")
    
    # ========== LEFT: Investigation Log ==========
    with left_col:
        st.subheader("📜 Investigation Log")
        
        # Chat messages
        chat_container = st.container(height=420, border=True)
        with chat_container:
            if not st.session_state.chat_history:
                st.info("You have arrived at the scene. Use the chat below or the Quick Actions on the right to begin investigating.")
                st.caption("Example first actions: 'Search the room where the body was found' or 'Talk to the first person you see'.")
            
            for msg in st.session_state.chat_history:
                if msg["role"] == "player":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant", avatar="🕵️"):
                        st.write(msg["content"])
        
        # Chat input
        action = st.chat_input(
            "What do you do or ask? (e.g. 'Search the library desk' or 'Ask Mrs. Blackwood about the argument')"
        )
        if action:
            process_player_action(action)
        
        # Quick starter if no history
        if not st.session_state.chat_history:
            if st.button("🚀 Begin Investigation — Arrive at the Scene", type="secondary", use_container_width=True):
                initial_action = (
                    f"You have just arrived at {mystery['setting']['location_name']}. "
                    f"The body of {mystery['victim']['name']} was discovered {mystery['victim']['body_discovery']}. "
                    "Describe the initial atmosphere, who is present, and any obvious details about the scene or body. "
                    "Do not reveal any hidden clues yet unless logically visible on first inspection."
                )
                process_player_action(initial_action)
    
    # ========== RIGHT: Quick Actions + Evidence + Notebook ==========
    with right_col:
        # Quick Actions
        with st.container(border=True):
            st.markdown("**⚡ Quick Actions**")
            
            st.markdown("**🔍 Search Locations**")
            for loc in mystery["setting"].get("key_locations", []):
                if st.button(f"Search {loc}", key=f"quick_loc_{loc}", use_container_width=True):
                    process_player_action(f"Search the {loc} thoroughly for clues, documents, or anything out of place.")
            
            st.markdown("**🗣️ Interview Suspects**")
            for s in mystery.get("suspects", []):
                btn_label = f"Interview {s['name']}"
                if st.button(btn_label, key=f"quick_int_{s['id']}", use_container_width=True):
                    process_player_action(
                        f"Interview {s['name']}. Ask about their relationship with the victim, "
                        f"where they were at the time of death, and if they noticed anything suspicious."
                    )
        
        # Discovered Evidence
        with st.container(border=True):
            st.markdown("**🔎 Discovered Evidence**")
            if st.session_state.discovered_clues:
                for clue_id in sorted(st.session_state.discovered_clues):
                    clue = next((c for c in mystery["clues"] if c["id"] == clue_id), None)
                    if clue:
                        with st.expander(f"📌 {clue['name']}", expanded=False):
                            st.write(clue["description"])
                            st.caption(f"Found in: {clue['location']}")
                            if st.checkbox("📝 Add to notebook", key=f"add_note_{clue_id}"):
                                note_text = f"\n• [{clue['name']}] {clue['description']}"
                                if note_text not in st.session_state.notebook:
                                    st.session_state.notebook += note_text
                                    st.rerun()
            else:
                st.caption("No evidence logged yet. Start searching!")
        
        # Notebook
        with st.container(border=True):
            st.markdown("**📓 Your Notebook**")
            st.session_state.notebook = st.text_area(
                label="Private notes & theories",
                value=st.session_state.get("notebook", ""),
                height=160,
                key="notebook_live",
                label_visibility="collapsed",
                placeholder="Track alibis, contradictions, timelines, and your emerging theory here..."
            )
    
    # ========== ACCUSATION SECTION ==========
    st.divider()
    
    with st.expander("⚖️ **MAKE YOUR FORMAL ACCUSATION** (When you are ready to name the killer)", expanded=False):
        st.warning("This is the moment of truth. Be as specific as possible about **method**, **motive**, and **opportunity**.")
        
        accused_name = st.selectbox(
            "Who is the killer?",
            options=[s["name"] for s in mystery["suspects"]],
            index=0
        )
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            method = st.text_area(
                "Method — How exactly was the victim killed? (Include any staging, weapon, poison, etc.)",
                height=90,
                placeholder="e.g. Poisoned with digitalis in the evening brandy, then the scene was staged to look like a heart attack..."
            )
        with col_m2:
            motive = st.text_area(
                "Motive — Why did they do it?",
                height=90,
                placeholder="e.g. To inherit the estate before the victim could change the will and cut them out..."
            )
        
        opportunity = st.text_area(
            "Opportunity — How did they commit the crime despite alibis, witnesses, or locked doors?",
            height=90,
            placeholder="e.g. They slipped away during the 10-minute power outage caused by the storm and used the hidden passage..."
        )
        
        extra = st.text_area(
            "Key evidence or reasoning that supports your theory (optional but recommended)",
            height=70
        )
        
        if st.button("📤 Submit Accusation to the Case Reviewer", type="primary", use_container_width=True):
            if not method or not motive or not opportunity:
                st.error("Please fill in method, motive, and opportunity for a proper judgment.")
            else:
                accusation = {
                    "accused_name": accused_name,
                    "method": method,
                    "motive": motive,
                    "opportunity": opportunity,
                    "extra_reasoning": extra
                }
                with st.spinner("The Case Reviewer is carefully weighing your evidence against the facts..."):
                    judgment = evaluate_accusation(accusation)
                st.session_state.last_judgment = judgment
                st.rerun()
    
    # Show last judgment
    if st.session_state.get("last_judgment"):
        j = st.session_state.last_judgment
        st.subheader("📋 Case Reviewer Verdict")
        
        verdict = j.get("verdict", "INCORRECT")
        score = j.get("overall_score", 0)
        
        if verdict == "CORRECT":
            st.success(f"🎉 **EXCELLENT DETECTIVE WORK!** You solved it. Score: {score}/100")
            if j.get("reveal_solution"):
                with st.expander("**FULL SOLUTION — YOU EARNED THIS**", expanded=True):
                    sol = mystery["solution"]
                    st.markdown(f"### The killer was **{sol['killer_name']}**")
                    st.markdown(f"**Motive:** {sol['motive']}")
                    st.markdown(f"**Method:** {sol['method']}")
                    st.markdown(f"**Opportunity:** {sol['opportunity']}")
                    st.markdown("### Complete Explanation")
                    st.write(sol.get("full_explanation", "The pieces all fit together perfectly."))
                    st.balloons()
                    st.success("Case closed. Thank you for playing. Generate a new mystery from the sidebar to continue.")
        elif verdict == "PARTIALLY_CORRECT":
            st.info(f"**Very close.** You have some of it right. Score: {score}/100")
            st.write(j.get("feedback_narrative", ""))
            if j.get("strengths"):
                st.success(f"**Strengths:** {j['strengths']}")
            if j.get("weaknesses"):
                st.warning(f"**Still missing:** {j['weaknesses']}")
        else:
            st.error(f"**Not quite.** Score: {score}/100")
            st.write(j.get("feedback_narrative", ""))
        
        st.caption("You can refine your theory and submit again, or gather more evidence first.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    initialize_session_state()
    render_sidebar()
    
    if st.session_state.get("game_active") and st.session_state.get("mystery_bible"):
        render_game_ui()
    else:
        render_welcome_screen()
    
    # Footer
    st.markdown("---")
    st.caption(
        "Built for coherent, infinite, fairly solvable AI murder mysteries. "
        "All stories are generated on-demand and remain consistent thanks to the hidden Mystery Bible architecture."
    )

if __name__ == "__main__":
    main()