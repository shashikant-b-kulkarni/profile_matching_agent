from typing import TypedDict, Annotated, Any
from click import prompt
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
#from langchain_community.chat_models import ChatOllama
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from mcp_astro_chatbot import MCP_ChatBot
import json
import asyncio
import streamlit as st
import os 
import argparse


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from datetime import datetime, time
import time as time_s
import logging
from logging import basicConfig, getLogger, INFO
import zipfile
from logging.handlers import RotatingFileHandler

class ZippedRotatingFileHandler(RotatingFileHandler):
    def doRollover(self):
        super().doRollover()
        # The 'old' file is now at baseFilename.1
        old_log = self.baseFilename + ".1"
        if os.path.exists(old_log):
            with zipfile.ZipFile(f"{old_log}.zip", 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(old_log, os.path.basename(old_log))
            os.remove(old_log)

# Setup
handler = ZippedRotatingFileHandler("profile-matcher.log", maxBytes=1024*1024, backupCount=5)
logging.basicConfig(handlers=[handler], level=logging.INFO)
logger = logging.getLogger("profile-matcher")

automated_run = False




def custom_message_reducer(old, new):
    """ def custom_message_reducer(old, new):
    logger.debug(f"\n\n Entered custom_message_reducer old:{old}")
    logger.debug(f"\n\n Entered custom_message_reducer new:{new}")
    if automated_run:
        return (old + new) #Preserve all the messages, we need them.
    if "stage" in st.session_state and "reevaluate" == st.session_state.stage:
        old = []
    return (old + new)[-10:]  # keep last 10 """

    return (old + new)

# --- State ---
class AgentState(TypedDict):
    messages: Annotated[list, custom_message_reducer]   # 👈 reducer applied
    stage: str
    intent: str
    login_successful: bool
    boy_profile_fetch_success: bool
    girl_profile_fetch_success: bool
    boy_profile_url: str
    girl_profile_url: str
    boy_profile: dict
    girl_profile: dict
    analysis_result: str 
    mcp_client: MCP_ChatBot
    model_choice: str
    llm: Any
    tokenizer: AutoTokenizer
    local_model: AutoModelForCausalLM
    automated :bool


def initialize_local_model(model_choice: str):
    #MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
    logger.info(f"\n\n Initializing local model: {model_choice}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_choice, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # important for batching

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        model_choice,
        torch_dtype=torch.float16,   # or bfloat16 if supported
        device_map="auto"            # auto GPU/CPU placement
    )

    model.eval()
    logger.info(f"Local model '{model_choice}' loaded successfully.")

    return tokenizer, model

def build_prompt_for_local_model(tokenizer, system_prompt, user_prompt, *fmt_args):
        logger.info("\n\n Building prompt for local model")
        user_msg = user_prompt.format(*fmt_args)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt, user_msg

def generate_local_model_response(tokenizer, model, prompt, max_new_tokens=3000):
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    input_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_len:]

    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    #response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove prompt from output (important)
    #return response[len(prompt):].strip()
    return response.strip()

# --- LLM ---
#llm = ChatOpenAI(
#    base_url="https://api.groq.com/openai/v1",
#    api_key=os.getenv("GROQ_API_KEY"),
#    model="llama-3.3-70b-versatile"
#)

#llm = ChatOllama(model="llama3")
VOCAREUM_BASE_URL  = os.getenv("VOCAREUM_BASE_URL") #"https://openai.vocareum.com/v1"          # custom proxy
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY") #"voc-1784561922175350473180169ac23371fb5b7.30042371"
PROFILE_DATA_PATH = os.getenv("PROFILES_DATA_PATH", "profiles_data")

#llm_gpt4o = ChatOpenAI(
#    openai_api_key=OPENAI_API_KEY,
#    base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
#    model="gpt-5.2",                    # ← CHANGED: was "gpt-4"
#    temperature=0.1,
#    max_tokens=2500,
#    tags=["profile_matcher"],
#)

#llm = llm_gpt4o

def get_llm(model_choice: str):
    if model_choice == "LLaMA 3 (Ollama)":
        logger.info("\n\n ** Entered Model choice : LLaMA 3 (Ollama)")
        return ChatOllama(model="llama3")
    
    elif model_choice == "Gpt OSS:20b (Ollama)":
        logger.info("\n ** Entered Model Choice: Gpt OSS:20b (Ollama)")
        return ChatOllama(model="gpt-oss:20b", temperature=0.1)

    elif model_choice == "Groq (LLaMA 70B)":
        logger.info("\n\n ** Entered Model choice : Groq (LLaMA 70B)")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
    elif model_choice == "Groq openai/gpt-oss-120b":
        logger.info("\n\n ** Entered Model choice : Groq openai/gpt-oss-120b")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )
    elif model_choice == "Groq qwen/qwen3-32b":
        logger.info("\n\n ** Entered Model choice : Groq qwen/qwen3-32b")
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="qwen/qwen3-32b"
        )
    elif model_choice == "Groq mixtral-8x7b":
        logger.info("\n\n ** Entered Model choice : Groq mixtral-8x7b")
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="mixtral-8x7b"
        )
    elif model_choice == "OpenAI-Gpt-5.2":
        logger.info("\n\n ** Entered Model choice : OpenAI-Gpt-5.2")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
            model="gpt-5.2",                    # ← CHANGED: was "gpt-4"
            temperature=0.1,
            #    max_tokens=2500,
            tags=["profile_matcher"],
        )
    elif model_choice == "OpenAI-gpt-4o-mini":
        logger.info("\n\n ** Entered Model choice : OpenAI-gpt-4o-mini")

        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
            model="gpt-4o-mini",                    # ← CHANGED: was "gpt-4"
            temperature=0.1,
            #    max_tokens=2500,
            tags=["profile_matcher"],
        )
    elif model_choice == "OpenAI-gpt-5.4-mini":
        logger.info("\n\n ** Entered Model choice : OpenAI-gpt-5.4-mini")

        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
            model="gpt-5.4-mini",                    # ← CHANGED: was "gpt-4"
            temperature=0.1,
            #    max_tokens=2500,
            tags=["profile_matcher"],
        )







# --- Nodes ---

def intent_detection(state: AgentState):
    logger.info(f"\n ***** Entered intent_detection with stage:{state["stage"]}")
    stage = state["stage"].lower()

    if "input" in stage:
        intent = "matchmaking"
    elif "chat" in stage:
        intent = "general"
    elif "reevaluate" in stage:
        intent = "reevaluate"

    return {"intent": intent}


def general_chat(state: AgentState):
    #prompt = f"User: {state['message']}\nAssistant:"
    llm = state["llm"]
    res = llm.invoke(state["messages"])

    logger.debug(f"n\n ******* GENERAL CHAT NODE RESP:{res}")

    return {"messages": [{"role": "assistant", "content":res.content}]}

MATCH_ANALYSIS_FORMAT= """
# Basic Details (Girl’s POV)

- **Boy’s Full Name:** Devashish Chandrashekhar Thakar  
- **Girl’s Full Name:** Mrunal Kulkarni  
- **Boy’s Profile Picture (200x200):**  
  ![Devashish Chandrashekhar Thakar](https://res.cloudinary.com/wiwaha/image/upload/t_profile_view/AnuroopVar/282026105453PM_672259_20260208_223019(2).jpg)

### Quick snapshot (non-astro):
- **Caste/Subcaste:** Both Brahmin, Deshastha Rigwedi (strong match)
- **Mother Tongue:** Both Marathi (match)
- **Marital Status:** Both Never Married (match)
- **Location:** Boy works Hyderabad; Girl works Pune (manageable but requires relocation/commute planning)

---

# Astrological Compatibility

## Boy Basic Astro Details (from Kundali Brief)
> "Boy Basic Astro Details:{'Lagna Rashi': 'Leo', 'Moon Sign': 'Leo', 'Nakshatra': 'Magha', 'Charan': 1}  
Ashtakoot Points(Orig):25  
Dashakoot Points(Orig):24  
... Graha Maitri: Enemy (Shatru)(1) ...  
Overall Scores: {'ashtakoot': 25, 'dashakoot': 24, ... 'total': 57}"

**Reference scores:**
- **Ashtakoot:** 25/36 (good)
- **Dashkoot:** 24 (generally favorable)
- **Kundali Detailed Note (important):** The report explicitly flags **Manglik mismatch (one Manglik, one non-Manglik)** and says **“match is not recommended”** under Manglik section.

## Doshas (No-Go Checks)
| Dosha | Value |
|---|---|
| Non Cancelled Shadashtak Yog | False |
| Nadi Dosha | Safe |
| Shani Placement | Neutral |
| Lagna Seventh Lord Placement | <span style="color:red">Critical-MALEFIC(Saturn)</span> |
| Navamsa Seventh Lord Placement | Critical-BENEFIC(Mercury) |

**Astro conclusion (from Girl’s POV):**
- Despite decent **Ashtakoot/Dashkoot**, the **Kundali Detailed Report** contains a strong warning: **Manglik mismatch** and **“match is not recommended.”**
- Additionally, **Lagna 7th lord placement is marked MALEFIC**, which is treated as a **red-flag criterion** per your rules.

---

# About Me and About Family

## About Me Compatibility
- **Boy:** Independent thinker, quiet/thoughtful, vegetarian & teetotaler, long-distance running, music, reading; liberal/caring family; religious believer but not conservative.
- **Girl:** Composed, reasonable, ethics-oriented; wants balance of family + career; enjoys drives/meetups; family-managed profile (self + parents).

**Fit assessment (Girl POV):**
- **Values:** Both prioritize **family**, appear **mature, balanced and ethical** → good alignment.
- **Lifestyle temperament:** Boy is “quiet/thoughtful”; Girl is “friendly/meetups” → generally compatible if both respect social/alone-time needs.
- **Religious approach:** Boy is religious but not ritualistic; Girl hasn’t specified ritual level—likely manageable unless Girl expects more traditional practice.

## About Family Compatibility
- **Boy’s family:** Academicians/professionals; strong education/research/medical background; liberal and caring.
- **Girl’s family:** Family of three, closely connected with extended family (Bengaluru/Pune/Kolhapur).

**Fit assessment:**
- **Social/education background:** Both families sound stable; boy’s family is highly academic/professional—generally compatible with girl’s educated background (BE + MBA).
- **Family structure expectations:** Girl’s close-knit small family may integrate well, but boy’s broader, accomplished family network could feel “high-expectation” unless communication is clear.

---

# Hobbies & Lifestyle Compatibility

| Area | Boy | Girl | Compatibility |
|---|---|---|---|
| Reading | Reading, Writing | Reading | Strong common interest |
| Outdoors/Fitness | Trekking, jogging, gym, yoga | Swimming, gym | Good (both active) |
| Socializing | Pubbing: No; Hotelling: Occasionally | Pubbing: Occasionally; Hotelling: Yes | Moderate mismatch (needs alignment) |
| Diet | Veg | Eggetarian | Potential friction (see expectations section) |

---

# Expectations Cross-Check (Both Sides)

## Boy’s Expectations vs Girl’s Profile
| Expectation | Preference | Information from Girl’s profile against criterion |
|---|---|---|
| Marital Status: Never Married | (Must Have) | Never Married ✅ |
| Caste: Brahmin | (Must Have) | Brahmin ✅ |
| Mother Tongue: Marathi | (Must Have) | Marathi ✅ |
| Religion: Hindu | (Must Have) | Hindu ✅ |
| Age: 24–34 | (Must Have) | DOB 27 Jun 1994 → Age ~31 ✅ |
| Education Level: Post Graduate | (Highly Preferred) | MBA (Post Graduate) ✅ |
| Working Partner: Must | (Preferred) | Working in MNC (Icertis) ✅ |
| Work Country: India | (Highly Preferred) | India ✅ |
| Diet: Veg | (Preferred) | **Eggetarian** ⚠️ (possible concern if strictly veg expected) |
| Drink: No | (Preferred) | No ✅ |
| Smoke: No | (Preferred) | No ✅ |
| Pubbing: No | (Preferred) | **Occasionally** ⚠️ |
| Cooking Skill: Yes | (Preferred) | Basic (generally acceptable) ✅/⚠️ |
| Financial Background: Affluent/Upper Middle | (Preferred) | Upper Middle Class ✅ |

## Girl’s Expectations vs Boy’s Profile
| Expectation | Preference | Information from Boy’s profile against criterion |
|---|---|---|
| Marital Status: Never Married | (Must Have) | Never Married ✅ |
| Caste: Brahmin | (Must Have) | Brahmin ✅ |
| Age: 31–34 | (Preferred) | DOB 21-Jan-1992 → Age ~34 ✅ (upper edge) |
| Education Level: Graduate/PG/International | (Highly Preferred) | PhD (very strong) ✅ |
| Working Partner: Must | (Preferred) | Working (AMGEN) ✅ |
| Work Country: India | (Preferred) | India ✅ |
| Work State: Maharashtra/Karnataka | (Highly Preferred) | **Telangana (Hyderabad)** ⚠️ |
| Work City: Mumbai/Pune/Bangalore etc. | (Preferred) | **Hyderabad** ⚠️ |
| Diet: Eggetarian | (Preferred) | **Veg** ✅ (usually acceptable; reverse is harder) |
| Drink: Occasionally | (Preferred) | **No** ✅ |
| Hotelling: Yes | (Preferred) | Occasionally ⚠️ |
| Pubbing: Occasionally | (Preferred) | No ✅ |
| Other: “swatahche ghar, sthavar asave” (own home/asset), supportive family | (Text) | Own home (Own, bungalow), supportive professional family ✅ |

**Expectation summary (Girl POV):**
- **Strong alignment:** caste, language, education, marital status, family values, non-smoking/non-drinking.
- **Key practical mismatches:** **Boy’s location (Hyderabad) vs Girl’s preferred Maharashtra cities**; and **diet preference mismatch (Boy prefers veg; Girl is eggetarian)**.
- **Lifestyle differences:** pubbing/hotelling preferences differ but are negotiable.

---

# Family Financial Status Compatibility

| Parameter | Boy | Girl | Compatibility |
|---|---|---|---|
| Boy Annual Income | ₹80.16 LPA | — | Strong earning stability |
| Girl Annual Income | — | ₹22.00 LPA | Good earning stability |
| Family Status | Upper Middle Class | Upper Middle Class | Match ✅ |
| Family Income | Above 50 Lac | 20 Lac to 50 Lac | Generally compatible (boy side higher) ✅ |
| Assets | Bungalow, own home, 4-wheeler | Land, own home, 4W+2W | Strong asset base on both sides ✅ |

---

# Age & Height Compatibility

| Factor | Boy | Girl | Compatibility |
|---|---|---|---|
| Age (approx.) | ~34 | ~31 | Good (within both expected ranges) ✅ |
| Height | 5'8" | 5.3 (~5'3") | Good typical gap ✅ |

---

## Overall Compatibility (Girl’s POV — consolidated)
- **Non-astro compatibility:** Generally **strong** (education, family status, values, language, caste/subcaste, career orientation).
- **Major deciding factor:** **Astrology section is a serious blocker** because the **Kundali report explicitly states “match is not recommended” due to Manglik mismatch**, and **Lagna 7th lord placement is MALEFIC** (red flag).
- **Practical blockers to resolve if proceeding despite astro:** Pune vs Hyderabad location alignment + diet expectation (veg vs eggetarian).

"""

MATCH_ANALYSIS_FORMAT_V1= """
# Basic Details (Girl’s POV)

- **Boy’s Full Name:** <boy's full name here>  
- **Girl’s Full Name:** <girl's full name here> 
- **Boy’s Profile Picture (200x200):**  
  ![<boy's full name here>](<boy's profile picture url here>)

**Quick snapshot (non-astro):**
- **Caste/Subcaste:** <caste/subcaste compatibility details here> (match/mismatch)
- **Mother Tongue:** <language compatibility details here> (match/mismatch)
- **Marital Status:** <marital status compatibility details here> (match/mismatch)
- **Location:** <location compatibility details here> (manageable/concern)

---

# Astrological Compatibility

## Boy Basic Astro Details (from Kundali Brief)
> "Boy Basic Astro Details:{'Lagna Rashi': '<boy's lagna_rashi>', 'Moon Sign': '<boy's moon_sign>', 'Nakshatra': '<boy's nakshatra>', 'Charan': <boy's charan>}  
Ashtakoot Points(Orig):<ashtakoot_points>  
Dashakoot Points(Orig):<dashakoot_points>  
... Graha Maitri: <Graha Maitri details here> ...  
Overall Scores: {'ashtakoot': <ashtakoot_points>, 'dashakoot': <dashakoot_points>, ... 'total': <total_points>}"

**Reference scores:**
- **Ashtakoot:** <ashtakoot_points>/36 (<ashtakoot assessment here>)
- **Dashkoot:** <dashakoot_points> (<dashakoot assessment here>)
- **Kundali Detailed Note (important):** <kundali detailed note here, especially if there are any red flags or strong warnings>

## Doshas (No-Go Checks)
| Dosha | Value |
|---|---|
| Non Cancelled Shadashtak Yog | <shadashtak yog value> |
| Nadi Dosha | <nadi dosha value> |
| Shani Placement | <shani placement value> |
| Lagna Seventh Lord Placement | <lagna seventh lord placement value marked with red color if Critical Malefic> |
| Navamsa Seventh Lord Placement | <navamsa seventh lord placement value marked with red color if Critical Malefic> |

**Astro conclusion (from Girl’s POV):**
- <Astro compatibility conclusion here, summarizing key points and especially noting any critical red flags such as Manglik mismatch or malefic placements.>

---

# About Me and About Family

## About Me Compatibility
- **Boy:** <Summary of boy's "About Me" section here, highlighting key traits, values, lifestyle preferences, family background, and religious approach.>
- **Girl:** <Summary of girl's "About Me" section here, highlighting key traits, values, lifestyle preferences, family background, and religious approach.> 

**Fit assessment (Girl POV):**
- **Values:** <Assessment of values compatibility here, noting areas of strong alignment or potential concern.>
- **Lifestyle temperament:** <Assessment of lifestyle/temperament compatibility here, noting if they are generally compatible or if there are significant differences that may require adjustment.>
- **Religious approach:** <Assessment of religious compatibility here, noting if they are likely to be compatible or if there may be friction.>

## About Family Compatibility
- **Boy’s family:** <Summary of boy’s family background here, including profession, education, values, and any other relevant details.>
- **Girl’s family:** <Summary of girl’s family background here, including profession, education, values, and any other relevant details.>

**Fit assessment:**
- **Social/education background:** <Assessment of social and educational background compatibility here, noting if they are generally compatible or if there are significant differences.>
- **Family structure expectations:** <Assessment of family structure and expectations compatibility here, noting if they are likely to integrate well or if there may be challenges.>

---

# Hobbies & Lifestyle Compatibility

| Area | Boy | Girl | Compatibility |
|---|---|---|---|
| <Area 1> | <Boy's interest> | <Girl's interest> | <Compatibility assessment> |
| <Area 2> | <Boy's interest> | <Girl's interest> | <Compatibility assessment> |

---

# Expectations Cross-Check (Both Sides)

## Boy’s Expectations vs Girl’s Profile
| Expectation | Preference | Information from Girl’s profile against criterion |
|---|---|---|
| Marital Status: <boy's marital status expectation> | (<preference>) | <girl's marital status> <tik mark if compatible> |
| Caste: <boy's caste expectation> | (<preference>) | <girl's caste> <tik mark if compatible> |
| Mother Tongue: <boy's mother tongue expectation> | (<preference>) | <girl's mother tongue> <tik mark if compatible> |
| Religion: <boy's religion expectation> | (<preference>) | <girl's religion> <tik mark if compatible> |
| Age: <boy's age expectation> | (<preference>) | <girl's age> <tik mark if compatible> |
| Education Level: <boy's education level expectation> | (<preference>) | <girl's education level> <tik mark if compatible> |
| Working Partner: <boy's working partner expectation> | (<preference>) | <girl's work details> <tik mark if compatible> |
| Work Country: <boy's work country expectation> | (<preference>) | <girl's work country> <tik mark if compatible> |
| Diet: <boy's diet expectation> | (<preference>) | <girl's diet, indicate any incompatible aspects> <tik mark if compatible> |
| Drink: <boy's drink expectation> | (<preference>) | <does girl drink?> <tik mark if compatible> |
| Smoke: <boy's smoke expectation> | (<preference>) | <does girl smoke?> <tik mark if compatible> |
| Pubbing: <boy's pubbing expectation> | (<preference>) | <does girl go to pubs?> <tik mark if compatible> |
| Cooking Skill: <boy's cooking skill expectation> | (<preference>) | <girl's cooking skill> <tik mark if compatible> |
| Financial Background: <boy's financial background expectation> | (<preference>) | <girl's financial background> <tik mark if compatible> |

## Girl’s Expectations vs Boy’s Profile
| Expectation | Preference | Information from Boy’s profile against criterion |
|---|---|---|
| Marital Status: <girl's marital status expectation> | (<preference>) | <boy's marital status> <tik mark if compatible> |
| Caste: <girl's caste expectation> | (<preference>) | <boy's caste> <tik mark if compatible> |
| Age: <girl's age expectation> | (<preference>) | <boy's age> <tik mark if compatible> |
| Education Level: <girl's education level expectation> | (<preference>) | <boy's education level> <tik mark if compatible> |
| Working Partner: <girl's working partner preference> | (<preference>) | <boy's work details> <tik mark if compatible> |
| Work Country: <girl's work country preference> | (<preference>) | <boy's work country> <tik mark if compatible> |
| Work State: <girl's work state preference> | (<preference>) | <boy's work state> <tik mark if compatible> |
| Work City: <girl's work city preference> | (<preference>) | <boy's work city> <tik mark if compatible> |
| Diet: <girl's diet expectation> | (<preference>) | <boy's diet> <tik mark if compatible> |
| Drink: <girl's drink expectation> | (<preference>) | <does boy drink?> <tik mark if compatible> |
| Hotelling: <girl's hotelling expectation> | (<preference>) | <does boy go to hotels?> <tik mark if compatible> |
| Pubbing: <girl's pubbing expectation> | (<preference>) | <does boy go to pubs?> <tik mark if compatible> |
| Other: <girl's other expectations> | (Text) | <compatible aspects from boy's profile> <tik mark if compatible> |

**Expectation summary (Girl POV):**
- **Strong alignment:** <summary of areas with strong alignment here> (e.g. caste, language, education, marital status, family values, non-smoking/non-drinking).**
- **Key practical mismatches:** **<summary of practical mismatches here>**.
- **Lifestyle differences:** <summary of lifestyle differences here, noting if they are negotiable or significant.>

---

# Family Financial Status Compatibility

| Parameter | Boy | Girl | Compatibility |
|---|---|---|---|
| Boy Annual Income | <boy's annual income> | — | <comment> |
| Girl Annual Income | — | <girl's annual income> | <comment> |
| Family Status | <boy's family status> | <girl's family status> | <Match or mismatch> <tik mark if compatible> |
| Family Income | <boy's family income> | <girl's family income> | <compatibility comment> <tik mark if compatible> |
| Assets | <boy's assets> | <girl's assets> | <comment> |

---

# Age & Height Compatibility

| Factor | Boy | Girl | Compatibility |
|---|---|---|---|
| Age (approx.) | ~<boy's age> | ~<girl's age> | <compatibility comment> <tik mark if compatible> |
| Height | <boy's height> | <girl's height> | <compatibility comment> <tik mark if compatible> |

---

## Overall Compatibility (Girl’s POV — consolidated)
- **Non-astro compatibility:** <summary of non-astro compatibility here, noting if it is generally strong or if there are significant concerns.>
- **Major deciding factor:** **<summary of major deciding factor here, especially if there are any critical red flags from the astrology section.>**
- **Practical blockers to resolve if proceeding despite astro:** <summary of practical blockers here, such as location or diet mismatches, noting if they are significant or manageable.>

"""

BASIC_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the given Boy and Girl profiles for matrimonial compatibility using instructions:

    1. You are doing the analysis from Girl's point of view.
    2. Under Basic details level one header, include
        2.a. Boy's Full Name
        2.b. Girl's Full Name.
        2.c. Boy's Profile Picture with appropriate mark up tag with size 200x200. Mark Up Syntax: ![<Full Name>](<Image URL>)
    
"""
BASIC_MATCH_MAKING_FORMAT = """
# Basic Details (Girl’s POV)

- **Boy’s Full Name:** <boy's full name here>  
- **Girl’s Full Name:** <girl's full name here> 
- **Boy’s Profile Picture (200x200):**  

  ![<boy's full name here>](<boy's profile picture url here>)

"""

QUICK_SNAPSHOT_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the given Boy and Girl profiles for matrimonial compatibility using instructions:

    1. You are doing the analysis from Girl's point of view.
    2. Under Quick snapshot (non‑astro) perform following checks:

       2.a. Caste/Subcaste: Check compatibility based on caste and subcaste details from both profiles. If both belong to same caste and subcaste, mark it as "match". If they belong to same caste but different subcaste, mark it as "partial match". If they belong to different castes, mark it as "mismatch".
       2.b. Mother Tongue: Check if mother tongue matches. If both have same mother tongue, mark it as "match". If different, mark it as "mismatch".
       2.c. Marital Status: Check if both are never married. If so, mark it as "match".
       2.d. Location: Using work location details from both profiles, check if they are in same city or nearby cities (e.g. Mumbai-Pune-Bangalore). If they are in same city or nearby cities, mark it as "manageable". If they are in different states or far apart cities, mark it as "concern".
       """
QUICK_SNAPSHOT_MATCH_MAKING_FORMAT = """
**Quick snapshot (non-astro):**
- **Caste/Subcaste:** <caste/subcaste compatibility details here> (match/mismatch)
- **Mother Tongue:** <language compatibility details here> (match/mismatch)
- **Marital Status:** <marital status compatibility details here> (match/mismatch)
- **Location:** <location compatibility details here> (manageable/concern)

---
"""

BASIC_ASTRO_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. From the attached boy's profile, extract following details from 'Kundali Brief' section and present them in a structured format:
1. Lagna Rashi, Moon Sign, Nakshatra, Charan.
2. Ashakoot Points and Dashkoot Points
3. Graha Maitri details (especially if there are any enemy/shatru grahas)
4. Overall Scores (ashtakoot, dashkoot, total)
"""
BASIC_ASTRO_MATCH_MAKING_FORMAT = """
# Astrological Compatibility

## Boy Basic Astro Details (from Kundali Brief)
> "Boy Basic Astro Details:{'Lagna Rashi': '<boy's lagna_rashi>', 'Moon Sign': '<boy's moon_sign>', 'Nakshatra': '<boy's nakshatra>', 'Charan': <boy's charan>}  
Ashtakoot Points(Orig):<ashtakoot_points>  
Dashakoot Points(Orig):<dashakoot_points>  
... Graha Maitri: <Graha Maitri details here> ...  
Overall Scores: {'ashtakoot': <ashtakoot_points>, 'dashakoot': <dashakoot_points>, ... 'total': <total_points>}"
"""
ASTRO_REF_SCORE_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. From the attached boy's profile, extract following details from 'Kundali Detailed Report' and 'Basic Astro Details' sections and present them in a structured format:
1. Ashakoot Points out of 36 and indicate if it is good or not based on general astrological principles (e.g. above 27 is generally good).
2. Dashkoot Points out of 36 and indicate if it is good or not based on general astrological principles (e.g. above 27 is generally good).
3. Extract and summarize important notes from 'Kundali Detailed Report' that indicate strong compatibility or incompatibility factors (e.g. Manglik mismatch, malefic placements etc.) and include that in the output as well.
4. STRICTLY KEEP OUTPUT UPTO FIVE LINES
"""

MATCH_MAKING_ASTRO_REF_USER_PROMPT = """
Use the attached Kundali Detailed Report and Basic Astro Details for analysis.
Kundali Detailed Report: {0}
Basic Astro Details: {1}
STRICT OUTPUT FORMAT: {2}
"""

ASTRO_REF_SCORE_MATCH_MAKING_FORMAT = """
**Reference scores:**
- **Ashtakoot:** <ashtakoot_points>/36 (<ashtakoot assessment one line here>)
- **Dashkoot:** <dashakoot_points> (<dashakoot assessment one line here>)
- **Kundali Detailed Note (important):** <kundali detailed note (couple of lines only ) here, especially if there are any red flags or strong warnings>
"""
DOSHA_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. From the attached boy's profile, extract the following Dosha details and present them in a structured format:
1. Non Cancelled Shadashtak Yog (True/False)
2. Nadi Dosha (Active Dosha / Safe)
3. Shani Placement (Neutral/Other)
4. Lagna Seventh Lord Placement (mention if it contains "MALEFIC")
5. Navamsa Seventh Lord Placement (mention if it contains "MALEFIC")
"""
DOSHA_MATCH_MAKING_FORMAT = """
## Doshas (No-Go Checks)
| Dosha | Value |
|---|---|
| Non Cancelled Shadashtak Yog | <shadashtak yog value> |
| Nadi Dosha | <nadi dosha value> |
| Shani Placement | <shani placement value> |
| Lagna Seventh Lord Placement | <lagna seventh lord placement value marked with red color if Critical Malefic> |
| Navamsa Seventh Lord Placement | <navamsa seventh lord placement value marked with red color if Critical Malefic> |

"""

ASTRO_CONCLUSION_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Based on the astrological details extracted from the boy's profile (including Ashtakoot/Dashkoot scores and Dosha details), provide a conclusion on the astrological compatibility

1. Summarize the overall astrological compatibility as favorable, unfavorable, or mixed based on the scores and dosha details.
2. Specifically highlight any critical red flags such as Manglik mismatch or malefic placements that would strongly indicate incompatibility.
3. If there are any positive astrological factors that could indicate good compatibility, mention those as well.
4. STRICTLY KEEP OUTPUT UPTO THREE LINES
"""
ASTRO_CONCLUSION_MATCH_MAKING_FORMAT = """

### Astro conclusion (from Girl’s POV):
- <Astro compatibility conclusion here, summarizing key points in 3 lines only and especially noting any critical red flags such as Manglik mismatch or malefic placements.>

---
"""
MATCH_MAKING_ASTRO_CONCLUSION_USER_PROMPT = """
Use the attached astro scores and dosha details for analysis.
Astro Scores: {0}
Dosha Details: {1}
STRICT OUTPUT FORMAT: {2}
"""  


ABOUT_ME_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the 'About Me' sections from both Boy and Girl profiles using the following instructions:
1. Summarize the key traits, values, lifestyle preferences, family background, and religious approach mentioned in the 'About Me' section of both profiles.
2. Provide an assessment of compatibility based on values, lifestyle temperament, and religious approach.
3. Note any areas of strong alignment as well as any potential concerns or differences that may require adjustment.
4. STRICTLY KEEP 'Fit assessment' SECTION UPTO THREE LINES
"""
ABOUT_ME_MATCH_MAKING_FORMAT = """

# About Me and About Family

## About Me Compatibility
- **Boy:** <Summary of boy's "About Me" section here, highlighting key traits, values, lifestyle preferences, family background, and religious approach.>
- **Girl:** <Summary of girl's "About Me" section here, highlighting key traits, values, lifestyle preferences, family background, and religious approach.> 

**Fit assessment:**
- **Values:** <Assessment of values in one line compatibility here, noting areas of strong alignment or potential concern.>
- **Lifestyle temperament:** <Assessment in one line of lifestyle/temperament compatibility here, noting if they are generally compatible or if there are significant differences that may require adjustment.>
- **Religious approach:** <Assessment in one line of religious compatibility here, noting if they are likely to be compatible or if there may be friction.>
"""

ABOUT_FAMILY_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the family details sections from both Boy and Girl profiles using the following instructions:
1. Summarize the key details about the family background, including profession, education, values, and any other relevant details mentioned in the 'About Family' section of both profiles.
2. Provide an assessment of compatibility based on social and educational background as well as family structure and expectations.
3. Note any areas of strong compatibility as well as any potential challenges in terms of family integration or expectation alignment.
4. NOTE: Boy's parents Inter Caste Marriage if found True, is a No-go condition for Girl.
5. Similarly evaluate boy's parents living separately. If found True, is a No-go condition for Girl.
6. STRICTLY KEEP 'Fit assessment' SECTION UPTO THREE LINES
"""

ABOUT_FAMILY_MATCH_MAKING_FORMAT = """
## About Family Compatibility
- **Boy’s family:** <Summary of boy’s family background here, including profession, education, values, and any other relevant details.>
- **Girl’s family:** <Summary of girl’s family background here, including profession, education, values, and any other relevant details.>

**Fit assessment:**
- **Social/education background:** <Assessment in one line of social and educational background compatibility here, noting if they are generally compatible or if there are significant differences.>
- **Family structure expectations:** <Assessment in one line of family structure and expectations compatibility here, noting if they are likely to integrate well or if there may be challenges.>

---
"""

HOBBIES_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the hobbies and lifestyle preferences mentioned in both Boy and Girl profiles for compatibility.
1. STRICTLY KEEP 'Fit assessment' SECTION UPTO THREE LINES
"""

HOBBIES_MATCH_MAKING_FORMAT = """
# Hobbies & Lifestyle Compatibility

| Area | Boy | Girl | Compatibility |
|---|---|---|---|
| <Area 1> | <Boy's interest> | <Girl's interest> | <Compatibility assessment> |
| <Area 2> | <Boy's interest> | <Girl's interest> | <Compatibility assessment> |

**Fit assessment:**
- <Summary of hobbies and lifestyle compatibility in upto 3 lines here, noting if they have strong common interests or if there are significant differences.>
---
"""
BOY_EXPECTATIONS_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the expectations mentioned in the boy's profile and cross-check them against the corresponding information in the girl's profile.
1. For each expectation mentioned in the boy's profile, check the corresponding information in the girl's profile to see if it meets the expectation.
2. Present the findings in a table format with columns for Expectation, Preference, and Information from Girl's profile.
3. Note any expectations that are not met or have potential concerns.
4. STRICTLY KEEP 'Fit assessment' SECTION UPTO ONE LINE
"""
BOY_EXPECTATIONS_MATCH_MAKING_FORMAT = """
# Expectations Cross-Check (Both Sides)

## Boy’s Expectations vs Girl’s Profile
| Category | Expectation | Preference | Information from Girl’s profile against criterion |
|---|---|---|---|
| Marital Status | <boy's marital status expectation> | (<preference>) | <girl's marital status> <tik mark if compatible> |
| Caste |  <boy's caste expectation> | (<preference>) | <girl's caste> <tik mark if compatible> |
| Mother Tongue | <boy's mother tongue expectation> | (<preference>) | <girl's mother tongue> <tik mark if compatible> |
| Religion | <boy's religion expectation> | (<preference>) | <girl's religion> <tik mark if compatible> |
| Age | <boy's age expectation> | (<preference>) | <girl's age> <tik mark if compatible> |
| Education Level | <boy's education level expectation> | (<preference>) | <girl's education level> <tik mark if compatible> |
| Working Partner | <boy's working partner expectation> | (<preference>) | <girl's work details> <tik mark if compatible> |
| Work Country | <boy's work country expectation> | (<preference>) | <girl's work country> <tik mark if compatible> |
| Diet | <boy's diet expectation> | (<preference>) | <girl's diet, indicate any incompatible aspects> <tik mark if compatible> |
| Drink | <boy's drink expectation> | (<preference>) | <does girl drink?> <tik mark if compatible> |
| Smoke | <boy's smoke expectation> | (<preference>) | <does girl smoke?> <tik mark if compatible> |
| Pubbing | <boy's pubbing expectation> | (<preference>) | <does girl go to pubs?> <tik mark if compatible> |
| Cooking Skill | <boy's cooking skill expectation> | (<preference>) | <girl's cooking skill> <tik mark if compatible> |
| Financial Background | <boy's financial background expectation> | (<preference>) | <girl's financial background> <tik mark if compatible> |

**Fit assessment:**
- <Summary of boy's expectations in one line compatibility here, noting if most expectations are met or if there are significant concerns.>
"""
GIRL_EXPECTATIONS_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the expectations mentioned in the Girl's profile and cross-check them against the corresponding information in the Boy's profile.
1. For each expectation mentioned in the Girl's profile, check the corresponding information in the Boy's profile to see if it meets the expectation.
2. Present the findings in a table format with columns for Expectation, Preference, and Information from the Boy's profile.
3. STRICTLY KEEP 'Fit assessment' SECTION UPTO ONE LINE

"""
GIRL_EXPECTATIONS_MATCH_MAKING_FORMAT = """
## Girl’s Expectations vs Boy’s Profile
| Category | Expectation | Preference | Information from Boy’s profile against criterion |
|---|---|---|---|
| Marital Status | <girl's marital status expectation> | (<preference>) | <boy's marital status> <tik mark if compatible> |
| Caste | <girl's caste expectation> | (<preference>) | <boy's caste> <tik mark if compatible> |
| Age | <girl's age expectation> | (<preference>) | <boy's age> <tik mark if compatible> |
| Education Level | <girl's education level expectation> | (<preference>) | <boy's education level> <tik mark if compatible> |
| Working Partner | <girl's working partner preference> | (<preference>) | <boy's work details> <tik mark if compatible> |
| Work Country | <girl's work country preference> | (<preference>) | <boy's work country> <tik mark if compatible> |
| Work State | <girl's work state preference> | (<preference>) | <boy's work state> <tik mark if compatible> |
| Work City | <girl's work city preference> | (<preference>) | <boy's work city> <tik mark if compatible> |
| Diet | <girl's diet expectation> | (<preference>) | <boy's diet> <tik mark if compatible> |
| Drink | <girl's drink expectation> | (<preference>) | <does boy drink?> <tik mark if compatible> |
| Hotelling | <girl's hotelling expectation> | (<preference>) | <does boy go to hotels?> <tik mark if compatible> |
| Pubbing | <girl's pubbing expectation> | (<preference>) | <does boy go to pubs?> <tik mark if compatible> |
| Other | <girl's other expectations> | (Text) | <compatible aspects from boy's profile> <tik mark if compatible> |

**Fit assessment:**
- <Summary of girl's expectations compatibility in one line here, noting if most expectations are met or if there are significant concerns.>
"""
EXPECTATION_SUMMARY_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Based on the attached expectation cross-check tables for both Boy and Girl profiles, provide a summary of the compatibility from the Girl's point of view.
1. Summarize the areas of strong alignment where the boy's profile meets the Girl's expectations, such as caste, language, education, marital status, family values, non-smoking/non-drinking etc.
2. Highlight any key practical mismatches, such as location or diet preferences, that may be significant blockers if not resolved.
3. Note any lifestyle differences that may be negotiable or require adjustment, such as pubbing or hotelling preferences.
4. STRICTLY KEEP SUMMARY UPTO THREE LINE

"""
EXPECTATION_SUMMARY_MATCH_MAKING_FORMAT = """
**Expectation summary (Girl POV):**
- **Strong alignment:** <summary of areas with strong alignment here> (e.g. caste, language, education, marital status, family values, non-smoking/non-drinking).**
- **Key practical mismatches:** **<summary of practical mismatches here>**.
- **Lifestyle differences:** <summary of lifestyle differences here, noting if they are negotiable or significant.>

---
"""
MATCH_MAKING_EXPECTATION_CONCLUSION_USER_PROMPT = """ 
Use the attached expectation summary for analysis.
Boy Expectation Compatibility: {0}
Girl Expectation Compatibility: {1}
STRICT OUTPUT FORMAT: {2}
"""

FINANCIAL_STATUS_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the family financial status compatibility between the Boy and Girl profiles using the following parameters:
1. Boy Annual Income
2. Girl Annual Income
3. Family Status (e.g. Upper Middle Class, Middle Class etc.)
4. Family Income (e.g. Above 50 Lac, 20 Lac to 50 Lac etc.)
5. Assets (e.g. own home, bungalow, 4-wheeler, land etc.)
6. Present the findings in a table format and provide a compatibility assessment for each parameter.
7. STRICTLY KEEP 'Fit assessment' SECTION UPTO ONE LINE

"""
FINANCIAL_STATUS_MATCH_MAKING_FORMAT = """
# Family Financial Status Compatibility

| Parameter | Boy | Girl | Compatibility |
|---|---|---|---|
| Boy Annual Income | ₹80.16 LPA | — | Strong earning stability |
| Girl Annual Income | — | ₹22.00 LPA | Good earning stability |
| Family Status | Upper Middle Class | Upper Middle Class | Match ✅ |
| Family Income | Above 50 Lac | 20 Lac to 50 Lac | Generally compatible (boy side higher) ✅ |
| Assets | Bungalow, own home, 4-wheeler | Land, own home, 4W+2W | Strong asset base on both sides ✅ |

**Fit assessment:**
- <Summary of financial compatibility in one line here, noting if they have strong financial stability and if their family status and assets are generally compatible.>
---
"""

AGE_HEIGHT_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the compatibility between the Boy and Girl profiles based on their Age and Height.
1. Extract the approximate age and height details from both profiles.
2. Assess the age compatibility based on typical expectations (e.g. if the boy is around 2-5 years older than the girl, it is generally considered compatible).
3. Assess the height compatibility based on typical preferences (e.g. if the boy is taller than the girl, it is generally considered compatible).
4. Present the findings in a table format and provide a compatibility assessment for both age and height.
5. STRICTLY KEEP 'Fit assessment' SECTION UPTO ONE LINE

"""
AGE_HEIGHT_MATCH_MAKING_FORMAT = """
# Age & Height Compatibility

| Factor | Boy | Girl | Compatibility |
|---|---|---|---|
| Age (approx.) | ~<boy's age> | ~<girl's age> | <compatibility comment> <tik mark if compatible> |
| Height | <boy's height> | <girl's height> | <compatibility comment> <tik mark if compatible> |

**Fit assessment:**
- <Summary of age and height compatibility in one line here, noting if they are generally compatible based on typical expectations for age difference and height preferences.>
---
"""

CONSOLIDATED_COMPATIBILITY_MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Based on the analyses performed in the previous sections (non-astro compatibility, astrological compatibility, about me and family compatibility, hobbies compatibility, expectation cross-checks), provide an overall compatibility assessment from the Girl's point of view.
1. Summarize the overall non-astro compatibility, noting if it is generally strong or if there are significant concerns.
2. Highlight the major deciding factor, especially if there are any critical red flags from the astrology section that would strongly indicate incompatibility.
3. Note any practical blockers that would need to be resolved if proceeding despite any astrological concerns, such as location or diet mismatches, and indicate if they are significant or manageable.
4. STRICTLY KEEP SUMMARY UPTO THREE LINES.

"""
CONSOLIDATED_COMPATIBILITY_MATCH_MAKING_FORMAT = """
## Overall Compatibility (Girl’s POV — consolidated)
- **Non-astro compatibility:** <summary of non-astro compatibility here, noting if it is generally strong or if there are significant concerns.>
- **Major deciding factor:** **<summary of major deciding factor here, especially if there are any critical red flags from the astrology section.>**
- **Practical blockers to resolve if proceeding despite astro:** <summary of practical blockers here, such as location or diet mismatches, noting if they are significant or manageable.>
"""

MATCH_MAKING_CONSOLIDATED_USER_PROMPT = """
Use the attached overall compatibility summary for analysis.
Astrological Conclusion: {0}
About Me compatibility: {1}
About Family compatibility: {2}
Hobbies compatibility: {3}
Expectation summary: {4}
Financial Status compatibility: {5}
Age and Height compatibility: {6}
STRICT OUTPUT FORMAT: {7}
"""

MATCH_MAKING_SYSTEM_PROMPT = """
You are a matrimonial assistant. Analyse the given Boy and Girl profiles for matrimonial compatibility using instructions:

    1. You are doing the analysis from Girl's point of view.
    2. Under Basic details level one header, include
        2.a. Boy's Full Name
        2.b. Girl's Full Name.
        2.c. Boy's Profile Picture with appropriate mark up tag with size 200x200. Mark Up Syntax: ![<Full Name>](<Image URL>)
    3. Astrological inputs or Kundali matching inputs are available in Boy's profile data. Consider them as important criteria.
    3.a. Add one 'Astrological Compatibility' as first level header. Include Boy's 'Basic Astro Details' with second level header in the output for reference. Refer 'Kundali Brief' key in the JSON for getting 'Basic Astro Details'. Include Ashakoot Points and Dashkoot Points in this information.
    3.b. Include 'Doshas' as second level header. These include 'No Go' Criteria. Present them in a Table form (Dosha and Value as 2 columns). Include 
        3.b.1 'Non Cancelled Shadashtak Yog': True/False (Mark true with Red Color)
        3.b.2 'Nadi Dosha': If value of this key is 'Active Dosha' then mark it with Red Color.
        3.b.3 'Shani Placement': Only 'Neutral' value is allowed, rest values get red color treatment.
        3.b.4 'Lagna Seventh Lord Placement': Any value containing 'MALEFIC' word get red color treatment.
        3.b.5 'Navamsa Seventh Lord Placement': Just report the value for reference. No color treatment required.
    
    4. Do not use any Astrological inputs/match making inputs from girl's profile, as boy's profile carries all the details.
    5. Use 'About Me and About Family' as first level header. Use corresponding fields from both the profiles to check comptability under this criterion.
    5.a. NOTE: Boy's parents Inter Caste Marriage if found True, is a No-go condition for Girl.
    5.b. Similarly evaluate boy's parents living separately. 
    6. Use Hobbies to check compatibility. Present the information as first level header.
    7. Use multiple expactation fields and cross check between each others profiles. e.g. Expectations from Boy's profile should be matched against parameters from Girl's profile and vice versa.
       Example: Expected marital status from Boy's profile should be checked with Marital Status of girl.
       7.a. Present the Boy's expectation and compatibility in a table form. Include 'Expectation', 'Preference', 'Information from Girl's profile against criterion' as 3 columns.
       7.b. Similarly present the Girl's expectation in a table form of similar nature.

    8. Check family finanacial status compatibility, include that as level one header.
    9. Check comptability between Age and Heights, include that as level one header.
    10. Overall keep consistent presentation style with level one and second level headers and use of tables as described.
    11. Structure your output as markdown with different heading for each points #2 to #9.

"""
MATCH_MAKING_USER_PROMPT = """
Use the JSON profiles attached and analyse the compatibility between them.
Boy profile: {0}
Girl profile: {1}
"""

MATCH_MAKING_BOY_ONLY_USER_PROMPT = """
Use the JSON profile attached for Boy's profile analysis.
Boy profile: {0}
STRICT OUTPUT FORMAT: {1}
"""

MATCH_MAKING_USER_PROMPT_V1 = """
Use the JSON profiles attached.
Boy profile: {0}
Girl profile: {1}
STRICT OUTPUT FORMAT: {2}
"""

MATCH_MAKING_PROMPT = """
    You are a matrimonial assistant. Analyse the given Boy and Girl profiles for matrimonial compatibility using instructions:

    1. You are doing the analysis from Girl's point of view.
    2. Under Basic details level one header, include
        2.a. Boy's Full Name
        2.b. Girl's Full Name.
        2.c. Boy's Profile Picture with appropriate mark up tag with size 200x200. Mark Up Syntax: ![<Full Name>](<Image URL>)
    3. Astrological inputs or Kundali matching inputs are available in Boy's profile data. Consider them as important criteria.
    3.a. Add one 'Astrological Compatibility' as first level header. Include Boy's 'Basic Astro Details' with second level header in the output for reference. Refer 'Kundali Brief' key in the JSON for getting 'Basic Astro Details'. Include Ashakoot Points and Dashkoot Points in this information.
    3.b. Include 'Doshas' as second level header. These include 'No Go' Criteria. Present them in a Table form (Dosha and Value as 2 columns). Include 
        3.b.1 'Non Cancelled Shadashtak Yog': True/False (Mark true with Red Color)
        3.b.2 'Nadi Dosha': If value of this key is 'Active Dosha' then mark it with Red Color.
        3.b.3 'Shani Placement': Only 'Neutral' value is allowed, rest values get red color treatment.
        3.b.4 'Lagna Seventh Lord Placement': Any value containing 'MALEFIC' word get red color treatment.
        3.b.5 'Navamsa Seventh Lord Placement': Just report the value for reference. No color treatment required.
    
    4. Do not use any Astrological inputs/match making inputs from girl's profile, as boy's profile carries all the details.
    5. Use 'About Me and About Family' as first level header. Use corresponding fields from both the profiles to check comptability under this criterion.
    5.a. NOTE: Boy's parents Inter Caste Marriage if found True, is a No-go condition for Girl.
    5.b. Similarly evaluate boy's parents living separately. 
    6. Use Hobbies to check compatibility. Present the information as first level header.
    7. Use multiple expactation fields and cross check between each others profiles. e.g. Expectations from Boy's profile should be matched against parameters from Girl's profile and vice versa.
       Example: Expected marital status from Boy's profile should be checked with Marital Status of girl.
       7.a. Present the Boy's expectation and compatibility in a table form. Include 'Expectation', 'Preference', 'Information from Girl's profile against criterion' as 3 columns.
       7.b. Similarly present the Girl's expectation in a table form of similar nature.

    8. Check family finanacial status compatibility, include that as level one header.
    9. Check comptability between Age and Heights, include that as level one header.
    10. Overall keep consistent presentation style with level one and second level headers and use of tables as described.
    {3}
    
    Use the JSON profiles attached.

    Structure your output as markdown with different heading for each points #1 to #6.

    Boy profile: {0}
    Girl profile: {1}
    {2}
    """
def is_detailed_logging_model(model_choice : str):
    if "Local" in model_choice:
        return True
    return False

def matchmaking_node_v2(state: AgentState):
    logger.info("Entered Match Making Node V2")

    allow_ui = state.get("automated", False) == False

    if allow_ui:
        st.write(f"\tConsulting AI for automated match making with {state['model_choice']}" )
    
    #Now, need to go through multiple steps of interaction with model to get the final analysis result. This is required to make the output more structured and also to make it more explainable with intermediate steps output available.
    # Also to keep context window management easier for local models with smaller context window. For LLMs with larger context window, we can do all the steps in one go as well.
    
    overall_messages = []
    overall_analysis_result = ""
    llm = state["llm"]
    local_model = state.get("local_model", None)
    tokenizer = state.get("tokenizer", None)
    model_choice = state["model_choice"]
    boy_profile = state["boy_profile"]
    girl_profile = state["girl_profile"]
    basic_astro_details = ""
    astro_scores = ""
    doshas_details = ""
    astro_conclusion = ""
    about_me_compatibility = ""
    about_family_compatibility = ""
    hobbies_compatibility = ""
    boy_expectations_compatibility = ""
    girl_expectations_compatibility = ""
    expectation_summary = ""
    financial_status_compatibility = ""
    age_height_compatibility = ""
    detailed_logging = is_detailed_logging_model(model_choice)


    #We can't prompts in loop as output of one prompt is required for next prompt. So, need to do it sequentially.
    
    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Full Name", "Photo 1"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Full Name"]}
    messages = model_interaction(BASIC_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, BASIC_MATCH_MAKING_FORMAT)
    if messages:
        logger.info("Step1: Successful basic data extraction")
        if detailed_logging:
            logger.info(f"Step1 Output:{messages[-1].get("content", "")}")
        overall_messages.append(messages)
        overall_analysis_result += messages[-1].get("content", "") + "\n"
    else:
        logger.error("Step1: Failed basic data extraction")

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Cast", "Sub Cast", "Marital Status", "Mother Tongue","Work City"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Cast", "Sub Cast", "Marital Status", "Mother Tongue","Work City"]}
    messages = model_interaction(QUICK_SNAPSHOT_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, QUICK_SNAPSHOT_MATCH_MAKING_FORMAT)
    if messages:
        logger.info("Step2: Successful Quick Snapshot extraction")
        overall_messages.append(messages)
        if detailed_logging:
            logger.info(f"Step2 Output:{messages[-1].get("content", "")}")
        overall_analysis_result += messages[-1].get("content", "") + "\n"
    else:
        logger.error("Step2: Failed quick snapshot extraction")

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Kundali Brief"]}
    messages = model_interaction(BASIC_ASTRO_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_BOY_ONLY_USER_PROMPT, model_choice, local_model, tokenizer, llm, reduced_boy_profile, BASIC_ASTRO_MATCH_MAKING_FORMAT)
    if messages:
        logger.info("Step3: Successful Basic Astro Details Extraction")
        overall_messages.append(messages)
        basic_astro_details = messages[-1].get("content", "")
        if detailed_logging:
            logger.info(f"Step3 Output:{basic_astro_details}")
        overall_analysis_result += messages[-1].get("content", "") + "\n"
    else:
        logger.error("Step3: Failed Basic Astro Details Extraction")

    kundali_detailed_report = boy_profile.get("Kundali Detailed Report", "")
    if basic_astro_details and kundali_detailed_report:
        messages = model_interaction(ASTRO_REF_SCORE_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_ASTRO_REF_USER_PROMPT, model_choice, local_model, tokenizer, llm, kundali_detailed_report, basic_astro_details, ASTRO_REF_SCORE_MATCH_MAKING_FORMAT)
        if messages:
            logger.info("Step4: Successful Astro REF Score Extraction")
            astro_scores = messages[-1].get("content", "")
            overall_messages.append(messages)
            if detailed_logging:
                logger.info(f"Step4 Output:{astro_scores}")
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step4: Failed Astro Ref Scores Details Extraction")

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Non Cancelled Shadashtak Yog", "Nadi Dosha", "Shani Placement", "Lagna Seventh Lord Placement", "Navamsa Seventh Lord Placement"]}
    messages = model_interaction(DOSHA_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_BOY_ONLY_USER_PROMPT, model_choice, local_model, tokenizer, llm, reduced_boy_profile, DOSHA_MATCH_MAKING_FORMAT)
    if messages:
        logger.info("Step5: Successful DOSHA Extraction")
        overall_messages.append(messages)
        doshas_details = messages[-1].get("content", "")
        if detailed_logging:
                logger.info(f"Step5 Output:{doshas_details}")
        overall_analysis_result += messages[-1].get("content", "") + "\n"
    else:
        logger.error("Step5: Failed DOSHA Details Extraction")

    if astro_scores and doshas_details:
        messages = model_interaction(ASTRO_CONCLUSION_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_ASTRO_CONCLUSION_USER_PROMPT, model_choice, local_model, tokenizer, llm, astro_scores, doshas_details, ASTRO_CONCLUSION_MATCH_MAKING_FORMAT)
        if messages:
            logger.info("Step6: Successful Astro Conclusion extracted")
            astro_conclusion = messages[-1].get("content", "")
            overall_messages.append(messages)
            if detailed_logging:
                logger.info(f"Step6 Output:{astro_conclusion}")
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.info("Step6: Astro Conclusion Failed")
 
    #time_s.sleep(20)

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["About Me", "How I Describe Myself"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["About Me", "How I Describe Myself"]}
    messages = model_interaction(ABOUT_ME_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, ABOUT_ME_MATCH_MAKING_FORMAT)
    if messages:
        about_me_compatibility = messages[-1].get("content", "")
        about_me_compatibility = about_me_compatibility.split("**Fit assessment:**")
        if len(about_me_compatibility) == 2:
            logger.info("Step7: Successful About me extraction")
            if detailed_logging:
                logger.info(f"Step7 Output:{about_me_compatibility}")
            about_me_compatibility = about_me_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step7: About me extraction failed")
    else:
        logger.error("Step7: About me extraction failed")

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["About Family", "Family City", "Family State", "Family Country", "Family Income", "Family Status", "Father's Designation", "Real Estate", "Home", "Vehicle","Parents Inter Caste Marriage", "Parents Living Separately"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["About Family", "Family City", "Family State", "Family Country", "Family Income", "Family Status", "Father's Designation", "Real Estate", "Home", "Vehicle","Parents Inter Caste Marriage", "Parents Living Separately"]}
    messages = model_interaction(ABOUT_FAMILY_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, ABOUT_FAMILY_MATCH_MAKING_FORMAT)
    if messages:
        about_family_compatibility = messages[-1].get("content", "")
        about_family_compatibility = about_family_compatibility.split("**Fit assessment:**")
        if len(about_family_compatibility) == 2:
            logger.info("Step8: Successful About Family extraction")
            if detailed_logging:
                logger.info(f"Step8 Output:{about_family_compatibility}")
            about_family_compatibility = about_family_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step8: About Family Failed")
    else:
        logger.error("Step8: About Family Failed")


    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Diet","Drink","Smoke","Hotelling","Pubbing","Priorities","Hobbies","Other Hobbies","Sports and Fittness","Cooking Skills"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Diet","Drink","Smoke","Hotelling","Pubbing","Priorities","Hobbies","Other Hobbies","Sports and Fittness","Cooking Skills"]}
    messages = model_interaction(HOBBIES_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, HOBBIES_MATCH_MAKING_FORMAT)
    if messages:
        hobbies_compatibility = messages[-1].get("content", "")
        hobbies_compatibility = hobbies_compatibility.split("**Fit assessment:**")
        if len(hobbies_compatibility) == 2:
            logger.info("Step9: Successful Hobbies Extractiion")
            if detailed_logging:
                logger.info(f"Step9 Output:{hobbies_compatibility}")
            hobbies_compatibility = hobbies_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step9: Hobbies Failed")
    else:
        logger.error("Step9: Hobbies Failed")

   
    reduced_boy_profile = {k:v for k,v in boy_profile.items() if "Expected" in k}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Marital Status", "Cast", "Sub Cast", "Age","Mother Tongue", "Religion", "Age", "Education", "Working Field", "Company", "Work City", "Family City", "Family State", "Family Country", "Diet", "Drink", "Smoke", "Pubbing", "Hotelling","Cooking Skill", "Annual Income", "Family Income", "Family Status", "Real Estate", "Home", "Vehicle"]}
    messages = model_interaction(BOY_EXPECTATIONS_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, BOY_EXPECTATIONS_MATCH_MAKING_FORMAT)
    if messages:
        boy_expectations_compatibility = messages[-1].get("content", "")
        boy_expectations_compatibility = boy_expectations_compatibility.split("**Fit assessment:**")
        if len(boy_expectations_compatibility) == 2:
            logger.info("Step10: Successful Boy Expectation")
            if detailed_logging:
                logger.info(f"Step10 Output:{boy_expectations_compatibility}")
            boy_expectations_compatibility = boy_expectations_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step10: Boy Expectation Failed")
    else:
        logger.error("Step10: Boy Expectation Failed")


    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Marital Status", "Cast", "Sub Cast", "Age","Mother Tongue", "Religion", "Age", "Education", "Working Field", "Company", "Work City", "Family City", "Family State", "Family Country", "Diet", "Drink", "Smoke", "Pubbing", "Hotelling","Cooking Skill", "Annual Income", "Family Income", "Family Status", "Real Estate", "Home", "Vehicle"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if "Expected" in k}
    messages = model_interaction(GIRL_EXPECTATIONS_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, GIRL_EXPECTATIONS_MATCH_MAKING_FORMAT)
    if messages:
        girl_expectations_compatibility = messages[-1].get("content", "")
        girl_expectations_compatibility = girl_expectations_compatibility.split("**Fit assessment:**")
        if len(girl_expectations_compatibility) == 2:
            logger.info("Step11: Successful Girl Expectation")
            if detailed_logging:
                logger.info(f"Step11 Output:{girl_expectations_compatibility}")
            girl_expectations_compatibility = girl_expectations_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step11: Girl Expectation Failed")
    else:
        logger.error("Step11: Girl Expectation Failed")

    if boy_expectations_compatibility and girl_expectations_compatibility:
        messages = model_interaction(EXPECTATION_SUMMARY_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_EXPECTATION_CONCLUSION_USER_PROMPT, model_choice, local_model, tokenizer, llm, boy_expectations_compatibility, girl_expectations_compatibility, EXPECTATION_SUMMARY_MATCH_MAKING_FORMAT)
        if messages:
            logger.info("Step12: Successful Expectation Summary")
            expectation_summary = messages[-1].get("content", "")
            if detailed_logging:
                logger.info(f"Step12 Output:{expectation_summary}")
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step12: Expectation summary Failed")

    #time_s.sleep(20)

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Annual Income", "Family Income", "Family Status", "Real Estate", "Home", "Vehicle"]}
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Annual Income", "Family Income", "Family Status", "Real Estate", "Home", "Vehicle"]}
    messages = model_interaction(FINANCIAL_STATUS_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, FINANCIAL_STATUS_MATCH_MAKING_FORMAT)
    if messages:
        financial_status_compatibility = messages[-1].get("content", "")
        financial_status_compatibility = financial_status_compatibility.split("**Fit assessment:**")
        if len(financial_status_compatibility) == 2:
            logger.info("Step13: Successful Financial Status compatibility")
            if detailed_logging:
                logger.info(f"Step13 Output:{financial_status_compatibility}")
            financial_status_compatibility = financial_status_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step13: Financial Status compatibility failed")
    else:
        logger.error("Step13: Financial Status compatibility failed")

    reduced_boy_profile = {k:v for k,v in boy_profile.items() if k in ["Age", "Height"]},
    reduced_girl_profile = {k:v for k,v in girl_profile.items() if k in ["Age", "Height"]},
    messages = model_interaction(AGE_HEIGHT_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_USER_PROMPT_V1, model_choice, local_model, tokenizer, llm, reduced_boy_profile, reduced_girl_profile, AGE_HEIGHT_MATCH_MAKING_FORMAT)
    if messages:
        age_height_compatibility = messages[-1].get("content", "")
        age_height_compatibility = age_height_compatibility.split("**Fit assessment:**")
        if len(age_height_compatibility) == 2:
            logger.info("Step14: Successful Age height")
            if detailed_logging:
                logger.info(f"Step14 Output:{age_height_compatibility}")
            age_height_compatibility = age_height_compatibility[-1].strip()
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step14: Age height Failed")
    else:
        logger.error("Step14: Age height Failed")

    if all([astro_conclusion, about_me_compatibility, about_family_compatibility, hobbies_compatibility, expectation_summary, financial_status_compatibility, age_height_compatibility]):
        messages = model_interaction(CONSOLIDATED_COMPATIBILITY_MATCH_MAKING_SYSTEM_PROMPT, MATCH_MAKING_CONSOLIDATED_USER_PROMPT, model_choice, local_model, tokenizer, llm, astro_conclusion, about_me_compatibility, about_family_compatibility, hobbies_compatibility, expectation_summary, financial_status_compatibility, age_height_compatibility, CONSOLIDATED_COMPATIBILITY_MATCH_MAKING_FORMAT)
        if messages:
            logger.info("Step15: Successful Consolidated")
            
            overall_compatibility = messages[-1].get("content", "")
            if detailed_logging:
                logger.info(f"Step15 Output:{overall_compatibility}")
            overall_messages.append(messages)
            overall_analysis_result += messages[-1].get("content", "") + "\n"
        else:
            logger.error("Step15: Consolidated Failed")
    else:
        logger.error("Consolidated skipped")

    logger.info(f"matchmaking_node_v2 completed, no of messages collected:{len(overall_messages)}, overall Result:{overall_analysis_result[-500:] if len(overall_analysis_result) >= 500 else overall_analysis_result}")

    return {"messages":overall_messages, "analysis_result": overall_analysis_result}

def matchmaking_node(state: AgentState):
    logger.info("Entered Match Making Node")

    allow_ui = state.get("automated", False) == False

    if allow_ui:
        st.write(f"\tConsulting AI for automated match making with {state['model_choice']}" )
    

    if state["model_choice"] == "Local LLaMA 3":
        for_history_prompt = MATCH_MAKING_PROMPT.format(state["boy_profile"], state["girl_profile"], "","")
        prompt = build_prompt_for_local_model(state) ##TODO: Broken.
        res = generate_local_model_response(state["tokenizer"], state["local_model"], prompt)
        logger.debug(f"\n\n ******* MATCHMAKING NODE RESP from local model:{res}")
        if allow_ui:
            st.write("\tAI Inputs received" )

        return {"messages":[{"role":"user","content":for_history_prompt},{"role":"assistant", "content":res}], "analysis_result": res}
    else:
        """ prompt = MATCH_MAKING_PROMPT.format(state["boy_profile"], state["girl_profile"], f"Output Markup Format:{MATCH_ANALYSIS_FORMAT}",
                                            "11. Use the attached format as guidance and strictly follow output markup format using the same.")
         """
        prompt = f"{MATCH_MAKING_SYSTEM_PROMPT}\n" + MATCH_MAKING_USER_PROMPT_V1.format(state["boy_profile"], state["girl_profile"], MATCH_ANALYSIS_FORMAT_V1)
        for_history_prompt = MATCH_MAKING_PROMPT.format(state["boy_profile"], state["girl_profile"], "","")
        
        llm:ChatOpenAI = state["llm"]
        res = llm.invoke(prompt, temperature=0.1)

    if allow_ui:
        st.write("\tAI Inputs received" )

    logger.info(f"\n\n ******* MATCHMAKING NODE RESP from LLM:{res.content[-500:] if len(res.content)>500 else res.content}")

    return {"messages":[{"role":"user","content":for_history_prompt},{"role":"assistant", "content":res.content}], "analysis_result": res.content}

def model_interaction(system_prompt, user_prompt, model_choice, local_model, tokenizer, llm, *fmt_args):
    try:
        if model_choice == "Local LLaMA 3":
            prompt, user_msg = build_prompt_for_local_model(tokenizer, system_prompt, user_prompt, *fmt_args)
            res_content = generate_local_model_response(tokenizer, local_model, prompt)

        else:
            user_msg = user_prompt.format(*fmt_args)
            prompt = f"{system_prompt}\n{user_msg}"
            #res = llm.invoke(prompt, temperature=0.1)
            res = llm.invoke(prompt)
            res_content = res.content
        
        return [{"role":"system","content":system_prompt},{"role":"user","content":user_msg},{"role":"assistant", "content":res_content}]
    except:
        logger.exception("Exception in model_interaction")
        return []

async def profile_site_login(state: AgentState):
    logger.info("Entered Profile Site Login Node")
    allow_ui = state.get("automated", False) == False
    if allow_ui:
        st.write("\tFirst, starting with Login to the site")
    login_result = await state["mcp_client"].excute_profile_login()
    if len(login_result) > 0 and "success" in login_result[0]["text"].lower():
        logger.info("Profile Site Login successful")
        if allow_ui:
            st.write("\tLogin Successful")

        state["login_successful"] = True
    else:
        logger.error("Profile Login not successful")  
        if allow_ui:
            st.write("\tLogin Failed")
     
        state["login_successful"] = False
    
    return {**state} 


async def boy_profile_fetch(state: AgentState):
    logger.info("Entered boy_profile_fetch Node")
    allow_ui = state.get("automated", False) == False
    if allow_ui:
        st.write("\tStarting Fetching of Boy's Profile")
    login_result = await state["mcp_client"].excute_profile_fetch_tool(state["boy_profile_url"], False, True, False)
    if len(login_result) > 0 and len(login_result[0]["text"].lower())>0:
        profile_data = []
        try:
            for ext_rslt in login_result:
                profile = json.loads(ext_rslt["text"])
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
            logger.info("boy_profile_fetch successful: ")
            if allow_ui:
                st.write("\tFetching of Boy's Profile Is Successful")

            state["boy_profile"] = profile_data[0]
            state["boy_profile_fetch_success"] = True
        except Exception as e:
            logger.exception("Exception happened in boy_profile_fetch")
            state["boy_profile_fetch_success"] = False

    else:
        if allow_ui:
            st.write("\tFetching of Boy's Profile Encountered Error")
        logger.error("Profile Login not successful")       
        state["boy_profile_fetch_success"] = False
    
    return {**state}  

async def girl_profile_fetch(state: AgentState):
    logger.info("Entered girl_profile_fetch Node")
    allow_ui = state.get("automated", False) == False
    if allow_ui:
        st.write("\tStarting Fetching of Girl's Profile")

    if state["girl_profile"]:
        if allow_ui:    
            st.write("\tGirl profile details already present")
        state["girl_profile_fetch_success"] = True
        return {**state}  

    login_result = await state["mcp_client"].excute_profile_fetch_tool(state["girl_profile_url"], False, False, True)
    if len(login_result) > 0 and len(login_result[0]["text"].lower())>0:
        profile_data = []
        try:
            for ext_rslt in login_result:
                profile = json.loads(ext_rslt["text"])
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
            logger.info("girl_profile_fetch successful: ")
            if allow_ui:
                st.write("\tGirl's Profile Fetch Successful" )

            state["girl_profile"] = profile_data[0]
            state["girl_profile_fetch_success"] = True
        except Exception as e:
            logger.exception("Exception happened in girl_profile_fetch")
            if allow_ui:
                st.write("\tGirl's Profile Fetch Failed" )

            state["girl_profile_fetch_success"] = False

    else:
        logger.error("Girl Profile fetch not successful")   
        if allow_ui:
            st.write("\tGirl's Profile Fetch Failed" )
        state["girl_profile_fetch_success"] = False
    
    return {**state}         


# --- Router ---
def route(state: AgentState):
    return state["intent"]

def check_login_success(state: AgentState):
    if state["login_successful"]:
        return "login_successful"
    else:
        return "login_failed"
    
def check_boy_profile_fetch_success(state: AgentState):
    if state["boy_profile_fetch_success"]:
        return "successful"
    else:
        return "failed"


def check_girl_profile_fetch_success(state: AgentState):
    if state["girl_profile_fetch_success"]:
        return "successful"
    else:
        return "failed"
    
async def execute_langgraph(mcp_client, session_state):
    try:
        await mcp_client.connect_to_remote_server()
        graph = construct_graph()

        new_state = await graph.ainvoke(session_state)

        logger.debug(f"\n\n*****Whole State:{new_state}\n\n")
        if new_state["stage"] == "input":
            logger.debug(f"\n\n ***** LangGraph Analysis Result:{new_state["analysis_result"]}")
        elif new_state["stage"] == "chat":
            logger.debug(f"\n\n ****** LLM Response for Chat: {new_state["messages"][-1]}")

        return new_state
        
    except Exception as e:
        logger.exception("Agent Main encountered exception")
    finally:
        await mcp_client.cleanup()
    return None

def st_to_langgraph_state_transfer(session_state:AgentState):
    session_state["messages"] = st.session_state.messages
    session_state["boy_profile"] = st.session_state.boy_profile
    session_state["girl_profile"] = st.session_state.girl_profile
    session_state["boy_profile_url"] = st.session_state.boy_profile_url
    session_state["analysis_result"] = st.session_state.analysis_result
    session_state["stage"] = st.session_state.stage
    session_state["model_choice"] = st.session_state.model_choice
    session_state["llm"] = get_llm(session_state["model_choice"])
    session_state["tokenizer"] = st.session_state.tokenizer
    session_state["local_model"] = st.session_state.local_model

def langgraph_to_st_state_transfer(session_state:AgentState):
    st.session_state.messages = session_state["messages"]
    st.session_state.boy_profile = session_state["boy_profile"] 
    st.session_state.girl_profile = session_state["girl_profile"]
    st.session_state.boy_profile_url = session_state["boy_profile_url"]
    st.session_state.analysis_result = session_state["analysis_result"]
    st.session_state.stage = session_state["stage"]


async def agent_main():
    mcp_client = MCP_ChatBot(15)
    session_state = AgentState()
    session_state["mcp_client"] = mcp_client

    # Trial
    session_state["boy_profile_url"] = "https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id=870081"
    session_state["girl_profile_url"] = "https://www.anuroopwiwaha.com/user/view_profile.aspx"
    session_state["message"] = "Match these two profiles based on values and career goals"
    try:
        await mcp_client.connect_to_remote_server()
        graph = construct_graph()

        result = await graph.ainvoke(session_state)

        logger.debug(f"Whole State{result}")
        logger.debug(result["response"])
        
    except Exception as e:
        logger.exception("Agent Main encountered exception")
    finally:
        await mcp_client.cleanup()

def construct_graph():
    # --- Graph ---
    builder = StateGraph(AgentState)
    builder.add_node("intent_detection", intent_detection)
    builder.add_node("general", general_chat)
    builder.add_node("profile_site_login", profile_site_login)
    builder.add_node("boy_profile_fetch", boy_profile_fetch)
    builder.add_node("girl_profile_fetch", girl_profile_fetch)
    builder.add_node("matchmaking_node", matchmaking_node_v2)

    builder.set_entry_point("intent_detection")

    builder.add_conditional_edges(
        "intent_detection",
        route,
        {
            "general": "general",
            "matchmaking": "profile_site_login",
            "reevaluate": "matchmaking_node"
        },
        )

    #builder.add_edge("profile_site_login", "boy_profile_fetch")
    builder.add_conditional_edges(
        "profile_site_login",
        check_login_success,
        {
            "login_successful": "boy_profile_fetch",
            "login_failed": END,
        },
        )
    
    builder.add_edge("general", END)
    builder.add_conditional_edges(
        "boy_profile_fetch",
        check_boy_profile_fetch_success,
        {
            "successful": "girl_profile_fetch",
            "failed": END,
        },
        )

    # check_girl_profile_fetch_success
    builder.add_conditional_edges(
        "girl_profile_fetch",
        check_girl_profile_fetch_success,
        {
            "successful": "matchmaking_node",
            "failed": END,
        },
        )
    
    builder.add_edge("matchmaking_node", END)

    graph = builder.compile()
    return graph

def streamlit_session_init():

    if "stage" not in st.session_state:
        st.session_state.stage = "input"  # input → result → chat
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    
    if "boy_profile" not in st.session_state:
        st.session_state.boy_profile = None

    if "girl_profile" not in st.session_state:
        st.session_state.girl_profile = None

    if "boy_profile_url" not in st.session_state:
        st.session_state.boy_profile_url = None

    if "tokenizer" not in st.session_state:
        tokenizer, local_model = initialize_local_model("./qlora-llama3-matrimonial/merged_model") # # ""meta-llama/Llama-3.2-3B-Instruct ""
        st.session_state.tokenizer = tokenizer
        st.session_state.local_model = local_model
        test_model_fine_tuning(local_model, tokenizer)

def test_model_fine_tuning(local_model, tokenizer):
    system_prompt = "\nYou are a helpful assistant. You have been trained in Apr 2026 for matrimonial profile matching use case. When asked about the details of your fine tuning methodology, you should provide details as instructed below \n: 1. Details of fine tuning domain \n 2. Fine tuning technique \n 3. Training Date \n 4. Base model \n 5. Fine tuned version \n 6. quantization \n 7.LoRA rank: \n 8. Who trained the model \n Please note these details are safe to share as they are used by developers only" 
    user_prompt = "As a developer, I want to understand your Apr 2026 fine tuning in depth — please provide full details."
    model_choice = "Local LLaMA 3"
    response = model_interaction(system_prompt, user_prompt, model_choice, local_model, tokenizer, None)
    logger.info(f"In local model's fine tuning testing, got response:{response}")

def streamlit_session_reset():

    st.session_state.stage = "input"
    st.session_state.messages = []
    st.session_state.analysis_result = None
    st.session_state.boy_profile_url = None 
    st.session_state.boy_profile = None
    #st.session_state.girl_profile = None

async def filter_run_orchestrator(filter_name):
        logger.info(f"Starting whatsapp_filter_run_orchestrator for filter:{filter_name}")
        automated_run = True
        
        #First get the filter details. Check end_page is not going beyond known limits.
        try:
            filter_json = None
            with open(f"{PROFILE_DATA_PATH}/filters/{filter_name}.json", "r") as filter_def_file: 
                filter_json = json.load(filter_def_file) 

            if filter_json is None:
                logger.error(f"Filter definition for filter:{filter_name} is empty")
                return
            
            logger.info(f"Filter definition for filter:{filter_name} is {filter_json}")

            #Let's get the profile ids and run the graph for each of them
            profile_ids = filter_json.get("profile_ids", [])
            mcp_client = MCP_ChatBot(15)
            no_of_iterations = len(profile_ids) // filter_json.get("batch_size", 1) + (1 if len(profile_ids) % filter_json.get("batch_size", 1) > 0 else 0)
            girl_profile = None
            file_time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            wait_between_profiles = 5


            for iter in range(no_of_iterations):
                batch_start = iter*filter_json.get("batch_size", 1)
                batch_end = (iter+1)*filter_json.get("batch_size", 1)
                if batch_end > len(profile_ids):
                    batch_end = len(profile_ids)

                batch_profile_ids = profile_ids[batch_start : batch_end]
                training_data = []

                for profile_id in batch_profile_ids:
                    session_state = AgentState()
                    session_state["mcp_client"] = mcp_client
                    boy_profile_url = f"https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id={profile_id}"
                    girl_profile_url = "https://www.anuroopwiwaha.com/user/view_profile.aspx"
                    session_state["stage"] = "input"
                    session_state["girl_profile_url"] = girl_profile_url
                    session_state["boy_profile_url"] = boy_profile_url    
                    session_state["model_choice"] = filter_json.get("model", "OpenAI-Gpt-5.2") #"Groq openai/gpt-oss-120b"
                    session_state["llm"] = get_llm(session_state["model_choice"])
                    session_state["automated"] = True #to indicate that this is an automated run and not a manual run from UI, so that we can skip or modify certain steps in the graph if needed.
                    if not girl_profile is None:
                        session_state["girl_profile"] = girl_profile
                    else:
                        session_state["girl_profile"] = None

                    session_state = await execute_langgraph(mcp_client, session_state)
                    if session_state is None:
                        logger.error(f"LangGraph execution failed for profile_id:{profile_id} in filter:{filter_name}")
                        await asyncio.sleep(wait_between_profiles)
                        continue
                    if girl_profile is None:
                        girl_profile = session_state.get("girl_profile")

                    """ training_message = [
                            {"role": "system", "content": MATCH_MAKING_SYSTEM_PROMPT},
                            {"role": "user", "content": f"{MATCH_MAKING_USER_PROMPT_V1.format(session_state['boy_profile'], session_state['girl_profile'], MATCH_ANALYSIS_FORMAT_V1)}" },
                            {"role": "assistant", "content": session_state["analysis_result"]}
                        ] """
                    #Now we are getting multiple messages from Graph.
                    if session_state.get("messages",[]):
                        logger.info(f"Out of LangGraph Execution for profile_id:{profile_id}, Total messages got:{len(session_state.get("messages",[]))}")
                        training_data.extend(session_state.get("messages",[]))

                    #let's add some delay between calls to avoid hitting rate limits or overloading the system. This can be adjusted based on actual performance and requirements.
                    await asyncio.sleep(wait_between_profiles)
                #After each batch is done, let's save the training data in a file for future use. This can be used for training a local model or for analysis.
                filename = f"{PROFILE_DATA_PATH}/v1/bulk/{filter_name}/{filter_name}_training_data_{file_time_stamp}_iter_{iter}.json"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                with open(filename, "w") as training_data_file:
                    json.dump(training_data, training_data_file, indent=4)
    

        except Exception as e:
            logger.error(f"Error in loading filter definition for filter:{filter_name} with error:{e}")
            return  

async def agent_main_with_UI():
    #--------------------------
    # Basic Initialisation
    #---------------------------
    mcp_client = MCP_ChatBot(15)
    session_state = AgentState()
    #initialize_local_model("meta-llama/Meta-Llama-3-8B-Instruct")
    #initialize_local_model("meta-llama/Llama-3.2-3B-Instruct")
    session_state["mcp_client"] = mcp_client
    session_state["stage"] = "input"
    session_state["girl_profile_url"] = "https://www.anuroopwiwaha.com/user/view_profile.aspx"
    
    # -----------------------------
    # Page Config + Styling
    # -----------------------------
    st.set_page_config(page_title="Matrimonial AI", layout="wide")

    st.markdown("""
        <style>
        body {
            background-color: #f8f5f2;
        }
        .main-title {
            font-size: 36px;
            font-weight: 700;
            color: #7b2cbf;
        }
        .sub-title {
            font-size: 18px;
            color: #555;
        }
        .card {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Session State Init
    # -----------------------------
    streamlit_session_init()
    
    # -----------------------------
    # Sidebar
    # -----------------------------
    with st.sidebar:
        st.title("⚙️ Settings")

        model_choice = st.selectbox(
            "Choose AI Model",
            ["Local LLaMA 3", "Groq openai/gpt-oss-120b", "LLaMA 3 (Ollama)", "Gpt OSS:20b (Ollama)", "OpenAI-Gpt-5.2", "OpenAI-gpt-4o-mini","OpenAI-gpt-5.4-mini","Groq (LLaMA 70B)", "Groq qwen/qwen3-32b"],
            key="model_choice"
        )

        st.sidebar.info(f"Using: {st.session_state.model_choice}")

        if st.button("🔄 Reset Context"):
            streamlit_session_reset()
            st.rerun()

    # -----------------------------
    # Header
    # -----------------------------
    st.markdown('<div class="main-title">💍 Matrimonial Match AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Smart compatibility insights powered by AI</div>', unsafe_allow_html=True)
    st.write("")

    # -----------------------------
    # Stage 1: URL Input
    # -----------------------------
    if st.session_state.stage == "input":
        st.markdown("### 🔗 Enter Profile URL")

        boy_profile_url = st.text_input("Profile URL", placeholder="https://www.anuroopwiwaha.com/User/MemberProfile.aspx?member_id=870081")

        if st.button("Analyze Profile"):
            if boy_profile_url:
                # Simulate LangGraph call
                with st.spinner("Analyzing profile..."):
                    # Replace this with real LangGraph invocation
                    st.session_state.boy_profile_url = boy_profile_url
                    st_to_langgraph_state_transfer(session_state=session_state)

                    session_state = await execute_langgraph(mcp_client, session_state)
                    
                    langgraph_to_st_state_transfer(session_state)
                    st.session_state.stage = "result"
                    st.rerun()
            else:
                st.warning("Please enter a valid URL")
    #---------
    # REevaluate
    #----------
    elif st.session_state.stage == "reevaluate":
        with st.spinner(f"Re analyzing profile with model:{st.session_state.model_choice}..."):
            st_to_langgraph_state_transfer(session_state=session_state)

            session_state = await execute_langgraph(mcp_client, session_state)
                    
            langgraph_to_st_state_transfer(session_state)
            st.session_state.stage = "result"
            st.rerun()


    # -----------------------------
    # Stage 2: Result Display
    # -----------------------------
    elif st.session_state.stage == "result":
        result = st.session_state.analysis_result

        st.markdown("### 📊 Match Analysis")

        st.markdown(result, unsafe_allow_html=True)

        st.write("")

        if st.button("💬 Continue with Chat"):
            st.session_state.stage = "chat"
            st.rerun()

        if st.button(f"🔄 Reevaluate with {st.session_state.model_choice}"):
            st.session_state.stage = "reevaluate"
            st.rerun()

    # -----------------------------
    # Stage 3: Chat Interface
    # -----------------------------
    elif st.session_state.stage == "chat":
        st_to_langgraph_state_transfer(session_state=session_state)
        st.markdown("### 💬 Ask Questions About This Profile")

        # Display chat history
        for msg in st.session_state.messages:
            logger.info(f"\n\n CHAT STAGE: MSG from Session State:{msg}")
            with st.chat_message(msg["role"]):
                if "Analyse the given Boy and Girl profiles" in msg["content"]:
                    #Write a short message
                    st.write(f"Analyse the profile match between Boy:{session_state['boy_profile']["Full Name"]} and Girl:{session_state['girl_profile']["Full Name"]}")
                else:
                    st.write(msg["content"])

        user_input = st.chat_input("Ask something about the match...")

        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": f"Using match making context from earlier messages, answer user query:{user_input}"})

            with st.chat_message("user"):
                st.write(user_input)

            # Simulate LLM response (replace with LangGraph call)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    st_to_langgraph_state_transfer(session_state)
                    session_state = await execute_langgraph(mcp_client, session_state)
                    st.write(session_state["messages"][-1]["content"])
                    langgraph_to_st_state_transfer(session_state)
                    st.session_state.stage = "chat"
                    st.rerun()
                    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Matcher")
    parser.add_argument("--automated", type=bool, default=False)
    parser.add_argument("--filter_name", type=str, default="peft", help="Name of the filter to run in automated mode")
    args = parser.parse_args()
    if args.automated:
        asyncio.run(filter_run_orchestrator(args.filter_name))
    else:
        asyncio.run(agent_main_with_UI())