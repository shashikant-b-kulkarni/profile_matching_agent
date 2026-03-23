from typing import TypedDict, Annotated, Any
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

# --- LLM ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

#llm = ChatOllama(model="llama3")
VOCAREUM_BASE_URL  = "https://openai.vocareum.com/v1"          # custom proxy
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY") #"voc-1784561922175350473180169ac23371fb5b7.30042371"

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
        print("\n\n ** Entered Model choice : LLaMA 3 (Ollama)")
        return ChatOllama(model="llama3")

    elif model_choice == "Groq (LLaMA 70B)":
        print("\n\n ** Entered Model choice : Groq (LLaMA 70B)")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
    elif model_choice == "Groq openai/gpt-oss-120b":
        print("\n\n ** Entered Model choice : Groq openai/gpt-oss-120b")

        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="openai/gpt-oss-120b"
        )
    elif model_choice == "Groq qwen/qwen3-32b":
        print("\n\n ** Entered Model choice : Groq qwen/qwen3-32b")
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="qwen/qwen3-32b"
        )
    elif model_choice == "Groq mixtral-8x7b":
        print("\n\n ** Entered Model choice : Groq mixtral-8x7b")
        return ChatOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            model="mixtral-8x7b"
        )
    elif model_choice == "OpenAI-Gpt-5.2":
        print("\n\n ** Entered Model choice : OpenAI-Gpt-5.2")
        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
            model="gpt-5.2",                    # ← CHANGED: was "gpt-4"
            temperature=0.1,
            #    max_tokens=2500,
            tags=["profile_matcher"],
        )
    elif model_choice == "OpenAI-gpt-4o-mini":
        print("\n\n ** Entered Model choice : OpenAI-gpt-4o-mini")

        return ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            base_url=VOCAREUM_BASE_URL,         # ← NEW: Vocareum endpoint
            model="gpt-4o-mini",                    # ← CHANGED: was "gpt-4"
            temperature=0.1,
            #    max_tokens=2500,
            tags=["profile_matcher"],
        )

def custom_message_reducer(old, new):
    print(f"\n\n Entered custom_message_reducer old:{old}")
    print(f"\n\n Entered custom_message_reducer new:{new}")
    if "stage" in st.session_state and "reevaluate" == st.session_state.stage:
        old = []
    return (old + new)[-10:]  # keep last 10

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




# --- Nodes ---

def intent_detection(state: AgentState):
    print(f"\n ***** Entered intent_detection with stage:{state["stage"]}")
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

    print(f"n\n ******* GENERAL CHAT NODE RESP:{res}")

    return {"messages": [{"role": "assistant", "content":res.content}]}

MATCH_ANALYSIS_FORMAT= """
# Basic Details (Girl’s POV)

- **Boy’s Full Name:** Devashish Chandrashekhar Thakar  
- **Girl’s Full Name:** Mrunal Kulkarni  
- **Boy’s Profile Picture (200x200):**  
  ![Devashish Chandrashekhar Thakar](https://res.cloudinary.com/wiwaha/image/upload/t_profile_view/AnuroopVar/282026105453PM_672259_20260208_223019(2).jpg)

**Quick snapshot (non-astro):**
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


def matchmaking_node(state: AgentState):
    st.write("\tConsulting AI for. Match making inputs" )
    prompt = MATCH_MAKING_PROMPT.format(state["boy_profile"], state["girl_profile"], f"Output Markup Format:{MATCH_ANALYSIS_FORMAT}",
                                        "11. Use the attached format as guidance and strictly follow output markup format using the same.")
    
    for_history_prompt = MATCH_MAKING_PROMPT.format(state["boy_profile"], state["girl_profile"], "","")
    
    llm = state["llm"]
    res = llm.invoke(prompt)

    st.write("\tAI Inputs received" )


    return {"messages":[{"role":"user","content":for_history_prompt},{"role":"assistant", "content":res.content}], "analysis_result": res.content}


async def profile_site_login(state: AgentState):
    print("Entered Profile Site Login Node")
    st.write("\tFirst, starting with Login to the site")
    login_result = await state["mcp_client"].excute_profile_login()
    if len(login_result) > 0 and "success" in login_result[0]["text"].lower():
        print("Profile Site Login successful")
        st.write("\tLogin Successful")

        state["login_successful"] = True
    else:
        print("Profile Login not successful")  
        st.write("\tLogin Failed")
     
        state["login_successful"] = False
    
    return {**state} 


async def boy_profile_fetch(state: AgentState):
    print("Entered boy_profile_fetch Node")
    st.write("\tStarting Fetching of Boy's Profile")
    login_result = await state["mcp_client"].excute_profile_fetch_tool(state["boy_profile_url"], False, True, False)
    if len(login_result) > 0 and len(login_result[0]["text"].lower())>0:
        profile_data = []
        try:
            for ext_rslt in login_result:
                profile = json.loads(ext_rslt["text"])
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
            print("boy_profile_fetch successful: ")
            st.write("\tFetching of Boy's Profile Is Successful")

            state["boy_profile"] = profile_data[0]
            state["boy_profile_fetch_success"] = True
        except Exception as e:
            print("Exception happened in boy_profile_fetch", e)
            state["boy_profile_fetch_success"] = False

    else:
        st.write("\tFetching of Boy's Profile Encountered Error")
        print("Profile Login not successful")       
        state["boy_profile_fetch_success"] = False
    
    return {**state}  

async def girl_profile_fetch(state: AgentState):
    print("Entered girl_profile_fetch Node")
    st.write("\tStarting Fetching of Girl's Profile")

    if state["girl_profile"]:
        st.write("\tGirl profile details already present")
        return {}

    login_result = await state["mcp_client"].excute_profile_fetch_tool(state["girl_profile_url"], False, False, True)
    if len(login_result) > 0 and len(login_result[0]["text"].lower())>0:
        profile_data = []
        try:
            for ext_rslt in login_result:
                profile = json.loads(ext_rslt["text"])
                if len(profile.keys()) > 0:       
                    profile_data.append(profile)
            print("girl_profile_fetch successful: ")
            st.write("\tGirl's Profile Fetch Successful" )

            state["girl_profile"] = profile_data[0]
            state["girl_profile_fetch_success"] = True
        except Exception as e:
            print("Exception happened in girl_profile_fetch", e)
            st.write("\tGirl's Profile Fetch Failed" )

            state["girl_profile_fetch_success"] = False

    else:
        print("Girl Profile fetch not successful")   
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

        print(f"\n\n*****Whole State:{new_state}\n\n")
        if new_state["stage"] == "input":
            print(f"\n\n ***** LangGraph Analysis Result:{new_state["analysis_result"]}")
        elif new_state["stage"] == "chat":
            print(f"\n\n ****** LLM Response for Chat: {new_state["messages"][-1]}")

        return new_state
        
    except Exception as e:
        print("Agent Main encountered exception",e)
    finally:
        await mcp_client.cleanup()

def st_to_langgraph_state_transfer(session_state:AgentState):
    session_state["messages"] = st.session_state.messages
    session_state["boy_profile"] = st.session_state.boy_profile
    session_state["girl_profile"] = st.session_state.girl_profile
    session_state["boy_profile_url"] = st.session_state.boy_profile_url
    session_state["analysis_result"] = st.session_state.analysis_result
    session_state["stage"] = st.session_state.stage
    session_state["model_choice"] = st.session_state.model_choice
    session_state["llm"] = get_llm(session_state["model_choice"])

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

        print("Whole State", result)
        print(result["response"])
        
    except Exception as e:
        print("Agent Main encountered exception",e)
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
    builder.add_node("matchmaking_node", matchmaking_node)

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


def streamlit_session_reset():

    st.session_state.stage = "input"
    st.session_state.messages = []
    st.session_state.analysis_result = None
    st.session_state.boy_profile_url = None 
    st.session_state.boy_profile = None
    #st.session_state.girl_profile = None

   

async def agent_main_with_UI():
    #--------------------------
    # Basic Initialisation
    #---------------------------
    mcp_client = MCP_ChatBot(15)
    session_state = AgentState()
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
            ["Groq openai/gpt-oss-120b", "LLaMA 3 (Ollama)", "OpenAI-Gpt-5.2", "OpenAI-gpt-4o-mini","Groq (LLaMA 70B)", "Groq qwen/qwen3-32b"],
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
            print(f"\n\n CHAT STAGE: MSG from Session State:{msg}")
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
    asyncio.run(agent_main_with_UI())