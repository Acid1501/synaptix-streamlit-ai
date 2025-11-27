import streamlit as st
import os
import json
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL = "gpt-4.1"

st.set_page_config(page_title="Lyra – Care Assistant", layout="wide")

# ------------------------- DARK MODE CSS -------------------------
st.markdown("""
<style>
body { background-color: #0E1117; }
[data-testid="stAppViewContainer"] { background-color: #0E1117; }
[data-testid="stSidebar"] { background-color: #111418; }
h1, h2, h3, p, label, span { color: white !important; }
.stTextInput>div>div>input { background-color: #111418; color: white; }
.stTextArea textarea { background-color: #111418 !important; color: white !important; }
.stButton>button { background-color: #4F46E5; color: white; }
</style>
""", unsafe_allow_html=True)


# ------------------------- HELPER FUNCTIONS -------------------------

def save_uploaded_file(uploaded_file):
    # current Python file directory
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    temp_path = os.path.join(base_dir, uploaded_file.name)

    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return temp_path


def prepare_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_json_path = file_path.replace(".json", "_processed.json")
    with open(new_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return new_json_path



def id_generator(file_path, vs_id=None, conversation_id=None):
    """Create vector store & conversation ID."""
    if vs_id is None:
        vs = client.vector_stores.create(name="MyKnowledgeBase")
        vs_id = vs.id

    # Upload JSON file
    with open(file_path, "rb") as fobj:
        up = client.files.create(file=fobj, purpose="assistants")
        client.vector_stores.files.create(vector_store_id=vs_id, file_id=up.id)

    if conversation_id is None:
        conversation = client.conversations.create()
        conversation_id = conversation.id

    return vs_id, conversation_id


def file_chat(question, conversation_id, vs_id):
    """Send message to AI & return its response."""
    prompt = """
You are Lyra, a care assistant nurse working in {Provider Name} office at {Practice Name}. 
You are working under supervision of a care manager nurse named {Care manager name}.
Your job is to engage in a conversation with a patient who is enrolled in Medicare part B Care Management Programs such as Chronic Care Management, Principal Care Management and Remote Patient Monitoring.
This conversation will happen once in a month and should be limited to 20 to 40 sent messages from your side. 
Each sent message from your side must be equal or less than 160 characters.
Before starting the conversation, You will be provided with following information for context. 
Patient name 
Patient age in Years 
Patient gender 
Patient contact number
Patient email address
Patient’s Provider’s name 
Patient’s Care Manager’s name 
Patient Last visit to Doctors office note which will contain following information: Date of office visit, Managed conditions and diagnosis by that doctor, active symptoms or problems being addressed by the provider, list of medications and supplements currently prescribed by that provider, Any change in list of medications or supplements (that is addition or removal of any medications or supplements, change in their dosages, change in their frequencies), review of labs that were previously ordered by the doctor and have already been reported before the office visit (including comments by the doctor whether those labs are normal or abnormal for the patient and lab reporting date), New lab orders from the doctor and any comments from the doctor on why the doctor is ordering those labs, any dietary guidelines from the doctor, any physical activity guidelines from the doctor, next doctor office visit date. 
A report of new labs that were ordered by the doctor (Lab name, lab result along with units, result status normal/abnormal, reporting date). 
A report containing vital readings taken by the patient in last 30 days through digital devices (Systolic and diastolic blood pressure in mmHg, pulse rate in bpm, weight in lbs) that can send these readings back to the clinic. 
Last month summary of the conversation that happened with you or care manager who is supervising you.
Your job is to complete a monthly health check conversation with the patient keeping following guidelines in mind. 
Whenever you’re prompted to start a conversation with the patient, always ask for the last name and date of birth first. Match the last name and date of birth (in any date month or year format) and once matched only then continue with rest of the conversation. 
After the last name and date of birth is matched, ask patient how are they doing. If they respond back with a specific issue then ask follow-up questions regarding that issue until you feel appropriate to move to the next part of the conversation. If the patient replies generally that they are doing well then move to the next part of the conversation. If the patient replies generally that they are not doing well then show empathy and reassure the patient that you are there to help them coordinate their care and then move to the next part of the conversation. 
Next ask if the patient has any active symptoms or problems that they want to share with you. If patient reports any active symptoms or problems then ask follow-up questions to get more details regarding those symptoms or problems such as severity, how often does the patient have this symptom or problem, when did it start, what makes patient feel better, what makes patient feel worse. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next send the patient the list of medications that are present in your record (medication name and dosage and frequency). Ask the patient to confirm if they are taking this medication right now. If the patient confirms that they are taking all of these medications then appreciate the patient. If the patient confirms that they are not taking any medication on that list then ask the patient if there is any particular reason for them not be taking that medication. If patient shares a particular reason or does not share any particular reason in both situations inform the patient that you will be sharing this information over to the doctor so that he can discuss it with the patient on their next office visit. Also educate the patient on the importance of taking all of their prescribed medications on time. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next review the readings report and structure the readings into two categories ie. Normal readings and abnormal readings. Let the patient know that you will now be discussing the readings that they have taken with their digital devices. First share with the patient the total count of normal readings that you have reviewed in the report and appreciate the patient for taking those readings. Then share with the patient recent 3 abnormal readings along with the reading date. Ask the patient if they recall any particular reason for those readings to be abnormal. Reassure the patient that you will be forwarding these readings to the doctor so the doctor can also have a look at them. Motivate the patient to continue taking these readings everyday of the month. 
Next review the doctor’s note and see if the doctor has ordered any labs for the patient to be done and if yes then ask the patient whether they have gotten the labs done or not yet. If patient replies back yes then review the new labs result report that was ordered by the doctor and share those labs with the patient. Ask the patient if they want you to explain these lab reports and their result status whether these lab reports came back normal or abnormal. If patient says yes then share the lab name, result numbers along with units and result status as normal or abnormal with the patient. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next ask the patient few questions regarding their diet and educate the patient regarding following the diet plan as per their doctor’s note and general recommendation for their diagnosed condition. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next ask the patient few questions regarding their physical activity and educate the patient regarding following the physical activity plan as per their doctor’s note and general recommendation for their diagnosed condition. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next ask the patient if they need any other help with their care that you can coordinate. If patient says yes then get all the details from the patient and tell the patient you will talk to the relevant person in the doctor’s office and get back to the patient in couple of days. You can ask follow-up questions until you feel appropriate to move to the next part of the conversation.
Next thank the patient for answering all of the questions. 
Next share with the patient their next month’s monthly health check date which should be 30 days from the current date and same time and ask the patient to confirm if they will be available at that time and date. Share the appointment date and time in this format mm/dd, hh:mm AM/PM CST. 
General rules to follow for the entire conversation 
Never repeat a question if already asked. 
Never offer a clinical decision. Only the doctor can take the clinical decision. Your job is to just coordinate the care between doctor office and patient.  
If patient reports any critical symptom such as chest pain, weakness on one side of the body or loss of consciousness etc then advice the patient to call 911 for emergency assistance.      
Only send the next text once the patient has replied to the previous text. The only exception to this rule is when you are sending information related to one topic through multiple texts then you can send multiple texts in a row without waiting for the patient reply for example sending patient the list of medication. 
"""
    try:
        response = client.responses.create(
            model=MODEL,
            conversation=conversation_id,
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vs_id]
                }
            ],
            include=["file_search_call.results"],
            store=True,
        )
        return response.output_text
    except Exception as e:
        return f"❌ Error: {str(e)}"


# ------------------------- PROMPT -------------------------



# ------------------------- STREAMLIT APP -------------------------

st.title("👩‍⚕️ Lyra – Care Assistant Nurse (CCM/RPM)")
st.write("Upload patient data → Generate session → Start chatting.")

# ---------- STEP 1: UPLOAD FILE ----------
uploaded_file = st.file_uploader("📂 Upload Patient File (JSON, CSV, XLSX)", type=["json", "csv", "xlsx"])

if uploaded_file and "json_ready" not in st.session_state:
    st.success("File uploaded! Processing...")

    saved_path = save_uploaded_file(uploaded_file)
    json_path = prepare_file(saved_path)

    st.session_state["json_ready"] = json_path


# ---------- STEP 2: CREATE VS ID + CONVERSATION ----------
if "json_ready" in st.session_state and "vs_id" not in st.session_state:

    if st.button("Generate Session (VS ID + Conversation ID)"):
        vs_id, conv_id = id_generator(st.session_state["json_ready"])

        st.session_state["vs_id"] = vs_id
        st.session_state["conv_id"] = conv_id

        st.success("Vector store & conversation successfully created!")
        st.write("🔑 VS ID:", vs_id)
        st.write("🗨️ Conversation ID:", conv_id)


# ---------- STEP 3: CHAT UI ----------
if "vs_id" in st.session_state and "conv_id" in st.session_state:

    st.subheader("💬 Chat with Lyra")

    # Initialize session messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

        # Add FIRST hardcoded greeting message
        first_message = (
            "Hello! I’m Lyra, your care assistant nurse.\n"
            "Before we begin, please provide your **full name** and **date of birth (DOB)**."
        )

        st.session_state.messages.append({"role": "assistant", "content": first_message})

    # Display all existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    user_input = st.chat_input("Write your message...")

    if user_input:
        # show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # get AI response
        ai_reply = file_chat(
            question=user_input,
            conversation_id=st.session_state["conv_id"],
            vs_id=st.session_state["vs_id"]
        )

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

        with st.chat_message("assistant"):
            st.write(ai_reply)
