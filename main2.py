import streamlit as st
import Func2
from predibase import Predibase

predibase_key="pb_13anNxIDUiuuUmmhYkEHCw"
st.set_page_config(page_title="User/Company Multi-Page App", layout="wide")


col1, col2 = st.columns([1, 5]) 
with col1:
    st.image("1.png", width=100)  


with col2:
    st.title("RefundRangers💪 - your Tax Return Co-pilot")

# Input field for occupation
occupation = st.text_input("What is your occupation?")

# Email address input (optional)
email = st.text_input(
    "What is your email address? (optional)", placeholder="example@example.com"
)

# Initialize session state
if "response_rag_text" not in st.session_state:
    st.session_state.response_rag_text = None

# Add CSS for button styling
submit_button_style = """
    <style>
    div.stButton > button {
        width: 100%;
        background-color: #abebc6;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    div.stButton > button:hover {
        background-color: #82e0aa;
    }
    </style>
"""
st.markdown(submit_button_style, unsafe_allow_html=True)

# Execute RAG when the Submit button is clicked
if st.button("Submit"):
    if occupation:
        result = Func2.main(occupation)  
        st.session_state.response_rag_text = result  # Store the result in the session
    else:
        st.warning("Please enter your occupation.")

# Display RAG result
if st.session_state.response_rag_text:
    st.title("Your Occupation's Deduction Strategy")
    st.write(st.session_state.response_rag_text)

# Initialize session state
if "image_upload_count" not in st.session_state:
    st.session_state.image_upload_count = 1
if "image_files" not in st.session_state:
    st.session_state.image_files = [None] * st.session_state.image_upload_count

st.title("Upload Your Receipts")

# Function to add a new image upload field
def add_image_upload_field():
    st.session_state.image_upload_count += 1
    st.session_state.image_files.append(None)

# Add another Receipt button (default style)
if st.button("Add another Receipt"):
    add_image_upload_field()

# Create file upload fields
for i in range(st.session_state.image_upload_count):
    st.session_state.image_files[i] = st.file_uploader(
        f"Receipt {i+1}", type=["png", "jpg", "jpeg"], key=f"file_uploader_{i}"
    )
    # Display the uploaded image (if any)
    if st.session_state.image_files[i] is not None:
        st.image(st.session_state.image_files[i], caption=f"Uploaded Receipt {i+1}", use_column_width=True)

st.title("Chat for more information")

# Initialize message state 
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "From your occupation, I can suggest some potential deductions. What would you like to know?",
        }
    ]

# Function to display chat history
def display_chat_history():
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        elif msg["role"] == "assistant":
            st.chat_message("assistant").write(msg["content"])


###### I need to fix this part, The result format is not enoguth. 

# Prompt for deductibility analysis
def handle_receipt_analysis(uploaded_file, occupation):
    # Extract receipt description using OCR
    receipt_result = Func2.extract_text_from_receipts_with_predibase(uploaded_file)
    description = receipt_result.get('description', 'Unknown')

    # Generate a prompt for deductibility assessment
    deductibility_prompt = f"""
    You are a helpful tax advisor. YOU only answer with tax-related questions. 
    A user working as a {occupation} has uploaded a receipt image. 
    The receipt description is as follows: "{description}".
    Based on the user's profession ({occupation}), please assess whether this expense is likely to be tax-deductible.
    From the {st.session_state.response_rag_text}, you can use this guidelines for deductions.
    """

    # Call the Predibase fine tuning model and process the response
    pb = Predibase(api_token=predibase_key)
    adapter_id = "qna-guides-pro-model/2"
    lorax_client = pb.deployments.client("solar-pro-preview-instruct")

    response = lorax_client.generate(
        deductibility_prompt,
        adapter_id=adapter_id,
        max_new_tokens=1000
    ).generated_text

    return response


def handle_chat():
    if st.session_state.image_files and occupation:
        for i, uploaded_file in enumerate(st.session_state.image_files):
            if uploaded_file:
                receipt_analysis = handle_receipt_analysis(uploaded_file, occupation)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Image {i+1}:\n{receipt_analysis}"
                })
                st.chat_message("assistant").write(f"Image {i+1}:\n{receipt_analysis}")
    
    # General text-based Q&A (always available, separated logic)
    if prompt := st.chat_input("Ask your tax-related questions here"):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Call Predibase model and handle the response
        pb = Predibase(api_token=predibase_key)
        adapter_id = "qna-guides-pro-model/2"
        lorax_client = pb.deployments.client("solar-pro-preview-instruct")

        # Query Predibase for general Q&A
        response = lorax_client.generate(
            prompt,
            adapter_id=adapter_id,
            max_new_tokens=1000
        ).generated_text

        # Append and display the assistant's response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# 4. Display chat history on the screen
display_chat_history()

# 5. Execute the main logic
handle_chat()
