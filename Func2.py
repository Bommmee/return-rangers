import json 
import os
import pdfplumber
import time
import re
import base64  
import requests  
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain.schema import Document 


upstage_api_key = "up_KnnEhzk56BNSfxepFC4wHwyO5W2p5"

# Function to extract text from PDFs
def extract_text_from_pdfs(selected_files):
    text_data = ""
    for file_path in selected_files:
        if os.path.exists(file_path):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text_data += page.extract_text()
        else:
            print(f"File {file_path} not found.")
    return text_data

# Function to post-process extracted text
def clean_extracted_text(text):
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'([a-zA-Z])([A-Z])', r'\1. \2', text)  
    return text.strip()

# Function to extract text from receipt images using the predibase_model
def extract_text_from_receipts_with_predibase(uploaded_file, occupation=None):
    if uploaded_file is None:
        return {
            "description": "No image uploaded."
        }

    # Solar DocVision API URL
    api_url = "https://api.upstage.ai/v1/solar/chat/completions"

    # API headers - Auth token needed (Bearer format)
    headers = {
        "Authorization": f"Bearer {upstage_api_key}",
        "Content-Type": "application/json"
    }

    # Encode the uploaded file as base64
    base64_image = base64.b64encode(uploaded_file.read()).decode('utf-8')

    # Add a detailed prompt for the image description, using occupation if available
    if occupation:
        text_prompt = f"""
        You are a smart assistant that specializes in understanding the content of images. A user who works as a {occupation} has uploaded an image. Your task is to describe what this image is about in a concise, clear manner. Please focus on the core details and provide a summary that includes the main elements visible in the image.

        **Response Format**:
        - Description: [Provide a brief description of what this image contains.]
        """
    else:
        text_prompt = """
        You are a smart assistant that specializes in understanding the content of images. A user has uploaded an image. Your task is to describe what this image is about in a concise, clear manner. Please focus on the core details and provide a summary that includes the main elements visible in the image.

        **Response Format**:
        - Description: [Provide a brief description of what this image contains.]
        """

    # Request data including the question and base64 image
    data = {
        "model": "solar-docvision",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text_prompt  # The prompt describing the image
                    }
                ]
            }
        ]
    }

 
    response = requests.post(api_url, headers=headers, json=data)

 
    if response.status_code == 200:
        result = response.json()

        # Extract and process fields from the response
        choices = result.get("choices", [])
        if choices:
            message_content = choices[0].get("message", {}).get("content", "")

            return {
                "description": message_content.strip() or "No description available."
            }
        else:
            return {
                "description": "No valid response from API."
            }
    else:
        return {
            "description": f"Error: {response.status_code}, {response.text}"
        }


# Function to select files matching a specific occupation category
def select_files_for_occupation(category, folder_path):
    selected_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf') and category.lower().replace(" ", "_") in filename.lower().replace(" ", "_"):
            selected_files.append(os.path.join(folder_path, filename))
    return selected_files

# Mapping occupations to categories using the Upstage model
def map_occupation_to_category(occupation):
    occupation_categories = [
        "Adult industry workers",
        "Agricultural workers",
        "Apprentices and trainees",
        "Australian Defence Force members",
        "Building and construction employees",
        "Bus drivers",
        "Call centre operators",
        "Cleaners",
        "Community workers and direct carers",
        "Doctor specialist and other medical professionals",
        "Engineers",
        "Factory workers",
        "Fire fighters",
        "Fitness and sporting industry employees",
        "Flight crew",
        "Gaming attendants",
        "Guards and security employees",
        "Hairdressers and beauty professionals",
        "Hospitality industry workers",
        "IT professionals",
        "Lawyers",
        "Meat workers",
        "Media professionals",
        "Mining site employees",
        "Nurses and midwives",
        "Office workers",
        "Paramedics",
        "Performing artists",
        "Pilots",
        "Police",
        "Professional Sportsperson",
        "Real estate employees",
        "Recruitment consultants",
        "Retail industry workers",
        "Sales and marketing managers",
        "Teacher and education professionals",
        "Tradesperson",
        "Train drivers",
        "Travel agent employees",
        "Truck drivers"
    ]

    # Create a prompt template
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert in job classification. Your task is to classify the given occupation into one of the 40 occupation categories provided below.
        You are provided with an input occupation and must match it to the most appropriate category based on the nature and responsibilities of the job.

        # Occupation Categories:
        {categories}

        Match the input occupation to the closest category. If you are unsure or there is ambiguity, select the category that best describes the overall role.

        ### Examples:
        - Input: 'web designer' -> Output: 'IT professionals'
        - Input: 'software engineer' -> Output: 'IT professionals'
        - Input: 'Basketball player' -> Output: 'Professional Sportsperson'
        - Input: 'stewardess' -> Output: 'Flight crew'
        - Input: 'Plumber' -> Output: 'Tradesperson'
        - Input: 'Adult Video Director' -> Output: 'Adult industry workers'

        Input: "{occupation}"
        Output the matched occupation category in JSON format.
        Example JSON:
        {{
          "occupation": "{occupation}",
          "category": "IT professionals"
        }}
        """
    )


    prompt = prompt_template.format(categories=", ".join(occupation_categories), occupation=occupation)

    # Use the Upstage model to generate a response 
    llm = ChatUpstage(api_key=upstage_api_key)
    response = llm.invoke(prompt)  # Passed as a string

    # Parse the response content into JSON
    try:
        response_data = json.loads(response.content)
        if "category" in response_data:
            return response_data["category"]
        else:
            return "Unclassified"
    except json.JSONDecodeError:
        return "Unclassified"

# Function to generate a summary 
def summarize_with_RAG(text_data, occupation):
    embedding = UpstageEmbeddings(model="solar-pro-preview-instruct", api_key=upstage_api_key)
    
    # Convert to a Document object
    document = Document(page_content=text_data)

    # Split the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    splits = text_splitter.split_documents([document])

    # Configure the retriever (BM25)
    retriever = BM25Retriever.from_documents(splits)

    # Set the query for the search
    query = f"Summarize tax deduction guidelines for {occupation}"

    # Summarize the content of the retrieved documents
    context_docs = retriever.invoke(query)

    llm = ChatUpstage(api_key=upstage_api_key)
    prompt_template = PromptTemplate.from_template(
"""
You are a tax consultant providing specific tax deduction guidelines for {occupation}. Please structure the summary based on the following format:

🌟 Key Tax Deduction Guidelines for {occupation}
1️⃣ Income and Allowances:
   - Include: What categories of income are relevant, such as salary, bonuses, and overtime pay.
   - Exclude: What income or allowances are tax-exempt or should not be included in tax returns.

2️⃣ Allowances and Deductions:
   - How to Handle: Describe how work-related allowances should be included in the tax return and what types of allowances may be deductible.
   - Examples: List examples of deductible and non-deductible allowances.

3️⃣ Deductible Work Expenses:
   - Golden Rules: Explain the criteria for work expenses that can be deducted (e.g., must be directly related to earning income).
   - Examples: Provide examples of deductible expenses and non-deductible expenses.

4️⃣ Car and Travel Expenses:
   - Claimable: Outline when car and travel expenses can be claimed.
   - Non-Claimable: Mention cases where these expenses are not deductible.

5️⃣ Clothing and Uniform Expenses:
   - Claimable: Describe what kind of work-related clothing or uniforms can be deducted.
   - Non-Claimable: List personal clothing items that cannot be deducted.

6️⃣ Phone, Data, and Internet Expenses:
   - Claimable: How should work-related phone, data, and internet expenses be claimed?

7️⃣ Self-Education Expenses:
   - Claimable: Under what conditions can self-education expenses be deducted?
   - Examples: Provide examples of deductible and non-deductible education expenses.

🚫 Non-Deductible Items:
   - Personal expenses and items provided or reimbursed by the employer.

📜 Summary:
   - Only work-related expenses can be deducted, and personal or reimbursed costs are not eligible for deductions. Ensure all expenses are properly documented.
"""
)

    
    chain = prompt_template | llm

    # Generate the summary based on the retrieved documents
    summary = chain.invoke({"occupation": occupation, "context": context_docs})

    return summary.content

# Main function
def main(occupation):
    start_time = time.time()

    # Step 1: Map the occupation category
    category = map_occupation_to_category(occupation)

    # Step 2: Select files for the given occupation and extract text
    pdf_folder_path = 'deduction_pdfs'
    selected_files = select_files_for_occupation(category, pdf_folder_path)
    
    if not selected_files:
        return f"No files found for occupation: {occupation}"
    
    pdf_text = extract_text_from_pdfs(selected_files)
    if not pdf_text:
        return f"No text extracted for occupation: {occupation}"

    # Step 3: Clean the extracted text
    cleaned_text = clean_extracted_text(pdf_text)

    # Step 4: Use RAG to generate summarized guidelines
    answer = summarize_with_RAG(cleaned_text, occupation)

    # Step 5: Return the result
    return f"Tax guidelines Summary for {category}:\n\n{answer}\n"