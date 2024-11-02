# import os
# import streamlit as st
# import google.generativeai as genai
# from PyPDF2 import PdfReader
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores.faiss import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from io import BytesIO  # Import BytesIO for handling binary file objects


# # Define the directory where PDFs are stored
# PDF_DIRECTORY = "/Users/tusharsharma/Downloads/Chat-With-PDF-main/data"  # Update this path to your PDF folder location
# def get_single_pdf_chunks(pdf, text_splitter):
#     try:
#         # Use BytesIO to wrap the PDF for PyPDF2 compatibility
#         pdf_bytes = BytesIO(pdf.read())
#         pdf_reader = PdfReader(pdf_bytes)
#         pdf_chunks = []
#         for page in pdf_reader.pages:
#             page_text = page.extract_text()
#             page_chunks = text_splitter.split_text(page_text)
#             pdf_chunks.extend(page_chunks)
#         return pdf_chunks
#     except Exception as e:
#         st.warning(f"Error processing PDF: {e}")
#         return []

# def get_all_pdfs_chunks(pdf_docs):
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=2500,
#         chunk_overlap=500
#     )

#     all_chunks = []
#     for pdf in pdf_docs:
#         pdf_chunks = get_single_pdf_chunks(pdf, text_splitter)
#         all_chunks.extend(pdf_chunks)
#     return all_chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     try:
#         vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#         return vectorstore
#     except:
#         st.warning("Issue with reading the PDF/s. Your File might be scanned so there will be nothing in chunks for embeddings to work on")

# def get_response(context, question, model):

#     if "chat_session" not in st.session_state:
#         st.session_state.chat_session = model.start_chat(history=[])

#     prompt_template = f"""
#     You are a helpful and informative bot that answers questions using text from the reference context included below. \
#     Be sure to respond in a complete sentence, providing in depth, in detail information and including all relevant background information. \
#     However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
#     strike a friendly and conversational tone. \
#     If the passage is irrelevant to the answer, you may ignore it also as a Note: Based on User query Try to Look into your chat History as well
    
#     Context: {context}?\n
#     Question: {question}\n
#     """

#     try:
#         response = st.session_state.chat_session.send_message(prompt_template)
#         return response.text

#     except Exception as e:
#         st.warning(e)

# def working_process(generation_config):

#     system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user request"

#     model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

#     vectorstore = st.session_state['vectorstore']

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#         AIMessage(content="Hello! I'm a PDF Assistant. Ask me anything about your PDFs or Documents")
#     ]
    
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.markdown(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.markdown(message.content)

#     user_query = st.chat_input("Enter Your Query....")
#     if user_query is not None and user_query.strip() != "":
#         st.session_state.chat_history.append(HumanMessage(content=user_query))

#         with st.chat_message("Human"):
#             st.markdown(user_query)
        
#         with st.chat_message("AI"):
#             try:
#                 relevant_content = vectorstore.similarity_search(user_query, k=10)
#                 result = get_response(relevant_content, user_query, model)
#                 st.markdown(result)
#                 st.session_state.chat_history.append(AIMessage(content=result))
#             except Exception as e:
#                 st.warning(e)

# def load_pdfs_and_build_vectorstore():
#     # Load PDFs from the directory
#     pdf_docs = []
#     for filename in os.listdir(PDF_DIRECTORY):
#         if filename.endswith(".pdf"):
#             pdf_path = os.path.join(PDF_DIRECTORY, filename)
#             try:
#                 with open(pdf_path, "rb") as pdf_file:
#                     pdf_docs.append(BytesIO(pdf_file.read()))  # Read as BytesIO object
#             except Exception as e:
#                 st.warning(f"Failed to load PDF {filename}: {e}")

#     # Process PDF text into chunks
#     text_chunks = get_all_pdfs_chunks(pdf_docs)
    
#     # Create the vectorstore with the chunks
#     vectorstore = get_vector_store(text_chunks)
    
#     # Store the vectorstore in session state
#     st.session_state.vectorstore = vectorstore
# def main():
#     load_dotenv()

#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
#     st.header("Chat with Multiple PDFs :books:")

#     # Directly set the API key (not recommended for production)
#     google_api_key = "AIzaSyCo1l1R3mI1aWAPGbtUq_DzgGH5XpoXfzA"
#     genai.configure(api_key=google_api_key)

#     generation_config = {
#         "temperature": 0.2,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 8000,
#     }

#     # Initialize state variables
#     if "vectorstore" not in st.session_state:
#         st.session_state.vectorstore = None

#     # Show a spinner while processing PDFs
#     with st.spinner("Processing PDFs, please wait..."):
#         load_pdfs_and_build_vectorstore()

#     # Once PDFs are processed, allow interaction
#     if st.session_state.vectorstore is not None:
#         working_process(generation_config)

# if __name__ == "__main__":
#     main()




# import os
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.vectorstores.faiss import FAISS
# import pickle

# VECTORSTORE_PATH = "vectorstore.pkl"

# def load_vectorstore():
#     # Load preprocessed vectorstore
#     with open(VECTORSTORE_PATH, "rb") as file:
#         return pickle.load(file)

# def get_response(context, question, model):

#     if "chat_session" not in st.session_state:
#         st.session_state.chat_session = model.start_chat(history=[])

#     prompt_template = f"""
#     You are a helpful and informative bot that answers questions using text from the reference context included below. \
#     Be sure to respond in a complete sentence, providing in depth, in detail information and including all relevant background information. \
#     However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
#     strike a friendly and conversational tone. \
#     If the passage is irrelevant to the answer, you may ignore it also as a Note: Based on User query Try to Look into your chat History as well
    
#     Context: {context}?\n
#     Question: {question}\n
#     """

#     try:
#         response = st.session_state.chat_session.send_message(prompt_template)
#         return response.text

#     except Exception as e:
#         st.warning(e)

# def working_process(generation_config):

#     system_instruction = "You are a helpful document answering assistant. You care about user and user experience. You always make sure to fulfill user request"

#     model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)

#     vectorstore = st.session_state['vectorstore']

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = [
#         AIMessage(content="Hello! I'm a PDF Assistant. Ask me anything about your PDFs or Documents")
#     ]
    
#     for message in st.session_state.chat_history:
#         if isinstance(message, AIMessage):
#             with st.chat_message("AI"):
#                 st.markdown(message.content)
#         elif isinstance(message, HumanMessage):
#             with st.chat_message("Human"):
#                 st.markdown(message.content)

#     user_query = st.chat_input("Enter Your Query....")
#     if user_query is not None and user_query.strip() != "":
#         st.session_state.chat_history.append(HumanMessage(content=user_query))

#         with st.chat_message("Human"):
#             st.markdown(user_query)
        
#         with st.chat_message("AI"):
#             try:
#                 relevant_content = vectorstore.similarity_search(user_query, k=10)
#                 result = get_response(relevant_content, user_query, model)
#                 st.markdown(result)
#                 st.session_state.chat_history.append(AIMessage(content=result))
#             except Exception as e:
#                 st.warning(e)

# def main():
#     load_dotenv()

#     st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
#     st.header("Chat with Multiple PDFs :books:")

#     # Directly set the API key (not recommended for production)
#     google_api_key = "AIzaSyCo1l1R3mI1aWAPGbtUq_DzgGH5XpoXfzA"
#     genai.configure(api_key=google_api_key)

#     generation_config = {
#         "temperature": 0.2,
#         "top_p": 1,
#         "top_k": 1,
#         "max_output_tokens": 8000,
#     }

#     # Load preprocessed vectorstore
#     if "vectorstore" not in st.session_state:
#         with st.spinner("Loading preprocessed data..."):
#             st.session_state.vectorstore = load_vectorstore()

#     # Once the vectorstore is loaded, allow interaction
#     if st.session_state.vectorstore is not None:
#         working_process(generation_config)

# if __name__ == "__main__":
#     main()


# import os
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.vectorstores.faiss import FAISS
# import json
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from datetime import datetime

# VECTORSTORE_INDEX_PATH = "faiss_index.index"
# METADATA_PATH = "metadata.json"
# CHAT_HISTORY_PATH = "chat_history.json"

# st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

# def load_vectorstore():
#     vectorstore = FAISS.load_local(
#         VECTORSTORE_INDEX_PATH, 
#         embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
#         allow_dangerous_deserialization=True
#     )
#     with open(METADATA_PATH, "r") as file:
#         texts = json.load(file)
#     vectorstore.texts = texts
#     return vectorstore

# def load_chat_history():
#     """Load chat history from a JSON file."""
#     if os.path.exists(CHAT_HISTORY_PATH):
#         try:
#             with open(CHAT_HISTORY_PATH, "r") as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             st.warning("Could not decode chat history file.")
#             return []
#     return []

# def save_chat_history():
#     """Save current chat to history file if it's a new session."""
#     chat_history = load_chat_history()
#     if "current_chat" in st.session_state and st.session_state.current_chat:
#         chat_history.append({
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "messages": st.session_state.current_chat
#         })
#         with open(CHAT_HISTORY_PATH, "w") as f:
#             json.dump(chat_history, f)

# def update_chat_history():
#     """Update the selected chat with new messages and save the history file."""
#     chat_history = load_chat_history()
#     if "selected_chat_index" in st.session_state and st.session_state.current_chat:
#         selected_chat_index = st.session_state.selected_chat_index
#         if 0 <= selected_chat_index < len(chat_history):
#             chat_history[selected_chat_index]["messages"] = st.session_state.current_chat
#             with open(CHAT_HISTORY_PATH, "w") as f:
#                 json.dump(chat_history, f)
#         else:
#             st.warning("Selected chat index is out of range.")
#     else:
#         st.warning("No chat selected or no current chat to update.")

# def get_response(context, question, model):
#     if "chat_session" not in st.session_state or st.session_state.chat_session is None:
#         st.session_state.chat_session = model.start_chat(history=[])

#     if "current_chat" not in st.session_state:
#         st.session_state.current_chat = []

#     chat_history_context = ""
#     for past_query, past_response in st.session_state.current_chat:
#         chat_history_context += f"User asked: {past_query}\nAI responded: {past_response}\n\n"

#     prompt_template = f"""
#     You are a helpful assistant who provides structured, friendly answers in response to user queries. When explaining concepts, follow this structure:

#     - Begin with a **simple definition**.
#     - Use an **analogy** for clarity, if applicable.
#     - Break down key points in bullet points:
#         - Example: 
#             - **Regular Income**: Describe predictable cash flow.
#             - **Flexibility**: Explain customizable options.
#             - **Capital Preservation**: Discuss growth potential.
#     - End with a friendly note.

#     **Previous Conversations**:
#     {chat_history_context}

#     **Current Context**: {context}
#     **User's Question**: {question}

#     For follow-up queries or definitions, provide concise, relevant summaries or details based on past interactions.
#     """

#     try:
#         response = st.session_state.chat_session.send_message(prompt_template)
#         st.session_state.current_chat.append((question, response.text))
#         return response.text

#     except AttributeError:
#         st.warning("Chat session could not be initialized. Please restart the app.")
#     except Exception as e:
#         st.warning(e)

# def sidebar_ui():
#     st.sidebar.title("Settings")
#     if "api_key" not in st.session_state:
#         st.session_state.api_key = ""

#     api_key = st.sidebar.text_input("Enter your Google API Key", type="password", value=st.session_state.api_key)
#     st.session_state.api_key = api_key
    
#     st.sidebar.markdown("### Previous Chats")
#     chat_history = load_chat_history()
#     chat_options = [f"{i+1}. {chat['timestamp']}" for i, chat in enumerate(chat_history)]
    
#     # Allow selecting a previous chat
#     selected_chat = st.sidebar.selectbox("Select a previous chat to view", options=["Start a New Chat"] + chat_options, index=0)
#     if selected_chat == "Start a New Chat":
#         st.session_state.current_chat = []  # Clear current chat history
#         st.session_state.new_chat = True
#         st.session_state.selected_chat = None
#         st.session_state.chat_session = None
#     else:
#         selected_chat_index = int(selected_chat.split(".")[0]) - 1
#         st.session_state.selected_chat = chat_history[selected_chat_index]["messages"]
#         st.session_state.selected_chat_index = selected_chat_index
#         st.session_state.current_chat = st.session_state.selected_chat.copy()
#         st.session_state.new_chat = False

# def working_process(generation_config):
#     api_key = st.session_state.api_key

#     if not api_key:
#         st.warning("Please enter a valid API key in the sidebar to continue.")
#         return

#     try:
#         genai.configure(api_key=api_key)
#     except Exception as e:
#         st.error(f"Error configuring API: {e}")
#         return

#     system_instruction = "You are a helpful document answering assistant."
#     model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest", generation_config=generation_config, system_instruction=system_instruction)
#     vectorstore = st.session_state['vectorstore']

#     # Display the current chat history
#     if "current_chat" in st.session_state:
#         for user_query, ai_response in st.session_state.current_chat:
#             with st.chat_message("Human"):
#                 st.markdown(user_query)
#             if ai_response:  # Check if the response is not None
#                 with st.chat_message("AI"):
#                     st.markdown(ai_response)

#     # Ensure input box is available for new queries, below the chat messages
#     user_query = st.chat_input("Enter Your Query....")
#     if user_query and user_query.strip():
#         # Add user query to the chat history before processing it
#         st.session_state.current_chat.append((user_query, None))  # Append without a response yet

#         with st.chat_message("Human"):
#             st.markdown(user_query)  # Show user input in chat

#         try:
#             # Use the relevant context from the vectorstore
#             relevant_content = vectorstore.similarity_search(user_query, k=10)
#             result = get_response(relevant_content, user_query, model)

#             # Update the last entry with the AI response
#             st.session_state.current_chat[-1] = (user_query, result)  
            
#             # Display the AI response in a new block
#             with st.chat_message("AI"):
#                 st.markdown(result)  

#         except Exception as e:
#             st.warning(f"Error generating response: {e}")

#     # Save or update chat history depending on the chat type
#     if st.session_state.new_chat:
#         save_chat_history()
#     else:
#         update_chat_history()




#     # If revisiting a previous chat, ensure the input box is still available
#     if "selected_chat" in st.session_state and st.session_state.selected_chat:
#         st.session_state.new_chat = False  # Set to false when selecting a previous chat

# generation_config = {
#     "temperature": 0.2,
#     "top_p": 1,
#     "top_k": 1,
#     "max_output_tokens": 8000,
# }

# def main():
#     load_dotenv()
#     st.header("Chat with Multiple PDFs :books:")

#     if "vectorstore" not in st.session_state:
#         with st.spinner("Loading preprocessed data..."):
#             st.session_state.vectorstore = load_vectorstore()

#     if "current_chat" not in st.session_state:
#         st.session_state.current_chat = []

#     sidebar_ui()

#     if st.session_state.vectorstore is not None:
#         working_process(generation_config)

# if __name__ == "__main__":
#     main()





# import os
# import streamlit as st
# import google.generativeai as genai
# from dotenv import load_dotenv
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_community.vectorstores.faiss import FAISS
# import json
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from datetime import datetime

# # Constants for paths
# VECTORSTORE_INDEX_PATH = "faiss_index.index"
# METADATA_PATH = "metadata.json"
# CHAT_HISTORY_PATH = "chat_history.json"

# # Streamlit configuration
# st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

# # Load VectorStore
# def load_vectorstore():
#     vectorstore = FAISS.load_local(
#         VECTORSTORE_INDEX_PATH, 
#         embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
#         allow_dangerous_deserialization=True
#     )
#     with open(METADATA_PATH, "r") as file:
#         texts = json.load(file)
#     vectorstore.texts = texts
#     return vectorstore

# # Load and save chat history
# def load_chat_history():
#     if os.path.exists(CHAT_HISTORY_PATH):
#         with open(CHAT_HISTORY_PATH, "r") as f:
#             return json.load(f)
#     return []

# def save_chat_history():
#     chat_history = load_chat_history()
#     if "current_chat" in st.session_state and st.session_state.current_chat:
#         chat_history.append({
#             "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
#             "messages": st.session_state.current_chat
#         })
#         with open(CHAT_HISTORY_PATH, "w") as f:
#             json.dump(chat_history, f)

# def update_chat_history():
#     chat_history = load_chat_history()
#     if "selected_chat_index" in st.session_state and st.session_state.current_chat:
#         selected_chat_index = st.session_state.selected_chat_index
#         if 0 <= selected_chat_index < len(chat_history):
#             chat_history[selected_chat_index]["messages"] = st.session_state.current_chat
#             with open(CHAT_HISTORY_PATH, "w") as f:
#                 json.dump(chat_history, f)

# # Function to get AI response
# def get_response(context, question, model):
#     # Initialize chat session if not already active
#     if "chat_session" not in st.session_state or st.session_state.chat_session is None:
#         st.session_state.chat_session = model.start_chat(history=[])

#     # Initialize current chat if not already set
#     if "current_chat" not in st.session_state:
#         st.session_state.current_chat = []

#     # Collect previous conversation to maintain context
#     chat_history_context = ""
#     for past_query, past_response in st.session_state.current_chat:
#         chat_history_context += f"User asked: {past_query}\nAI responded: {past_response}\n\n"

#     # Compose prompt with structured response guidance and accumulated history context
#     prompt_template = f"""
#     You are a knowledgeable assistant who responds to user queries with detailed and context-aware answers. When users ask for elaboration, provide relevant, informative responses based on previous interactions without asking for more context. Use the following structure:

#     - Provide an **in-depth explanation** or **details**.
#     - Relate the answer to previous interactions if applicable.
#     - Break down key points in bullet points if necessary.

#     **Previous Conversations**:
#     {chat_history_context}

#     **Current Context**: {context}
#     **User's Question**: {question}

#     Respond with relevant details and examples based on the user's question.
#     """

#     try:
#         # Generate response
#         response = st.session_state.chat_session.send_message(prompt_template)

#         # Append new question and response to current chat history
#         st.session_state.current_chat.append((question, response.text))
#         return response.text

#     except AttributeError:
#         st.warning("Chat session could not be initialized. Please restart the app.")
#     except Exception as e:
#         st.warning(f"An error occurred: {e}")




# # Sidebar UI for chat history and settings
# def sidebar_ui():
#     st.sidebar.title("Settings")
#     api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
#     st.session_state.api_key = api_key
    
#     st.sidebar.markdown("### Previous Chats")
#     chat_history = load_chat_history()
#     chat_options = ["Start a New Chat"] + [f"{i+1}. {chat['timestamp']}" for i, chat in enumerate(chat_history)]
    
#     selected_chat = st.sidebar.selectbox("Select a previous chat", chat_options, index=0)
#     if selected_chat == "Start a New Chat":
#         st.session_state.current_chat = []  # Clear current chat history
#         st.session_state.selected_chat = None
#         st.session_state.chat_session = None
#     else:
#         selected_chat_index = int(selected_chat.split(".")[0]) - 1
#         st.session_state.selected_chat = chat_history[selected_chat_index]["messages"]
#         st.session_state.selected_chat_index = selected_chat_index
#         st.session_state.current_chat = st.session_state.selected_chat.copy()



# # Main working process
# # Main working process
# def working_process(generation_config):
#     # Ensure the API key is set
#     if not st.session_state.api_key:
#         st.warning("Please enter a valid API key in the sidebar.")
#         return
    
#     # Configure the GenAI model
#     try:
#         genai.configure(api_key=st.session_state.api_key)
#     except Exception as e:
#         st.error(f"Error configuring API: {e}")
#         return

#     # Initialize the model with instructions
#     model = genai.GenerativeModel(
#         model_name="gemini-1.5-pro-latest",
#         generation_config=generation_config,
#         system_instruction="You are a helpful document answering assistant."
#     )
    
#     vectorstore = st.session_state['vectorstore']

#     # Display each interaction in its own chat block
#     if "current_chat" in st.session_state:
#         # Clear previous output
#         for user_query, ai_response in st.session_state.current_chat:
#             with st.chat_message("Human"):
#                 st.markdown(user_query)
#             if ai_response:
#                 with st.chat_message("AI"):
#                     st.markdown(ai_response)

#     # Capture new user query
#     user_query = st.chat_input("Enter Your Query....")
#     if user_query:
#         # Append the new query to current chat history without a response yet
#         st.session_state.current_chat.append((user_query, None))

#         # Display the user's query in its own block immediately
#         with st.chat_message("Human"):
#             st.markdown(user_query)

#         try:
#             # Retrieve relevant context from the vectorstore and generate response
#             relevant_content = vectorstore.similarity_search(user_query, k=10)
#             result = get_response(relevant_content, user_query, model)

#             # Append the AI response to current chat history and display it
#             st.session_state.current_chat[-1] = (user_query, result)  # Update with the AI's response
#             with st.chat_message("AI"):
#                 st.markdown(result)

#         except Exception as e:
#             st.warning(f"Error generating response: {e}")

#     # Save or update chat history based on whether it's a new or existing chat session
#     if st.session_state.new_chat:
#         save_chat_history()
#         st.session_state.new_chat = False  # Reset to ensure new chat only triggers once
#     else:
#         update_chat_history()

import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores.faiss import FAISS
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from datetime import datetime

# Constants for paths
VECTORSTORE_INDEX_PATH = "faiss_index.index"
METADATA_PATH = "metadata.json"
CHAT_HISTORY_PATH = "chat_history.json"

# Streamlit configuration
st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")

# Load VectorStore
def load_vectorstore():
    vectorstore = FAISS.load_local(
        VECTORSTORE_INDEX_PATH, 
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"), 
        allow_dangerous_deserialization=True
    )
    with open(METADATA_PATH, "r") as file:
        texts = json.load(file)
    vectorstore.texts = texts
    return vectorstore

# Load and save chat history
def load_chat_history():
    if os.path.exists(CHAT_HISTORY_PATH):
        with open(CHAT_HISTORY_PATH, "r") as f:
            return json.load(f)
    return []

def save_chat_history():
    chat_history = load_chat_history()
    if "current_chat" in st.session_state and st.session_state.current_chat:
        chat_history.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": st.session_state.current_chat
        })
        with open(CHAT_HISTORY_PATH, "w") as f:
            json.dump(chat_history, f)

def update_chat_history():
    chat_history = load_chat_history()
    if "selected_chat_index" in st.session_state and st.session_state.current_chat:
        selected_chat_index = st.session_state.selected_chat_index
        if 0 <= selected_chat_index < len(chat_history):
            chat_history[selected_chat_index]["messages"] = st.session_state.current_chat
            with open(CHAT_HISTORY_PATH, "w") as f:
                json.dump(chat_history, f)

# Function to get AI response
def get_response(context, question, model):
    # Initialize chat session if not already active
    if "chat_session" not in st.session_state or st.session_state.chat_session is None:
        st.session_state.chat_session = model.start_chat(history=[])

    # Initialize current chat if not already set
    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    # Collect previous conversation to maintain context
    chat_history_context = ""
    for past_query, past_response in st.session_state.current_chat:
        chat_history_context += f"User asked: {past_query}\nAI responded: {past_response}\n\n"

    # Compose prompt with structured response guidance and accumulated history context
    prompt_template = f"""
    You are a knowledgeable assistant who responds to user queries with detailed and context-aware answers. When users ask for elaboration, provide relevant, informative responses based on previous interactions without asking for more context. Use the following structure:

    - Provide an **in-depth explanation** or **details**.
    - Relate the answer to previous interactions if applicable.
    - Break down key points in bullet points if necessary.
    - Every new chat come in new chat area. 
    - History should be maintaine for forever in the chats.

    **Previous Conversations**:
    {chat_history_context}

    **Current Context**: {context}
    **User's Question**: {question}

    Respond with relevant details and examples based on the user's question.
    """

    try:
        # Generate response
        response = st.session_state.chat_session.send_message(prompt_template)

        # Append new question and response to current chat history
        st.session_state.current_chat.append((question, response.text))
        return response.text

    except AttributeError:
        st.warning("Chat session could not be initialized. Please restart the app.")
    except Exception as e:
        st.warning(f"An error occurred: {e}")

# Sidebar UI for chat history and settings
# Sidebar UI for chat history and settings
def sidebar_ui():
    st.sidebar.title("Settings")
    api_key = st.sidebar.text_input("Enter your Google API Key", type="password")
    st.session_state.api_key = api_key
    
    st.sidebar.markdown("### Previous Chats")
    chat_history = load_chat_history()

    # Display each previous chat with the first question as its title
    chat_options = ["Start a New Chat"] + [
        f"{i+1}. {chat['messages'][0][0]}" if chat["messages"] else f"{i+1}. (No question)"
        for i, chat in enumerate(chat_history)
    ]
    
    selected_chat = st.sidebar.selectbox("Select a previous chat", chat_options, index=0)
    if selected_chat == "Start a New Chat":
        st.session_state.current_chat = []  # Clear current chat history
        st.session_state.selected_chat = None
        st.session_state.chat_session = None
    else:
        selected_chat_index = int(selected_chat.split(".")[0]) - 1
        st.session_state.selected_chat = chat_history[selected_chat_index]["messages"]
        st.session_state.selected_chat_index = selected_chat_index
        st.session_state.current_chat = st.session_state.selected_chat.copy()


# Main working process
def working_process(generation_config):
    # Ensure the API key is set
    if not st.session_state.api_key:
        st.warning("Please enter a valid API key in the sidebar.")
        return
    
    # Configure the GenAI model
    try:
        genai.configure(api_key=st.session_state.api_key)
    except Exception as e:
        st.error(f"Error configuring API: {e}")
        return

    # Initialize the model with instructions
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        system_instruction="You are a helpful document answering assistant."
    )
    
    vectorstore = st.session_state['vectorstore']

    # Display each interaction in its own chat block
    if "current_chat" in st.session_state:
        for user_query, ai_response in st.session_state.current_chat:
            with st.chat_message("Human"):
                st.markdown(user_query)
            if ai_response:
                with st.chat_message("AI"):
                    st.markdown(ai_response)

    # Capture new user query
    user_query = st.chat_input("Enter Your Query....")
    if user_query:
        # Display the user's query in its own block immediately
        with st.chat_message("Human"):
            st.markdown(user_query)

        # Append the new query to current chat history without a response yet
        st.session_state.current_chat.append((user_query, None))

        try:
            # Retrieve relevant context from the vectorstore and generate response
            relevant_content = vectorstore.similarity_search(user_query, k=10)
            result = get_response(relevant_content, user_query, model)

            # Update the current chat with the AI response
            st.session_state.current_chat[-1] = (user_query, result)  # Update with the AI's response
            with st.chat_message("AI"):
                st.markdown(result)

        except Exception as e:
            st.warning(f"Error generating response: {e}")

    # Save or update chat history based on whether it's a new or existing chat session
    if st.session_state.new_chat:
        save_chat_history()
        st.session_state.new_chat = False  # Reset to ensure new chat only triggers once
    else:
        update_chat_history()

# Main Function
# Main Function
def main():
    load_dotenv()
    st.header("Chat with Multiple PDFs :books:")

    # Initialize session state variables if they do not exist
    if "vectorstore" not in st.session_state:
        with st.spinner("Loading preprocessed data..."):
            st.session_state.vectorstore = load_vectorstore()

    if "current_chat" not in st.session_state:
        st.session_state.current_chat = []

    if "new_chat" not in st.session_state:
        st.session_state.new_chat = True  # Initialize as True to handle the first chat session correctly

    sidebar_ui()
    if st.session_state.vectorstore:
        generation_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 8000,
        }
        working_process(generation_config)


if __name__ == "__main__":
    main()










