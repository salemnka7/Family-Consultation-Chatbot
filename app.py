import streamlit as st
import os
import hashlib
import pickle
import PyPDF2
import docx
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import arabic_reshaper
from bidi.algorithm import get_display
import time
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Arabic Documents QA Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Fixed OpenAI API key - replace with your actual API key
OPENAI_API_KEY = "x"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Create necessary directories
Path("data").mkdir(exist_ok=True)
Path("data/files").mkdir(exist_ok=True)
Path("data/users").mkdir(exist_ok=True)
Path("data/vectorstore").mkdir(exist_ok=True)

# Constants
VECTORSTORE_PATH = "data/vectorstore/faiss_index"
USERS_DB_PATH = "data/users/users.pkl"
ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "admin123"  # You should change this in production

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "openai_client" not in st.session_state:
    # Initialize OpenAI client with the fixed API key
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Helper functions for authentication
def load_users():
    if os.path.exists(USERS_DB_PATH):
        with open(USERS_DB_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # Create default admin user
        admin_password_hash = hashlib.sha256(DEFAULT_ADMIN_PASSWORD.encode()).hexdigest()
        users = {ADMIN_USERNAME: {"password": admin_password_hash, "role": "admin"}}
        save_users(users)
        return users

def save_users(users):
    with open(USERS_DB_PATH, "wb") as f:
        pickle.dump(users, f)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    users = load_users()
    if username in users and users[username]["password"] == hash_password(password):
        return users[username]["role"]
    return None

def add_user(username, password, role):
    users = load_users()
    if username in users:
        return False
    
    users[username] = {
        "password": hash_password(password),
        "role": role
    }
    save_users(users)
    return True

# Helper functions for document processing
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def process_documents(uploaded_files):
    all_text = ""
    saved_files = []
    
    for uploaded_file in uploaded_files:
        file_path = f"data/files/{uploaded_file.name}"
        
        # Save the uploaded file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        saved_files.append(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Extract text based on file type
        if file_extension == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif file_extension == ".docx":
            text = extract_text_from_docx(file_path)
        elif file_extension == ".txt":
            text = extract_text_from_txt(file_path)
        else:
            text = ""
            st.warning(f"Unsupported file type: {file_extension}")
        
        all_text += text + "\n\n"
    
    return all_text, saved_files

def normalize_arabic_text(text):
    # Basic normalization for Arabic text
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    return text

def create_and_save_vector_db(text):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings()
    
    # Create vector database
    db = FAISS.from_texts(chunks, embeddings)
    
    # Save the vector store
    db.save_local(VECTORSTORE_PATH)
    
    return db

def load_vector_db():
    if os.path.exists(f"{VECTORSTORE_PATH}/index.faiss"):
        embeddings = OpenAIEmbeddings()
        return FAISS.load_local(VECTORSTORE_PATH, embeddings)
    return None

def get_answer_from_openai(query, context, openai_client):
    try:
        # Format the prompt with the retrieved context
        messages = [
            {"role": "system", "content": 
             "أنت مساعد ذكي متخصص في الإجابة على الأسئلة باللغة العربية. استخدم المعلومات التالية فقط للإجابة على السؤال. "
             "إذا كانت المعلومات غير كافية، فقل إنك لا تعرف الإجابة. يجب أن تكون إجاباتك دقيقة ومفيدة وباللغة العربية."
             "لا تقدم معلومات من خارج النص المعطى لك."
             f"\n\nمعلومات السياق:\n{context}\n\n"},
            {"role": "user", "content": query}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # or use a different model as needed
            messages=messages,
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"حدث خطأ: {str(e)}"

# Login and authentication UI
def login_ui():
    st.title("تسجيل الدخول")
    
    login_tab, register_tab = st.tabs(["تسجيل الدخول", "إنشاء حساب مستخدم جديد"])
    
    with login_tab:
        username = st.text_input("اسم المستخدم", key="login_username")
        password = st.text_input("كلمة المرور", type="password", key="login_password")
        
        if st.button("تسجيل الدخول"):
            role = authenticate(username, password)
            if role:
                st.session_state.logged_in = True
                st.session_state.user_role = role
                st.session_state.username = username
                st.success(f"تم تسجيل الدخول بنجاح كـ {role}")
                st.rerun()
            else:
                st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
    
    with register_tab:
        if st.session_state.user_role == "admin":
            new_username = st.text_input("اسم المستخدم الجديد")
            new_password = st.text_input("كلمة المرور الجديدة", type="password")
            role = st.selectbox("الدور", ["user", "admin"])
            
            if st.button("إنشاء المستخدم"):
                if add_user(new_username, new_password, role):
                    st.success(f"تم إنشاء المستخدم {new_username} بنجاح")
                else:
                    st.error("اسم المستخدم موجود بالفعل")
        else:
            st.info("فقط المسؤولون يمكنهم إنشاء مستخدمين جدد")

# Admin interface
def admin_interface():
    st.title(f"لوحة المسؤول - مرحباً {st.session_state.username}")
    
    tabs = st.tabs(["رفع المستندات", "إدارة المستخدمين", "اختبار النظام"])
    
    with tabs[0]:
        st.header("رفع المستندات")
        uploaded_files = st.file_uploader("اختر ملفات بتنسيق PDF أو DOCX أو TXT", 
                                        type=["pdf", "docx", "txt"], 
                                        accept_multiple_files=True)
        
        if uploaded_files and st.button("معالجة المستندات"):
            with st.spinner("جاري معالجة المستندات..."):
                # Process documents
                all_text, saved_files = process_documents(uploaded_files)
                normalized_text = normalize_arabic_text(all_text)
                
                # Create and save vector database
                try:
                    db = create_and_save_vector_db(normalized_text)
                    st.success(f"تمت معالجة {len(uploaded_files)} مستندات بنجاح وإنشاء قاعدة البيانات المتجهة!")
                    st.write(f"تم حفظ الملفات في: {', '.join(saved_files)}")
                except Exception as e:
                    st.error(f"حدث خطأ أثناء معالجة المستندات: {str(e)}")
    
    with tabs[1]:
        st.header("إدارة المستخدمين")
        users = load_users()
        
        # Display current users
        user_data = []
        for username, user_info in users.items():
            user_data.append({"اسم المستخدم": username, "الدور": user_info["role"]})
        
        st.write("المستخدمون الحاليون:")
        st.table(user_data)
        
        # Add new user
        st.subheader("إضافة مستخدم جديد")
        new_username = st.text_input("اسم المستخدم الجديد", key="admin_new_username")
        new_password = st.text_input("كلمة المرور الجديدة", type="password", key="admin_new_password")
        role = st.selectbox("الدور", ["user", "admin"])
        
        if st.button("إضافة مستخدم"):
            if add_user(new_username, new_password, role):
                st.success(f"تم إنشاء المستخدم {new_username} بنجاح")
                st.rerun()
            else:
                st.error("اسم المستخدم موجود بالفعل")
    
    with tabs[2]:
        st.header("اختبار النظام")
        db = load_vector_db()
        if db:
            test_query = st.text_input("اكتب سؤالًا للاختبار:")
            if test_query and st.button("اختبار"):
                with st.spinner("جاري البحث..."):
                    docs = db.similarity_search(test_query, k=4)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    
                    st.subheader("السياق المسترجع:")
                    st.text_area("النص:", value=context, height=200)
                    
                    response = get_answer_from_openai(test_query, context, st.session_state.openai_client)
                    
                    st.subheader("إجابة النظام:")
                    st.write(response)
        else:
            st.warning("لم يتم العثور على قاعدة بيانات المتجهات. الرجاء رفع ومعالجة المستندات أولاً.")

# User interface
def user_interface():
    st.title(f"نظام الأسئلة والأجوبة - مرحباً {st.session_state.username}")
    
    # Check if vector store exists
    db = load_vector_db()
    if db is None:
        st.warning("لم يتم تحميل أي مستندات بعد. يرجى الاتصال بالمسؤول.")
        return
    
    # Display chat interface
    st.subheader("المحادثة")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("اسأل سؤالًا عن المستندات..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("جاري التفكير..."):
                # Search the vector store for relevant context
                docs = db.similarity_search(query, k=4)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # Get response from OpenAI
                response = get_answer_from_openai(query, context, st.session_state.openai_client)
                
                st.write(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main application
def main():
    # Add logout button in sidebar
    if st.session_state.logged_in:
        if st.sidebar.button("تسجيل الخروج"):
            st.session_state.logged_in = False
            st.session_state.user_role = None
            st.session_state.username = None
            st.session_state.messages = []
            st.rerun()
    
    # Display appropriate interface based on login status and role
    if not st.session_state.logged_in:
        login_ui()
    else:
        if st.session_state.user_role == "admin":
            admin_interface()
        else:
            user_interface()

if __name__ == "__main__":
    main()
