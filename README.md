# RAG-based-Search-app
This project provides a powerful, conversational AI chatbot that runs on your local machine. It reads documents (PDFs, DOCX, TXT files) from a local folder, builds a searchable knowledge base using AI embeddings, and answers your questions based on the content of those documents.

The chatbot uses a RAG architecture with LangChain, ensuring that its answers are grounded in the information you provide.

Features
Conversational Memory: Remembers the last few turns of the conversation.

Local & Private: All documents and knowledge base remain on your local machine.

Streaming Responses: AI answers appear token-by-token.

Source Citation: Lists the documents used for its answers.

Persistent Knowledge Base: Saves the knowledge base locally for fast startups.

Multi-File Support: Handles PDFs, Microsoft Word (.docx), and plain text (.txt) files.

Robust Error Handling: Skips individual files that fail to load without crashing.

Flexible AI Models: Configurable to use local models (Ollama) or cloud APIs (Groq).

Project Structure
/ai_chatbot_project
|
|-- enhanced_chatbot.py     # The main Python script for the chatbot
|-- requirements.txt        # A list of all required Python packages
|-- readme.txt              # This file
|
|-- /documents/             # <-- Place all your source documents here
|   |-- report.pdf
|   |-- notes.docx
|   |-- ...
|
|-- /vector_store/          # <-- The local knowledge base will be saved here automatically

Setup and Installation Guide
Follow these steps carefully to set up and run the chatbot on your Windows machine.

--- Prerequisites ---
Before you begin, ensure you have the following installed:

Anaconda: A Python environment manager.
Download: https://www.anaconda.com/download

API Key (if using a cloud LLM): If you plan to use Groq (the default), get a free API key.
GroqCloud Console: https://console.groq.com/keys

**--- Step 1: Install System-Level Dependencies ---**
The chatbot needs two external programs to read complex PDFs. These must be installed on Windows itself.

A) Install Tesseract (for OCR)
The Tesseract OCR Engine allows the chatbot to read text from scanned documents or images inside PDFs.

Download: Go to the official installer page: https://github.com/UB-Mannheim/tesseract/wiki
Download the latest .exe installer.

Run Installer: Run the installer and follow the setup wizard.

Add to PATH: During installation, ensure the option to "Add Tesseract to system PATH" is CHECKED.

B) Install Poppler (for PDFs)
The Poppler utility helps the chatbot handle PDF files.

Download: Go to the latest Poppler releases page: https://github.com/oschwartz10612/poppler-windows/releases/
Download the .7z file (e.g., poppler-24.02.0-win.7z).

Extract: Use 7-Zip to extract the file. We recommend extracting it to C:, which will create a folder like C:\poppler-24.02.0.

Add to PATH Manually:

Copy the full path to the bin folder inside your new Poppler directory (e.g., C:\poppler-24.02.0\bin).


IMPORTANT: After installing Tesseract and Poppler, you MUST close and re-open your Command Prompt or Anaconda Prompt for the changes to take effect.
You can verify the setup by running tesseract --version and pdfinfo -v in the new terminal. Both should show version information.

**--- Step 2: Set Up the Python Environment ---**

Open a new Anaconda Prompt.

Navigate to your project folder (e.g., cd C:\path\to\your\ai_chatbot_project).

Create a new, clean Conda environment:
conda create -n chatbot_env python=3.10 -y

Activate the new environment:
conda activate chatbot_env

Install all required Python packages from the requirements.txt file (ensure this file is in your project folder):
pip install -r requirements.txt

**--- Step 3: Configure Your API Key (for Groq) ---**
The chatbot needs an API key to connect to the Groq AI service. The most secure method is to set a permanent environment variable.

Press the Windows Key, type env, and click "Edit the system environment variables".

Click the "Environment Variables..." button.

In the top "User variables" section, click "New...".

Enter the following:

Variable name: GROQ_API_KEY

Variable value: Paste your actual secret key from the Groq console.

Click "OK" on all windows to save.

**--- Step 4: Run the Chatbot! ---**

Close and re-open your Anaconda Prompt one last time to ensure it loads your new API key variable.

In the new terminal, activate the environment and navigate to your project folder:
conda activate chatbot_env
cd C:\path\to\your\ai_chatbot_project

Run the script:
python enhanced_chatbot.py

The first time you run it, it will process your documents and create the vector_store directory. Subsequent runs will be much faster. When you see the You: prompt, the chatbot is ready!

How to Use
Simply type your question and press Enter.

Type reset to clear the current conversation history and start fresh.

Type exit to quit the program.

Customization
You can easily switch the LLM provider by editing the ChatbotConfig class at the top of the enhanced_chatbot.py file. Change the LLM_PROVIDER variable to "Ollama" or "OpenAI" and ensure the corresponding Python packages from requirements.txt are installed. For Ollama, ensure the Ollama application is running and the desired model is pulled. For OpenAI, set the OPENAI_API_KEY environment variable.
