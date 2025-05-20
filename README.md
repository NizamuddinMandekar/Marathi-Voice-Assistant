# Marathi Voice Assistant

An interactive Marathi Voice Assistant that provides answers to questions based on the contents of a Marathi PDF document. The system extracts text from the PDF, processes it, and uses machine learning to provide the most relevant answers. The responses are then converted into speech using Google Text-to-Speech (gTTS).

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
- [Dependencies](#dependencies)
- [Usage Guide](#usage-guide)
- [System Architecture](#system-architecture)
- [Limitations](#limitations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)

---

## Prerequisites

To run this project, you'll need:

- Python 3.x
- A system with internet access (to install dependencies and download resources)

Ensure you have the following tools:
- **pip** (Python's package installer)
- A Jupyter notebook environment or a Python script executor to interact with the system.

---

## Setup Instructions

Follow these steps to set up the Marathi Voice Assistant on your local machine:

### 1. Clone the Repository

Clone the repository to your local system:

```bash
git clone https://github.com/your-repository-url/marathi-voice-assistant.git
cd marathi-voice-assistant
```
2. Install Required Dependencies

You can install the necessary dependencies by running:
```bash
pip install -r requirements.txt
```
Alternatively, install them individually:
```bash
pip install pymupdf scikit-learn gtts indic-nlp-library sentence-transformers
```
3. Set up Indic NLP Resources

The Indic NLP library requires external resources. To download them, run:
```bash
import os
from indicnlp import loader

INDIC_RESOURCES_PATH = "/path/to/indic_nlp_resources"
if not os.path.exists(INDIC_RESOURCES_PATH):
    !git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git {INDIC_RESOURCES_PATH}
os.environ['INDIC_RESOURCES_PATH'] = INDIC_RESOURCES_PATH
loader.load()
```
#Dependencies

The system relies on the following Python libraries:
PyMuPDF (fitz): For extracting text from PDF files.
Scikit-learn: For machine learning utilities (though it isn’t actively used in this implementation).
gTTS: Google Text-to-Speech for converting text to Marathi speech.
Indic NLP: Tokenizes and processes Marathi text.
Sentence Transformers: Converts sentences into embeddings for semantic search.

#Usage Guide
#1. Prepare Your PDF
Upload a Marathi-language PDF containing the text you want the assistant to answer questions about.

#2. Run the Assistant
Launch the interactive assistat by running the script in a Jupyter notebook or Python environment. The assistant will:
    Extract text from the PDF.
    Preprocess and split the text into sentences.
    Use a sentence transformer to encode the text into embeddings.
    Allow the user to ask questions related to the PDF's content.

Example usage:

# Load and process the PDF (upload first!)
```bash
pdf_path = "/content/Extra-Ordinary_CENTRAL-SECTION_Part-4-Marathi-.pdf"  # Change if needed
print("Extracting text...")
raw_text = extract_text_from_pdf(pdf_path)
clean_text = preprocess_text(raw_text)
sentences = split_sentences(clean_text)
print(f"Total sentences: {len(sentences)}")

# Load sentence transformer
print("Encoding sentences...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

# Start the assistant
query = input("\nतुमचा प्रश्न: ")
response = answer_query(query)
print("उत्तर:", response)
```
#3. Interaction
    The assistant processes the user's query and returns the most relevant sentence from the PDF.
    The response is converted into Marathi speech using gTTS.
    An audio file will be generated, which you can download or listen to directly in the notebook.

#4. Accuracy Feedback (Optional)
    After the assistant gives an answer, you can manually verify the correctness of the response.
    The system will keep track of the accuracy of its responses, providing feedback in real-time.

System Architecture
Components:
    PDF Text Extraction:
        Extracts text from the provided PDF file using PyMuPDF (fitz).
    Text Preprocessing:
        Cleans and normalizes the extracted text (removes non-Marathi characters, extra spaces, etc.).
    Sentence Tokenization:
        Splits the text into sentences using the Indic NLP library.
    Sentence Embeddings:
        Each sentence is encoded into a vector (embedding) using a pre-trained Sentence Transformer model (paraphrase-multilingual-MiniLM-L12-v2).
    Query Handling & Cosine Similarity:
        The assistant compares the user’s query with the sentence embeddings using cosine similarity to find the most relevant sentence.
    Text-to-Speech (TTS):
        Converts the relevant answer into speech using gTTS (Google Text-to-Speech).
        Provides a downloadable MP3 file or attempts to play the audio directly in the notebook.
    Interactive Query System:
        Users can interact with the system by typing questions and receiving answers in Marathi.
        
  #Limitations
    Accuracy: The assistant relies on sentence embeddings and cosine similarity, which may not always yield perfect results, especially for complex or ambiguous queries.
    PDF Quality: If the PDF is poorly formatted or contains complex layouts (images, tables), the text extraction may not work perfectly.
    Audio Playback: Some notebook environments may not support direct audio playback. However, a download link will be provided for MP3 files.

#Future Enhancements
    Multi-language Support: Extend support to other languages (e.g., Hindi, Gujarati) by adjusting the language resources and models.
    Contextual Understanding: Improve the assistant’s ability to handle more complex queries by implementing NLP techniques like named entity recognition (NER) or question-answering models.
    Real-time Updates: Allow dynamic updates to sentence embeddings if the content of the PDF changes.
