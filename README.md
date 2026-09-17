# 🎓 MotiveAI

### AI-Powered Academic Motivation Letter Assistant

**MotiveAI** is a RAG-powered academic writing assistant designed to help students transform their real academic background, scientific interests, achievements, goals, and study motivations into **personalized, professional, and academically structured motivation letters**.

Instead of generating a generic AI-written letter, MotiveAI combines **Retrieval-Augmented Generation (RAG)** with **GPT-OSS-120B through Groq** to provide relevant academic writing guidance and generate a letter based on the applicant's actual profile.

Users can also provide feedback on the generated letter and iteratively rebuild it until the content, tone, structure, and emphasis match their requirements.

---

## ✨ Features

### 🧠 RAG-Powered Generation

MotiveAI uses a lightweight Retrieval-Augmented Generation pipeline:

```text
Applicant Profile
       ↓
Query Construction
       ↓
TF-IDF Retrieval
       ↓
Cosine Similarity
       ↓
Relevant Academic Guidance
       ↓
GPT-OSS-120B
       ↓
Personalized Motivation Letter
```

Relevant academic-writing knowledge is retrieved before the LLM generates the letter.

---

### ✍️ Personalized Motivation Letters

The application considers important aspects of an applicant's academic journey, including:

* Scientific interests
* Academic background
* Personal goals
* Academic achievements
* Projects and research
* Certifications
* Internships
* Reason for choosing the subject
* Reason for choosing Russia
* Future academic and career goals

The generated letter is designed around the applicant's **actual information rather than generic templates**.

---

### 🔄 Iterative Letter Refinement

MotiveAI allows users to improve their letter through multiple revision cycles.

```text
Version 1
   ↓
User Feedback
   ↓
Version 2
   ↓
User Feedback
   ↓
Version 3
```

Users can request changes such as:

* Make it more academic
* Make it more personal
* Strengthen the introduction
* Improve the conclusion
* Emphasize research interests
* Improve the Russia section
* Reduce repetition
* Make it more concise
* Make it more natural
* Make it more formal

---

### 🔍 Quality Review

After generating a letter, the system evaluates important writing dimensions such as:

* Academic relevance
* Personalization
* Coherence
* Clarity
* Structure
* Repetition
* Generic language
* Unsupported claims
* Alignment between academic background and selected subject
* Alignment between goals and degree

The purpose is to help users identify areas that may need improvement before submitting their application.

---

### 🛡️ Authenticity & Anti-Hallucination

MotiveAI is designed to preserve the applicant's real story.

The system must not fabricate:

* Grades or GPA
* Awards
* Publications
* Research experience
* Internships
* Jobs
* Certifications
* Universities
* Professors
* Laboratories
* Scholarships
* Achievements

If information is unavailable, the system should omit it instead of inventing it.

> **Your story remains yours. MotiveAI helps structure and communicate it professionally.**

---

## 🚀 Technology Stack

| Technology                  | Purpose                      |
| --------------------------- | ---------------------------- |
| **Python**                  | Application development      |
| **Streamlit**               | Web interface                |
| **Groq API**                | Fast LLM inference           |
| **GPT-OSS-120B**            | Motivation-letter generation |
| **Scikit-learn**            | TF-IDF retrieval             |
| **Cosine Similarity**       | Semantic relevance matching  |
| **Streamlit Session State** | Revision/session management  |

---

## 🏗️ Architecture

```text
┌─────────────────────────────┐
│      Applicant Profile      │
│                             │
│ • Academic Background       │
│ • Scientific Interests      │
│ • Goals                     │
│ • Achievements              │
│ • Subject Motivation        │
│ • Russia Motivation         │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│     Retrieval Pipeline      │
│                             │
│ TF-IDF Vectorization        │
│          +                  │
│ Cosine Similarity           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Relevant Academic Guidance  │
│                             │
│ • Writing Structure         │
│ • Academic Tone             │
│ • Personalization           │
│ • Scientific Interests      │
│ • Goal Alignment            │
│ • Common Mistakes           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│        GPT-OSS-120B         │
│          via Groq           │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Personalized Motivation   │
│          Letter             │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       User Feedback         │
└──────────────┬──────────────┘
               │
               ▼
        Refine / Rebuild
```

---

## 🎯 Core Application Workflow

### 1. Build Your Profile

The user provides information about their:

* Education
* Scientific interests
* Personal goals
* Achievements
* Projects
* Selected subject
* Reason for choosing Russia
* Future plans

### 2. Generate

MotiveAI retrieves relevant academic-writing guidance and uses GPT-OSS-120B to create the initial motivation letter.

### 3. Review

The user reads the generated letter and reviews the quality indicators.

### 4. Refine

The user provides natural-language feedback.

For example:

> "Make my research interests more prominent and make the introduction stronger."

### 5. Rebuild

MotiveAI combines:

```text
Original Profile
+
Retrieved Guidance
+
Current Letter
+
User Feedback
```

to generate an improved version.

### 6. Finalize

The user can continue refining the letter until they are satisfied with the final version.

---

## 📋 Applicant Information

MotiveAI can work with information such as:

### Academic Background

* Current degree
* University
* Field of study
* Graduation year

### Scientific Interests

Examples:

* Cybersecurity
* Artificial Intelligence
* Machine Learning
* Data Science
* Computer Science
* Computer Vision
* Networks
* Cloud Security
* Digital Forensics

### Achievements

* Academic achievements
* Projects
* Final Year Project
* Research
* Publications
* Certifications
* Internships
* Competitions
* Leadership
* Technical skills

### Motivation

* Why this subject?
* Why this degree?
* Why Russia?
* What are your future goals?

---

## 🔐 Security

API credentials are never hardcoded into the application.

The application supports:

```python
st.secrets["GROQ_API_KEY"]
```

for Streamlit Cloud and environment variables for local development.

### Local Environment Variable

```bash
GROQ_API_KEY=your_api_key
```

### Streamlit Secrets

Add the following to your Streamlit application's Secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

> Never commit your API key to GitHub.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/motiveai.git
```

### 2. Navigate to the Project

```bash
cd motiveai
```

### 3. Create a Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure API Key

Set your Groq API key as an environment variable:

```bash
GROQ_API_KEY=your_api_key
```

Or configure it through Streamlit Secrets.

### 6. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Project Structure

The application is intentionally lightweight.

```text
motiveai/
│
├── app.py
└── requirements.txt
```

All application logic, UI, RAG implementation, prompts, and session management are contained within `app.py`.

---

## 🌐 Deployment on Streamlit Community Cloud

MotiveAI can be deployed directly on Streamlit Community Cloud.

### Steps

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new application.
4. Select the GitHub repository.
5. Set the main file to:

```text
app.py
```

6. Add your API key under **Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

7. Deploy.

No separate backend or database is required.

---

## 🧪 Example Refinement Requests

Users can provide natural-language instructions such as:

```text
Make the introduction more academically strong.
```

```text
Focus more on my scientific interests.
```

```text
Make the letter sound natural and less AI-generated.
```

```text
Improve the connection between my previous degree and my selected subject.
```

```text
Make my future career goals clearer.
```

```text
Strengthen the reason for choosing Russia without adding unsupported facts.
```

```text
Reduce repetition and make the letter more concise.
```

---

## 🎓 Intended Use Cases

MotiveAI can be used for:

* Master's applications
* Scholarship applications
* University admissions
* International study applications
* Research program applications
* Academic motivation letters
* Statement refinement

The generated content should always be reviewed by the applicant and adapted to the specific requirements of the target institution.

---

## ⚠️ Important Disclaimer

MotiveAI is an **AI-assisted academic writing tool**.

It does not guarantee:

* Admission
* Scholarship selection
* Professor approval
* Acceptance into a university

The application is designed to help applicants communicate their **real academic background, goals, and motivations** in a clearer and more professional way.

Applicants should verify institution-specific requirements and factual claims before submitting any application.

---

## 🔮 Future Improvements

Potential future versions could include:

* 📄 PDF/DOCX export
* 📚 External university knowledge bases
* 🎓 University-specific RAG
* 🔬 Professor research-profile matching
* 🌍 Country-specific application guidance
* 📑 Scholarship requirement analysis
* 🧠 Semantic embeddings and vector databases
* 📊 Advanced writing analytics
* 📝 Multiple motivation-letter templates
* 🔎 University/program-specific personalization
* 🌐 Multi-language support
* 👤 Applicant profile management

A future RAG architecture could evolve into:

```text
Applicant Profile
       +
University Information
       +
Program Information
       +
Scholarship Requirements
       +
Professor Research
       ↓
Advanced RAG
       ↓
GPT-OSS-120B
       ↓
Highly Contextualized Application Letter
```

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

If you would like to contribute:

```bash
git fork
git clone
git checkout -b feature/your-feature
```

Make your changes and submit a pull request.

---

## 📄 License

This project can be released under the **MIT License**.

---

## 👨‍💻 Author

**Kumail Abbas**

Computer Science Student
University of Baltistan, Skardu

Interested in:

* Cybersecurity
* Defensive Security
* Artificial Intelligence
* RAG Applications
* AI-powered Security Solutions
* Full-Stack Development

---

## ⭐ Support

If you find this project useful, consider giving the repository a ⭐ on GitHub.

---

### Built with Python, Streamlit, Groq & GPT-OSS-120B

> **Your academic journey. Your achievements. Your goals.
> MotiveAI helps turn them into a professional story.**
