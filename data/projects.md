My Projects

Project 1: MindbridgeRAG_CareerAnxiety | Python, Gradio, RAG, Gemini API

What it is: MindbridgeRAG_CareerAnxiety is an AI-powered career guidance and support system. It helps students understand career-related concerns and anxiety by giving relevant, context-based answers using Retrieval-Augmented Generation (RAG). A student can type a worry about their career, and the system answers using information it has retrieved, instead of answering from guesswork.

Why I built it: I wanted to explore how AI and cloud-based technologies can give accessible, personalized support to students who feel uncertain or confused about their career. Many students do not know who to ask, so I wanted a tool that is always available.

How I built it: I made an interactive web interface with Gradio and added a RAG pipeline that retrieves relevant information before the AI writes a response. I used the Gemini API to generate the answers. I also added a fallback mechanism so that the system keeps working when the AI service has a problem. This means the user still gets a useful reply even if the Gemini API fails.

The problem I faced: The hardest part was connecting the retrieval step with the response generation step while managing API dependencies. I had to make sure the system could continue to function through the fallback mechanism.

What I learned: I gained practical experience with RAG, LLM integration, API handling, Gradio interface development, prompt engineering and building AI-powered applications. I also learned that retrieval-based systems make AI answers more relevant and more accurate for the context.

---

Project 2: Campus Marketplace | React Native, Expo Router, TypeScript, Supabase

What it is: Campus Marketplace is a cross-platform mobile app that lets students buy and sell items to each other securely. It has authentication, product listings and real-time communication. It works on both Android and iOS because it is built with React Native.

Why I built it: I wanted a secure and practical marketplace made specifically for students, so they can trade items easily with other students in the same community.

How I built it: I used React Native, Expo Router, TypeScript and Supabase. Supabase handles the database, authentication and real-time features. I also developed a full Super Admin portal with live Supabase integration. It covers user management, listings, orders and analytics, with CSV export, so an admin can manage the whole platform and download data.

The problem I faced: I diagnosed and fixed a Row-Level Security (RLS) recursion bug in production. Row-Level Security decides which rows a user is allowed to read, and a recursion problem happens when a security policy ends up checking itself again and again. I solved it with a SECURITY DEFINER function pattern, which restored stable data access across the app.

What I learned: I learned advanced Supabase security rules, how to manage cross-platform UI states, and how to keep a centralized theming system so the whole app looks consistent. My database course helped me a lot with this project.

---

Project 3: SentinelGuard (Final Year Project) | AI-Based Intrusion Detection, Zero Trust Architecture

What it is: SentinelGuard is an AI-enhanced Zero Trust security framework that protects IoT devices. Zero Trust means no device is trusted automatically; every device must be checked. The framework combines a Trust Evaluation stage and a Policy Decision stage. It also uses RAG to explain detected threats in natural language, so the explanation is easy to read for people who are not security experts.

Why I built it: I wanted to design an advanced security framework that keeps IoT device operations safe. My interest in security, and my Cyber Forensics course, led me to this topic.

How I am validating it: I am testing the framework on an IoT irrigation system use case. I use two benchmark datasets, CICIoT2023 and IoTID20. It is still my ongoing final year project, so it is not finished yet.

---

Project 4: This Digital Clone

This is an AI chatbot that answers questions as me, using only documents I wrote about myself. It uses Retrieval-Augmented Generation. First it splits my documents into chunks. Then it searches them with BM25 keyword search and with dense embedding search, combines the results, and sends the best chunks to a language model that replies in first person. I built it for my Agentic AI course using Streamlit and the Groq API. It taught me the difference between keyword search and semantic search. It also taught me that a chatbot should say "I don't have that in my notes" when the answer is not in its documents.

---

Project 5: IoT Environmental Monitoring System

What it is: I designed real-time dashboards to monitor temperature, humidity and air quality.
Impact: It improved monitoring insight accuracy by 30%.

---

Project 6: RFID Smart Access Control System

What it is: I simulated a secure access mechanism for automated entry control.
Impact: It reduced unauthorized entry attempts by 40% in testing.

---

Project 7: Flight Delay Prediction (Machine Learning)

What it is: I built a predictive model using Pandas, Scikit-learn and Matplotlib.
Accuracy: The model achieved 87% accuracy in predictions.

---

Additional academic and research projects:
- C++ Student Management System: a record management system for student data.
- 8086 Assembly Temperature Alarm System: threshold-based monitoring with automated alerts.
- Mini-OS (Student Insight Task Manager): an interactive project that shows core operating system principles through modular design.
- Intruder Classification and Detection: a machine learning model for classifying and detecting network intrusions.
- VaultIQ-Inspired Student Knowledge Vault (Research): I researched a zero-trust digital vault architecture for secure student data storage.