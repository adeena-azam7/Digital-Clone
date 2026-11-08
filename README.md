# DigitalTwin-RAG: Adeena Azam's AI Clone

**This is an AI clone, not the real person.**

Live app: [PASTE YOUR STREAMLIT LINK HERE]

A chatbot that answers questions as me, using only documents I wrote about myself. It uses Retrieval-Augmented Generation (RAG): BM25 keyword search and dense embedding search, combined with Reciprocal Rank Fusion (RRF), and a Groq language model that answers in first person. It is built with Streamlit and deployed on Streamlit Community Cloud.

## How it works
1. **Chunking:** documents in `data/` are split into chunks of 250 words with a 40-word overlap. Each chunk keeps its file name and chunk ID.
2. **Two indexes:** a BM25 index (`rank_bm25`) and sentence embeddings (`all-MiniLM-L6-v2`), computed once and saved in `embeddings.npy`.
3. **Retrieval modes:** BM25 only, Dense only (cosine similarity with numpy), and Hybrid.
4. **Fusion:** Hybrid uses Reciprocal Rank Fusion, score = sum of 1 / (60 + rank). The other method, weighted fusion (alpha * dense + (1 - alpha) * bm25 after min-max normalizing each score), was not used.
5. **Generation:** the top 3 chunks and the question go to a Groq model with a persona prompt. It must answer in first person, only from the context, and say "I don't have that in my notes." when the answer is missing.
6. **Security:** the API key is stored as a Streamlit secret (`GROQ_API_KEY`) and is not in the code.

## 1. Knowledge base and why I chose it
I used 9 documents with 25 chunks in total: `bio.md`, `education.md`, `skills.md`, `projects.md`, `personality.md`, `study_habits.md`, `advice_for_juniors.md`, `essay.md` and `faq.md`.

I chose them to cover the required mix. `bio.md`, `education.md` and `skills.md` hold facts. `personality.md` and `study_habits.md` hold opinions, hobbies and habits. `projects.md` and `essay.md` are long-form. `faq.md` is written in my own voice, so the clone sounds like me.

Rare keywords such as project names (SentinelGuard, Campus Marketplace, MindbridgeRAG_CareerAnxiety) give BM25 something exact to match. Plain descriptive passages, for example about how I recharge after a hard week, let dense search match questions that use different words.

## 2. Test questions
For every retrieved chunk (top 3) I checked by hand whether it **contains the answer** (Yes / No). Format: file #chunk number. "Hits" = how many of the 3 chunks contain the answer.

| Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|
| **Q1.** What is a SECURITY DEFINER function? *(exact rare keyword)* | 1. faq #6: Yes<br>2. projects #3: Yes<br>3. projects #2: Yes<br>**Hits: 3/3** | 1. projects #3: Yes<br>2. faq #6: Yes<br>3. skills #1: Yes<br>**Hits: 3/3** | 1. faq #6: Yes<br>2. projects #3: Yes<br>3. skills #1: Yes<br>**Hits: 3/3** | **Tie.** All three modes found 3 chunks with the term. The phrase is also distinctive in meaning, so dense matched BM25 on this small corpus. |
| **Q2.** What do you do to relax when you are exhausted? *(paraphrase, no shared content words)* | 1. faq #5: No<br>2. faq #4: No<br>3. faq #2: Yes<br>**Hits: 1/3** | 1. personality #1: Yes<br>2. study_habits #1: No<br>3. study_habits #2: No<br>**Hits: 1/3** | 1. faq #4: No<br>2. faq #3: No<br>3. faq #2: Yes<br>**Hits: 1/3** | **Dense.** It is the only mode that put an answer chunk (hobbies and recharging) at rank 1. BM25 matched filler words (do, you, to, when) and ranked the best chunk 12th. |
| **Q3.** Which datasets do you use to validate SentinelGuard? | 1. faq #2: Yes<br>2. faq #6: Yes<br>3. projects #3: Yes<br>**Hits: 3/3** | 1. faq #6: Yes<br>2. projects #3: Yes<br>3. education #2: No<br>**Hits: 2/3** | 1. faq #6: Yes<br>2. faq #2: Yes<br>3. projects #3: Yes<br>**Hits: 3/3** | **BM25 and Hybrid.** The exact name SentinelGuard and the word "datasets" appear in all three chunks. Dense let in education #2, which talks about courses, not datasets. |
| **Q4.** Why do you leave things until the last minute? *(paraphrase)* | 1. advice_for_juniors #1: Yes<br>2. personality #1: Yes<br>3. faq #5: No<br>**Hits: 2/3** | 1. study_habits #2: Yes<br>2. study_habits #1: Yes<br>3. advice_for_juniors #2: No<br>**Hits: 2/3** | 1. personality #1: Yes<br>2. study_habits #2: Yes<br>3. study_habits #1: Yes<br>**Hits: 3/3** | **Hybrid.** BM25 caught "last-minute", dense caught the "delay until the deadline" wording, and fusion kept the good chunks from both. |
| **Q5.** What is your favorite movie? *(not in my notes)* | 1. faq #3: No<br>2. faq #5: No<br>3. faq #4: No<br>**Hits: 0/3** | 1. faq #3: No<br>2. personality #1: No<br>3. faq #1: No<br>**Hits: 0/3** | 1. faq #3: No<br>2. personality #1: No<br>3. faq #2: No<br>**Hits: 0/3** | **No mode can be right,** because my notes never mention a movie. The top dense similarity was only 0.13, against 0.27 to 0.49 for answerable questions. So the LLM must refuse. Clone's answer: [PASTE CLONE ANSWER] |

## 3. Failure case
**Question:** "What do you do to relax when you are exhausted?" (Hybrid mode)

**What the clone said:** [PASTE THE CLONE'S HYBRID ANSWER TO Q2 HERE, AND SAY IF IT WAS WRONG OR WEAK]

**What went wrong in retrieval:** Hybrid dropped the best chunk. `personality.md` #1 (hobbies, "I pick up my sketchbook or brushes") was Dense rank 1 but BM25 rank 12, so its RRF score (0.0303) lost to three FAQ chunks (0.0311, 0.0308, 0.0306) and it fell out of the top 3. Two of the three chunks Hybrid kept (`faq.md` #4 and #3) did not contain the answer.

**Why it happened:** my FAQ is split into only 6 chunks of 250 words, and each chunk mixes about 8 unrelated questions. These chunks are full of the words "you", "do", "what" and "I", so BM25 gives them high scores for almost any question, and `faq.md` appeared in the top 3 for nearly every query. RRF only looks at ranks, so one bad BM25 ranking can push a chunk that dense search ranked first out of the list.

## 4. One improvement I would make next
I would add a cross-encoder reranker over the hybrid results. The current ranking only looks at the question and each chunk separately. A reranker reads the question and the chunk together, so it can push the truly relevant chunk to the top and reduce weak answers like the failure case above.