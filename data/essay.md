Why I Am Interested in AI and RAG

I am a computer science student, and the area that excites me most is AI development. Over the last years my projects have slowly moved in this direction, and Retrieval-Augmented Generation (RAG) has become a common thread between them.

The problem with normal language models

A normal language model answers from what it learned during training. It can sound very confident even when it is wrong. This is dangerous when the answer matters, for example in career guidance or security. The model does not know when it is guessing, and the reader often cannot tell either.

How RAG helps

RAG changes this. First the system searches a set of trusted documents and finds the passages that match the question. Then the language model writes an answer using only those passages. I like this idea because the answers become more relevant, and I can check where the information came from. If the passages do not contain the answer, the system can honestly say so.

Where I used RAG

I first used RAG in MindbridgeRAG_CareerAnxiety, where students get context-based guidance about career worries. In my final year project SentinelGuard, RAG explains detected security threats in natural language. In this digital clone, I used it to answer questions about myself. Each project used RAG for a different purpose, which showed me how flexible the idea is.

Keyword search and semantic search

While building this clone I learned that there are two main ways to search. BM25 is keyword search. It is very good when a question contains an exact rare word, such as a project name. But it fails when the question uses different words from the document. Dense embedding search looks at meaning, so it can match a question to a passage even when they share no words. But it can miss an exact rare term. Each method fails in a different way, so I combined them with Reciprocal Rank Fusion to get a better ranking. Reciprocal Rank Fusion gives each chunk a score based on its rank in each list, so a chunk that ranks well in both lists rises to the top.

Honesty as a design rule

The most important lesson for me is honesty. A good AI system should say "I don't have that in my notes" instead of inventing facts. This matches my own values, because I believe it is better to say "I don't know" than to pretend.

What I want to learn next

I want to deepen my knowledge of machine learning, learn how to deploy AI applications online, and build AI agents that use tools. I believe this is the path that will help me grow into a skilled Software Engineer.