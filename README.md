# Study Coach: adaptive learning for any subject and curriculum

A student enters a subject, a level (KCSE, CBC, IGCSE, IB, university, anything) and optional notes.
The app builds a prerequisite graph of concepts, then asks questions chosen by an algorithm that
estimates what the student knows.

## How it works
1. **Topic map**: an LLM generates a concept graph with prerequisites for any subject (`/api/map`).
2. **Mastery tracking**: Bayesian Knowledge Tracing keeps a probability of mastery per concept.
3. **Adaptive selection**: ask where the student is most uncertain, only once prerequisites look solid.
   After a wrong answer, diagnose the weakest prerequisite.
4. **Verified questions**: each question is generated, then solved independently; mismatches are discarded.
   With pasted notes, questions are grounded only in that text.

## Results
`python simulate.py` simulates 500 students on random concept graphs and compares adaptive vs random order.
Result: adaptive reached full mastery with about 55% fewer questions (101 vs 224 on average) to reach mastery. Chart: `sim_results.png`.

## Run locally
```
pip install -r requirements.txt
export GEMINI_API_KEY=your_key
uvicorn server:app --reload
```
Open http://localhost:8000

## Deploy (free): Render or Railway
- New Web Service from your GitHub repo
- Build: `pip install -r requirements.txt`
- Start: `uvicorn server:app --host 0.0.0.0 --port $PORT`
- Add `GEMINI_API_KEY` as an environment variable
