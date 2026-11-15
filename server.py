"""Adaptive Study Coach - backend. Keeps the API key server-side."""
import json, os, re, time
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL = os.getenv("MODEL", "gemini-3.8-flash")
VERIFY = os.getenv("VERIFY", "1") == "1"  # set VERIFY=0 in Render to skip the second check (faster)
client = genai.Client()  # reads GEMINI_API_KEY
app = FastAPI()


def ask(prompt: str, max_tokens: int = 4000) -> dict:
    for attempt in range(5):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a precise curriculum expert. Reply with ONLY valid JSON, no prose, no markdown fences.",
                    response_mime_type="application/json", max_output_tokens=max_tokens),
            )
            break
        except Exception as e:
            busy = any(c in str(e) for c in ("503", "429", "UNAVAILABLE"))
            if busy and attempt < 4:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    text = resp.text or ""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text)


class MapReq(BaseModel):
    subject: str
    level: str
    notes: str = ""
    language: str = "English"


class QReq(MapReq):
    concept: str
    difficulty: str = "medium"
    avoid: list[str] = []


@app.post("/api/map")
def make_map(r: MapReq):
    src = f"Use ONLY this material to define the concepts:\n{r.notes[:6000]}" if r.notes.strip() else \
          "Use the standard syllabus for this subject and level."
    try:
        d = ask(f"""Subject: {r.subject}. Level/curriculum: {r.level}. {src}
Build a prerequisite graph of 8 to 12 core concepts a student must master.
Return JSON: {{"concepts":[{{"id":"c1","name":"short name","prereqs":["ids of concepts that must be learned first"]}}]}}
Rules: ids are c1..cN; prereqs only reference other ids; the graph must be acyclic; at least two concepts have no prereqs.
Write concept names in {r.language}.""")
    except Exception as e:
        print("MAP ERROR:", repr(e), flush=True)
        raise HTTPException(502, f"Could not build topic map: {e}")
    ids = {c["id"] for c in d["concepts"]}
    for c in d["concepts"]:
        c["prereqs"] = [p for p in c.get("prereqs", []) if p in ids and p != c["id"]]
    return d


@app.post("/api/question")
def make_question(r: QReq):
    grounding = f"Base the question ONLY on this material:\n{r.notes[:6000]}\n" if r.notes.strip() else ""
    prompt = f"""Subject: {r.subject}. Level: {r.level}. Concept: {r.concept}. Difficulty: {r.difficulty}.
{grounding}Write one multiple-choice question with exactly 4 options and exactly ONE correct answer.
Do not repeat these earlier questions: {r.avoid[-6:]}
Write everything in {r.language}. Explain WHY the correct answer is right and why a common wrong answer is wrong, in simple words.
Return JSON: {{"question":"...","options":["A","B","C","D"],"answer_index":0,"explanation":"..."}}"""
    last = None
    for _ in range(2):
        try:
            q = ask(prompt)
            assert len(q["options"]) == 4 and 0 <= q["answer_index"] < 4
            if not VERIFY:
                return q
            check = ask(f"""Solve this question yourself. Question: {q['question']} Options: {json.dumps(q['options'])}
Return JSON: {{"best_index":0,"exactly_one_correct":true}}""", 1500)
            if check["best_index"] == q["answer_index"] and check["exactly_one_correct"]:
                return q
            last = "verification mismatch"
        except Exception as e:
            last = repr(e)
            print("QUESTION ERROR:", last, flush=True)
    raise HTTPException(502, f"Could not generate a verified question ({last}). Click Try again.")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
