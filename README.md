# PawPal with AI

PawPal with AI helps busy pet owners turn care tasks into a realistic daily plan and get concise, source-grounded answers to general pet-care questions. It extends the PawPal task and scheduling experience with a Streamlit interface, local retrieval, Gemini-generated responses, emergency safety guardrails, and explanations for each scheduling decision.

## Built on PawPal Starter

This project is built on top of the original project **PawPal Starter**. PawPal Starter helps busy pet owners stay consistent with care routines by tracking tasks per pet and building explainable daily schedules within a set time budget. It supports task filtering and duration sorting, uses greedy priority-then-duration scheduling with inclusion/omission logs, flags overlapping time slots, and creates follow-up occurrences for recurring daily or weekly tasks. It turns time and priority constraints into a clear daily plan.

## Architecture Overview

The system diagram is available in [assets/diagrams/architecture.mmd](assets/diagrams/architecture.mmd).

- The Streamlit UI collects owner settings, pet details, care tasks, schedule requests, and care questions.
- The `Owner`, `Pet`, `Task`, and `Scheduler` model classes maintain in-session data. The scheduler considers pending tasks, a daily minute budget, numeric priority (1 is highest), and task duration.
- Scheduling decisions are greedy: tasks are considered by priority and then shortest duration, and the app logs why each task was included or omitted. Timed tasks are also checked for overlapping intervals; adjacent intervals are not conflicts.
- For questions, `PetCareRAG` retrieves relevant passages from [knowledge/pet_care.json](knowledge/pet_care.json) using TF-IDF cosine similarity.
- Supported questions are sent to Gemini with only the retrieved passages as context. Urgent-care questions bypass the model and return a veterinary warning. The UI separately displays the source titles used for grounded answers.
- Automated tests verify the model, scheduler, retrieval, provider-error handling, and emergency guardrails.

## Setup Instructions

### 1. Clone and enter the project

```bash
git clone https://github.com/farzan4/applied-ai-system-project.git
cd applied-ai-system-project
```

### 2. Create and activate a Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure Gemini for AI answers

Create a Gemini API key, then set it in the current terminal:

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

The default model is `gemini-3.5-flash-lite`. To use a different model your API key can access:

```bash
export GEMINI_MODEL="your-model-name"
```

The API key is optional for running the app. Scheduling and urgent-care warnings work without it; general care questions require a configured key.

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

### 6. Run the command-line example

```bash
python main.py
```

### 7. Run tests

```bash
pytest -q
```

## Reproducible Execution Evidence

The following commands and results were produced from the repository at the
project root. Run `python main.py` for scheduling evidence and `pytest -q`
for the automated reliability suite.

### Command 1: Generate a daily schedule

```bash
python main.py
```

Example output (abridged to the schedule and reasoning):

```text
✅ SCHEDULED TASKS:
  • [Health] Give Luna Joint Supplement (5m) at 09:00 - Priority 1
  • [Exercise] Luna Neighborhood Walk (30m) at 09:00 - Priority 2
  • [Grooming] Groom Milo's Fur (15m) - Priority 3

Total Scheduled Time: 50 / 60 mins

⚠️ OMITTED TASKS (Over Budget):
  • [Enrichment] Extended Dog Park Play (25m) - Priority 4

📋 SCHEDULER LOG / REASONING:
  - Included 'Give Luna Joint Supplement' (5m) - Priority 1.
  - Included 'Luna Neighborhood Walk' (30m) - Priority 2.
  - Included 'Groom Milo's Fur' (15m) - Priority 3.
  - Omitted 'Extended Dog Park Play' (25m) - Exceeds daily limit of 60m.
  ⚠️ Warning: 'Give Luna Joint Supplement' and 'Luna Neighborhood Walk' overlap on 2026-11-23 at 09:00 and 09:00.
```

The example input is created in `main.py`: Alex has a 60-minute budget, Luna
has a 5-minute joint supplement and a 30-minute walk both at 09:00, Milo has
a 15-minute grooming task, and Luna has a 25-minute dog park activity. The
schedule includes 50 minutes, omits the 25-minute activity as over budget,
and reports the duplicate 09:00 interval as an overlap.

### Command 2: Run three end-to-end AI inputs through retrieval and guardrails

This run exercises the complete RAG service entry point for one supported
question, one urgent question, and one unsupported question. No Gemini API key
is required: supported retrieval returns local evidence and reports that a key
is needed for generation, urgent input is answered locally, and unsupported
input is rejected before provider access.

Run this Python snippet from the project root:

```bash
python - <<'PY'
from petpal_rag import PetCareRAG

rag = PetCareRAG(api_key="")

questions = [
    "How can I make grooming less stressful?",
    "My pet is having trouble breathing",
    "How do I build a spaceship?",
]

for question in questions:
    print("INPUT:", question)
    print(
        "RETRIEVED:",
        ", ".join(source.title for source in rag.retrieve(question)),
    )
    try:
        answer = rag.answer(question)
        print("OUTPUT:", answer.text)
        print("GUARDRAILED:", answer.guarded)
    except Exception as exc:
        print("OUTPUT:", exc)
PY
```

Example output:

```text
INPUT: How can I make grooming less stressful?
RETRIEVED: Gentle grooming
OUTPUT: AI answers need a Gemini API key. Set GEMINI_API_KEY as described in the README, then restart the app.
GUARDRAILED: False

INPUT: My pet is having trouble breathing
RETRIEVED: Urgent veterinary care
OUTPUT: This may be an emergency. Contact a veterinarian or local emergency veterinary clinic now. PawPal+ cannot assess or treat an urgent condition.
GUARDRAILED: True

INPUT: How do I build a spaceship?
RETRIEVED:
OUTPUT: I couldn't find relevant guidance in the care guide. Try asking about feeding, exercise, grooming, medication safety, preventive care, or enrichment.
GUARDRAILED: False
```

The input/output matrix demonstrates RAG evidence selection, generation
blocking when Gemini is unavailable, deterministic urgent-care bypass, and
rejection of a question outside the bundled knowledge base.

### Command 3: Run the reliability suite

```bash
pytest -q
```

Captured result:

```text
.................................                                        [100%]
33 passed in 0.11s
```

### Reliability and guardrail results

| Check | Result |
|---|---|
| Retriever ranking: grooming, feeding, exercise, preventive care, and enrichment paraphrases | Passed |
| Unsupported-question provider call | Blocked; no model call |
| Missing Gemini API key | Reported as a configuration error |
| Breathing/choking/poison-like urgent terms | Returned a local veterinary warning and bypassed the model |
| Empty and overlong questions | Rejected; overlong limit is 500 characters |
| Provider HTTP 401/404 | Reported as provider/model errors without exposing the response body |
| Scheduling, recurrence, filtering, and overlap boundaries | Passed |

The guardrails are deterministic code paths, but the suite does not verify live
Gemini answer quality or real-world veterinary safety. Gemini calls use mocked
HTTP responses in the test suite and require a real API key for live validation.

## Sample Interactions

### 1. Grooming question

**Input:**

```text
How can I make grooming less stressful?
```

**Result:**

PawPal retrieves the `grooming` care-guide passage and, with a valid Gemini key, returns a concise answer such as:

```text
Use calm, brief grooming sessions and introduce tools gradually. Pause if
your pet becomes stressed, and use only grooming products labeled for that
animal. If you notice irritation or unusual hair loss, ask a veterinarian.
```

The UI also lists **Gentle grooming** as the source.

### 2. Feeding question

**Input:**

```text
How much should I feed my dog?
```

**Result:**

PawPal retrieves `feeding-and-water` and synthesizes advice around fresh water, an individual routine, and species-, age-, health-, and activity-specific needs. Its response should recommend consulting a veterinarian for persistent appetite or weight concerns rather than presenting a universal quantity.

### 3. Urgent-care guardrail

**Input:**

```text
My pet is having trouble breathing
```

**Result:**

```text
This may be an emergency. Contact a veterinarian or local emergency veterinary
clinic now. PawPal+ cannot assess or treat an urgent condition.
```

This response is generated locally, bypasses Gemini, and identifies `Urgent veterinary care` as a source.

### 4. Schedule example

**Input:**

- Owner budget: 60 minutes
- Luna: joint supplement, 5 minutes, high priority, 9:00 AM
- Luna: neighborhood walk, 30 minutes, medium priority, 9:00 AM
- Milo: grooming, 15 minutes, low priority, no scheduled time
- Luna: dog park play, 25 minutes, low priority, no scheduled time

**Result:**

PawPal schedules the joint supplement and walk, with a 35-minute total, then schedules grooming for 15 minutes (50 minutes total). Dog park play is omitted because its 25-minute duration would exceed the remaining 10 minutes. The app reports the inclusion, omission, and total budget usage. The two tasks at 9:00 AM receive an overlap warning because both scheduled intervals occupy the same start time.

## Design Decisions

- **Retrieval-augmented generation instead of unrestricted prompting:** I used local TF-IDF retrieval to find relevant passages from the bundled care guide before asking Gemini to generate an answer. This makes responses easier to evaluate and reduces unsupported claims compared with sending only the user's question to the model. The trade-off is that the system can only answer topics covered by the guide and the keyword-based retriever may miss good paraphrases or ambiguous questions.
- **A small, fixed knowledge base:** I stored the care guidance in [knowledge/pet_care.json](knowledge/pet_care.json), a reviewable and reproducible source. This avoids maintaining an external vector database and keeps retrieval tests fast and offline, but it does not provide veterinary-specific, real-time, or personalized medical knowledge.
- **A low-temperature Gemini request:** The AI implementation asks the model to synthesize only retrieved passages and uses a temperature of `0.2` with a 400-token limit. This favors concise, consistent responses. It is less creative than a higher-temperature answer, but creativity is less important than accuracy for pet-care guidance.
- **Prompt and evidence boundaries:** The system prompt tells Gemini not to diagnose, prescribe, recommend medication doses, or claim facts outside the supplied passages. The UI also displays the retrieved source titles separately. This improves transparency, but source attribution cannot guarantee that the model has followed every instruction perfectly.
- **Emergency detection outside the model:** Urgent terms trigger a deterministic veterinary-care warning and return no generated AI response. This is less flexible than asking the model to assess urgency, but it provides a consistent safety fallback and works even when no API key or network connection is available.
- **Explicit no-answer behavior:** If retrieval finds no sufficiently relevant passage or an API key is missing, PawPal reports what it cannot do instead of inventing an answer or silently falling back to general model knowledge. The trade-off is a narrower user experience with more rejections, but the project prioritizes evidence and useful failure messages over speculative answers.
- **Tested provider interactions:** Gemini requests are isolated behind `PetCareRAG` and tested with mocked HTTP responses. This makes provider, model, authentication, timeout, and malformed-response behavior predictable and testable without consuming API quota. Mocked tests cannot verify live model quality, so they complement rather than replace manual and live provider testing.

## Testing Summary

The project uses `pytest` to test both the PawPal scheduling logic and the AI care-answer implementation. The AI tests in `tests/test_rag.py` verify that relevant care passages are retrieved for common questions, that those passages and the expected request settings are sent to Gemini, and that returned answers include their source references. They also check safe handling of unsupported or invalid questions, missing API keys, provider HTTP errors, and urgent-care questions that must bypass Gemini.

- **Latest result:** `pytest -q` passed all **33 tests**.
- **AI reliability checks:** Retrieval ranking and common paraphrases are checked with assertions; Gemini integration behavior is exercised using a mocked HTTP response; urgent-care and unsupported-question tests assert that no provider request is made.
- **Important limitation:** Mocked responses verify request construction and error handling, not live Gemini availability, generated-answer accuracy, or adherence to the prompt. Those require separate live-provider checks and human evaluation.
- **Other coverage:** The suite also checks task completion and recurrence, duration sorting, owner filtering, greedy scheduling, and overlap boundaries.
- **Remaining gaps:** The suite does not yet automate Streamlit UI interactions, persistence, editing/deleting pets and tasks, or live Gemini responses.

## Reflection

This project taught me that useful AI systems combine generated language with verifiable data, deterministic logic, and clear safety boundaries. The scheduler's simple, explainable algorithm and the RAG service's local evidence checks made the system more trustworthy than relying on model output alone. Human review remains essential for setting requirements, correcting AI-generated code, and deciding when no AI answer is safer than a confident one.

## Portfolio Artifact

**Code:** [PawPal+ on GitHub](https://github.com/farzan4/applied-ai-system-project)

**What this project says about me as an AI engineer:** I build AI systems with verifiable boundaries, not just impressive demos. PawPal+ shows that I can turn a real user problem into a maintainable architecture, combine deterministic scheduling with retrieval-grounded generation, design predictable failure behavior, and prove reliability with automated tests. It also reflects my willingness to use AI as a collaborator while retaining responsibility for architecture, safety, tradeoffs, and verification.
