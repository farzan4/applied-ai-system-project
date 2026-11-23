# PawPal+ 5-7 Minute Presentation Talk Track

## Slide 1 — Title (20 seconds)
Hello, I'm presenting PawPal+, an applied AI system that helps busy pet owners turn care tasks into realistic daily plans and answer general pet-care questions with grounded guidance.

## Slide 2 — The Problem (45 seconds)
Pet owners may manage several tasks while working within a limited amount of time. A full chatbot answer is not always enough, and unrestricted model output can be especially risky for pet care. The goal of PawPal+ is to combine an understandable schedule with evidence-grounded answers and clear no-answer or emergency behavior.

## Slide 3 — Architecture (60 seconds)
The Streamlit interface collects owner settings, pets, tasks, and questions. The domain model maintains those records, and the scheduler considers priority, duration, and the available daily budget. For questions, TF-IDF retrieval finds relevant passages in a local care guide. Supported questions are sent to Gemini with only those passages as context.

The safety boundary is separate from the model: urgent terms activate a deterministic local veterinary warning and unsupported questions are rejected rather than answered from general model knowledge.

## Slide 4 — Demo (90 seconds)
In this example, Alex has a 60-minute budget. Luna has a five-minute supplement and a 30-minute walk, both at 9:00; Milo has 15 minutes of grooming; and Luna has 25 minutes of dog park play. PawPal schedules the first three tasks for 50 total minutes and omits dog park play because it would exceed the remaining budget. It also warns that both 9:00 tasks overlap.

For the question about making grooming less stressful, the system retrieves the Gentle grooming passage and returns an answer based on that evidence, while displaying the source title.

## Slide 5 — Grounded Answering (55 seconds)
The AI path has four stages: retrieve, guard, generate, and attribute. Retrieval uses TF-IDF cosine similarity over a fixed care guide. Urgent questions bypass Gemini. Supported questions go to Gemini at a low temperature with an instruction to avoid diagnosis or unsupported claims. Finally, the UI shows which care-guide source was used.

The deliberate tradeoff is a narrower knowledge base, but that makes the evidence reviewable and makes unsupported questions predictable.

## Slide 6 — Verification (60 seconds)
I used pytest to verify the scheduler, recurrence, filtering, overlap boundaries, retrieval ranking, provider error handling, urgent guardrails, and invalid input behavior. The latest run passes all 33 tests in 0.06 seconds.

Important limitations remain: mocked provider tests do not prove live Gemini answer quality, and the suite does not yet automate the Streamlit UI or persistence.

## Slide 7 — What I Learned (50 seconds)
I learned that useful AI systems need explicit boundaries and responsible human ownership. Separating model, scheduler, retrieval, and safety responsibilities made the system easier to understand and test. AI was a strong coding collaborator, but I remained accountable for architecture, verification, and deciding when not to answer.

Next, I would add persistence, edit and delete flows, UI automation, and live answer evaluation. The complete code is available on GitHub.

## Portfolio Artifact

**Code:** [PawPal+ on GitHub](https://github.com/farzan4/applied-ai-system-project)

**What this project says about me as an AI engineer:** I build AI systems with verifiable boundaries, not just impressive demos. PawPal+ shows that I can turn a real user problem into a maintainable architecture, combine deterministic scheduling with retrieval-grounded generation, design predictable failure behavior, and prove reliability with automated tests. It also reflects my willingness to use AI as a collaborator while retaining responsibility for architecture, safety, tradeoffs, and verification.
