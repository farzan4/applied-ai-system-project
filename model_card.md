# PawPal+ Model Card

## System Overview and Intended Use

PawPal+ combines a pet-care task scheduler with a retrieval-augmented question-answering feature. For care questions, it searches a small local guide using TF-IDF cosine similarity and, when relevant guidance is found and a Gemini API key is configured, asks Gemini to synthesize an answer from the retrieved passages. The app displays the source titles. Urgent-care questions detected by the app receive a local veterinary warning instead of being sent to Gemini.

The question-answering feature is intended for general, educational pet-care information. It is not a veterinary service, diagnostic tool, or substitute for advice from a veterinarian. The scheduler is intended to help organize tasks; its plan is not a determination that a pet's care needs have been met.

## Limitations and Potential Biases

- The bundled guide is small and general. It does not cover every species, condition, region, or care question, and it does not provide individualized or real-time veterinary guidance.
- Retrieval is based on word overlap, aliases, and TF-IDF scoring, not a deep understanding of meaning. It can miss relevant paraphrases, rank a merely related passage, or return no answer for a question the guide could help with.
- Gemini can still produce inaccurate, incomplete, or overconfident wording even when given relevant passages and instructions to stay grounded. Displaying source titles does not prove that every claim is supported.
- The emergency guardrail depends on a finite list of urgent terms. It can miss an emergency described in different words, and a matched word can trigger a warning when the situation is not urgent. It is a precaution, not reliable triage.
- The guide and interface are English-language and assume users can describe their question in terms represented by the guide. The quality of answers may therefore vary across language, terminology, and user context.
- The scheduler uses owner-entered task priorities and a greedy priority-then-duration rule. It does not account for all animal-specific needs, optimize every possible combination, or verify that the generated plan is clinically appropriate.
- The application has not been clinically validated. Automated tests cover local retrieval and mocked provider responses, not the accuracy of live Gemini answers or real-world health outcomes.

## Misuse Risks and Safeguards

Someone could treat a general answer as a diagnosis or treatment plan, use it instead of contacting a veterinarian, or rely on the schedule as proof that a pet's needs are covered. A confident generated answer could also be mistaken for professional advice.

Safeguards in the current system include an educational-use notice in the UI; instructions to Gemini not to diagnose, prescribe, recommend medication doses, or override a veterinarian; retrieval from a reviewable local guide; visible source titles; and a no-answer response when no sufficiently relevant passage is found. Recognized urgent terms bypass Gemini and return a message to contact a veterinarian or emergency clinic. The app also limits question length.

These safeguards reduce risk but cannot prevent misuse or guarantee that every urgent question is recognized. Users should contact a veterinarian for medical concerns and seek immediate veterinary care for possible emergencies rather than waiting for an AI response. Further risk reduction would require broader human-reviewed guidance, evaluation with diverse real-world questions, and continued review of missed and incorrectly triggered emergency cases.

## Reliability Testing and What Surprised Me

The tests check retrieval for common pet-care paraphrases, including grooming, feeding, activity, preventive care, and enrichment. They also check that a feeding question does not return an enrichment passage just because it mentions food puzzles; unsupported questions do not call the model; and recognized urgent questions bypass the provider. Gemini request construction and response handling are tested with mocked responses.

What surprised me was how much the system's apparent ability to understand a question depends on relatively simple lexical matching and a small set of aliases. The tests show that common paraphrases can work, but also why a seemingly related passage can be a misleading match. I also learned that passing mocked integration and guardrail tests says nothing conclusive about the accuracy of a live generated answer. The current tests support confidence in specific code paths, not a claim that the system is reliably safe or correct for every pet-care question.

## Collaboration with AI

I used an AI coding assistant to explore class responsibilities, scheduling rules, edge cases, tests, and documentation. I accepted the suggestion to keep scheduling policy in a dedicated `Scheduler` rather than mixing it into the `Owner`, `Pet`, and `Task` data classes; that separation made the rules easier to understand and test.

I did not keep a separate `DailyPlan` wrapper as originally suggested. In this project it would mainly have wrapped the scheduler's output, so I changed the design to return the scheduled tasks, omitted tasks, and reasoning log directly. That was a simpler fit for the implementation and the tests. I reviewed AI suggestions against the project needs and verified behavior with the code and tests rather than treating generated suggestions as automatically correct.
