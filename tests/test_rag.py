import json
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from petpal_rag import PetCareRAG, RAGServiceError


def test_retrieval_ranks_matching_passage_first():
    rag = PetCareRAG(api_key="test-key")

    sources = rag.retrieve("How can I make grooming less stressful?")

    assert sources
    assert sources[0].source_id == "grooming"
    assert all(source.score > 0 for source in sources)


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("How much should I feed my dog?", "feeding-and-water"),
        ("What food should I give my cat?", "feeding-and-water"),
        ("how do i groom my dog", "grooming"),
        ("How often should I brush my dog?", "grooming"),
        ("How do I make my cat more active?", "exercise"),
        ("What vaccinations does my puppy need?", "preventive-care"),
        ("How can I keep my dog entertained?", "enrichment"),
    ],
)
def test_retrieval_handles_common_pet_care_paraphrases(question, expected_source):
    sources = PetCareRAG(api_key="test-key").retrieve(question)

    assert sources
    assert sources[0].source_id == expected_source


def test_food_question_does_not_match_enrichment_food_puzzles():
    sources = PetCareRAG(api_key="test-key").retrieve(
        "What food should I give my cat?"
    )

    assert [source.source_id for source in sources] == ["feeding-and-water"]


def test_answer_sends_retrieved_passages_to_model_and_returns_sources():
    rag = PetCareRAG(api_key="test-key")
    response_body = json.dumps(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"text": "Use brief, calm sessions and pause if stressed."}
                        ]
                    }
                }
            ]
        }
    ).encode()

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return response_body

    with patch("petpal_rag.urlopen", return_value=FakeResponse()) as urlopen_mock:
        answer = rag.answer("How can I make grooming less stressful?")

    request = urlopen_mock.call_args.args[0]
    request_body = json.loads(request.data)
    user_prompt = request_body["contents"][0]["parts"][0]["text"]
    assert request.full_url.endswith("/models/gemini-3.5-flash-lite:generateContent")
    assert "Gentle grooming" in user_prompt
    assert "Introduce tools gradually" in user_prompt
    assert answer.text == "Use brief, calm sessions and pause if stressed."
    assert answer.sources[0].source_id == "grooming"
    assert request.get_header("X-goog-api-key") == "test-key"


def test_unsupported_question_does_not_call_model():
    rag = PetCareRAG(api_key="test-key")

    with patch("petpal_rag.urlopen") as urlopen_mock:
        with pytest.raises(RAGServiceError, match="couldn't find relevant guidance"):
            rag.answer("How do I build a spaceship?")

    urlopen_mock.assert_not_called()


def test_missing_gemini_api_key_is_reported_for_supported_question():
    with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
        rag = PetCareRAG()
    with pytest.raises(RAGServiceError, match="need a Gemini API key"):
        rag.answer("How can I make grooming less stressful?")


def test_provider_http_error_is_reported_without_exposing_response_body():
    rag = PetCareRAG(api_key="test-key")
    provider_error = HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent",
        401,
        "Unauthorized",
        hdrs=None,
        fp=None,
    )

    with patch("petpal_rag.urlopen", side_effect=provider_error):
        with pytest.raises(RAGServiceError, match="returned HTTP 401"):
            rag.answer("How can I make grooming less stressful?")


def test_provider_404_identifies_unavailable_model():
    rag = PetCareRAG(api_key="test-key", model="unavailable-model")
    provider_error = HTTPError(
        "https://generativelanguage.googleapis.com/v1beta/models/unavailable-model:generateContent",
        404,
        "Not Found",
        hdrs=None,
        fp=None,
    )

    with patch("petpal_rag.urlopen", side_effect=provider_error):
        with pytest.raises(RAGServiceError, match="couldn't find model 'unavailable-model'"):
            rag.answer("How can I make grooming less stressful?")


@pytest.mark.parametrize(
    "question",
    [
        "My pet is having trouble breathing",
        "My dog ate chocolate",
        "My pet may have been poisoned",
    ],
)
def test_urgent_question_returns_vet_warning_without_calling_model(question):
    rag = PetCareRAG(api_key="")

    with patch("petpal_rag.urlopen") as urlopen_mock:
        answer = rag.answer(question)

    urlopen_mock.assert_not_called()
    assert answer.guarded is True
    assert "emergency veterinary clinic now" in answer.text
    assert any(source.source_id == "urgent-care" for source in answer.sources)


def test_empty_and_overlong_questions_are_rejected():
    rag = PetCareRAG(api_key="test-key")

    with pytest.raises(RAGServiceError, match="Enter a question"):
        rag.answer("  ")
    with pytest.raises(RAGServiceError, match="500 characters or fewer"):
        rag.answer("x" * 501)


def test_invalid_knowledge_file_is_reported(tmp_path: Path):
    invalid_guide = tmp_path / "guide.json"
    invalid_guide.write_text('{"not": "a list"}', encoding="utf-8")

    with pytest.raises(RAGServiceError, match="contains no entries"):
        PetCareRAG(knowledge_path=invalid_guide, api_key="test-key")
