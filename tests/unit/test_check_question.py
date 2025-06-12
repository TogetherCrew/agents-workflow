import unittest
from unittest.mock import patch, Mock
import json

from tasks.hivemind.classify_question import ClassifyQuestion, QuestionClassificationResult, MessageClassificationResult


class TestClassifyQuestion(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-4o-mini-2024-07-18"
        self.rag_threshold = 0.5
        self.check_question = ClassifyQuestion(self.model, self.rag_threshold)
        self.check_question_with_reasoning = ClassifyQuestion(self.model, self.rag_threshold, enable_reasoning=True)

    def test_init_valid_threshold(self):
        # Test that valid thresholds work
        valid_thresholds = [0, 0.25, 0.5, 0.75, 1.0]
        for threshold in valid_thresholds:
            question_classifier = ClassifyQuestion(self.model, threshold)
            self.assertEqual(question_classifier.rag_threshold, threshold)

    def test_init_invalid_threshold_too_low(self):
        # Test that threshold below 0 raises ValueError
        with self.assertRaises(ValueError) as context:
            ClassifyQuestion(self.model, -0.1)
        self.assertIn("rag_threshold must be between 0 and 1", str(context.exception))

    def test_init_invalid_threshold_too_high(self):
        # Test that threshold above 1 raises ValueError
        with self.assertRaises(ValueError) as context:
            ClassifyQuestion(self.model, 1.5)
        self.assertIn("rag_threshold must be between 0 and 1", str(context.exception))

    def test_init_with_reasoning(self):
        # Test that enable_reasoning parameter works
        classifier = ClassifyQuestion(self.model, self.rag_threshold, enable_reasoning=True)
        self.assertTrue(classifier.enable_reasoning)
        
        classifier = ClassifyQuestion(self.model, self.rag_threshold, enable_reasoning=False)
        self.assertFalse(classifier.enable_reasoning)

    @patch("transformers.pipeline")
    def test_classify_message_statement(self, mock_pipeline):
        # Test that a statement is correctly classified as False

        # Mock the pipeline response
        mock_pipeline.return_value = [{"label": "LABEL_0", "score": 0.99}]  # STATEMENT

        result = self.check_question.classify_message("This is a statement.")
        self.assertFalse(result)

    @patch("transformers.pipeline")
    def test_classify_message_question(self, mock_pipeline):
        # Test that a question is correctly classified as True

        # Mock the pipeline response
        mock_pipeline.return_value = [{"label": "LABEL_1", "score": 0.99}]  # QUESTION

        result = self.check_question.classify_message("Is this a question?")
        self.assertTrue(result)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_question_lm_true_response(self, mock_openai):
        # Test that classify_question_lm returns True for positive responses

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({"result": True})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_question_lm("What is the weather?")
        self.assertIsInstance(result, QuestionClassificationResult)
        self.assertTrue(result.result)
        self.assertIsNone(result.reasoning)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_question_lm_false_response(self, mock_openai):
        # Test that classify_question_lm returns False for negative responses

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({"result": False})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_question_lm("Hello there!")
        self.assertIsInstance(result, QuestionClassificationResult)
        self.assertFalse(result.result)
        self.assertIsNone(result.reasoning)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_question_lm_with_reasoning(self, mock_openai):
        # Test classify_question_lm with reasoning enabled

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({
            "result": True,
            "reasoning": "This is clearly asking for information about weather conditions."
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question_with_reasoning.classify_question_lm("What is the weather?")
        self.assertIsInstance(result, QuestionClassificationResult)
        self.assertTrue(result.result)
        self.assertEqual(result.reasoning, "This is clearly asking for information about weather conditions.")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_question_lm_invalid_json_response(self, mock_openai):
        # Test that classify_question_lm raises JSONDecodeError for invalid JSON

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "invalid_json"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(json.JSONDecodeError):
            self.check_question.classify_question_lm("Is this valid?")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_high_score(self, mock_openai):
        # Test that the classify_message_lm method returns True for a score above threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({"score": 0.8})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm(
            "What is the capital of France?"
        )
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertTrue(result.result)
        self.assertEqual(result.score, 0.8)
        self.assertIsNone(result.reasoning)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_low_score(self, mock_openai):
        # Test that the classify_message_lm method returns False for a score below threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({"score": 0.2})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("I am going to the store.")
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertFalse(result.result)
        self.assertEqual(result.score, 0.2)
        self.assertIsNone(result.reasoning)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_exact_threshold(self, mock_openai):
        # Test that the classify_message_lm method returns True for a score equal to threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({"score": 0.5})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("Can you help me?")
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertTrue(result.result)
        self.assertEqual(result.score, 0.5)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_with_reasoning(self, mock_openai):
        # Test classify_message_lm with reasoning enabled

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = json.dumps({
            "score": 0.9,
            "reasoning": "This requires up-to-date information about cryptocurrency prices which would need RAG retrieval."
        })
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question_with_reasoning.classify_message_lm("What is the latest Bitcoin price?")
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertTrue(result.result)
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.reasoning, "This requires up-to-date information about cryptocurrency prices which would need RAG retrieval.")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_boundary_values(self, mock_openai):
        # Test boundary values 0 and 1

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Test with score 0
        mock_message.content = json.dumps({"score": 0.0})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("Hello there!")
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertFalse(result.result)
        self.assertEqual(result.score, 0.0)

        # Test with score 1
        mock_message.content = json.dumps({"score": 1.0})
        result = self.check_question.classify_message_lm("What is the latest news?")
        self.assertIsInstance(result, MessageClassificationResult)
        self.assertTrue(result.result)
        self.assertEqual(result.score, 1.0)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_invalid_json_response(self, mock_openai):
        # Test that classify_message_lm raises JSONDecodeError for invalid JSON

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "Invalid JSON"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(json.JSONDecodeError):
            self.check_question.classify_message_lm("Can you do something for me?")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_score_validation_still_works(self, mock_openai):
        # Test that score validation still works even with structured outputs
        # (This tests the additional validation we keep in the code)

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Test with value greater than 1 - should raise ValueError due to our validation
        mock_message.content = json.dumps({"score": 1.5})
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.check_question.classify_message_lm("Could you help me with this?")
        self.assertIn("Generated score must be between 0 and 1", str(context.exception))
