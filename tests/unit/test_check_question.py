import unittest
from unittest.mock import patch, Mock

from tasks.hivemind.classify_question import ClassifyQuestion


class TestClassifyQuestion(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-4o-mini-2024-07-18"
        self.rag_threshold = 0.5
        self.check_question = ClassifyQuestion(self.model, self.rag_threshold)

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
    def test_classify_message_lm_high_score(self, mock_openai):
        # Test that the classify_message_lm method returns True for a score above threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "0.8"  # Score above threshold (0.5)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm(
            "What is the capital of France?"
        )
        self.assertTrue(result)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_low_score(self, mock_openai):
        # Test that the classify_message_lm method returns False for a score below threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "0.2"  # Score below threshold (0.5)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("I am going to the store.")
        self.assertFalse(result)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_exact_threshold(self, mock_openai):
        # Test that the classify_message_lm method returns True for a score equal to threshold

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "0.5"  # Score equal to threshold (0.5)
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("Can you help me?")
        self.assertTrue(result)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_boundary_values(self, mock_openai):
        # Test boundary values 0 and 1

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Test with score 0
        mock_message.content = "0"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("Hello there!")
        self.assertFalse(result)

        # Test with score 1
        mock_message.content = "1"
        result = self.check_question.classify_message_lm("What is the latest news?")
        self.assertTrue(result)

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_score_below_zero(self, mock_openai):
        # Test that classify_message_lm raises ValueError for scores below 0

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "-0.1"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.check_question.classify_message_lm("Could you help me with this?")
        self.assertIn("Generated score must be between 0 and 1", str(context.exception))

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_invalid_response(self, mock_openai):
        # Test that classify_message_lm raises ValueError for an invalid response from OpenAI API

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "Invalid"  # Non-numeric response
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            self.check_question.classify_message_lm("Can you do something for me?")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_out_of_range_response(self, mock_openai):
        # Test that classify_message_lm raises ValueError for values outside 0-1 range

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Test with value greater than 1 - should raise ValueError due to score validation
        mock_message.content = "1.5"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.check_question.classify_message_lm("Could you help me with this?")
        self.assertIn("Generated score must be between 0 and 1", str(context.exception))

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_invalid_decimal_format(self, mock_openai):
        # Test that classify_message_lm raises ValueError for invalid decimal formats

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "0.5.2"  # Invalid decimal format
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            self.check_question.classify_message_lm("Could you help me with this?")

    @patch("tasks.hivemind.classify_question.OpenAI")
    def test_classify_message_lm_negative_response(self, mock_openai):
        # Test that classify_message_lm raises ValueError for negative responses

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "-0.5"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError) as context:
            self.check_question.classify_message_lm("Could you help me with this?")
        self.assertIn("Generated score must be between 0 and 1", str(context.exception))
