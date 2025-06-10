import unittest
from unittest.mock import patch, Mock

from tasks.hivemind.classify_question import ClassifyQuestion


class TestClassifyQuestion(unittest.TestCase):
    def setUp(self):
        self.model = "gpt-4o-mini-2024-07-18"
        self.rag_threshold = 0.5
        self.check_question = ClassifyQuestion(self.model, self.rag_threshold)

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
        # Test that classify_message_lm works correctly with values outside 0-1 range (current implementation allows this)

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        # Test with value greater than 1 - should not raise error but return True since 1.5 >= 0.5
        mock_message.content = "1.5"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        result = self.check_question.classify_message_lm("Could you help me with this?")
        self.assertTrue(result)  # 1.5 >= 0.5 threshold

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
        # Test that classify_message_lm raises ValueError for negative responses (regex doesn't match negative numbers)

        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "-0.5"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        mock_openai.return_value.chat.completions.create.return_value = mock_response

        with self.assertRaises(ValueError):
            self.check_question.classify_message_lm("Could you help me with this?")
