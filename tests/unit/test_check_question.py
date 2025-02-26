import unittest
from unittest.mock import patch
from tasks.hivemind.classify_question import ClassifyQuestion


class TestClassifyQuestion(unittest.TestCase):

    def setUp(self):
        self.model = "gpt-4o-mini"
        self.check_question = ClassifyQuestion(self.model)

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

    @patch("openai.ChatCompletion.create")
    def test_classify_message_lm_true(self, mock_openai):
        # Test that the classify_message_lm method returns True for a message that requires external knowledge

        # Mock OpenAI response
        mock_openai.return_value = {"choices": [{"message": {"content": "True"}}]}

        result = self.check_question.classify_message_lm(
            "What is the capital of France?"
        )
        self.assertTrue(result)

    @patch("openai.ChatCompletion.create")
    def test_classify_message_lm_false(self, mock_openai):
        # Test that the classify_message_lm method returns False for a statement or a non-question

        # Mock OpenAI response
        mock_openai.return_value = {"choices": [{"message": {"content": "False"}}]}

        result = self.check_question.classify_message_lm("I am going to the store.")
        self.assertFalse(result)

    @patch("openai.ChatCompletion.create")
    def test_classify_message_lm_invalid_response(self, mock_openai):
        # Test that classify_message_lm raises ValueError for an invalid response from OpenAI API

        # Mock OpenAI response
        mock_openai.return_value = {"choices": [{"message": {"content": "Invalid"}}]}

        with self.assertRaises(ValueError):
            self.check_question.classify_message_lm("Can you do something for me?")

    @patch("openai.ChatCompletion.create")
    def test_classify_message_lm_no_answer(self, mock_openai):
        # Test that classify_message_lm returns False if the OpenAI response doesn't contain "true" or "false"

        # Mock OpenAI response
        mock_openai.return_value = {"choices": [{"message": {"content": "Maybe."}}]}

        with self.assertRaises(ValueError):
            _ = self.check_question.classify_message_lm("Could you help me with this?")

    @patch("openai.ChatCompletion.create")
    def test_classify_message_lm_no_api_key(self, mock_openai):
        # Test that classify_message_lm raises an error if no API key is provided

        # Set API key to None (to simulate no API key in environment variables)
        self.check_question.api_key = None

        with self.assertRaises(ValueError):
            self.check_question.classify_message_lm("Where is the nearest hospital?")
