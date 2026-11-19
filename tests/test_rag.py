"""Lightweight tests using mocks; no model downloads required."""
import unittest
from unittest.mock import MagicMock, patch
import torch

from src.rag import RAGPipeline


class PipelineTests(unittest.TestCase):
    def test_empty_corpus_rejected(self):
        with self.assertRaises(ValueError):
            RAGPipeline(MagicMock(), MagicMock(), "cpu", MagicMock(), None, [])

    def test_retrieve_ranked_results(self):
        retriever = MagicMock()
        retriever.encode.side_effect = [
            torch.tensor([[1., 0.], [0., 1.]]),
            torch.tensor([[0.9, 0.1]]),
        ]
        with patch("src.rag.get_text_from_index", return_value={"d1": "one", "d2": "two"}):
            pipeline = RAGPipeline(retriever, MagicMock(), "cpu", MagicMock(), None, ["d1", "d2"])
        self.assertEqual(pipeline.retrieve("query"), [("d1", unittest.mock.ANY), ("d2", unittest.mock.ANY)])

    def test_generation_decodes_only_new_tokens(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt"
        inputs = {"input_ids": torch.tensor([[11, 12]])}
        class Encoded(dict):
            def to(self, device):
                return self
        tokenizer.return_value = Encoded(inputs)
        generator = MagicMock()
        generator.generate.return_value = torch.tensor([[11, 12, 42]])
        tokenizer.decode.return_value = "answer"
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline.generator, pipeline.gen_tokenizer, pipeline.device = generator, tokenizer, "cpu"
        self.assertEqual(pipeline.generate("q", "ctx"), "answer")
        self.assertEqual(tokenizer.decode.call_args.args[0].tolist(), [42])


if __name__ == "__main__":
    unittest.main()
