"""
Tests for ParsableContext per-turn usage tracking.
"""
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from vllm.outputs import CompletionOutput, RequestOutput


def _make_output(
    prompt_token_ids: list[int],
    output_token_ids: list[int],
    num_cached_tokens: int = 0,
    finished: bool = False,
    finish_reason: str | None = None,
) -> RequestOutput:
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=output_token_ids,
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=finish_reason,
    )
    return RequestOutput(
        request_id="test",
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
        num_cached_tokens=num_cached_tokens,
    )


def _make_parsable_context():
    """Build a ParsableContext with all dependencies mocked out."""
    # Mock the parser returned by get_responses_parser_for_simple_context
    mock_parser = MagicMock()
    mock_parser.finish_reason = None

    with patch(
        "vllm.entrypoints.openai.responses.context"
        ".get_responses_parser_for_simple_context",
        return_value=mock_parser,
    ):
        from vllm.entrypoints.openai.responses.context import ParsableContext

        ctx = ParsableContext.__new__(ParsableContext)

        # Manually initialise only the fields under test, bypassing the
        # heavyweight constructor (renderer, tokenizer, etc.).
        from vllm.entrypoints.openai.responses.context import TurnMetrics

        ctx.num_prompt_tokens = 0
        ctx.num_output_tokens = 0
        ctx.num_cached_tokens = 0
        ctx.num_reasoning_tokens = 0
        ctx.num_tool_output_tokens = 0
        ctx.all_turn_metrics = []
        ctx.current_turn_metrics = TurnMetrics()
        ctx.is_first_turn = True
        ctx._is_first_call_of_turn = True
        ctx.parser = mock_parser

        # Minimal request stub (enable_response_messages=False skips that branch)
        request_stub = MagicMock()
        request_stub.enable_response_messages = False
        ctx.request = request_stub

    return ctx


class TestParsableContextSingleTurn(unittest.TestCase):
    def setUp(self):
        self.ctx = _make_parsable_context()

    def test_single_streaming_chunk_not_finished(self):
        """Mid-stream chunk: no turn finalised yet."""
        output = _make_output(
            prompt_token_ids=list(range(10)),
            output_token_ids=[100, 101],
            num_cached_tokens=3,
            finished=False,
        )
        self.ctx.append_output(output)

        self.assertEqual(self.ctx.num_prompt_tokens, 10)
        self.assertEqual(self.ctx.num_output_tokens, 2)
        self.assertEqual(self.ctx.num_cached_tokens, 3)
        self.assertEqual(self.ctx.num_tool_output_tokens, 0)
        # Turn not finalised yet
        self.assertEqual(len(self.ctx.all_turn_metrics), 0)
        self.assertFalse(self.ctx._is_first_call_of_turn)

    def test_single_turn_finished(self):
        """Final chunk: turn is finalised and metrics recorded."""
        # First (non-final) chunk
        self.ctx.append_output(
            _make_output([0] * 10, [1, 2], num_cached_tokens=2, finished=False)
        )
        # Final chunk
        self.ctx.append_output(
            _make_output([0] * 10, [3], num_cached_tokens=2, finished=True)
        )

        self.assertEqual(self.ctx.num_prompt_tokens, 10)
        self.assertEqual(self.ctx.num_output_tokens, 3)   # 2 + 1
        self.assertEqual(self.ctx.num_cached_tokens, 2)
        self.assertEqual(self.ctx.num_tool_output_tokens, 0)

        self.assertEqual(len(self.ctx.all_turn_metrics), 1)
        turn = self.ctx.all_turn_metrics[0]
        self.assertEqual(turn.input_tokens, 10)
        self.assertEqual(turn.output_tokens, 3)
        self.assertEqual(turn.cached_input_tokens, 2)
        self.assertEqual(turn.tool_output_tokens, 0)

        # Flag reset for potential next turn
        self.assertTrue(self.ctx._is_first_call_of_turn)

    def test_prompt_tokens_counted_once_across_streaming_chunks(self):
        """Prompt tokens must not be double-counted for mid-stream chunks."""
        for i in range(5):
            finished = (i == 4)
            self.ctx.append_output(
                _make_output([0] * 20, [i], finished=finished)
            )

        # Still 20, not 20*5
        self.assertEqual(self.ctx.num_prompt_tokens, 20)
        self.assertEqual(self.ctx.num_output_tokens, 5)


class TestParsableContextMultiTurn(unittest.TestCase):
    def setUp(self):
        self.ctx = _make_parsable_context()

    def _run_turn(self, prompt_ids, output_ids, cached=0):
        """Simulate a complete generation turn (streaming)."""
        for i, tok in enumerate(output_ids):
            finished = i == len(output_ids) - 1
            self.ctx.append_output(
                _make_output(prompt_ids, [tok], num_cached_tokens=cached,
                             finished=finished)
            )

    def test_two_turns_tool_output_tokens(self):
        """
        Turn 1: prompt=10 tokens, output=4 tokens
        Turn 2: prompt=20 tokens (10 original + 4 output + 6 tool response)
        Expected tool_output_tokens = 20 - 10 - 4 = 6
        """
        self._run_turn(prompt_ids=list(range(10)), output_ids=[1, 2, 3, 4])
        self._run_turn(prompt_ids=list(range(20)), output_ids=[5, 6])

        self.assertEqual(len(self.ctx.all_turn_metrics), 2)

        t1 = self.ctx.all_turn_metrics[0]
        self.assertEqual(t1.input_tokens, 10)
        self.assertEqual(t1.output_tokens, 4)
        self.assertEqual(t1.tool_output_tokens, 0)

        t2 = self.ctx.all_turn_metrics[1]
        self.assertEqual(t2.input_tokens, 20)
        self.assertEqual(t2.output_tokens, 2)
        self.assertEqual(t2.tool_output_tokens, 6)   # 20 - 10 - 4

        self.assertEqual(self.ctx.num_tool_output_tokens, 6)
        self.assertEqual(self.ctx.num_prompt_tokens, 30)   # 10 + 20
        self.assertEqual(self.ctx.num_output_tokens, 6)    # 4 + 2

    def test_three_turns_accumulation(self):
        """Three tool-call turns accumulate correctly."""
        # Turn 1: prompt=10, output=5
        self._run_turn(list(range(10)), list(range(5)))
        # Turn 2: prompt=20 (+10 prompt, +5 output = 15; tool adds 5 → 20)
        self._run_turn(list(range(20)), list(range(3)))
        # Turn 3: prompt=30 (+20 prompt, +3 output = 23; tool adds 7 → 30)
        self._run_turn(list(range(30)), list(range(2)))

        self.assertEqual(len(self.ctx.all_turn_metrics), 3)

        # tool tokens turn 2: 20 - 10 - 5 = 5
        self.assertEqual(self.ctx.all_turn_metrics[1].tool_output_tokens, 5)
        # tool tokens turn 3: 30 - 20 - 3 = 7
        self.assertEqual(self.ctx.all_turn_metrics[2].tool_output_tokens, 7)

        self.assertEqual(self.ctx.num_tool_output_tokens, 12)  # 5 + 7
        self.assertEqual(self.ctx.num_prompt_tokens, 60)       # 10+20+30
        self.assertEqual(self.ctx.num_output_tokens, 10)       # 5+3+2

    def test_cached_tokens_accumulate_across_turns(self):
        self._run_turn(list(range(10)), [1, 2], cached=3)
        self._run_turn(list(range(15)), [3], cached=5)

        self.assertEqual(self.ctx.num_cached_tokens, 8)  # 3 + 5
        self.assertEqual(self.ctx.all_turn_metrics[0].cached_input_tokens, 3)
        self.assertEqual(self.ctx.all_turn_metrics[1].cached_input_tokens, 5)


class TestParsableContextNegativeToolTokensClampedToZero(unittest.TestCase):
    def test_negative_tool_tokens_clamped(self):
        """
        If the prompt shrinks (shouldn't happen but guard exists), tool tokens
        should be clamped to 0, not go negative.
        """
        ctx = _make_parsable_context()

        # Turn 1: prompt=20, output=5
        for i, tok in enumerate([1, 2, 3, 4, 5]):
            ctx.append_output(
                _make_output(list(range(20)), [tok], finished=(i == 4))
            )

        # Turn 2: prompt=10 (smaller — pathological case)
        ctx.append_output(
            _make_output(list(range(10)), [6], finished=True)
        )

        self.assertEqual(ctx.all_turn_metrics[1].tool_output_tokens, 0)
        self.assertEqual(ctx.num_tool_output_tokens, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)