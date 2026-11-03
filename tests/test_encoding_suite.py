# NOESIS-Σ Encoding, Reasoning Budget, DSML Tool-Call & Examples Golden Test Suite
# Path: tests/test_encoding_suite.py

from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import patch
import pytest

import encoding as enc
from test_encoding import *  # noqa: F401,F403 — exposes all 55 encoding & examples tests to pytest tests/


def test_encoding_all_validation_and_edge_branches(tmp_path: Path) -> None:
    """Cover all validation errors, fallback branches, and DSML parser edge cases in core/encoding/encoding.py."""
    # Lines 80-81: to_json fallback when ensure_ascii=False raises
    orig_dumps = json.dumps
    calls = 0

    def flaky_dumps(val, **kw):
        nonlocal calls
        calls += 1
        if calls == 1 and not kw.get("ensure_ascii", True):
            raise UnicodeEncodeError("utf-8", "", 0, 1, "mock")
        return orig_dumps(val, **kw)

    with patch.object(enc.json, "dumps", side_effect=flaky_dumps):
        assert enc.to_json({"a": 1}) == '{"a": 1}'

    # Line 263: sort_tool_results_by_call_order with mixed text and multiple tool_result blocks
    msgs_sort = [
        {"role": "assistant", "tool_calls": [{"id": "c1"}, {"id": "c2"}]},
        {
            "role": "user",
            "content_blocks": [
                {"type": "text", "text": "intro"},
                {"type": "tool_result", "tool_use_id": "c2", "content": "r2"},
                {"type": "tool_result", "tool_use_id": "c1", "content": "r1"},
            ],
        },
    ]
    sorted_msgs = enc.sort_tool_results_by_call_order(msgs_sort)
    assert sorted_msgs[1]["content_blocks"][1]["tool_use_id"] == "c1"

    # Lines 278, 289: parse_tagged_text malformed & empty path errors
    with pytest.raises(ValueError, match="Malformed"):
        enc.parse_tagged_text("hello <image>unclosed")
    with pytest.raises(ValueError, match="must not be empty"):
        enc.parse_tagged_text("hello <image></image>")

    # Lines 311, 315-317, 319: _extract_image string image_url, image dict keys, and invalid source error
    assert enc._extract_image({"type": "image_url", "image_url": "a.png"})["url"] == "a.png"
    assert enc._extract_image({"type": "image", "data": "abc"})["data"] == "abc"
    with pytest.raises(ValueError, match="valid source"):
        enc._extract_image({"type": "image"})

    # Lines 331-332, 345: _process_image_blocks non-dict block & placeholder in text block
    blks, _ = enc._process_image_blocks(["raw_str"])
    assert blks == ["raw_str"]
    with pytest.raises(ValueError, match="Text block contains image placeholder"):
        enc._process_image_blocks([{"type": "text", "text": f"bad {enc.IMAGE_PLACEHOLDER}"}])

    # Line 365: _validate_no_image_sp_tokens in reasoning_content
    with pytest.raises(ValueError, match="reasoning_content"):
        enc._validate_no_image_sp_tokens({"role": "assistant", "reasoning_content": enc.IMAGE_PLACEHOLDER})

    # Line 381: process_image_messages when content is a list and content_blocks not in msg
    proc, imgs = enc.process_image_messages([{"role": "user", "content": [{"type": "image_url", "image_url": "x.png"}]}])
    assert len(imgs) == 1

    # Lines 419-420: _read_until_stop when no stop string matches
    idx, content, matched = enc._read_until_stop(0, "hello world", ["NOT_FOUND"])
    assert idx == 11 and content == "hello world" and matched is None

    # Lines 539-540, 544: encode_arguments_to_dsml with non-JSON string and non-dict arguments
    dsml_non_json = enc.encode_arguments_to_dsml({"arguments": "{not_valid_json"})
    assert 'name="arguments"' in dsml_non_json
    dsml_list_arg = enc.encode_arguments_to_dsml({"arguments": [1, 2]})
    assert 'name="arguments"' in dsml_list_arg

    # Lines 629, 650, 654, 663, 698, 710: render_message branches
    # response_format on system message
    sys_rf = enc.render_message(
        0,
        [{"role": "system", "content": "sys", "response_format": {"type": "json_object"}}],
        thinking_mode="chat",
    )
    assert "json_object" in sys_rf

    # Unsupported block inside tool_result and top-level content_blocks
    u_unsupported = enc.render_message(
        0,
        [
            {
                "role": "user",
                "content_blocks": [
                    {"type": "tool_result", "content": [{"type": "audio"}]},
                    {"type": "video"},
                ],
            }
        ],
        thinking_mode="chat",
    )
    assert "[Unsupported audio]" in u_unsupported and "[Unsupported video]" in u_unsupported

    # Standalone tool role error in render_message
    with pytest.raises(NotImplementedError, match="merge_tool_messages"):
        enc.render_message(0, [{"role": "tool", "content": "x"}], thinking_mode="chat")

    # wo_eos=True on assistant message
    asst_wo = enc.render_message(
        1,
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "partial", "wo_eos": True}],
        thinking_mode="chat",
    )
    assert not asst_wo.endswith(enc.eos_token)

    # Unknown role error
    with pytest.raises(NotImplementedError, match="Unknown role"):
        enc.render_message(0, [{"role": "alien", "content": "x"}], thinking_mode="chat")

    # Line 861: load_cases empty messages with tools
    bad_case_file = tmp_path / "bad_case.json"
    bad_case_file.write_text(json.dumps({"messages": [], "tools": [{"function": {"name": "f"}}]}))
    with pytest.raises(ValueError, match="at least one message"):
        enc.load_cases(str(bad_case_file))

    # Lines 908, 914, 920, 929, 933, 938: parse_tool_calls error branches
    with pytest.raises(ValueError, match="Tool call format error"):
        enc.parse_tool_calls(0, "bad_prefix")
    with pytest.raises(ValueError, match="Missing special token"):
        enc.parse_tool_calls(0, ">\n")
    with pytest.raises(ValueError, match="Tool name format error"):
        enc.parse_tool_calls(0, f">\n<{enc.dsml_token}{enc.tool_call_tag_name} bad_name</{enc.dsml_token}{enc.tool_call_tag_name}")
    with pytest.raises(ValueError, match="Parameter format error"):
        enc.parse_tool_calls(
            0,
            f'>\n<{enc.dsml_token}{enc.tool_call_tag_name} name="fn">\n'
            f'<{enc.dsml_token}{enc.tool_parameter_tag_name} bad_param/{enc.dsml_token}{enc.tool_parameter_tag_name}',
        )
    with pytest.raises(ValueError, match="Duplicate parameter name"):
        enc.parse_tool_calls(
            0,
            f'>\n<{enc.dsml_token}{enc.tool_call_tag_name} name="fn">\n'
            f'<{enc.dsml_token}{enc.tool_parameter_tag_name} name="x" string="true">1</{enc.dsml_token}{enc.tool_parameter_tag_name}>\n'
            f'<{enc.dsml_token}{enc.tool_parameter_tag_name} name="x" string="true">2</{enc.dsml_token}{enc.tool_parameter_tag_name}>\n',
        )
    with pytest.raises(ValueError, match="expected '>\\\\n'"):
        enc.parse_tool_calls(
            0,
            f'>\n<{enc.dsml_token}{enc.tool_call_tag_name} name="fn">\n'
            f'<{enc.dsml_token}{enc.tool_parameter_tag_name} name="x" string="true">1</{enc.dsml_token}{enc.tool_parameter_tag_name}BAD'
            f'</{enc.dsml_token}{enc.tool_call_tag_name}',
        )
