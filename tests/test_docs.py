"""Documentation drift guards.

Two READMEs (English and Chinese) describe the same CLI, so the realistic failure is a
subcommand, a tier or a flag being added to the code while one of the two documents keeps
describing the old set.  These tests fail when that happens, in either language.

They deliberately check the *reverse* direction from a hand-written list -- "everything the
parser registers is documented" -- because that is the drift that actually occurs: nobody adds
a subcommand and forgets to document it on purpose, but it happens on the way past.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from qwen25vl_lite.cli import build_parser
from qwen25vl_lite.vision_budget import TIERS

REPO = Path(__file__).resolve().parent.parent
READMES = {"README.md": REPO / "README.md", "README.zh-CN.md": REPO / "README.zh-CN.md"}

# The flags a reader is most likely to copy straight out of the READMEs.  If one is renamed,
# both documents are wrong in a way that is invisible until someone runs the command.
DOCUMENTED_SERVE_FLAGS = [
    "--model", "--device", "--dtype", "--weight-dtype", "--group-size",
    "--kv-dtype", "--num-blocks", "--block-size", "--max-model-len",
    "--max-num-seqs", "--max-num-batched-tokens", "--vision-tier",
    "--int4-kernel", "--paged-attn-kernel", "--cuda-graph",
    "--attn-backend", "--preemption", "--no-prefix-cache", "--no-chunked-prefill",
    "--fastv", "--fastv-ratio", "--fastv-layer", "--quantize-kv",
]


def _subparsers():
    """The name -> subparser mapping, without importing argparse's private class name."""
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001 - argparse exposes no public accessor
        if action.__class__.__name__ == "_SubParsersAction":
            return action.choices
    raise AssertionError("build_parser() registered no subcommands")


@pytest.mark.parametrize("name", sorted(READMES))
def test_every_subcommand_is_documented(name):
    text = READMES[name].read_text(encoding="utf-8")
    missing = [cmd for cmd in sorted(_subparsers()) if f"`{cmd}`" not in text]
    assert not missing, f"{name} does not document: {missing}"


@pytest.mark.parametrize("name", sorted(READMES))
def test_every_vision_tier_is_documented(name):
    text = READMES[name].read_text(encoding="utf-8")
    missing = [tier for tier in sorted(TIERS) if tier not in text]
    assert not missing, f"{name} does not mention vision tier(s): {missing}"


@pytest.mark.parametrize("name", sorted(READMES))
def test_the_documented_serve_flags_all_exist(name):
    serve = _subparsers()["serve"]
    known = {opt for action in serve._actions for opt in action.option_strings}  # noqa: SLF001
    text = READMES[name].read_text(encoding="utf-8")
    documented = [flag for flag in DOCUMENTED_SERVE_FLAGS if flag in text]
    assert documented, f"{name} documents none of the expected serve flags"

    unknown = [flag for flag in documented if flag not in known]
    assert not unknown, f"{name} documents flag(s) the serve parser does not accept: {unknown}"


def test_the_camera_vision_budget_flags_are_documented():
    """These two are the only way to shrink the vision budget over HTTP -- the server's own
    --vision-tier flag does not set the image pixel budget -- so a document that omits them
    leaves the reader with no working lever."""
    camera = _subparsers()["camera"]
    known = {opt for action in camera._actions for opt in action.option_strings}  # noqa: SLF001
    assert {"--vision-tier", "--max-vision-tokens", "--width"} <= known

    for name, path in READMES.items():
        text = path.read_text(encoding="utf-8")
        for flag in ("--max-vision-tokens", "--width"):
            assert flag in text, f"{name} does not document {flag}"
