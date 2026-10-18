# [MODULE DOC]
# Tests pinning the example-documentation sync script (loaded by file path as a module at
# module scope): the readme converter (title extraction ignoring fenced comments, math and brace
# masking, escaping, linear-time handling of unterminated math, unpaired backticks, link
# rewriting to site pages, badge links, repo-escaping links, html tag handling, accordion
# conversion, void image tags), the index helpers (first sentence, index parsing, navigation
# ordering and grouping), page building over a miniature examples tree, and a drift gate that
# runs the script in check mode in a subprocess.
"""Pin the behavior of scripts/tools/sync_example_docs.py, the examples -> docs mirror.

The generator lives in scripts/tools because it writes files; these tests import it the
way tests/fast/examples/infra_features/p2p_weight_transfer/test_run.py imports run.py.
Each converter case here is a hazard the docs build either rejects (unescaped braces,
non-self-closed void tags) or silently drops (<details> content, corrupted inline math),
so the suite is what keeps README edits from breaking the published site.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "tools" / "sync_example_docs.py"


@pytest.fixture(scope="module")
def sync() -> ModuleType:
    # [FUNCTION DOC] HELPER: Module-scoped fixture. Builds a module spec from the script's file
    # path, creates and executes the module (not registered in the module table) and returns it.
    spec = importlib.util.spec_from_file_location("sync_example_docs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert(sync, readme, rel_dir="fully_async", mirrored=None, broken=None):
    # [FUNCTION DOC] HELPER: Calls the script's converter with the readme text, relative
    # directory (default "fully_async"), a mirrored-pages mapping (default empty) and a list that
    # collects broken-link reports (default a fresh empty list; pass one to inspect it). Returns
    # the converter's pair of title and body.
    return sync.convert(readme, rel_dir, mirrored or {}, [] if broken is None else broken)


class TestTitle:
    # [CLASS DOC] Title extraction tests.
    def test_fenced_comment_is_not_the_title(self, sync):
        # [FUNCTION DOC] A shell comment line inside a fenced block precedes the real level-1
        # heading; the title is the real heading and the comment line stays in the body.
        readme = "```bash\n# Download the model\nhf download x\n```\n\n# Real Title\n\nBody.\n"
        title, body = convert(sync, readme)
        assert title == "Real Title"
        assert "# Download the model" in body  # and the fence keeps its line

    def test_missing_title_is_an_error(self, sync):
        # [FUNCTION DOC] A readme with no level-1 heading raises the script's sync error whose
        # message matches "no level-1 heading".
        with pytest.raises(sync.SyncError, match="no level-1 heading"):
            convert(sync, "just prose\n")


class TestMasking:
    # [CLASS DOC] Tests of math masking and brace escaping in converted bodies.
    def test_inline_math_braces_survive(self, sync):
        # [FUNCTION DOC] Inline math with braces is left intact in the body and no escaped brace
        # appears anywhere.
        _, body = convert(sync, "# T\n\nRoles $\\pi_{\\text{old}}$ and $\\pi_{\\text{new}}$.\n")
        assert "$\\pi_{\\text{old}}$" in body
        assert "\\{" not in body

    def test_display_math_survives(self, sync):
        # [FUNCTION DOC] A display math block with braces appears unchanged in the body.
        _, body = convert(sync, "# T\n\n$$\nL_{\\text{PPO}}(\\theta)\n$$\n")
        assert "L_{\\text{PPO}}" in body

    def test_prose_braces_are_escaped(self, sync):
        # [FUNCTION DOC] In prose, a brace is backslash-escaped and an angle bracket opening an
        # unknown tag is converted to its entity form (the docs build rejects them raw).
        _, body = convert(sync, "# T\n\nParses <tool_call>{...}</tool_call> tokens.\n")
        assert "\\{" in body
        assert "&lt;tool_call>" in body

    def test_unterminated_math_span_is_linear(self, sync):
        # [FUNCTION DOC] A line with a dollar sign followed by 5000 repeated backslash-letter
        # pairs and no closing dollar must convert (no catastrophic backtracking) and keep the
        # text. The test relies on the call finishing; it has no explicit time limit.
        # Disjoint alternatives in the inline-math pattern; a pathological line must
        # fail to match in linear time rather than backtrack exponentially (CodeQL 133).
        readme = "# T\n\nPrice $" + "\\a" * 5000 + " end.\n"
        _, body = convert(sync, readme)
        assert "Price $" in body

    def test_unpaired_backtick_stays_within_its_paragraph(self, sync):
        # [FUNCTION DOC] An unpaired backtick in one paragraph must not hide the next paragraph:
        # a relative link to a script in the following paragraph is still reported as broken.
        readme = "# T\n\nBroken `tick here.\n\nNext [run](./run.py) paragraph.\n"
        broken = []
        _, body = convert(sync, readme, broken=broken)
        # The link in the following paragraph is still seen by the rewriter.
        assert any("run.py" in b for b in broken)


class TestLinks:
    # [CLASS DOC] Link rewriting tests.
    def test_mirrored_readme_resolves_to_site_page(self, sync):
        # [FUNCTION DOC] A relative link to a readme of a mirrored directory is rewritten to the
        # site page path with underscores turned into hyphens.
        mirrored = {"examples/fully_async": "fully_async"}
        _, body = convert(sync, "# T\n\nSee [it](./README.md).\n", mirrored=mirrored)
        assert "](/examples/fully-async)" in body

    def test_badge_in_link_rewrites_both_hrefs(self, sync):
        # [FUNCTION DOC] An image inside a link: the outer link is rewritten to the site page and
        # the inner image target (a missing local file) is reported as broken.
        mirrored = {"examples/fully_async": "fully_async"}
        broken = []
        _, body = convert(sync, "# T\n\n[![badge](./pic.png)](./README.md) end.\n", mirrored=mirrored, broken=broken)
        assert "](/examples/fully-async)" in body  # outer link
        assert any("pic.png" in b for b in broken)  # inner image checked too

    def test_repo_escaping_link_is_reported(self, sync):
        # [FUNCTION DOC] A relative link climbing out of the repository is reported with a message
        # saying it escapes the repository.
        broken = []
        convert(sync, "# T\n\nSee [up](../../../outside.md).\n", broken=broken)
        assert any("escapes the repository" in b for b in broken)


class TestHtml:
    # [CLASS DOC] Html handling tests.
    def test_common_tags_stay_markup(self, sync):
        # [FUNCTION DOC] A heading tag with an attribute remains markup and is not entity-escaped.
        _, body = convert(sync, '# T\n\n<h3 align="center">Hello</h3>\n')
        assert "<h3" in body and "&lt;h3" not in body

    def test_details_becomes_accordion_and_title_drops_braces(self, sync):
        # [FUNCTION DOC] A details block with a summary containing braces becomes an accordion
        # component whose title has the braces removed ("API v2"); the inner content is kept.
        readme = "# T\n\n<details>\n<summary>API {v2}</summary>\n\ncontent\n\n</details>\n"
        _, body = convert(sync, readme)
        assert '<Accordion title="API v2">' in body
        assert "content" in body

    def test_void_img_tag_becomes_markdown_image(self, sync):
        # [FUNCTION DOC] An img tag inside a centered paragraph tag is converted to a markdown
        # image with the alt text and source, and no img tag remains (void tags are rejected by
        # the docs build).
        _, body = convert(
            sync, '# T\n\n<p align="center">\n<img src="https://x.test/a.png" alt="A" width="800">\n</p>\n'
        )
        assert "![A](https://x.test/a.png)" in body
        assert "<img" not in body


class TestIndex:
    # [CLASS DOC] Tests of index parsing, description extraction and navigation building.
    def test_first_sentence_strips_markdown(self, sync):
        # [FUNCTION DOC] The first-sentence helper skips the title, removes bold markers and link
        # syntax (keeping link text) and stops after the first period.
        text = "# T\n\nThis **mirrors** [stuff](./run.py) nicely. Second sentence.\n"
        assert sync.first_sentence(text) == "This mirrors stuff nicely."

    def test_parse_index_reads_bullets_and_heading_links(self, sync, tmp_path):
        # [FUNCTION DOC] Parses an index file with a bullet entry (bold link plus description) and
        # a heading link. Returns a mapping from directory to description (bullets only) and a set
        # of all registered directories (bullets and headings).
        index = tmp_path / "README.md"
        index.write_text(
            "# Examples\n\n"
            "- **[fully_async](./fully_async)**: Async rollout.\n"
            "## [Infra Features](./infra_features)\n"
        )
        descriptions, registered = sync.parse_index(index)
        assert descriptions == {"fully_async": "Async rollout."}
        assert registered == {"fully_async", "infra_features"}

    def test_navigation_follows_bullet_order(self, sync):
        # [FUNCTION DOC] Navigation for pages {root, zeta, alpha, two unlisted} with bullet order
        # [zeta, alpha]: an index entry first, then one "Recipes" group listing zeta, alpha, then
        # unlisted pages alphabetically, with underscores shown as hyphens.
        pages = {"": None, "zeta": None, "alpha": None, "unlisted_b": None, "unlisted_a": None}
        entries = sync.build_navigation(pages, ["zeta", "alpha"])
        assert entries == [
            "examples/index",
            {
                "group": "Recipes",
                "pages": [
                    "examples/zeta",
                    "examples/alpha",
                    "examples/unlisted-a",
                    "examples/unlisted-b",
                ],
            },
        ]

    def test_section_directory_becomes_rootless_group_with_landing_first(self, sync):
        # [FUNCTION DOC] A directory with its own landing page and a child page becomes the last
        # navigation group, titled in title case ("Infra Features"), listing the landing page
        # before the child.
        pages = {"": None, "alpha": None, "infra_features": None, "infra_features/beta": None}
        entries = sync.build_navigation(pages, ["alpha"])
        assert entries[-1] == {
            "group": "Infra Features",
            "pages": ["examples/infra-features", "examples/infra-features/beta"],
        }


class TestBuildPages:
    # [CLASS DOC] Tests of the page builder over a miniature examples tree.
    def _mini_tree(self, tmp_path, monkeypatch, sync, index_text):
        # [FUNCTION DOC] HELPER: Creates an examples tree under the temporary path with an index
        # readme holding the given text and a "demo" directory with a small readme (title "Demo",
        # two sentences), and points the script module's examples root at it.
        examples = tmp_path / "examples"
        (examples / "demo").mkdir(parents=True)
        (examples / "README.md").write_text(index_text)
        (examples / "demo" / "README.md").write_text("# Demo\n\nA demo example. More prose.\n")
        monkeypatch.setattr(sync, "EXAMPLES", examples)

    def test_unregistered_directory_is_an_error(self, sync, tmp_path, monkeypatch):
        # [FUNCTION DOC] Building pages when the index lists no bullets raises the sync error
        # matching "not listed in examples/README.md".
        self._mini_tree(tmp_path, monkeypatch, sync, "# Examples\n\nNo bullets here.\n")
        with pytest.raises(sync.SyncError, match="not listed in examples/README.md"):
            sync.build_pages()

    def test_registered_directory_renders_with_its_bullet_description(self, sync, tmp_path, monkeypatch):
        # [FUNCTION DOC] With the demo directory registered by a bullet, building returns a triple
        # whose rendered pages include the demo page containing the front-matter description from
        # the bullet ("One-liner."), and whose order list is ["demo"].
        self._mini_tree(tmp_path, monkeypatch, sync, "# Examples\n\n- **[demo](./demo)**: One-liner.\n")
        _, rendered, order = sync.build_pages()
        page = next(text for path, text in rendered.items() if path.name == "demo.md")
        assert 'description: "One-liner."' in page
        assert order == ["demo"]


def test_docs_examples_matches_the_readmes():
    # [FUNCTION DOC] Runs the sync script in check mode with the current interpreter, from the
    # repository root, capturing text output (subprocess side effect; reads the repo's docs).
    # Asserts a zero exit status; on failure the message includes the script's error stream.
    """The committed mirror must be exactly what the generator produces (drift gate)."""
    result = subprocess.run([sys.executable, str(SCRIPT), "--check"], capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, f"docs/examples is out of sync with examples/:\n{result.stderr}"
