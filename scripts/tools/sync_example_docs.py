#!/usr/bin/env python3
# [MODULE DOC] Tool that mirrors every README under the examples directory (except experimental
# ones) into generated Markdown pages under the docs examples folder, and regenerates the Examples
# tab of the docs navigation JSON file. Pipeline: discover READMEs; read the index README for
# one-line descriptions and the registry of listed directories; convert each README (mask code
# regions, extract the title, drop comments and excluded blocks, rewrite images, links, details
# blocks, void tags, unknown tags and braces for the strict MDX parser, restore code); render a
# page with front matter; build the navigation entries. The main function either writes changed
# files and deletes stale pages, or in check mode only reports drift. Module-level constants hold
# paths, GitHub base URLs, the allowed HTML tag set, void tag set, section directories and the
# compiled regular expressions used by the converters.
"""Mirror the README files under examples/ into the Examples tab of the docs site.

examples/ is the single source of truth. Every README.md outside examples/experimental/
becomes one page under docs/examples/, and the Examples tab in docs/docs.json is
regenerated from the same tree, so the site cannot drift from the repository.

Usage:
    python scripts/tools/sync_example_docs.py            # regenerate
    python scripts/tools/sync_example_docs.py --check    # fail if anything is stale

The generated pages are Mintlify MDX, which is stricter than GitHub-flavored Markdown.
Two constructs in a README break the build and are rewritten here: unescaped braces
(parsed as JSX expressions) and non-self-closing void tags such as <img> and <br>.
Relative links and images are rewritten to the site page when the target is mirrored,
and to GitHub otherwise.

docs/docs.json is round-tripped through json.dumps on every run, so this script owns
that file's formatting (indent=1); hand-edits to other tabs keep their content but are
renormalized to that style.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXAMPLES = REPO / "examples"
DOCS = REPO / "docs"
DOCS_JSON = DOCS / "docs.json"
OUT_DIR = DOCS / "examples"

BRANCH = "main"
GITHUB_TREE = f"https://github.com/radixark/miles/tree/{BRANCH}"
GITHUB_BLOB = f"https://github.com/radixark/miles/blob/{BRANCH}"
GITHUB_RAW = f"https://raw.githubusercontent.com/radixark/miles/{BRANCH}"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
MAX_DESCRIPTION = 160

# Tags Mintlify renders as JSX. Anything else in prose (<tool_call>, <search>, ...) is a
# model token, not markup, and is escaped so acorn never sees it.
HTML_TAGS = {
    "a",
    "b",
    "blockquote",
    "br",
    "code",
    "details",
    "div",
    "em",
    "hr",
    "i",
    "img",
    "kbd",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "sub",
    "summary",
    "sup",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "caption",
    "center",
    "dd",
    "del",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "ins",
    "mark",
    "picture",
    "s",
    "small",
    "source",
    "u",
    "video",
    # Mintlify's own component, which <details> blocks are rewritten into.
    "accordion",
}
VOID_TAGS = {"br", "hr", "img", "input"}

# - **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation.
INDEX_BULLET = re.compile(r"^\s*[-*]\s+\*\*\[([^\]]+)\]\(([^)]+)\)\*\*:\s*(.+?)\s*$")
# ## [Infra Features](./infra_features) — a section directory registered as a heading link.
INDEX_HEADING = re.compile(r"^#{2,}\s+\[([^\]]+)\]\(([^)]+)\)\s*$")

# Directories rendered as label-only sidebar sections. Their own README becomes the
# section's first row, labeled "Overview" — never a group `root`, which would make the
# header itself navigate (banned by pre-commit; see docs/README.md).
SECTION_DIRS = ("infra_features",)
MD_LINK = re.compile(r"(!?)\[([^\]]*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
IMG_LINK = re.compile(r"!\[([^\]]*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
# Link text may carry one nested image ([![badge](img)](target)), which MD_LINK would
# mis-parse as text "![badge" with href "img".
OUTER_LINK = re.compile(r"(?<!!)\[((?:[^\[\]\n]|!\[[^\]]*\]\([^)]*\))*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)")
HTML_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9:-]*)((?:\"[^\"]*\"|'[^']*'|[^>\"'])*?)(/?)>")
IMG_TAG = re.compile(r"<img\b((?:\"[^\"]*\"|'[^']*'|[^>\"'])*)/?>", re.IGNORECASE)
ATTR = re.compile(r"([A-Za-z-]+)\s*=\s*\"([^\"]*)\"|([A-Za-z-]+)\s*=\s*'([^']*)'")
DETAILS = re.compile(r"<details[^>]*>\s*(?:<summary[^>]*>(.*?)</summary>)?(.*?)</details>", re.DOTALL | re.IGNORECASE)
# Content between these markers stays on GitHub but is left out of the docs site.
EXCLUDE_BLOCK = re.compile(r"<!--\s*docs:exclude:start\s*-->.*?<!--\s*docs:exclude:end\s*-->", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE = re.compile(r"^\s*(```+|~~~+)")


class SyncError(Exception):
    # [CLASS DOC] Error raised for any problem with the sources that the user must fix (missing
    # index or title, unlisted directory, bad description, slug collision, broken link, missing
    # navigation tab). The main function catches it, prints it and exits with status one.
    pass


def discover_pages():
    # [FUNCTION DOC] Finds every README file under the examples directory (sorted recursive
    # search), skipping any whose directory path includes an experimental or bytecode-cache
    # component. Returns a mapping from the slash-joined directory path relative to examples (empty
    # string for the root) to the README file path. Reads the filesystem only.
    """Map each mirrored directory (relative to examples/, "" for the root) to its README."""
    pages = {}
    for readme in sorted(EXAMPLES.rglob("README.md")):
        rel = readme.parent.relative_to(EXAMPLES)
        parts = [] if rel == Path(".") else list(rel.parts)
        if "experimental" in parts or "__pycache__" in parts:
            continue
        pages["/".join(parts)] = readme
    return pages


def slug_for(rel_dir):
    # [FUNCTION DOC] Maps a relative example directory to its page slug: the root becomes "index";
    # otherwise each path component is lower-cased with underscores turned into dashes, joined by
    # slashes.
    if not rel_dir:
        return "index"
    return "/".join(part.replace("_", "-").lower() for part in rel_dir.split("/"))


def site_url(rel_dir):
    # [FUNCTION DOC] Returns the docs site URL path of a directory's page: the examples root path
    # for the root, else the examples path followed by the slug.
    return "/examples" if not rel_dir else f"/examples/{slug_for(rel_dir)}"


def repo_dir_of(rel_dir):
    # [FUNCTION DOC] Returns the repository-relative path of an example directory: the examples
    # folder itself for the root, else examples joined with the relative directory.
    return "examples" if not rel_dir else f"examples/{rel_dir}"


def parse_index(index_readme):
    # [FUNCTION DOC] Scans the index README line by line. A line matching the bullet pattern (bold
    # link followed by a colon and description) or, failing that, the section-heading link pattern
    # registers a directory: the link target is stripped of any fragment and trailing slash,
    # joined under examples and normalized; targets that do not stay inside examples are skipped.
    # The directory (relative to examples) is added to the registered set, and for bullets its
    # description is stored (insertion order equals bullet order). Returns the descriptions mapping
    # and the registered set.
    """Read examples/README.md, the registry for every mirrored page.

    Returns (descriptions, registered): one-line descriptions keyed by directory in
    bullet order, and the set of every directory the index mentions — as a bullet or,
    for a section directory like infra_features, as a section-heading link.
    """
    descriptions, registered = {}, set()
    for line in index_readme.read_text().splitlines():
        bullet = INDEX_BULLET.match(line)
        heading = None if bullet else INDEX_HEADING.match(line)
        m = bullet or heading
        if not m:
            continue
        target = m.group(2).split("#")[0].strip().rstrip("/")
        rel = os.path.normpath(os.path.join("examples", target))
        if not rel.startswith("examples/"):
            continue
        rel_dir = rel[len("examples/") :]
        registered.add(rel_dir)
        if bullet:
            descriptions[rel_dir] = bullet.group(3)
    return descriptions, registered


def first_sentence(text):
    # [FUNCTION DOC] Extracts the first sentence of the first prose paragraph. Skips fenced code and
    # heading lines; collects stripped non-blank lines until a blank line ends the first paragraph;
    # joins them with spaces, replaces markdown links by their text (or target if the text is
    # empty), deletes emphasis, code and underscore characters, then returns the text up to the
    # first sentence-ending punctuation followed by whitespace or end (or the whole paragraph if
    # none), trimmed. Returns an empty string when there is no prose.
    """First prose sentence of a README, used when the index has no bullet for a page."""
    body = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence or line.startswith("#"):
            continue
        if line.strip():
            body.append(line.strip())
        elif body:
            break
    para = " ".join(body)
    para = MD_LINK.sub(lambda m: m.group(2) or m.group(3), para)
    para = re.sub(r"[*`_]", "", para)
    match = re.search(r"^(.+?[.!?])(\s|$)", para)
    return (match.group(1) if match else para).strip()


def mask_code(text):
    # [FUNCTION DOC] Hides regions that must not be rewritten behind numbered placeholders (a null
    # character, the index, a null character). Order: fenced code blocks found line by line (a
    # block closes on a line that starts with the same fence marker; an unclosed block runs to the
    # end of text), then display math between double dollar signs (may span lines), then inline
    # code spans (may wrap a line but not cross a blank line), then single-line inline math. Returns
    # the masked text and the list of stashed chunks in placeholder order. Pure function.
    # KNOWN ISSUE: the fence-closing check only requires the line to start with the opening marker,
    # so a longer closing fence or trailing text still closes it.
    """Replace fenced blocks, inline code and display math with placeholders."""
    stash = []

    def keep(chunk):
        # [FUNCTION DOC] HELPER: Appends the chunk to the stash and returns its placeholder string.
        stash.append(chunk)
        return f"\x00{len(stash) - 1}\x00"

    out_lines = []
    fence = None
    buffer = []
    for line in text.split("\n"):
        m = FENCE.match(line)
        if fence is None and m:
            fence = m.group(1)
            buffer = [line]
        elif fence is not None:
            buffer.append(line)
            if m and line.strip().startswith(fence):
                out_lines.append(keep("\n".join(buffer)))
                fence = None
        else:
            out_lines.append(line)
    if fence is not None:
        out_lines.append(keep("\n".join(buffer)))
    text = "\n".join(out_lines)

    text = re.sub(r"\$\$.*?\$\$", lambda m: keep(m.group(0)), text, flags=re.DOTALL)
    # A code span may wrap a line but not a paragraph, so an unpaired backtick cannot
    # swallow the rest of the document into the stash.
    text = re.sub(r"(`+)((?:(?!\n\n)[^`])+?)\1", lambda m: keep(m.group(0)), text)
    # Inline math, bounded to one line so a stray "$5" in prose stays inert. The two
    # alternatives are disjoint (an escape, or anything but a backslash), so an
    # unterminated span cannot trigger exponential backtracking.
    text = re.sub(r"(?<!\$)\$(?!\$)(?:\\[^\n]|[^\\$\n])+\$(?!\$)", lambda m: keep(m.group(0)), text)
    return text, stash


def unmask_code(text, stash):
    # [FUNCTION DOC] Restores the stashed chunks in place of their placeholders. Repeats the
    # substitution while any placeholder marker character remains, since stashed chunks may
    # themselves contain placeholders. Returns the restored text.
    # A stashed chunk can itself contain a placeholder (inline math around a code
    # span, say), so expand until none remain.
    while "\x00" in text:
        text = re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], text)
    return text


def resolve_link(href, cur_dir, mirrored, broken):
    # [FUNCTION DOC] Decides the replacement for a link target found in the README of directory
    # cur_dir. Returns none (keep unchanged) for empty targets, pure fragments, absolute paths and
    # URLs with a scheme, and for links that escape the repository (those are also appended to the
    # broken list with a message). Otherwise the target is joined to the README's directory and
    # normalized; a path to a mirrored directory (or its README) becomes the site page URL with the
    # fragment kept; a docs source Markdown/MDX file becomes its published path (dropping an
    # index suffix); an existing directory becomes a GitHub tree URL; an existing image file a
    # GitHub raw URL (fragment dropped); any other existing file a GitHub blob URL; a missing
    # target is appended to the broken list and none is returned. Reads the filesystem.
    """Rewrite a repo-relative href to a site page or a GitHub URL. None keeps it as is."""
    if not href or href.startswith(("#", "/")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", href):
        return None
    path, sep, fragment = href.partition("#")
    if not path:
        return None
    rel = os.path.normpath(os.path.join(repo_dir_of(cur_dir), path))
    if rel.startswith(".."):
        broken.append(f"{repo_dir_of(cur_dir)}/README.md -> {href} (escapes the repository)")
        return None
    suffix = "#" + fragment if sep else ""

    target = rel[: -len("/README.md")] if rel.endswith("/README.md") else rel
    if target in mirrored:
        return site_url(mirrored[target]) + suffix

    absolute = REPO / rel
    # A README pointing at a docs source file should point at the published page instead.
    if absolute.is_file() and rel.startswith("docs/") and absolute.suffix in {".md", ".mdx"}:
        page = rel[len("docs/") : -len(absolute.suffix)]
        return f"/{page[: -len('/index')] if page.endswith('/index') else page}{suffix}"
    if absolute.is_dir():
        return f"{GITHUB_TREE}/{rel}{suffix}"
    if absolute.is_file():
        if absolute.suffix.lower() in IMAGE_SUFFIXES:
            return f"{GITHUB_RAW}/{rel}"
        return f"{GITHUB_BLOB}/{rel}{suffix}"
    broken.append(f"{repo_dir_of(cur_dir)}/README.md -> {href}")
    return None


def convert_img_tags(text, cur_dir, mirrored, broken):
    # [FUNCTION DOC] Replaces each HTML image tag with a markdown image on its own paragraph, using
    # the tag's alt text and its source resolved with the link resolver (original source kept when
    # unresolved); width and other attributes are dropped. Afterwards removes all paragraph
    # open/close tags (replaced by a newline), blanks whitespace-only lines, and collapses runs of
    # three or more newlines into two. Returns the new text; may add to the broken list.
    """<img src=... alt=...> is not self-closing in most READMEs; markdown carries it fine."""

    def replace(match):
        # [FUNCTION DOC] HELPER: Substitution callback for one image tag: parses its quoted
        # attributes (lower-cased names; either quote style), resolves the source, and returns the
        # markdown image surrounded by blank lines.
        attrs = dict()
        for m in ATTR.finditer(match.group(1)):
            key = (m.group(1) or m.group(3)).lower()
            attrs[key] = m.group(2) if m.group(2) is not None else m.group(4)
        src = attrs.get("src", "")
        resolved = resolve_link(src, cur_dir, mirrored, broken)
        # A row of side-by-side images relies on the width attributes we just dropped, so
        # give each one its own paragraph rather than letting them run into each other.
        return f"\n\n![{attrs.get('alt', '')}]({resolved or src})\n\n"

    text = IMG_TAG.sub(replace, text)
    # The wrapper <p align="center"> around those images carries no meaning in markdown.
    text = re.sub(r"</?p\b[^>]*>", "\n", text)
    text = re.sub(r"^[ \t]+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text)


def convert_details_blocks(text):
    # [FUNCTION DOC] Rewrites every details block into an Accordion component block, with the
    # summary text (default "Details") as title and the stripped inner content as the body.
    # Returns the new text.
    """A <details> block renders empty on the docs site; Mintlify's Accordion keeps it."""

    def replace(match):
        # [FUNCTION DOC] HELPER: Substitution callback for one details block: cleans the summary
        # (removes emphasis, code, underscore, brace characters and any tags, trims, turns double
        # quotes into single quotes) to form the title and wraps the inner content.
        summary, inner = match.group(1) or "Details", match.group(2)
        # Braces would be escaped to \{ later and render literally inside the attribute.
        title = re.sub(r"[*`_{}]|<[^>]+>", "", summary).strip().replace('"', "'")
        return f'\n\n<Accordion title="{title}">\n\n{inner.strip()}\n\n</Accordion>\n\n'

    return DETAILS.sub(replace, text)


def escape_html(text):
    # [FUNCTION DOC] Makes tag-like text safe for MDX. For each tag match: a tag whose lower-cased
    # name is not in the allowed set gets its opening angle bracket replaced by an escaped entity
    # (so it renders as text); an allowed void tag that is neither self-closed nor a closing tag
    # is rewritten as self-closed with its attributes right-trimmed; everything else is unchanged.
    # Returns the new text.
    """Self-close void tags; escape anything that is a model token rather than markup."""

    def replace(match):
        # [FUNCTION DOC] HELPER: Substitution callback for one tag; applies the rules above.
        closing, name, attrs, self_closed = match.groups()
        if name.lower() not in HTML_TAGS:
            return "&lt;" + match.group(0)[1:]
        if name.lower() in VOID_TAGS and not self_closed and not closing:
            return f"<{name}{attrs.rstrip()} />"
        return match.group(0)

    return HTML_TAG.sub(replace, text)


def convert(readme_text, rel_dir, mirrored, broken):
    # [FUNCTION DOC] Converts one README into a page title and MDX-safe body. Steps in order:
    # remove docs-excluded blocks; mask code regions; find the first line starting with a level-1
    # heading marker, take it as the title (unmasked) and delete it (raises a sync error if none);
    # strip HTML comments; convert image tags; rewrite markdown image links and then link
    # (outer-link pattern handles one nested image) targets through the link resolver, keeping
    # titles; convert details blocks; escape HTML; escape curly braces with a backslash; collapse
    # three or more newlines; restore code; strip leading and trailing newlines. Returns the title
    # and body. May add messages to the broken list.
    readme_text = EXCLUDE_BLOCK.sub("", readme_text)
    # Mask before anything else, including the title scan: a "# comment" inside a
    # fenced block must not be mistaken for the page's level-1 heading.
    masked, stash = mask_code(readme_text)
    lines = masked.split("\n")
    title = None
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = unmask_code(line[2:].strip(), stash)
            del lines[i]
            break
    if title is None:
        raise SyncError(f"{repo_dir_of(rel_dir)}/README.md has no level-1 heading to use as the page title")
    body = "\n".join(lines)

    # GitHub-only annotations; also raw comments are not valid MDX.
    body = HTML_COMMENT.sub("", body)
    body = convert_img_tags(body, rel_dir, mirrored, broken)

    def rewrite(bang):
        # [FUNCTION DOC] HELPER: Builds a substitution callback for links; the argument is the
        # prefix ("!" for images, empty for plain links) that is re-emitted before the bracketed text.
        def replace(match):
            # [FUNCTION DOC] HELPER: Rebuilds one link with its resolved target (or the original)
            # and the optional title hint.
            text, href, hint = match.groups()
            resolved = resolve_link(href, rel_dir, mirrored, broken)
            return f"{bang}[{text}]({resolved or href}{hint or ''})"

        return replace

    body = IMG_LINK.sub(rewrite("!"), body)
    body = OUTER_LINK.sub(rewrite(""), body)
    body = convert_details_blocks(body)
    body = escape_html(body)
    body = body.replace("{", "\\{").replace("}", "\\}")
    body = re.sub(r"\n{3,}", "\n\n", body)
    body = unmask_code(body, stash)
    return title, body.strip("\n")


def render_page(title, description, rel_dir, body):
    # [FUNCTION DOC] Produces the full text of a generated page: front matter with the title and
    # description as quoted strings (non-ASCII kept), a sidebar title of "Overview" for the root and
    # section directories only, a comment naming the source README and the generator, then the body
    # and a final newline.
    source = f"{repo_dir_of(rel_dir)}/README.md"
    # Tab and section landing pages keep their full title (h1 / SEO) but show a short,
    # uniform "Overview" label in the sidebar (see the convention in docs/README.md).
    landing = rel_dir == "" or rel_dir in SECTION_DIRS
    sidebar = 'sidebarTitle: "Overview"\n' if landing else ""
    return (
        "---\n"
        f"title: {json.dumps(title, ensure_ascii=False)}\n"
        f"{sidebar}"
        f"description: {json.dumps(description, ensure_ascii=False)}\n"
        f"# Generated from {source} by scripts/tools/sync_example_docs.py. Edit that README, not this file.\n"
        "---\n"
        f"{body}\n"
    )


def build_pages():
    # [FUNCTION DOC] Builds every generated page in memory. Raises a sync error when the root
    # README is missing, when mirrored directories are not registered in the index (message lists
    # them), when a page has no description (index bullet, else the README's first sentence), when a
    # description exceeds 160 characters, on a slug collision, or when any link is broken (message
    # lists sorted unique entries). When the description came from the first sentence and the
    # converted body starts with the identical sentence (after stripping markdown), that leading
    # sentence is removed from the body to avoid duplication. Returns the pages mapping, a mapping
    # from output path to page text, and the list of directories in bullet order.
    pages = discover_pages()
    if "" not in pages:
        raise SyncError("examples/README.md is missing; it is the source of the Examples index page")
    mirrored = {repo_dir_of(rel): rel for rel in pages}
    descriptions, registered = parse_index(pages[""])

    unregistered = sorted(rel_dir for rel_dir in pages if rel_dir and rel_dir not in registered)
    if unregistered:
        raise SyncError(
            "mirrored but not listed in examples/README.md — add a bullet for:\n  "
            + "\n  ".join(repo_dir_of(d) for d in unregistered)
        )

    broken, rendered, slug_owner = [], {}, {}
    for rel_dir, readme in sorted(pages.items()):
        text = readme.read_text()
        title, body = convert(text, rel_dir, mirrored, broken)
        description = descriptions.get(rel_dir)
        if description is None:
            # Derived from the README's own first sentence, which Mintlify already renders
            # under the title as the description — drop the duplicate from the body. The
            # comparison strips markdown the same way first_sentence does, so a link or
            # emphasis in the opening sentence does not defeat the dedup.
            description = first_sentence(text)
            lead = re.match(r"\s*(.+?[.!?])(\s|$)", body, re.DOTALL)
            if lead:
                normalized = MD_LINK.sub(lambda m: m.group(2) or m.group(3), lead.group(1))
                if re.sub(r"[*`_]", "", normalized).strip() == description:
                    body = body[lead.end(1) :].lstrip()
        if not description:
            raise SyncError(f"{repo_dir_of(rel_dir)}/README.md has no description; add a bullet in examples/README.md")
        if len(description) > MAX_DESCRIPTION:
            raise SyncError(
                f"description for {repo_dir_of(rel_dir)} is {len(description)} characters, "
                f"over the {MAX_DESCRIPTION} the docs site allows; shorten it at the source"
            )
        out_path = OUT_DIR / f"{slug_for(rel_dir)}.md"
        if out_path in rendered:
            raise SyncError(
                f"slug collision: {repo_dir_of(slug_owner[out_path])} and {repo_dir_of(rel_dir)} "
                f"both map to {out_path.relative_to(REPO)}"
            )
        slug_owner[out_path] = rel_dir
        rendered[out_path] = render_page(title, description, rel_dir, body)

    if broken:
        raise SyncError("READMEs link to paths that do not exist:\n  " + "\n  ".join(sorted(set(broken))))
    # dicts preserve insertion order, so this is the bullet order of examples/README.md.
    return pages, rendered, list(descriptions)


def build_navigation(pages, bullet_order):
    # [FUNCTION DOC] Builds the list of navigation entries for the Examples tab. Directories are
    # sorted by rank in bullet order (unlisted ones last, then alphabetically); the root is
    # skipped; those under infra_features go to the infra list, others to recipes. Result: the
    # index page, a Recipes group, and, if any infra pages exist, an Infra Features group with the
    # infra landing page first then the rest.
    """Examples tab, mirroring the directory layout: top-level recipes, then infra_features.

    Sidebar order follows the bullet order in examples/README.md — the index README owns
    ordering along with titles and descriptions. Directories without a bullet sort last,
    alphabetically.

    Emits the tab's `pages` list directly: the index page first, then label-only
    sections whose landing page is their first row — no group `root` (see docs/README.md).
    """
    rank = {rel_dir: i for i, rel_dir in enumerate(bullet_order)}
    recipes, infra = [], []
    for rel_dir in sorted(pages, key=lambda d: (rank.get(d, len(rank)), d)):
        if not rel_dir:
            continue
        page = f"examples/{slug_for(rel_dir)}"
        (infra if rel_dir.startswith("infra_features") else recipes).append(page)
    entries = ["examples/index"]
    entries.append({"group": "Recipes", "pages": recipes})
    if infra:
        infra_root = "examples/infra-features"
        children = [p for p in infra if p != infra_root]
        entries.append({"group": "Infra Features", "pages": [infra_root] + children})
    return entries


def examples_tab(config):
    # [FUNCTION DOC] Returns the tab object named Examples from the navigation tabs of the parsed
    # docs config; raises a sync error if there is none.
    for tab in config["navigation"]["tabs"]:
        if tab.get("tab") == "Examples":
            return tab
    raise SyncError('docs.json has no "Examples" tab')


def render_docs_json(config):
    # [FUNCTION DOC] Serializes the docs config to JSON text with one-space indentation, non-ASCII
    # characters kept, and a trailing newline.
    return json.dumps(config, indent=1, ensure_ascii=False) + "\n"


def main():
    # [FUNCTION DOC] Command-line entry point returning a process exit code. Option: a check flag.
    # Builds pages (a sync error is printed to standard error as an error line and returns 1),
    # loads the docs JSON, replaces the Examples tab's pages with the new navigation and removes its
    # old groups key, renders the JSON. Computes stale generated pages (existing files under the
    # output directory not produced now), changed pages (missing or different content) and whether
    # the JSON changed. In check mode: prints each problem to standard error with a hint to rerun
    # the tool and returns 1 if any, else 0, writing nothing. Otherwise: deletes stale pages,
    # writes all pages (creating directories), rewrites the JSON if changed, prints removed/wrote
    # lines for each, and returns 0.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail instead of writing when something is stale")
    args = parser.parse_args()

    try:
        pages, rendered, bullet_order = build_pages()
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    config = json.loads(DOCS_JSON.read_text())
    try:
        tab = examples_tab(config)
    except SyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    tab.pop("groups", None)
    tab["pages"] = build_navigation(pages, bullet_order)
    docs_json_text = render_docs_json(config)

    existing = {p for p in OUT_DIR.rglob("*.md")} if OUT_DIR.exists() else set()
    stale = sorted(existing - set(rendered))
    changed = sorted(p for p, text in rendered.items() if not p.exists() or p.read_text() != text)
    json_changed = DOCS_JSON.read_text() != docs_json_text

    if args.check:
        problems = [f"stale: {p.relative_to(REPO)}" for p in stale]
        problems += [f"out of date: {p.relative_to(REPO)}" for p in changed]
        if json_changed:
            problems.append("out of date: docs/docs.json")
        if problems:
            print("error: docs/examples is out of sync with examples/:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            print("run: python scripts/tools/sync_example_docs.py", file=sys.stderr)
            return 1
        return 0

    for path in stale:
        path.unlink()
    for path, text in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    if json_changed:
        DOCS_JSON.write_text(docs_json_text)

    for path in stale:
        print(f"removed {path.relative_to(REPO)}")
    for path in changed:
        print(f"wrote {path.relative_to(REPO)}")
    if json_changed:
        print("wrote docs/docs.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
