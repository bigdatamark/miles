# [MODULE DOC] Tests for inferring the inline-block function extraction operation in the proof
# generator. Each test uses the shared repository fixture to commit a base version and a second
# version of a single module with a class, in which a block of statements of one method is moved
# into a new helper method and replaced by a call, then infers the recipe for the latest commit
# and checks the recorded extraction (and, in some tests, replays it against the commit).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from generator_testlib import _commit, _git, _write  # noqa: F401
from mechanical_refactor_proof_generator import build_repro, infer_recipe, recipe_to_script


def test_infer_extract_function_with_returned_local(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: class C whose dispatch method contains a conditional block that
    # builds a value named pool through a multi-line call, followed by a return of pool, plus a
    # sibling method keep. Second commit: that block becomes a new method _build_pool (keyword-
    # only parameter n) that ends by returning pool, and dispatch calls it. Asserts for the head
    # commit: recipe supported; no symbol moves; exactly one function extraction with name
    # _build_pool, source and destination both the module, enclosing class C, insertion anchor
    # keep, block indentation 12, body equal to the verbatim original block text, call text equal
    # to the new assignment line, return text equal to the helper's return line, and signature
    # equal to the helper's header line.
    """A block ending in ``pool = make(...)`` carved into a helper that returns ``pool`` infers
    an extract_function whose body is the verbatim block and whose return_text is authored.
    """
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            x = self.a\n"
                "            y = x + n\n"
                "            pool = make(\n"
                "                a=x,\n"
                "                b=y,\n"
                "            )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(n=n)\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        x = self.a\n"
                "        y = x + n\n"
                "        pool = make(\n"
                "            a=x,\n"
                "            b=y,\n"
                "        )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "extract _build_pool from dispatch")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert recipe.moves == []
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["name"] == "_build_pool"
    assert ex["src"] == "kv.py" and ex["dst"] == "kv.py"
    assert ex["into_class"] == "C"
    assert ex["before"] == "keep"
    assert ex["body_indent"] == 12
    assert ex["body"] == (
        "            x = self.a\n"
        "            y = x + n\n"
        "            pool = make(\n"
        "                a=x,\n"
        "                b=y,\n"
        "            )\n"
    )
    assert ex["call"] == "            pool = self._build_pool(n=n)\n"
    assert ex["return_text"] == "        return pool"
    assert ex["signature"] == "    def _build_pool(self, *, n):\n"


def test_infer_extract_function_keeps_leading_comment_in_body(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: a method with a conditional block whose first line is a
    # comment, then two statements. Second commit extracts the block, comment included, into a
    # helper method returning pool. Asserts exactly one extraction whose signature is only the
    # header line (no comment), whose body starts with the comment, and that replaying the
    # recipe against the commit leaves an empty residual diff.
    """A block whose first line is a comment extracts with that comment in the body, not
    absorbed into the authored signature (which is the def header through its colon only).
    """
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        if self.flag:\n"
                "            # pick the pool class for this backend\n"
                "            cls = PoolA\n"
                "            pool = cls(n)\n"
                "        return pool\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(n=n)\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        # pick the pool class for this backend\n"
                "        cls = PoolA\n"
                "        pool = cls(n)\n"
                "        return pool\n"
            )
        },
    )
    commit = _commit(repo, "extract _build_pool with a leading comment")
    recipe = infer_recipe(commit, str(repo))
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["signature"] == "    def _build_pool(self, *, n):\n"
    assert ex["body"].lstrip().startswith("# pick the pool class")
    assert build_repro(recipe, repo_root=str(repo)).run() == ""


def test_infer_extract_function_no_return_text_when_body_is_whole_helper(
    repo: Path,
) -> None:
    # [FUNCTION DOC] Base commit: a run method calling pre, two log calls, and post. Second
    # commit moves the two log calls into a new helper _emit with no return, replaced by a call
    # to it. Asserts the recipe is supported, has exactly one extraction named _emit, its return
    # text is nothing, and its call text is the single call line.
    """When the helper body reproduces the source block with no trailing return, return_text
    is None (the block is a side-effecting statement sequence, not a value producer)."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self.log(1)\n"
                "        self.log(2)\n"
                "        self.post()\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self._emit()\n"
                "        self.post()\n"
                "\n"
                "    def _emit(self):\n"
                "        self.log(1)\n"
                "        self.log(2)\n"
            )
        },
    )
    _commit(repo, "extract _emit from run")
    recipe = infer_recipe("HEAD", str(repo))
    assert recipe.supported
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["name"] == "_emit"
    assert ex["return_text"] is None
    assert ex["call"] == "        self._emit()\n"


def test_infer_extract_function_edited_body_does_not_pass(repo: Path) -> None:
    # [FUNCTION DOC] Base commit: run calls pre, log with argument 1, and post. Second commit
    # extracts the log call into helper _emit but changes its argument to 2. Infers the recipe
    # for that commit, replays it, and asserts the residual diff is non-empty (no false pass).
    """A helper whose body was edited (not a verbatim cut) never yields a false pass: the
    reproduction's byte-diff surfaces the bundled change as a non-empty residual."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self.log(1)\n"
                "        self.post()\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def run(self):\n"
                "        self.pre()\n"
                "        self._emit()\n"
                "        self.post()\n"
                "\n"
                "    def _emit(self):\n"
                "        self.log(2)\n"
            )
        },
    )
    commit = _commit(repo, "extract _emit but change the arg")
    recipe = infer_recipe(commit, str(repo))
    residual = build_repro(recipe, repo_root=str(repo)).run()
    assert residual != ""


def test_infer_extract_function_when_block_and_call_share_closing_paren(
    repo: Path,
) -> None:
    # [FUNCTION DOC] Base commit: dispatch assigns pool from a multi-line pool-creating call
    # inside a conditional. Second commit replaces it by a multi-line call to a new helper
    # _build_pool, whose body is the original call followed by a return of pool. Both the removed
    # block and its replacement end with a lone closing parenthesis line. Asserts exactly one
    # extraction, whose body (right-trimmed) still ends with the closing parenthesis, whose
    # return text is the helper's return line, and that replaying leaves an empty residual.
    """The removed block and its replacement call both end in a lone ``)``; the prefix/suffix
    split must not absorb that shared line, or the extracted body loses its final line.
    """
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        if self.flag:\n"
                "            pool = make_pool(\n"
                "                a=n,\n"
                "                b=self.b,\n"
                "            )\n"
                "        return pool\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(\n"
                "                n=n,\n"
                "            )\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        pool = make_pool(\n"
                "            a=n,\n"
                "            b=self.b,\n"
                "        )\n"
                "        return pool\n"
            )
        },
    )
    commit = _commit(repo, "extract _build_pool")
    recipe = infer_recipe(commit, str(repo))
    assert len(recipe.extract_functions) == 1
    ex = recipe.extract_functions[0]
    assert ex["body"].rstrip().endswith(")")
    assert ex["return_text"] == "        return pool"
    assert build_repro(recipe, repo_root=str(repo)).run() == ""


def test_emitted_script_passes_on_extract_function(repo: Path, tmp_path: Path) -> None:
    # [FUNCTION DOC] Base commit: dispatch with a conditional block building pool, plus a keep
    # method. Second commit extracts the block into a helper placed before keep. Infers the
    # recipe, replays it and asserts the residual is empty (the message shows the residual on
    # failure); also asserts the rendered script text mentions the function-extraction call.
    # The temporary path fixture argument is unused.
    """The recipe for an extract_function reproduces the commit byte-for-byte (bare repo, no
    formatter) so build_repro returns an empty residual."""
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            x = self.a\n"
                "            pool = make(\n"
                "                a=x,\n"
                "            )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    _commit(repo, "base")
    _write(
        repo,
        **{
            "kv.py": (
                "class C:\n"
                "    def dispatch(self, n):\n"
                "        base = self.setup()\n"
                "        if self.flag:\n"
                "            pool = self._build_pool(n=n)\n"
                "        return pool\n"
                "\n"
                "    def _build_pool(self, *, n):\n"
                "        x = self.a\n"
                "        pool = make(\n"
                "            a=x,\n"
                "        )\n"
                "        return pool\n"
                "\n"
                "    def keep(self):\n"
                "        return 0\n"
            )
        },
    )
    commit = _commit(repo, "extract _build_pool from dispatch")
    recipe = infer_recipe(commit, str(repo))
    residual = build_repro(recipe, repo_root=str(repo)).run()
    assert residual == "", residual
    assert "extract_function" in recipe_to_script(recipe, "extract")
