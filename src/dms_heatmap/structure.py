"""Secondary structure for the tracks drawn under the heatmap.

The heatmap answers "what happens at this position"; the secondary structure
answers "where is this position".  Putting the two on one figure, sharing an x
axis, is what lets a reader see that a stripe of intolerance is a buried helix
face rather than an arbitrary run of residues.

Everything here reduces a structure to one **simplified three-state code per
residue** -- helix (``H``), strand (``E``), loop (``L``) -- which is what
SSDraw-style diagrams draw, plus ``-`` for positions the model does not cover.
Four inputs are accepted, because "the structure" arrives in whatever form the
prediction or the PDB entry came in:

===========================  =================================================
``*.dssp``                   DSSP output; the eight-state code is collapsed
``*.pdb`` / ``*.ent``        ``HELIX``/``SHEET`` records, else DSSP is run
``*.cif`` / ``*.mmcif``      the ``_struct_conf`` loop, else DSSP is run
``*.ss``                     a bare ``HHHEEELL`` string, optionally FASTA-style
===========================  =================================================

Coordinate files with no secondary-structure annotation at all -- which is what
most predicted models are -- need ``mkdssp`` (or ``dssp``) on ``PATH``.  It is
called only as a fallback and its absence is a warning, never an error.

**Numbering is checked, not assumed.**  A structure numbered by its own author
numbering, a model of one domain, or a construct with a tag all put residue *i*
of the model at a different position of the assayed protein, and a silently
shifted secondary-structure track is worse than none at all.  So when the
dataset carries a wild-type protein sequence, :meth:`SecondaryStructure.aligned_to`
finds the offset that matches the model's own sequence to it and reports the
identity it achieved; a structure that cannot be placed says so instead of
guessing.
"""

from __future__ import annotations

import logging
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "HELIX",
    "LOOP",
    "STRAND",
    "STRUCTURE_PATTERNS",
    "UNMODELLED",
    "Run",
    "SecondaryStructure",
    "StructureError",
    "discover_structures",
    "match_structures",
    "read_structure",
]

log = logging.getLogger("dms_heatmap")

#: The three states an SSDraw-style diagram draws.
HELIX = "H"
STRAND = "E"
LOOP = "L"

#: A position the structure does not cover: outside the model, or a residue the
#: model leaves out.  Distinct from ``LOOP``, which is a positive observation.
UNMODELLED = "-"

#: DSSP's eight states collapsed to three.  Helices of every pitch draw as a
#: helix and both strand states as a strand; turns and bends are loop, because
#: a diagram that drew them differently would imply more than DSSP asserts.
_DSSP_SIMPLE = {
    "H": HELIX,   # alpha helix
    "G": HELIX,   # 3-10 helix
    "I": HELIX,   # pi helix
    "E": STRAND,  # extended strand in a sheet
    "B": STRAND,  # residue in an isolated beta bridge
    "T": LOOP,    # hydrogen-bonded turn
    "S": LOOP,    # bend
    "P": LOOP,    # kappa helix (polyproline II), DSSP 4 only
}

#: What the same three states are called in a mmCIF ``_struct_conf`` loop.
_CIF_PREFIXES = ((("HELX",), HELIX), (("STRN", "BETA", "SHEET"), STRAND))

#: Accepted in a bare ``*.ss`` string, so DSSP output, PSIPRED-style ``C`` and a
#: dashed loop all read the same way.
_SS_TEXT = {**_DSSP_SIMPLE, "C": LOOP, "L": LOOP, "-": LOOP, ".": LOOP, " ": LOOP}

#: Filename globs searched for structures.
STRUCTURE_PATTERNS: tuple[str, ...] = (
    "*.dssp",
    "*.pdb",
    "*.ent",
    "*.cif",
    "*.mmcif",
    "*.ss",
)

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M",
}

#: Sequence identity at which an offset is taken as certainly right.
_ACCEPT_IDENTITY = 0.90

#: Identity below which the best offset found is rejected as no alignment.
_MIN_IDENTITY = 0.80

#: Fraction of the model that must land inside the assayed protein for an
#: offset to be considered at all.  A handful of residues can match anywhere.
_MIN_COVERAGE = 0.50


class StructureError(ValueError):
    """A structure file cannot be read, or cannot be placed on the protein."""


@dataclass(frozen=True)
class Run:
    """A stretch of consecutive positions in the same state.

    A run is clipped to the window it was asked for, which for a wrapped
    heatmap is one block of the protein.  ``origin`` is where the element
    really starts and ``open_end`` says it really continues past ``end``, so a
    helix split across two blocks can keep the phase of its coil and a strand
    does not grow a second arrowhead at the fold.
    """

    code: str
    start: int
    end: int
    origin: int | None = None
    open_end: bool = False

    def __post_init__(self) -> None:
        if self.origin is None:
            object.__setattr__(self, "origin", self.start)

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True)
class SecondaryStructure:
    """Three-state secondary structure, placed on the assayed protein.

    ``codes[i]`` is the state of the residue numbered ``resnums[i]`` in the
    structure's own numbering; ``offset`` is what must be added to that to get
    a position in the dataset, and :meth:`code_map` does it.
    """

    name: str
    source: Path | None
    codes: tuple[str, ...]
    resnums: tuple[int, ...]
    sequence: str
    chain: str | None = None
    offset: int = 0
    identity: float | None = None
    #: Provenance: how this structure was read and placed.  Recorded, not raised.
    notes: tuple[str, ...] = ()
    #: Things the reader should know about: an unverifiable numbering, a
    #: doubtful placement, a track that will draw as flat loop.  These reach
    #: the run's warning list and the warning count in ``summary.tsv``.
    warnings: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.codes)

    def code_map(self) -> dict[int, str]:
        """State by dataset position."""
        return {num + self.offset: code for num, code in zip(self.resnums, self.codes)}

    def runs(self, start: int, end: int) -> list[Run]:
        """Segment positions ``start..end`` into runs of one state each.

        Positions the model does not cover come back as ``UNMODELLED`` runs
        rather than being dropped, so a drawn track is continuous and a gap in
        the model is visible as a gap rather than as a loop.  The first and
        last runs are marked as clipped when the element they belong to
        extends beyond the window.
        """
        codes = self.code_map()
        runs: list[Run] = []
        for pos in range(start, end + 1):
            code = codes.get(pos, UNMODELLED)
            if runs and runs[-1].code == code and runs[-1].end == pos - 1:
                runs[-1] = Run(code, runs[-1].start, pos, runs[-1].origin)
            else:
                runs.append(Run(code, pos, pos))
        if not runs:
            return runs

        # Walk back to where the first element really began.  Only for a
        # modelled state: ``UNMODELLED`` is the default for every position in
        # and out of the model, so walking it back would never terminate.
        origin = runs[0].start
        if runs[0].code != UNMODELLED:
            while codes.get(origin - 1, UNMODELLED) == runs[0].code:
                origin -= 1
        runs[0] = Run(runs[0].code, runs[0].start, runs[0].end, origin)

        last = runs[-1]
        open_end = codes.get(last.end + 1, UNMODELLED) == last.code
        runs[-1] = Run(last.code, last.start, last.end, last.origin, open_end)
        return runs

    def composition(self) -> dict[str, float]:
        """Fraction of modelled residues in each state."""
        total = len(self.codes)
        if not total:
            return {}
        return {
            state: round(self.codes.count(state) / total, 4)
            for state in (HELIX, STRAND, LOOP)
        }

    def aligned_to(
        self, wt_sequence: str | None, offset: int | None = None
    ) -> "SecondaryStructure":
        """Place this structure on ``wt_sequence``, returning a placed copy.

        An explicit ``offset`` is obeyed and merely scored.  Otherwise the
        offset is searched for: the structure's own numbering first, then the
        model sequence as a substring of the protein, then every offset that
        puts enough of the model inside the protein.  Raises
        :class:`StructureError` if nothing reaches
        :data:`_MIN_IDENTITY`, because the alternative is a track that is
        confidently wrong.
        """
        if offset is not None:
            identity, matched = self._score(wt_sequence, offset)
            if identity is None:
                return self._placed(
                    offset,
                    None,
                    f"numbering offset {offset:+d} applied as given",
                    warning="the structure's numbering could not be checked "
                    "against the assayed protein, so a shifted track would go "
                    "unnoticed",
                )
            note = (
                f"numbering offset {offset:+d} applied as given; {identity:.0%} "
                f"sequence identity over {matched} residue(s)"
            )
            return self._placed(
                offset,
                identity,
                note,
                warning=(
                    f"the structure matches only {identity:.0%} of the assayed "
                    f"protein at the offset given ({offset:+d}); it may not be "
                    "this protein, or the offset may be wrong"
                    if identity < _MIN_IDENTITY
                    else None
                ),
            )

        if not wt_sequence or not self.sequence.strip("X"):
            return self._placed(
                0,
                None,
                "structure numbering used as-is",
                warning="no sequence to verify the structure's numbering against, "
                "so the strips assume its residue numbers are dataset positions; "
                "pass --structure-offset if they are not",
            )

        best_offset, best_identity, matched = self._search(wt_sequence)
        if best_identity is None or best_identity < _MIN_IDENTITY:
            raise StructureError(
                f"cannot place {self.source.name if self.source else self.name} on the "
                f"assayed protein: the best numbering offset matches only "
                f"{(best_identity or 0):.0%} of residues; pass an explicit "
                f"--structure-offset if the numbering is known"
            )
        note = (
            f"placed by sequence at offset {best_offset:+d} "
            f"({best_identity:.0%} identity over {matched} residue(s))"
        )
        return self._placed(best_offset, best_identity, note)

    def as_dict(self) -> dict:
        return {
            "source": str(self.source) if self.source else None,
            "chain": self.chain,
            "n_residues": len(self.codes),
            "first_position": min(self.code_map(), default=None),
            "last_position": max(self.code_map(), default=None),
            "offset": self.offset,
            "sequence_identity": self.identity,
            "composition": self.composition(),
            "notes": list(self.notes),
            "warnings": list(self.warnings),
        }

    # -- placement internals ------------------------------------------------

    def _placed(
        self,
        offset: int,
        identity: float | None,
        note: str,
        warning: str | None = None,
    ):
        from dataclasses import replace

        return replace(
            self,
            offset=offset,
            identity=identity,
            notes=(*self.notes, note),
            warnings=self.warnings + ((warning,) if warning else ()),
        )

    def _score(self, wt_sequence: str | None, offset: int) -> tuple[float | None, int]:
        """(identity, residues compared) for one candidate offset."""
        if not wt_sequence:
            return None, 0
        matched = compared = 0
        for num, aa in zip(self.resnums, self.sequence):
            index = num + offset - 1
            if aa == "X" or not 0 <= index < len(wt_sequence):
                continue
            compared += 1
            matched += wt_sequence[index] == aa
        if compared == 0:
            return None, 0
        return matched / compared, compared

    def _search(self, wt_sequence: str) -> tuple[int, float | None, int]:
        """Best offset placing this structure on ``wt_sequence``."""
        best = (0, None, 0)

        def consider(offset: int) -> bool:
            nonlocal best
            identity, compared = self._score(wt_sequence, offset)
            if identity is None or compared < _MIN_COVERAGE * len(self.codes):
                return False
            if best[1] is None or identity > best[1]:
                best = (offset, identity, compared)
            return identity >= _ACCEPT_IDENTITY

        # The structure's own numbering, then the model sequence read as a
        # contiguous substring of the protein -- between them these cover a
        # full-length model and a domain model, which is nearly every case.
        if consider(0):
            return best
        found = wt_sequence.find(self.sequence)
        if found >= 0 and consider(found + 1 - self.resnums[0]):
            return best

        low = 1 - max(self.resnums)
        high = len(wt_sequence) - min(self.resnums)
        for offset in range(low, high + 1):
            if consider(offset):
                break
        return best


# -- discovery -------------------------------------------------------------


def discover_structures(
    root: Path, patterns: tuple[str, ...] = STRUCTURE_PATTERNS
) -> list[Path]:
    """Recursively find candidate structure files under ``root`` (or ``root``)."""
    root = Path(root)
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise StructureError(f"structure path does not exist: {root}")
    found: set[Path] = set()
    for pattern in patterns:
        found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found)


def match_structures(paths: list[Path], names: list[str]) -> dict[str, Path]:
    """Pair structure files with dataset names by filename.

    A structure belongs to dataset ``tna1`` if its stem is ``tna1`` or starts
    with ``tna1`` followed by a separator (``tna1_alphafold.pdb``), matched
    case-insensitively.  The one exception is a single structure and a single
    dataset, which are paired whatever they are called -- that is what makes
    ``--input one.tsv --structures model.pdb`` behave the way it reads.

    Where one dataset has several candidates, the format that carries the most
    secondary-structure information wins, in the order of
    :data:`STRUCTURE_PATTERNS`: a DSSP file states every residue's state, a
    coordinate file only its helices and sheets, and a bare string has no
    sequence to check the numbering against.
    """
    if len(paths) == 1 and len(names) == 1:
        return {names[0]: paths[0]}

    order = {p.lstrip("*"): index for index, p in enumerate(STRUCTURE_PATTERNS)}

    def preference(path: Path) -> tuple[int, str]:
        return (order.get(path.suffix.lower(), len(order)), str(path))

    out: dict[str, Path] = {}
    for name in names:
        key = name.lower()
        candidates = [
            path
            for path in paths
            if (stem := path.stem.lower()) == key
            or (stem.startswith(key) and not stem[len(key) :][:1].isalnum())
        ]
        if candidates:
            out[name] = min(candidates, key=preference)
    return out


# -- reading ---------------------------------------------------------------


def read_structure(
    path: Path, chain: str | None = None, name: str | None = None
) -> SecondaryStructure:
    """Read one structure file into three-state codes.

    The returned structure is *unplaced* (offset 0); call
    :meth:`SecondaryStructure.aligned_to` to put it on the assayed protein.
    """
    path = Path(path)
    if not path.is_file():
        raise StructureError(f"structure file does not exist: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(errors="replace")
    notes: list[str] = []
    warnings: list[str] = []

    if suffix == ".dssp":
        records = _parse_dssp(text, chain)
    elif suffix == ".ss":
        records = _parse_ss_text(text)
    elif suffix in {".pdb", ".ent"}:
        records = _parse_pdb(text, chain)
        if records and not _annotated(records):
            records = _via_dssp(path, chain, records, notes, warnings)
    elif suffix in {".cif", ".mmcif"}:
        records = _parse_mmcif(text, chain)
        if records and not _annotated(records):
            records = _via_dssp(path, chain, records, notes, warnings)
    else:
        raise StructureError(
            f"{path.name}: unsupported structure format {suffix!r}; expected one of "
            + ", ".join(p.lstrip("*") for p in STRUCTURE_PATTERNS)
        )

    if not records:
        raise StructureError(f"{path.name}: no residues found")

    chains = {c for c, _, _, _ in records if c}
    return SecondaryStructure(
        name=name or path.stem,
        source=path,
        codes=tuple(code for _, _, _, code in records),
        resnums=tuple(num for _, num, _, _ in records),
        sequence="".join(aa for _, _, aa, _ in records),
        chain=sorted(chains)[0] if len(chains) == 1 else chain,
        notes=tuple(notes),
        warnings=tuple(warnings),
    )


#: A residue record: (chain, residue number, one-letter amino acid, state).
_Record = tuple[str, int, str, str]


def _annotated(records: list[_Record]) -> bool:
    """Whether anything but loop was actually asserted."""
    return any(code in {HELIX, STRAND} for _, _, _, code in records)


def _via_dssp(
    path: Path,
    chain: str | None,
    fallback: list[_Record],
    notes: list[str],
    warnings: list[str],
) -> list[_Record]:
    """Re-read ``path`` through DSSP, or keep ``fallback`` and say why not.

    A coordinate file with no secondary-structure records is the normal case
    for a predicted model, so this is a routine path and not an error: without
    DSSP the track would be a flat loop from end to end, which says nothing, so
    the caller gets a warning it can put in the manifest.
    """
    text = _run_dssp(path)
    if text is None:
        warnings.append(
            f"{path.name} carries no secondary-structure records and neither mkdssp "
            "nor dssp is on PATH, so every residue reads as loop; install DSSP or "
            "supply a .dssp/.ss file to get real secondary structure"
        )
        return fallback
    notes.append(f"secondary structure assigned by running DSSP on {path.name}")
    return _parse_dssp(text, chain)


def _run_dssp(path: Path) -> str | None:
    """DSSP output for ``path``, or ``None`` if DSSP cannot be run.

    DSSP 4 renamed the flag that selects classic output and insists on an
    explicit destination, while DSSP 2/3 took neither, so each calling
    convention is tried in turn.
    """
    executable = shutil.which("mkdssp") or shutil.which("dssp")
    if executable is None:
        return None

    with tempfile.TemporaryDirectory() as scratch:
        out = Path(scratch) / "out.dssp"
        attempts = (
            [executable, "--output-format", "dssp", str(path), str(out)],
            [executable, str(path), str(out)],
            [executable, "-i", str(path), "-o", str(out)],
        )
        for command in attempts:
            try:
                done = subprocess.run(
                    command, capture_output=True, text=True, timeout=600, check=False
                )
            except (OSError, subprocess.SubprocessError):
                continue
            if done.returncode == 0 and out.is_file() and out.stat().st_size:
                return out.read_text(errors="replace")
            log.debug("DSSP call failed: %s: %s", " ".join(command), done.stderr[:200])
    return None


def _select_chain(records: list[_Record], chain: str | None) -> list[_Record]:
    """Keep one chain: the requested one, or the first that appears.

    Silently concatenating chains would run two different numbering schemes
    into one track, so a multi-chain file without ``--structure-chain`` takes
    the first chain rather than all of them.
    """
    if not records:
        return records
    wanted = chain or records[0][0]
    kept = [r for r in records if r[0] == wanted]
    if not kept:
        present = sorted({r[0] for r in records})
        raise StructureError(
            f"chain {chain!r} not in the structure; found {', '.join(present) or 'none'}"
        )
    return kept


def _deduplicate(records: list[_Record]) -> list[_Record]:
    """One record per residue number, keeping the first seen."""
    seen: set[int] = set()
    out: list[_Record] = []
    for record in records:
        if record[1] in seen:
            continue
        seen.add(record[1])
        out.append(record)
    return out


def _parse_dssp(text: str, chain: str | None = None) -> list[_Record]:
    """Residue records from DSSP output."""
    lines = text.splitlines()
    start = next(
        (i + 1 for i, line in enumerate(lines) if line.lstrip().startswith("#  RESIDUE")),
        None,
    )
    if start is None:
        raise StructureError("not DSSP output: no '#  RESIDUE' header line")

    records: list[_Record] = []
    for line in lines[start:]:
        if len(line) < 17 or line[13] == "!":  # '!' marks a chain break
            continue
        try:
            resnum = int(line[5:10])
        except ValueError:
            continue
        amino = line[13].upper()
        records.append(
            (
                line[11],
                resnum,
                amino if amino.isalpha() else "X",
                _DSSP_SIMPLE.get(line[16], LOOP),
            )
        )
    return _deduplicate(_select_chain(records, chain))


def _parse_pdb(text: str, chain: str | None = None) -> list[_Record]:
    """Residue records from a PDB file's CA atoms and HELIX/SHEET records."""
    residues: list[_Record] = []
    spans: list[tuple[str, int, int, str]] = []

    for line in text.splitlines():
        tag = line[:6]
        if tag == "ENDMDL":
            break  # first model only; NMR ensembles repeat every residue
        if tag == "HELIX ":
            span = _pdb_span(line, 19, 21, 25, 33, 37, HELIX)
            if span:
                spans.append(span)
        elif tag == "SHEET ":
            span = _pdb_span(line, 21, 22, 26, 33, 37, STRAND)
            if span:
                spans.append(span)
        elif tag in {"ATOM  ", "HETATM"} and line[12:16].strip() == "CA":
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            residues.append(
                (
                    line[21],
                    resnum,
                    _THREE_TO_ONE.get(line[17:20].strip().upper(), "X"),
                    LOOP,
                )
            )

    return _apply_spans(_deduplicate(_select_chain(residues, chain)), spans)


def _pdb_span(
    line: str, chain_col: int, from_a: int, from_b: int, to_a: int, to_b: int, code: str
):
    """One HELIX/SHEET record as (chain, start, end, code), or ``None``."""
    try:
        return (line[chain_col], int(line[from_a:from_b]), int(line[to_a:to_b]), code)
    except (ValueError, IndexError):
        return None


def _parse_mmcif(text: str, chain: str | None = None) -> list[_Record]:
    """Residue records from an mmCIF's ``_atom_site`` and ``_struct_conf``."""
    residues: list[_Record] = []
    for row in _cif_category(text, "atom_site"):
        if row.get("label_atom_id", "").strip('"') != "CA":
            continue
        if row.get("pdbx_PDB_model_num", "1") != "1":
            continue
        try:
            resnum = int(row.get("auth_seq_id") or row["label_seq_id"])
        except (KeyError, ValueError):
            continue
        residues.append(
            (
                row.get("auth_asym_id") or row.get("label_asym_id", ""),
                resnum,
                _THREE_TO_ONE.get(row.get("label_comp_id", "").upper(), "X"),
                LOOP,
            )
        )

    spans: list[tuple[str, int, int, str]] = []
    for row in _cif_category(text, "struct_conf"):
        kind = row.get("conf_type_id", "").upper()
        code = next(
            (c for prefixes, c in _CIF_PREFIXES if kind.startswith(prefixes)), None
        )
        if code is None:
            continue
        try:
            spans.append(
                (
                    row.get("beg_auth_asym_id") or row.get("beg_label_asym_id", ""),
                    int(row.get("beg_auth_seq_id") or row["beg_label_seq_id"]),
                    int(row.get("end_auth_seq_id") or row["end_label_seq_id"]),
                    code,
                )
            )
        except (KeyError, ValueError):
            continue

    return _apply_spans(_deduplicate(_select_chain(residues, chain)), spans)


def _cif_category(text: str, category: str) -> list[dict[str, str]]:
    """Rows of one mmCIF category, from either a ``loop_`` or key-value form."""
    prefix = f"_{category}."
    lines = text.splitlines()
    rows: list[dict[str, str]] = []
    single: dict[str, str] = {}
    index = 0

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        if stripped.lower() == "loop_":
            keys: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip().startswith("_"):
                keys.append(lines[index].strip())
                index += 1
            if not keys or not keys[0].startswith(prefix):
                continue
            names = [k[len(prefix) :] for k in keys]
            while index < len(lines):
                body = lines[index]
                stripped = body.strip()
                if not stripped or stripped.startswith(("#", "_")) or stripped == "loop_":
                    break
                if stripped.startswith(";"):  # multi-line value: not used here
                    index += 1
                    while index < len(lines) and not lines[index].startswith(";"):
                        index += 1
                    index += 1
                    continue
                values = _cif_split(body)
                if len(values) == len(names):
                    rows.append(dict(zip(names, values)))
                index += 1
            continue

        if stripped.startswith(prefix):
            parts = _cif_split(stripped)
            if len(parts) >= 2:
                single[parts[0][len(prefix) :]] = parts[1]
        index += 1

    if single:
        rows.append(single)
    return rows


def _cif_split(line: str) -> list[str]:
    """Split an mmCIF data line, honouring quotes."""
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _apply_spans(
    residues: list[_Record], spans: list[tuple[str, int, int, str]]
) -> list[_Record]:
    """Overlay HELIX/SHEET-style ranges onto per-residue loop defaults."""
    if not residues or not spans:
        return residues
    chain = residues[0][0]
    assigned: dict[int, str] = {}
    for span_chain, start, end, code in spans:
        if span_chain and chain and span_chain != chain:
            continue
        if start > end:
            start, end = end, start
        for num in range(start, end + 1):
            assigned[num] = code
    return [
        (c, num, aa, assigned.get(num, code)) for c, num, aa, code in residues
    ]


def _parse_ss_text(text: str) -> list[_Record]:
    """Residue records from a bare three-state string, positions from 1.

    FASTA-style headers are dropped, so DSSP-derived ``.ss`` files and a
    hand-written string both work.  There is no sequence to verify the
    numbering against, which is recorded as a note when the structure is placed.
    """
    body = "".join(
        line.strip() for line in text.splitlines() if not line.startswith(">")
    )
    if not body:
        raise StructureError("empty secondary-structure string")

    unknown = sorted({c for c in body.upper() if c not in _SS_TEXT})
    if unknown:
        raise StructureError(
            "not a secondary-structure string: unexpected character(s) "
            + ", ".join(repr(c) for c in unknown)
            + "; expected H/G/I (helix), E/B (strand), C/L/T/S/- (loop)"
        )
    return [
        ("", index, "X", _SS_TEXT[code])
        for index, code in enumerate(body.upper(), start=1)
    ]
