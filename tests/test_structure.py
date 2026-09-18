"""Reading secondary structure, and placing it on the assayed protein."""

from __future__ import annotations

import pytest

from dms_heatmap.structure import (
    HELIX,
    LOOP,
    STRAND,
    UNMODELLED,
    SecondaryStructure,
    StructureError,
    discover_structures,
    match_structures,
    read_structure,
)

from .conftest import WT_SEQUENCE

THREE = {
    "M": "MET", "K": "LYS", "V": "VAL", "L": "LEU", "A": "ALA", "G": "GLY",
    "T": "THR", "W": "TRP", "S": "SER", "Y": "TYR",
}


def dssp_text(codes: str, sequence: str = WT_SEQUENCE, chain: str = "A", first: int = 1):
    """Classic DSSP output for one chain, in its fixed columns."""
    lines = ["==== Secondary Structure Definition ====", "  #  RESIDUE AA STRUCTURE BP1 BP2  ACC"]
    for index, (code, residue) in enumerate(zip(codes, sequence)):
        number = first + index
        lines.append(f"{index + 1:5d}{number:5d} {chain} {residue}  {code}")
    return "\n".join(lines) + "\n"


def _columns(record: str, fields: dict[int, str]) -> str:
    """A fixed-column PDB record, written by column so the spec is visible.

    ``fields`` maps a 0-based column index to the text starting there.  PDB
    records are positional and a record built by counting spaces in an f-string
    is a record that parses by luck.
    """
    line = list(record.ljust(80))
    for index, value in fields.items():
        line[index : index + len(value)] = value
    return "".join(line).rstrip()


def pdb_text(spans=(), sequence: str = WT_SEQUENCE, chain: str = "A", first: int = 1):
    """A PDB with CA atoms and optional HELIX/SHEET records."""
    lines = []
    for kind, start, end in spans:
        if kind == HELIX:
            # HELIX: initChainID col 20, initSeqNum 22-25, endChainID 32,
            # endSeqNum 34-37 (1-based, as the PDB format describes them).
            lines.append(
                _columns(
                    "HELIX",
                    {
                        7: "  1", 11: "  1", 15: "ALA", 19: chain,
                        21: f"{start:4d}", 27: "ALA", 31: chain,
                        33: f"{end:4d}", 38: " 1",
                    },
                )
            )
        else:
            # SHEET: initChainID col 22, initSeqNum 23-26, endChainID 33,
            # endSeqNum 34-37 -- one column along from HELIX, which is exactly
            # the kind of difference a hand-spaced fixture gets wrong.
            lines.append(
                _columns(
                    "SHEET",
                    {
                        7: "  1", 11: "  A", 14: " 1", 17: "ALA", 21: chain,
                        22: f"{start:4d}", 28: "ALA", 32: chain,
                        33: f"{end:4d}",
                    },
                )
            )
    for index, residue in enumerate(sequence):
        # ATOM: name cols 13-16, resName 18-20, chainID 22, resSeq 23-26.
        lines.append(
            _columns(
                "ATOM",
                {
                    6: f"{index + 1:5d}", 12: " CA ", 17: THREE[residue],
                    21: chain, 22: f"{first + index:4d}",
                    30: f"{index * 1.5:8.3f}{0.0:8.3f}{0.0:8.3f}",
                    54: "  1.00 20.00           C",
                },
            )
        )
    lines.append("END")
    return "\n".join(lines) + "\n"


def cif_text(spans=(), sequence: str = WT_SEQUENCE, chain: str = "A", first: int = 1):
    """An mmCIF with an ``_atom_site`` loop and a ``_struct_conf`` loop."""
    lines = ["data_test"]
    if spans:
        lines += [
            "loop_",
            "_struct_conf.conf_type_id",
            "_struct_conf.beg_auth_asym_id",
            "_struct_conf.beg_auth_seq_id",
            "_struct_conf.end_auth_asym_id",
            "_struct_conf.end_auth_seq_id",
        ]
        for kind, start, end in spans:
            name = "HELX_RH_AL_P" if kind == HELIX else "STRN"
            lines.append(f"{name} {chain} {start} {chain} {end}")
        lines.append("#")
    lines += [
        "loop_",
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    for index, residue in enumerate(sequence):
        lines.append(f"CA {THREE[residue]} {chain} {first + index} 1")
    lines.append("#")
    return "\n".join(lines) + "\n"


def write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.write_text(text)
    return path


# --- parsing --------------------------------------------------------------


def test_dssp_eight_states_collapse_to_three(tmp_path):
    # H/G/I are all helices, E/B both strands, T/S/blank all loop.
    path = write(tmp_path, "gene1.dssp", dssp_text("HGIEBTS  H"))
    structure = read_structure(path)
    assert "".join(structure.codes) == "HHHEELLLLH"
    assert structure.sequence == WT_SEQUENCE
    assert structure.chain == "A"


def test_dssp_skips_chain_breaks(tmp_path):
    text = dssp_text("HHHHHHHHHH")
    lines = text.splitlines()
    lines.insert(5, f"{99:5d}{0:5d} {' '} !  ")
    path = write(tmp_path, "gene1.dssp", "\n".join(lines) + "\n")
    assert len(read_structure(path)) == 10


def test_a_file_that_is_not_dssp_is_rejected(tmp_path):
    path = write(tmp_path, "gene1.dssp", "just some text\n")
    with pytest.raises(StructureError, match="not DSSP output"):
        read_structure(path)


def test_pdb_helix_and_sheet_records_become_codes(tmp_path):
    path = write(tmp_path, "gene1.pdb", pdb_text([(HELIX, 2, 4), (STRAND, 7, 9)]))
    structure = read_structure(path)
    assert "".join(structure.codes) == "LHHHLLEEEL"
    assert structure.sequence == WT_SEQUENCE


def test_pdb_reads_only_the_first_model(tmp_path):
    # The second model numbers its residues 101.. , so reading both would give
    # twenty residues rather than ten -- an NMR ensemble would give twenty
    # copies of every one.
    first = pdb_text([(HELIX, 1, 10)]).replace("END\n", "ENDMDL\n")
    path = write(tmp_path, "gene1.pdb", first + pdb_text([], first=101))
    assert len(read_structure(path)) == 10


def test_an_unannotated_pdb_without_dssp_warns_rather_than_failing(tmp_path, monkeypatch):
    monkeypatch.setattr("dms_heatmap.structure.shutil.which", lambda _: None)
    path = write(tmp_path, "gene1.pdb", pdb_text())
    structure = read_structure(path)

    # Every residue reads as loop, and the figure says so instead of implying
    # the protein has no helices.
    assert set(structure.codes) == {LOOP}
    assert any("neither mkdssp nor dssp" in w for w in structure.warnings)


def test_mmcif_struct_conf_becomes_codes(tmp_path):
    path = write(tmp_path, "gene1.cif", cif_text([(HELIX, 1, 3), (STRAND, 6, 8)]))
    structure = read_structure(path)
    assert "".join(structure.codes) == "HHHLLEEELL"


def test_a_bare_secondary_structure_string_is_accepted(tmp_path):
    path = write(tmp_path, "gene1.ss", ">gene1\nHHHCCCEEEL\n")
    structure = read_structure(path)
    assert "".join(structure.codes) == "HHHLLLEEEL"
    assert structure.resnums == tuple(range(1, 11))


def test_a_sequence_is_not_mistaken_for_a_structure_string(tmp_path):
    path = write(tmp_path, "gene1.ss", ">gene1\nMKVLAGTWSY\n")
    with pytest.raises(StructureError, match="unexpected character"):
        read_structure(path)


def test_an_unsupported_extension_is_reported(tmp_path):
    path = write(tmp_path, "gene1.xyz", "whatever\n")
    with pytest.raises(StructureError, match="unsupported structure format"):
        read_structure(path)


# --- chains ---------------------------------------------------------------


def test_a_multi_chain_file_takes_the_first_chain(tmp_path):
    text = dssp_text("HHHHHHHHHH", chain="A") + dssp_text(
        "EEEEEEEEEE", chain="B"
    ).split("ACC\n")[1]
    path = write(tmp_path, "gene1.dssp", text)
    assert set(read_structure(path).codes) == {HELIX}


def test_a_chain_can_be_chosen(tmp_path):
    text = dssp_text("HHHHHHHHHH", chain="A") + dssp_text(
        "EEEEEEEEEE", chain="B"
    ).split("ACC\n")[1]
    path = write(tmp_path, "gene1.dssp", text)
    assert set(read_structure(path, chain="B").codes) == {STRAND}


def test_an_absent_chain_is_an_error(tmp_path):
    path = write(tmp_path, "gene1.dssp", dssp_text("HHHHHHHHHH", chain="A"))
    with pytest.raises(StructureError, match="chain 'Z' not in the structure"):
        read_structure(path, chain="Z")


# --- runs -----------------------------------------------------------------


def structure_from(codes: str, first: int = 1, sequence: str | None = None):
    return SecondaryStructure(
        name="test",
        source=None,
        codes=tuple(codes),
        resnums=tuple(range(first, first + len(codes))),
        sequence=sequence if sequence is not None else "X" * len(codes),
    )


def test_runs_segment_a_window_into_elements():
    runs = structure_from("LLHHHHEEL").runs(1, 9)
    assert [(r.code, r.start, r.end) for r in runs] == [
        (LOOP, 1, 2),
        (HELIX, 3, 6),
        (STRAND, 7, 8),
        (LOOP, 9, 9),
    ]


def test_a_run_clipped_by_the_window_knows_where_its_element_began():
    # Positions 5..9 of a helix that really starts at 3: the coil has to keep
    # its phase, and the strand must not gain an arrowhead at the fold.
    structure = structure_from("LLHHHHHHEE")
    first, *_, last = structure.runs(5, 9)
    assert (first.code, first.start, first.origin) == (HELIX, 5, 3)
    assert (last.code, last.end, last.open_end) == (STRAND, 9, True)


def test_a_run_that_really_ends_inside_the_window_is_not_open():
    (run,) = structure_from("EEEE").runs(1, 4)
    assert run.open_end is False


def test_positions_outside_the_model_are_marked_not_modelled():
    structure = structure_from("HHH", first=4)
    runs = structure.runs(1, 8)
    assert [(r.code, r.start, r.end) for r in runs] == [
        (UNMODELLED, 1, 3),
        (HELIX, 4, 6),
        (UNMODELLED, 7, 8),
    ]


def test_a_window_entirely_outside_the_model_terminates():
    # UNMODELLED is the default for every position, inside the model or not,
    # so walking back to find where the run began must not follow it.
    (run,) = structure_from("HHH", first=100).runs(1, 10)
    assert (run.code, run.origin) == (UNMODELLED, 1)


def test_composition_is_the_fraction_of_modelled_residues():
    composition = structure_from("HHHHEELL").composition()
    assert composition == {HELIX: 0.5, STRAND: 0.25, LOOP: 0.25}


# --- placement on the protein --------------------------------------------


def test_matching_numbering_is_accepted_at_offset_zero():
    structure = structure_from("HHHHHHHHHH", sequence=WT_SEQUENCE).aligned_to(WT_SEQUENCE)
    assert structure.offset == 0
    assert structure.identity == pytest.approx(1.0)
    assert not structure.warnings


def test_a_domain_model_is_found_by_its_sequence():
    # A model of residues 4..10 that numbers them 1..7: without the search the
    # track would sit three positions to the left of the biology.
    structure = structure_from(
        "HHHEEEL", first=1, sequence=WT_SEQUENCE[3:]
    ).aligned_to(WT_SEQUENCE)
    assert structure.offset == 3
    assert structure.identity == pytest.approx(1.0)
    assert structure.code_map()[4] == HELIX


def test_a_structure_of_another_protein_is_refused():
    with pytest.raises(StructureError, match="cannot place"):
        structure_from("HHHHHHHHHH", sequence="CCCCCCCCCC").aligned_to(WT_SEQUENCE)


def test_an_explicit_offset_is_obeyed_and_scored():
    structure = structure_from("HHHHHHHHHH", sequence=WT_SEQUENCE).aligned_to(
        WT_SEQUENCE, offset=2
    )
    assert structure.offset == 2
    assert structure.identity < 0.8
    assert any("may not be this protein" in w for w in structure.warnings)


def test_a_structure_with_no_sequence_is_placed_but_flagged():
    structure = structure_from("HHHHHHHHHH").aligned_to(WT_SEQUENCE)
    assert structure.offset == 0
    assert structure.identity is None
    assert any("no sequence to verify" in w for w in structure.warnings)


def test_placement_notes_are_provenance_not_warnings():
    structure = structure_from("HHHHHHHHHH", sequence=WT_SEQUENCE).aligned_to(WT_SEQUENCE)
    assert any("placed by sequence" in n for n in structure.notes)
    assert structure.as_dict()["sequence_identity"] == pytest.approx(1.0)


# --- discovery and matching ----------------------------------------------


def test_structures_are_discovered_recursively(tmp_path):
    (tmp_path / "nested").mkdir()
    write(tmp_path, "gene1.pdb", pdb_text())
    (tmp_path / "nested" / "gene2.ss").write_text("HHHH\n")
    write(tmp_path, "notes.txt", "ignore me")

    found = discover_structures(tmp_path)
    assert [p.name for p in found] == ["gene1.pdb", "gene2.ss"]


def test_a_missing_structure_path_is_an_error(tmp_path):
    with pytest.raises(StructureError, match="does not exist"):
        discover_structures(tmp_path / "nope")


@pytest.mark.parametrize(
    "filename,matches",
    [
        ("tna1.pdb", True),
        ("TNA1.dssp", True),
        ("tna1_alphafold.pdb", True),
        ("tna1-af2.cif", True),
        ("tna10.pdb", False),      # a different gene, not a suffixed tna1
        ("other.pdb", False),
    ],
)
def test_structures_are_matched_to_datasets_by_filename(tmp_path, filename, matches):
    path = tmp_path / filename
    paired = match_structures([path, tmp_path / "decoy.pdb"], ["tna1"])
    assert ("tna1" in paired) == matches


def test_one_structure_and_one_dataset_are_paired_whatever_they_are_called(tmp_path):
    paired = match_structures([tmp_path / "model.pdb"], ["tna1"])
    assert paired == {"tna1": tmp_path / "model.pdb"}


def test_the_most_informative_format_wins_when_several_match(tmp_path):
    # A DSSP file states every residue's state, a PDB only its helices and
    # sheets, and a bare string carries no sequence to check numbering against.
    candidates = [
        tmp_path / "tna1.ss",
        tmp_path / "tna1.pdb",
        tmp_path / "tna1.dssp",
        tmp_path / "tna1.cif",
    ]
    paired = match_structures(candidates, ["tna1", "other"])
    assert paired["tna1"].name == "tna1.dssp"

    without_dssp = [p for p in candidates if p.suffix != ".dssp"]
    assert match_structures(without_dssp, ["tna1", "other"])["tna1"].name == "tna1.pdb"


def test_the_exact_name_beats_a_better_format_under_a_prefixed_name(tmp_path):
    # Naming a file tna1.ss is how you say "this is the structure for tna1";
    # it must not lose to a tna1_alphafold.pdb lying beside it just because a
    # PDB carries more than a bare string.
    paired = match_structures(
        [tmp_path / "tna1_alphafold.pdb", tmp_path / "tna1.ss"], ["tna1"]
    )
    assert paired["tna1"].name == "tna1.ss"


def test_a_prefixed_name_is_still_used_when_nothing_matches_exactly(tmp_path):
    paired = match_structures([tmp_path / "tna1_alphafold.pdb"], ["tna1", "other"])
    assert paired["tna1"].name == "tna1_alphafold.pdb"
