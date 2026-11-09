"""Empreintes des donnees : fins de ligne ignorees, tables non lues ignorees, ecart de donnees qui arrete l'execution."""
import duckdb

from meandre.domain.manifest import Manifest, fingerprint, fingerprint_bytes, fingerprint_duckdb


def test_text_fingerprint_ignores_line_endings(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x;1\r\ny;2\r\n")
    b.write_bytes(b"x;1\ny;2\n")
    assert fingerprint(a) == fingerprint(b)


def test_duckdb_fingerprint_reads_content_of_read_tables_only(tmp_path):
    p1, p2 = tmp_path / "a.duckdb", tmp_path / "b.duckdb"
    for p, extra in ((p1, False), (p2, True)):
        c = duckdb.connect(str(p))
        c.execute("CREATE TABLE nodes AS SELECT range AS id FROM range(5)")
        if extra:
            c.execute("CREATE TABLE dams AS SELECT 1 AS x")
        c.close()
    assert fingerprint_duckdb(p1) == fingerprint_duckdb(p2)


def test_differences_by_file_name_and_legacy_bytes(tmp_path):
    f = tmp_path / "territorial-raw-QC.csv"
    f.write_bytes(b"a\r\n")
    m = Manifest("gasp")
    m.add("attributes", f)
    legacy = {"attributes": {"path": "/autre/machine/territorial-raw-QC.csv", "fingerprint": fingerprint_bytes(f)}}
    assert m.differences(legacy) == []
    changed = {"attributes": {"path": "/autre/machine/territorial-raw-QC.csv", "fingerprint": "0" * 16}}
    assert len(m.differences(changed)) == 1
