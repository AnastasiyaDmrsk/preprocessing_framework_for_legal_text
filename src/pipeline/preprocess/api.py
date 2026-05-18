from pathlib import Path
from typing import Tuple, List

from src.pipeline.preprocess.preprocess import RegulatoryTextPreprocessor


def preprocess_legal_text(reg_text: str, path: Path) -> Tuple[str, List[str]]:
    """
    Preprocess regulatory text and write outputs to *path*.

    Outputs
    -------
    preprocess.txt  — one obligation clause per line, with gateway markers
    references.csv  — Location,Reference rows for all detected cross-references

    Returns
    -------
    (preprocessed_text, references_list)
    """
    reg_preprocessor       = RegulatoryTextPreprocessor()
    preprocessed_text, references = reg_preprocessor.preprocess(reg_text)

    path.mkdir(parents=True, exist_ok=True)
    (path / "preprocess.txt").write_text(preprocessed_text, encoding="utf-8")
    with open(path / "references.csv", "w", encoding="utf-8") as f:
        f.write("Location,Reference\n")
        f.writelines(ref + "\n" for ref in references)

    return preprocessed_text, references


