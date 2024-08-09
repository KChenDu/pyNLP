from pathlib import Path
from re import compile


def convert2SentenceList(corpus_path: Path) -> list[list[str]]:
    with open(corpus_path, encoding='utf-8') as corpus:
        pattern = compile("\\s+")
        return [pattern.split(line.strip()) for line in corpus]
