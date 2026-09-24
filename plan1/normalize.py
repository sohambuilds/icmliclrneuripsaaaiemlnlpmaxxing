"""Small, fixed text cleanup (Plan 1 section 3).

Per record: a conservative name, a reduced name (legal forms and and/the/of removed anywhere), a normalized
address, the address number tokens, and the retrieval text (reduced name + address). Raw fields stay untouched
in the raw cache. Other scripts and their combining marks are kept; only Latin accents are folded.
"""
import polars as pl

NORMALIZE_VERSION = 2  # bump whenever the output of normalize_records changes (cache files carry it)

LEGAL_FORMS = [
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp", "pllc", "pc",
    "ltd", "limited", "pvt", "private", "sarl", "sas", "sasu", "eurl", "sci",
]
FUNCTION_WORDS = ["and", "the", "of"]
# dotted forms such as L.L.C. become "l l c" after punctuation -> spaces; join them so the list above catches them
DOTTED_FORMS = {
    "l l c": "llc", "p l l c": "pllc", "l l p": "llp", "s a r l": "sarl", "e u r l": "eurl", "s a s u": "sasu",
    "s a s": "sas", "s c i": "sci",
}
ADDRESS_ABBREVIATIONS = {
    "road": "rd", "street": "st", "saint": "st", "avenue": "av", "ave": "av",
    "boulevard": "blvd", "drive": "dr", "lane": "ln", "rue": "r",
}
NULL_PLACEHOLDERS = r"(?i)(?:\bnull\b|\bn/a\b)"
ZERO_WIDTH_RE = r"[\x{200B}-\x{200D}\x{2060}\x{FEFF}]"  # joiners inside Indic words: delete, never space
NONLATIN_RE = r"[^\p{Latin}\p{Common}\p{Inherited}]"  # a letter from any non-Latin script


def _ws(e: pl.Expr) -> pl.Expr:
    return e.str.replace_all(r"\s+", " ").str.strip_chars()


def basic_clean(e: pl.Expr) -> pl.Expr:
    """Unicode normalization, Latin accent folding, lowercase, & -> and, punctuation -> spaces.

    Accent folding removes only U+0300-U+036F (Latin combining diacritics); Indic vowel signs live in their
    own blocks and survive. Apostrophes are dropped rather than spaced so "Orelee's" == "Orelees".
    """
    e = e.fill_null("").str.normalize("NFKC").str.replace_all(ZERO_WIDTH_RE, "")
    e = e.str.normalize("NFKD").str.replace_all(r"[\x{0300}-\x{036f}]", "").str.normalize("NFC")
    e = e.str.to_lowercase()
    e = e.str.replace_all("&", " and ").str.replace_all(r"['’‘`´]", "")
    e = e.str.replace_all(r"[^\p{L}\p{M}\p{N}]+", " ")
    return _ws(e)


def word_map(e: pl.Expr, mapping: dict[str, str]) -> pl.Expr:
    """Replace whole words/phrases in one pass (longest first).

    The text's spaces are doubled so neighbouring matches don't share a separator; multi-word keys are doubled
    the same way so they still match.
    """
    keys = sorted(mapping, key=len, reverse=True)
    padded = " " + e.str.replace_all(" ", "  ") + " "
    patterns = [" " + k.replace(" ", "  ") + " " for k in keys]
    return _ws(padded.str.replace_many(patterns, [f" {mapping[k]} " for k in keys]))


def number_tokens(raw_address: pl.Expr) -> pl.Expr:
    """Address tokens containing a digit, taken before punctuation cleanup.

    Attached letters and internal slashes/hyphens stay (12B, 12/3, A-5); edge punctuation goes (#3422 -> 3422,
    135. -> 135); leading zeros in digit runs go (002078 -> 2078, 12/003 -> 12/3). No house/unit/postal typing.
    """
    tokens = raw_address.str.to_lowercase().str.extract_all(r"[^\s,;:()\[\]{}|<>\"]+")
    return tokens.list.eval(
        pl.element()
        .str.replace_all(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "")
        .str.replace_all(r"\b0+(\d)", "${1}")
        .filter(pl.element().str.contains(r"\d"))
    ).list.unique().list.sort()


def normalize_records(records: pl.DataFrame) -> pl.DataFrame:
    """records: entity_id, business_name, business_address, country, source (raw)."""
    name_cons = basic_clean(pl.col("business_name"))
    drop_re = r"\b(?:" + "|".join(LEGAL_FORMS + FUNCTION_WORDS) + r")\b"
    address_raw = pl.col("business_address").fill_null("").str.replace_all(NULL_PLACEHOLDERS, " ")
    out = records.select(
        "entity_id",
        "source",
        "country",
        name_cons=name_cons,
        name_red=_ws(word_map(name_cons, DOTTED_FORMS).str.replace_all(drop_re, " ")),
        addr_norm=word_map(basic_clean(address_raw), ADDRESS_ABBREVIATIONS),
        addr_nums=number_tokens(address_raw),
        name_nonlatin=pl.col("business_name").fill_null("").str.contains(NONLATIN_RE),
    ).with_columns(
        name_red=pl.when(pl.col("name_red") == "").then(pl.col("name_cons")).otherwise(pl.col("name_red")),
        addr_missing=pl.col("addr_norm") == "",
    )
    return out.with_columns(retrieval_text=_ws(pl.col("name_red") + " " + pl.col("addr_norm")))
