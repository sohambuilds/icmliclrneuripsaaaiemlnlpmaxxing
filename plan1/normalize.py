"""Text cleanup, applied identically to train and test.

Per record: a conservative name, a reduced name (legal forms and filler words removed anywhere), a normalized
address, the address number tokens, and the retrieval text (reduced name + address). Raw fields stay untouched
in the raw cache.

v3 (run p1-v2) adds: Indian-script words -> English letters (translit.py), state names in one form, number
markers stripped (N°49 -> 49), French/web filler words, more legal forms and address abbreviations.
v4 (run p1-v3): same rules, better dictionary. Each v2 fix can be switched off for ablations (normalize_records).
"""
import unicodedata

import polars as pl

NORMALIZE_VERSION = 4  # bump whenever the output of normalize_records changes (cache files carry it)

LEGAL_FORMS = [
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "llp", "lp", "pllc", "pc",
    "ltd", "limited", "pvt", "private", "sarl", "sas", "sasu", "eurl", "sci",
]
LEGAL_FORMS_EXTRA = ["sa", "ets", "etablissements", "cie", "opc", "plc"]  # added in v3 (switch: fillers)
FUNCTION_WORDS = ["and", "the", "of"]
FUNCTION_WORDS_EXTRA = [  # added in v3 (switch: fillers)
    "de", "la", "le", "les", "du", "des", "d", "l", "et",  # French fillers
    "www", "com", "net", "org",  # pieces of website names
]
# dotted forms such as L.L.C. become "l l c" after punctuation -> spaces; join them so the list above catches them
DOTTED_FORMS = {
    "l l c": "llc", "p l l c": "pllc", "l l p": "llp", "s a r l": "sarl", "e u r l": "eurl", "s a s u": "sasu",
    "s a s": "sas", "s c i": "sci",
}
ADDRESS_ABBREVIATIONS = {
    "road": "rd", "street": "st", "saint": "st", "avenue": "av", "ave": "av", "boulevard": "blvd", "bd": "blvd",
    "drive": "dr", "lane": "ln", "rue": "r", "court": "ct", "highway": "hwy", "parkway": "pkwy", "circle": "cir",
    "place": "pl", "square": "sq", "terrace": "ter", "trail": "trl", "route": "rte", "suite": "ste",
    "apartment": "apt", "appt": "apt", "appartement": "apt", "floor": "fl", "flr": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w", "allee": "all", "impasse": "imp", "chemin": "ch",
}
US_STATES = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca", "colorado": "co",
    "connecticut": "ct", "delaware": "de", "district of columbia": "dc", "florida": "fl", "georgia": "ga",
    "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks", "kentucky": "ky",
    "louisiana": "la", "maine": "me", "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh",
    "new jersey": "nj", "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd",
    "ohio": "oh", "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
    "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
IN_STATES = {
    "maharashtra": "mh", "delhi": "dl", "uttar pradesh": "up", "karnataka": "ka", "tamil nadu": "tn",
    "west bengal": "wb", "gujarat": "gj", "telangana": "tg", "haryana": "hr", "rajasthan": "rj", "kerala": "kl",
    "keralam": "kl", "madhya pradesh": "mp", "bihar": "br", "andhra pradesh": "ap", "punjab": "pb", "odisha": "od",
    "orissa": "od", "goa": "ga", "assam": "as", "jharkhand": "jh", "chhattisgarh": "cg", "uttarakhand": "uk",
    "himachal pradesh": "hp", "jammu and kashmir": "jk", "chandigarh": "ch", "puducherry": "py",
}
NATIVE_STATES = {  # state names written in Indian scripts; keys are NFC-normalized below to match cleaned text
    "महाराष्ट्र": "maharashtra", "दिल्ली": "delhi", "उत्तर प्रदेश": "uttar pradesh", "ಕರ್ನಾಟಕ": "karnataka",
    "தமிழ்நாடு": "tamil nadu", "পশ্চিমবঙ্গ": "west bengal", "ગુજરાત": "gujarat", "తెలంగాణ": "telangana",
    "हरियाणा": "haryana", "राजस्थान": "rajasthan", "കേരളം": "kerala", "बिहार": "bihar",
    "मध्य प्रदेश": "madhya pradesh", "ఆంధ్రప్రదేశ్": "andhra pradesh", "ਪੰਜਾਬ": "punjab", "ଓଡ଼ିଶା": "odisha",
}
NATIVE_STATES = {unicodedata.normalize("NFC", k.replace("\u200c", "").replace("\u200d", "")): v for k, v in NATIVE_STATES.items()}
NULL_PLACEHOLDERS = r"(?i)(?:\bnull\b|\bn/a\b)"
ZERO_WIDTH_RE = r"[\x{200B}-\x{200D}\x{2060}\x{FEFF}]"  # joiners inside Indic words: delete, never space
NONLATIN_RE = r"[^\p{Latin}\p{Common}\p{Inherited}]"  # a letter from any non-Latin script
INDIC_RE = r"[\x{0900}-\x{0DFF}]"


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


def number_tokens(raw_address: pl.Expr, strip_markers: bool = True) -> pl.Expr:
    """Address tokens containing a digit, taken before punctuation cleanup.

    Attached letters and internal slashes/hyphens stay (12B, 12/3, A-5); edge punctuation goes (#3422 -> 3422,
    135. -> 135); number markers go (N°49 -> 49, No.5/257 -> 5/257); leading zeros go (002078 -> 2078).
    """
    tokens = raw_address.str.to_lowercase().str.extract_all(r"[^\s,;:()\[\]{}|<>\"]+")
    e = pl.element().str.replace_all(r"^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$", "")
    if strip_markers:
        e = e.str.replace_all(r"^(?:n°|nº|no\.?)(\d)", "${1}")
    e = e.str.replace_all(r"\b0+(\d)", "${1}")
    return tokens.list.eval(e.filter(pl.element().str.contains(r"\d"))).list.unique().list.sort()


def _convert_indic(df: pl.DataFrame, col: str, to_latin) -> pl.DataFrame:
    """Apply to_latin to the distinct values of `col` that contain Indian script (done once per value)."""
    if to_latin is None:
        return df
    vals = df.select(pl.col(col).filter(pl.col(col).str.contains(INDIC_RE)).unique()).to_series()
    if vals.len() == 0:
        return df
    new = pl.Series([to_latin(v) for v in vals.to_list()], dtype=pl.String)
    return df.with_columns(pl.col(col).replace_strict(vals, new, default=pl.col(col)))


def normalize_records(records: pl.DataFrame, to_latin=None, states: bool = True, number_markers: bool = True,
                      fillers: bool = True) -> pl.DataFrame:
    """records: entity_id, business_name, business_address, country, source (raw).

    to_latin: converter from translit.make_converter; None keeps Indian script as is (and native state names).
    The switches turn off one v2 fix each, for ablations only; the pipeline always uses the defaults.
      states: state names -> one form (codes); number_markers: N°49 -> 49; fillers: French/web words and the
      extra legal forms (sa, ets, cie, opc, plc).
    """
    address_raw = pl.col("business_address").fill_null("").str.replace_all(NULL_PLACEHOLDERS, " ")
    addr_clean = basic_clean(address_raw)
    if to_latin is not None:
        addr_clean = word_map(addr_clean, NATIVE_STATES)
    df = records.select(
        "entity_id",
        "source",
        "country",
        name_cons=basic_clean(pl.col("business_name")),
        addr_clean=addr_clean,
        addr_nums=number_tokens(address_raw, strip_markers=number_markers),
        name_nonlatin=pl.col("business_name").fill_null("").str.contains(NONLATIN_RE),
    )
    df = _convert_indic(_convert_indic(df, "name_cons", to_latin), "addr_clean", to_latin)
    drop_words = LEGAL_FORMS + FUNCTION_WORDS + ((LEGAL_FORMS_EXTRA + FUNCTION_WORDS_EXTRA) if fillers else [])
    drop_re = r"\b(?:" + "|".join(drop_words) + r")\b"
    addr = pl.col("addr_clean")
    if states:  # states first (so "west bengal" is caught before "west" -> "w"), then street words
        addr = word_map(addr, {**US_STATES, **IN_STATES})
    df = df.with_columns(
        name_red=_ws(word_map(pl.col("name_cons"), DOTTED_FORMS).str.replace_all(drop_re, " ")),
        addr_norm=word_map(addr, ADDRESS_ABBREVIATIONS),
    ).drop("addr_clean")
    df = df.with_columns(
        name_red=pl.when(pl.col("name_red") == "").then(pl.col("name_cons")).otherwise(pl.col("name_red")),
        addr_missing=pl.col("addr_norm") == "",
    )
    return df.with_columns(retrieval_text=_ws(pl.col("name_red") + " " + pl.col("addr_norm")))
