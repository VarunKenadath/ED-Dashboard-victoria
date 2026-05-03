"""
ETL Script: Hospital Data CSV files -> SQLite (victoria_ed.db)
Processes all tables from the hospital_data directory.
Tables beginning with 'Table_A' or 'Table_S' are ignored.
"""

import logging
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, inspect, text

# =============================================================================
# CONFIGURATION
# =============================================================================

CSV_DIR = Path(r"C:\Users\Varun\Desktop\hospital_data")
DB_PATH = CSV_DIR / "victoria_ed.db"   # SQLite database file
IF_TABLE_EXISTS = "replace"            # "skip" | "replace" | "append"

# =============================================================================
# SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STATES = ["nsw", "vic", "qld", "wa", "sa", "tas", "act", "nt"]
STATE_RENAME = {
    "nsw": "NSW", "vic": "VIC", "qld": "Qld", "wa": "WA",
    "sa": "SA", "tas": "Tas", "act": "ACT", "nt": "NT",
}


def get_engine():
    return create_engine(f"sqlite:///{DB_PATH}", echo=False)


# =============================================================================
# SHARED HELPERS
# =============================================================================

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def normalise_name(name: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^\w]", "_", str(name).strip().lower())).strip("_")


def replace_na_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace common null representations (n.p, .., '. .', empty) with pd.NA."""
    na_vals = ["n.p", "n.p.", "..", ". .", "N/A", "n/a", "NA", "None",
               "none", "NULL", "null", "-"]
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(na_vals, pd.NA)
    return df


def drop_footnotes(df: pd.DataFrame, id_col: str | None = None) -> pd.DataFrame:
    """Drop footnote/note rows at the bottom of a table."""
    first_col = df.columns[0]
    if id_col:
        first_col = id_col
    mask = (
        df[first_col].astype(str).str.strip().str.startswith("(")
        | df[first_col].astype(str).str.strip().str.lower().str.startswith("note")
        | df[first_col].astype(str).str.strip().str.lower().str.startswith("http")
        | df[first_col].astype(str).str.strip().str.lower().str.startswith("technical")
        | df[first_col].astype(str).str.strip().str.lower().str.startswith("back to")
        | df[first_col].astype(str).str.strip().isin(["", "nan"])
    )
    first_bad = mask.idxmax() if mask.any() else None
    if first_bad is not None and mask[first_bad]:
        df = df.loc[:first_bad].iloc[:-1]
    return df


def clean_string_values(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all string values and normalise special chars."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col].astype(str)
            .str.strip()
            .str.replace(r"Ã¢â‚¬â€", "_", regex=False)
            .str.replace(r"Ã¢â¬â", "_", regex=False)
            .str.replace(r"\s+", " ", regex=True)
        )
        df[col] = df[col].replace("nan", pd.NA)
    return df


def get_state_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that correspond to Australian states/territories."""
    return [c for c in df.columns if re.match(r"^(nsw|vic|qld|wa|sa|tas|act|nt)(_\w+)?$", c)]


def melt_states(
    df: pd.DataFrame,
    id_vars: list[str],
    value_name: str = "presentations",
) -> pd.DataFrame:
    state_cols = get_state_cols(df)
    df = df[id_vars + state_cols].copy()
    df = df.melt(id_vars=id_vars, value_vars=state_cols, var_name="state", value_name=value_name)
    # Normalise state name back to display form
    df["state"] = df["state"].str.replace(r"_\w+$", "", regex=True).map(
        lambda x: STATE_RENAME.get(x, x.upper())
    )
    return df


def melt_years(
    df: pd.DataFrame,
    id_vars: list[str],
    value_name: str = "value",
) -> pd.DataFrame:
    year_cols = [c for c in df.columns if re.match(r"^20\d\d", c)]
    df = df[id_vars + year_cols].copy()
    df = df.melt(id_vars=id_vars, value_vars=year_cols, var_name="year", value_name=value_name)
    return df


def read_csv(filepath: Path, skip_rows: int = 0) -> pd.DataFrame:
    return pd.read_csv(filepath, skiprows=skip_rows, encoding="utf-8", low_memory=False)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    if before - len(df):
        log.info(f"  Removed {before - len(df)} duplicate row(s)")
    return df


def drop_total_rows(df: pd.DataFrame, col: str, patterns: list[str] | None = None) -> pd.DataFrame:
    """Remove rows where a column value starts with 'total' or matches patterns."""
    defaults = [r"^total"]
    pats = (patterns or []) + defaults
    combined = "|".join(pats)
    mask = df[col].astype(str).str.lower().str.strip().str.match(combined)
    return df[~mask]


def drop_total_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose cleaned name starts with 'total'."""
    drop = [c for c in df.columns if re.match(r"^total", c.lower())]
    return df.drop(columns=drop, errors="ignore")


def cast_numeric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def finalise(df: pd.DataFrame, value_col: str = "presentations") -> pd.DataFrame:
    df = replace_na_values(df)
    df = clean_string_values(df)
    df = deduplicate(df)
    if value_col in df.columns:
        df = cast_numeric(df, value_col)
    df = df.dropna(subset=[value_col] if value_col in df.columns else [])
    log.info(f"  Shape after transforms: {df.shape}")
    return df


# =============================================================================
# LOAD HELPER
# =============================================================================

def load_to_postgres(df: pd.DataFrame, table_name: str, engine, existing_tables: set) -> None:
    if IF_TABLE_EXISTS == "skip" and table_name in existing_tables:
        log.info(f"  Table '{table_name}' already exists – skipping load")
        return
    action = "replace" if IF_TABLE_EXISTS == "replace" else "append"
    df.to_sql(table_name, engine, if_exists=action, index=False)
    log.info(f"  Loaded {len(df)} rows into '{table_name}'")


# =============================================================================
# TABLE_2_1 and TABLE_2_2 (original logic preserved)
# =============================================================================

def reshape_presentations(
    df: pd.DataFrame,
    id_column: str = "peer_group",
    extra_id_columns: list[str] | None = None,
) -> pd.DataFrame:
    extra_id_columns = extra_id_columns or []
    year_columns = [col for col in df.columns if col.startswith("20")]
    df = df.dropna(subset=[id_column], how="all")
    df = df.dropna(subset=year_columns, how="all")
    df = df[
        ~df[id_column].astype("string").str.lower().isin(
            ["average_annual_change_last_five_years", "change_since_last_year"]
        )
    ]
    id_cols = [id_column, *extra_id_columns]
    df = df[[*id_cols, *year_columns]]
    df = df.melt(
        id_vars=id_cols,
        value_vars=year_columns,
        var_name="years",
        value_name="presentations",
    )
    df = df.dropna(subset=["presentations"])
    return df


def apply_renames(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    cleaned_map = {normalise_name(k): v for k, v in rename_map.items()}
    return df.rename(columns=cleaned_map)


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for column, value in filters.items():
        cleaned_column = normalise_name(column)
        if cleaned_column not in df.columns:
            log.warning(f"  Filter skipped – column '{cleaned_column}' not found")
            continue
        df = df[df[cleaned_column].astype("string").str.lower() == str(value).lower()]
    return df


def cast_columns(df: pd.DataFrame, cast_map: dict) -> pd.DataFrame:
    for col, dtype in cast_map.items():
        if col not in df.columns:
            log.warning(f"  Cast skipped – column '{col}' not found")
            continue
        try:
            if dtype == "datetime":
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            elif dtype == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif dtype == "float":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "bool":
                df[col] = df[col].map({"True": True, "False": False, "1": True, "0": False})
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            log.warning(f"  Could not cast '{col}' to {dtype}: {e}")
    return df


FILE_CONFIGS_2X = [
    {
        "source_file": "Table_2_1.csv",
        "table_name": "emergency_department_presentations_by_peer_group",
        "rename_columns": {"Peer group": "peer_group"},
        "filters": {},
        "cast_columns": {
            "peer_group": "string",
            "years": "string",
            "presentations": "float",
        },
    },
    {
        "source_file": "Table_2_2.csv",
        "table_name": "emergency_department_presentations_by_state_territory",
        "rename_columns": {
            "State/territory": "states/territories",
            "Measure": "measure",
        },
        "id_column": "states/territories",
        "extra_id_columns": ["measure"],
        "cast_columns": {
            "states/territories": "string",
            "years": "string",
            "measure": "string",
            "presentations": "float",
        },
    },
]


def handle_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.replace(["N/A", "n/a", "NA", "None", "none", "NULL", "null", "-"], pd.NA)
    return df


def process_2x_file(file_config: dict) -> pd.DataFrame:
    source_file = file_config["source_file"]
    id_column = file_config.get("id_column", "peer_group")
    filepath = CSV_DIR / source_file
    if not filepath.exists():
        log.warning(f"File not found: {filepath}")
        return pd.DataFrame()
    log.info(f"Processing: {source_file}")
    df = read_csv(filepath, 1)
    df = handle_nulls(df)
    df = clean_column_names(df)
    rename_columns = file_config.get("rename_columns", {})
    if rename_columns:
        df = apply_renames(df, rename_columns)
    filters = file_config.get("filters", {})
    if filters:
        df = apply_filters(df, filters)
    df = reshape_presentations(
        df,
        id_column=id_column,
        extra_id_columns=file_config.get("extra_id_columns", []),
    )
    cast_map = file_config.get("cast_columns", {})
    if cast_map:
        df = cast_columns(df, cast_map)
    df = deduplicate(df)
    log.info(f"  Shape after transforms: {df.shape}")
    return df


# =============================================================================
# TABLE 2.3
# =============================================================================

def process_table_2_3() -> pd.DataFrame:
    """Presentations by peer group and state."""
    log.info("Processing: Table_2_3.csv")
    df = read_csv(CSV_DIR / "Table_2_3.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "peer_group")
    df = replace_na_values(df)

    # Remove rows where measure == 'Proportion of total (%)' and 'All hospitals'
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = df[~df["peer_group"].astype(str).str.lower().str.strip().str.startswith("all hospital")]

    # Melt states (drop Total column)
    id_vars = ["peer_group"]
    df = melt_states(df, id_vars=id_vars, value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 3.1
# =============================================================================

def process_table_3_1() -> pd.DataFrame:
    """Emergency department presentations by state and demographics."""
    log.info("Processing: Table_3_1.csv")
    df = read_csv(CSV_DIR / "Table_3_1.csv", skip_rows=0)
    df = clean_column_names(df)
    df = replace_na_values(df)

    # Remove Total rows
    df = df[~df["sex"].astype(str).str.lower().str.strip().isin(["persons"])]
    df = df[~df["age_group"].astype(str).str.lower().str.strip().str.startswith("total")]

    # Melt states
    id_vars = ["sex", "age_group"]
    df = melt_states(df, id_vars=id_vars, value_name="presentations")

    # Normalise en-dash/em-dash to hyphen in age_group (e.g. "0–4" -> "0-4")
    df["age_group"] = (
        df["age_group"].astype(str)
        .str.replace("\u2013", "-", regex=False)   # en-dash
        .str.replace("\u2014", "-", regex=False)   # em-dash
        .str.strip()
    )

    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 3.2
# =============================================================================

def process_table_3_2() -> pd.DataFrame:
    """Emergency department presentations by indigenous status."""
    log.info("Processing: Table_3_2.csv")
    df = read_csv(CSV_DIR / "Table_3_2.csv", skip_rows=0)
    df = clean_column_names(df)
    df = drop_footnotes(df, "indigenous_status")
    df = replace_na_values(df)

    # Remove Total rows
    df = df[~df["indigenous_status"].astype(str).str.lower().str.strip().str.startswith("total")]

    # Melt states
    id_vars = ["indigenous_status", "measure"]
    df = melt_states(df, id_vars=id_vars, value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 3.3 (standalone)
# =============================================================================

def _load_table_3_3_raw() -> pd.DataFrame:
    """Load Table_3_3 and return cleaned long-format DataFrame."""
    df = read_csv(CSV_DIR / "Table_3_3.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "triage_category")
    df = replace_na_values(df)

    # Remove rows where triage_category starts with 'total'
    df = df[~df["triage_category"].astype(str).str.lower().str.strip().str.startswith("total")]

    # Remove 'Presentation rate ratio' rows
    df = df[~df["measure"].astype(str).str.lower().str.contains("rate ratio")]

    # Clean '(c)' suffix from measure names
    df["measure"] = df["measure"].astype(str).str.replace(r"\([a-z]\)$", "", regex=True).str.strip()

    # Remoteness area columns (keep everything except 'total')
    remoteness_cols = [c for c in df.columns if c not in ["triage_category", "measure"]
                       and not c.startswith("total")]
    df = df.melt(
        id_vars=["triage_category", "measure"],
        value_vars=remoteness_cols,
        var_name="remoteness_area",
        value_name="value",
    )
    # Clean remoteness area names
    df["remoteness_area"] = df["remoteness_area"].str.replace("_", " ").str.title().str.strip()
    return df


def process_table_3_3() -> pd.DataFrame:
    """ED presentations by triage category and remoteness area."""
    log.info("Processing: Table_3_3.csv")
    df = _load_table_3_3_raw()
    df = finalise(df, "value")
    return df


# =============================================================================
# TABLE 3.4
# =============================================================================

def process_table_3_4() -> pd.DataFrame:
    """ED presentations by triage category and socioeconomic status."""
    log.info("Processing: Table_3_4.csv")
    df = read_csv(CSV_DIR / "Table_3_4.csv", skip_rows=1)
    df = clean_column_names(df)

    # Keep only first 7 meaningful columns (drop formula/unnamed cols)
    meaningful_cols = [c for c in df.columns if not c.startswith("unnamed")]
    df = df[meaningful_cols]

    df = drop_footnotes(df, "triage_category")
    df = replace_na_values(df)

    # Remove Total(a) column and Total(b) triage rows
    df = drop_total_cols(df)
    df = df[~df["triage_category"].astype(str).str.lower().str.strip().str.startswith("total")]

    # Keep only Presentations measure
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]

    # Melt socioeconomic status columns (everything except triage_category and measure)
    ses_cols = [c for c in df.columns if c not in ["triage_category", "measure"]]
    df = df.melt(
        id_vars=["triage_category", "measure"],
        value_vars=ses_cols,
        var_name="socioeconomic_status",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.1
# =============================================================================

def process_table_4_1() -> pd.DataFrame:
    """ED presentations by type of visit and state."""
    log.info("Processing: Table_4_1.csv")
    df = read_csv(CSV_DIR / "Table_4_1.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "type_of_visit")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = df[~df["type_of_visit"].astype(str).str.lower().str.strip().str.startswith("total")]
    df = melt_states(df, id_vars=["type_of_visit"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.2
# =============================================================================

def process_table_4_2() -> pd.DataFrame:
    """ED presentations by triage category and arrival mode."""
    log.info("Processing: Table_4_2.csv")
    df = read_csv(CSV_DIR / "Table_4_2.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "triage_category")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")
    df = drop_total_rows(df, "arrival_mode")
    df = melt_states(df, id_vars=["triage_category", "arrival_mode"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.3
# =============================================================================

def process_table_4_3() -> pd.DataFrame:
    """ED presentations by age group, sex and triage category."""
    log.info("Processing: Table_4_3.csv")
    df = read_csv(CSV_DIR / "Table_4_3.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "sex")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = drop_total_rows(df, "sex")
    df = drop_total_rows(df, "age_group")

    # Melt triage category columns (everything except sex, age_group, measure, total)
    triage_cols = [c for c in df.columns
                   if c not in ["sex", "age_group", "measure"]
                   and not c.startswith("total")]
    df = df.melt(
        id_vars=["sex", "age_group"],
        value_vars=triage_cols,
        var_name="triage_category",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.4
# =============================================================================

def process_table_4_4() -> pd.DataFrame:
    """ED presentations by day of week and time of presentation."""
    log.info("Processing: Table_4_4.csv")
    df = read_csv(CSV_DIR / "Table_4_4.csv", skip_rows=1)
    # Keep only meaningful columns (drop unnamed/formula cols)
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df = clean_column_names(df)
    df = drop_footnotes(df, "time_of_presentation")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]

    day_cols = [c for c in df.columns
                if c not in ["time_of_presentation", "measure"]
                and not c.startswith("total")]
    df = df.melt(
        id_vars=["time_of_presentation"],
        value_vars=day_cols,
        var_name="day_of_week",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.5
# =============================================================================

def process_table_4_5() -> pd.DataFrame:
    """ED presentations by principal diagnosis and state."""
    log.info("Processing: Table_4_5.csv")
    df = read_csv(CSV_DIR / "Table_4_5.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = drop_total_rows(df, "principal_diagnosis")
    df = melt_states(df, id_vars=["principal_diagnosis"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.6
# =============================================================================

def process_table_4_6() -> pd.DataFrame:
    """ED presentations by principal diagnosis and triage category."""
    log.info("Processing: Table_4_6.csv")
    df = read_csv(CSV_DIR / "Table_4_6.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = drop_total_rows(df, "principal_diagnosis")

    triage_cols = [c for c in df.columns
                   if c != "principal_diagnosis" and not c.startswith("total")]
    df = df.melt(
        id_vars=["principal_diagnosis"],
        value_vars=triage_cols,
        var_name="triage_category",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.7
# =============================================================================

def process_table_4_7() -> pd.DataFrame:
    """ED presentations by principal diagnosis and admission status."""
    log.info("Processing: Table_4_7.csv")
    df = read_csv(CSV_DIR / "Table_4_7.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = drop_total_rows(df, "principal_diagnosis")

    admission_cols = [c for c in df.columns
                      if c not in ["principal_diagnosis", "measure"]
                      and not c.startswith("total")
                      and not c.startswith("all")]
    df = df.melt(
        id_vars=["principal_diagnosis"],
        value_vars=admission_cols,
        var_name="admission_status",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.8
# =============================================================================

def process_table_4_8() -> pd.DataFrame:
    """ED presentations by principal diagnosis and age group."""
    log.info("Processing: Table_4_8.csv")
    df = read_csv(CSV_DIR / "Table_4_8.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = drop_total_rows(df, "principal_diagnosis")

    age_cols = [c for c in df.columns
                if c != "principal_diagnosis" and not c.startswith("total")]
    df = df.melt(
        id_vars=["principal_diagnosis"],
        value_vars=age_cols,
        var_name="age_group",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.9
# =============================================================================

def process_table_4_9() -> pd.DataFrame:
    """Top 20 principal diagnoses by state."""
    log.info("Processing: Table_4_9.csv")
    df = read_csv(CSV_DIR / "Table_4_9.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = drop_total_rows(df, "principal_diagnosis")
    df = melt_states(df, id_vars=["principal_diagnosis"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.10
# =============================================================================

def process_table_4_10() -> pd.DataFrame:
    """Top 20 principal diagnoses for admitted patients by triage category."""
    log.info("Processing: Table_4_10.csv")
    df = read_csv(CSV_DIR / "Table_4_10.csv", skip_rows=1)
    # Drop unnamed/formula columns
    df = df[[c for c in df.columns if not str(c).startswith("Unnamed")]]
    df = clean_column_names(df)
    df = drop_footnotes(df, "principal_diagnosis")
    df = replace_na_values(df)
    df = drop_total_rows(df, "principal_diagnosis")

    triage_cols = [c for c in df.columns
                   if c != "principal_diagnosis" and not c.startswith("total")]
    df = df.melt(
        id_vars=["principal_diagnosis"],
        value_vars=triage_cols,
        var_name="triage_category",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.11
# =============================================================================

def process_table_4_11() -> pd.DataFrame:
    """Top 5 principal diagnoses by age group (already long format)."""
    log.info("Processing: Table_4_11.csv")
    df = read_csv(CSV_DIR / "Table_4_11.csv", skip_rows=1)
    df = clean_column_names(df)
    # Column may carry a footnote suffix (e.g. age_group_c)
    age_col = next((c for c in df.columns if c.startswith("age_group")), df.columns[0])
    df = df.rename(columns={age_col: "age_group"})
    df = drop_footnotes(df, "age_group")
    df = replace_na_values(df)
    df = drop_total_rows(df, "age_group")
    df = cast_numeric(df, "total")
    df = deduplicate(df)
    df = df.dropna(subset=["total"])
    log.info(f"  Shape after transforms: {df.shape}")
    return df


# =============================================================================
# TABLE 4.12
# =============================================================================

def process_table_4_12() -> pd.DataFrame:
    """ED presentations by triage category and episode end status."""
    log.info("Processing: Table_4_12.csv")
    df = read_csv(CSV_DIR / "Table_4_12.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "episode_end_status")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = drop_total_rows(df, "episode_end_status")

    triage_cols = [c for c in df.columns
                   if c not in ["episode_end_status", "measure"]
                   and not c.startswith("total")]
    df = df.melt(
        id_vars=["episode_end_status"],
        value_vars=triage_cols,
        var_name="triage_category",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.13
# =============================================================================

def process_table_4_13() -> pd.DataFrame:
    """ED presentations by episode end status and state."""
    log.info("Processing: Table_4_13.csv")
    df = read_csv(CSV_DIR / "Table_4_13.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "episode_end_status")
    df = replace_na_values(df)
    df = drop_total_rows(df, "episode_end_status")
    df = melt_states(df, id_vars=["episode_end_status"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.14
# =============================================================================

def process_table_4_14() -> pd.DataFrame:
    """Proportion admitted by triage category and state."""
    log.info("Processing: Table_4_14.csv")
    df = read_csv(CSV_DIR / "Table_4_14.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "triage_category")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")
    df = melt_states(df, id_vars=["triage_category"], value_name="proportion_admitted_pct")
    df = finalise(df, "proportion_admitted_pct")
    return df


# =============================================================================
# TABLE 4.15
# =============================================================================

def process_table_4_15() -> pd.DataFrame:
    """ED presentations by age group and episode end status."""
    log.info("Processing: Table_4_15.csv")
    df = read_csv(CSV_DIR / "Table_4_15.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "age_group")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = drop_total_rows(df, "age_group")

    end_status_cols = [c for c in df.columns
                       if c not in ["age_group", "measure"]
                       and not c.startswith("total")]
    df = df.melt(
        id_vars=["age_group"],
        value_vars=end_status_cols,
        var_name="episode_end_status",
        value_name="presentations",
    )
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 4.16
# =============================================================================

def process_table_4_16() -> pd.DataFrame:
    """ED presentations by funding source and state."""
    log.info("Processing: Table_4_16.csv")
    df = read_csv(CSV_DIR / "Table_4_16.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "funding_source")
    df = replace_na_values(df)
    df = df[df["measure"].astype(str).str.lower().str.strip() == "presentations"]
    df = drop_total_rows(df, "funding_source")
    df = melt_states(df, id_vars=["funding_source"], value_name="presentations")
    df = finalise(df, "presentations")
    return df


# =============================================================================
# TABLE 5.1
# =============================================================================

def process_table_5_1() -> pd.DataFrame:
    """Emergency presentation waiting time statistics over years."""
    log.info("Processing: Table_5_1.csv")
    df = read_csv(CSV_DIR / "Table_5_1.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "state_territory")
    df = replace_na_values(df)

    id_vars = ["state_territory", "measure"]
    df = melt_years(df, id_vars=id_vars, value_name="value")
    df = finalise(df, "value")
    return df


# =============================================================================
# TABLE 5.5
# =============================================================================

def process_table_5_5() -> pd.DataFrame:
    """Proportion seen on time by indigenous status and triage category."""
    log.info("Processing: Table_5_5.csv")
    df = read_csv(CSV_DIR / "Table_5_5.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "indigenous_status")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")
    df = melt_states(df, id_vars=["indigenous_status", "triage_category"],
                     value_name="proportion_seen_on_time_pct")
    df = finalise(df, "proportion_seen_on_time_pct")
    return df


# =============================================================================
# TABLE 5.6
# =============================================================================

def process_table_5_6() -> pd.DataFrame:
    """ED presentations and median waiting time by indigenous status and triage."""
    log.info("Processing: Table_5_6.csv")
    df = read_csv(CSV_DIR / "Table_5_6.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "indigenous_status")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")
    df = melt_states(df, id_vars=["indigenous_status", "triage_category", "measure"],
                     value_name="value")
    df = finalise(df, "value")
    return df


# =============================================================================
# TABLE 6.1
# =============================================================================

def process_table_6_1() -> pd.DataFrame:
    """90th percentile ED length of stay by admission status over years."""
    log.info("Processing: Table_6_1.csv")
    df = read_csv(CSV_DIR / "Table_6_1.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "admission_status")
    df = replace_na_values(df)

    id_vars = ["admission_status", "state_territory"]
    df = melt_years(df, id_vars=id_vars, value_name="90th_pct_length_of_stay_hhmm")
    df = replace_na_values(df)
    df = clean_string_values(df)
    df = deduplicate(df)
    df = df.dropna(subset=["90th_pct_length_of_stay_hhmm"])
    log.info(f"  Shape after transforms: {df.shape}")
    return df


# =============================================================================
# TABLE 6.2
# =============================================================================

def process_table_6_2() -> pd.DataFrame:
    """90th percentile ED length of stay by triage category, admission status and state."""
    log.info("Processing: Table_6_2.csv")
    df = read_csv(CSV_DIR / "Table_6_2.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "admission_status")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")
    df = melt_states(df, id_vars=["admission_status", "triage_category"],
                     value_name="90th_pct_length_of_stay_hhmm")
    df = replace_na_values(df)
    df = clean_string_values(df)
    df = deduplicate(df)
    df = df.dropna(subset=["90th_pct_length_of_stay_hhmm"])
    log.info(f"  Shape after transforms: {df.shape}")
    return df


# =============================================================================
# TABLE 6.3
# =============================================================================

def process_table_6_3() -> pd.DataFrame:
    """Proportion of presentations with LOS <= 4 hours, by state and year."""
    log.info("Processing: Table_6_3.csv")
    df = read_csv(CSV_DIR / "Table_6_3.csv", skip_rows=1)
    df = clean_column_names(df)
    # Column may carry a footnote suffix (e.g. admission_status_b)
    adm_col = next((c for c in df.columns if c.startswith("admission_status")), df.columns[0])
    df = df.rename(columns={adm_col: "admission_status"})
    df = drop_footnotes(df, "admission_status")
    df = replace_na_values(df)

    id_vars = ["admission_status", "state_territory"]
    df = melt_years(df, id_vars=id_vars, value_name="proportion_4hr_or_less_pct")
    df = finalise(df, "proportion_4hr_or_less_pct")
    return df


# =============================================================================
# COMBINED: Tables 6.4 + 6.5 + 6.6 + 5.2 + 5.3
#   -> ed_performance_by_peer_group_and_state
#   PK: (state, peer_group, triage_category, metric_type)
# =============================================================================

def _load_peer_group_state_table(filename: str, metric_type: str) -> pd.DataFrame:
    """Load a peer-group x state wide table and return long format with metric_type."""
    df = read_csv(CSV_DIR / filename, skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "peer_group")
    df = replace_na_values(df)
    # Remove total triage rows
    df = drop_total_rows(df, "triage_category")
    # Remove total peer group rows
    df = drop_total_rows(df, "peer_group")
    # Melt states
    df = melt_states(df, id_vars=["peer_group", "triage_category"], value_name="value")
    df["metric_type"] = metric_type
    return df


def process_combined_performance() -> pd.DataFrame:
    """Combine Tables 6.4, 6.5, 6.6, 5.2, 5.3 into one performance table."""
    log.info("Processing: Combined performance table (6.4+6.5+6.6+5.2+5.3)")
    parts = [
        _load_peer_group_state_table("Table_6_4.csv", "less_4hr_all"),
        _load_peer_group_state_table("Table_6_5.csv", "less_4hr_admitted"),
        _load_peer_group_state_table("Table_6_6.csv", "less_4hr_not_admitted"),
        _load_peer_group_state_table("Table_5_2.csv", "admission_rate"),
        _load_peer_group_state_table("Table_5_3.csv", "seen_on_time"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = finalise(df, "value")
    return df


# =============================================================================
# COMBINED: Tables 6.7 + 6.8
#   -> ed_duration_of_clinical_care
#   PK: (triage_category, duration, admission_status)
# =============================================================================

def _load_duration_table(filename: str, admission_status: str) -> pd.DataFrame:
    """Load a duration x triage-category table and return long format."""
    df = read_csv(CSV_DIR / filename, skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "duration")
    df = replace_na_values(df)
    # Keep only Presentations rows (exclude 'Proportion of total')
    mask = (
        df["measure"].astype(str).str.lower().str.contains("presentations")
        & ~df["measure"].astype(str).str.lower().str.contains("proportion")
    )
    df = df[mask]
    # Remove Total duration rows
    df = drop_total_rows(df, "duration")

    # Melt triage category columns
    triage_cols = [c for c in df.columns
                   if c not in ["duration", "measure"] and not c.startswith("total")]
    df = df.melt(
        id_vars=["duration"],
        value_vars=triage_cols,
        var_name="triage_category",
        value_name="presentations",
    )
    df["admission_status"] = admission_status
    return df


def process_combined_duration() -> pd.DataFrame:
    """Combine Tables 6.7 and 6.8 into one duration-of-clinical-care table."""
    log.info("Processing: Combined duration table (6.7+6.8)")
    parts = [
        _load_duration_table("Table_6_7.csv", "admitted"),
        _load_duration_table("Table_6_8.csv", "not_admitted"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df = finalise(df, "presentations")
    return df


# =============================================================================
# COMBINED: Tables 3.3 + 5.4
#   -> ed_presentations_by_triage_and_remoteness
#   PK: (triage_category, remoteness_area, measure)
# =============================================================================

def _load_table_5_4_raw() -> pd.DataFrame:
    """Load Table_5_4 and return long format with measure='median_time'."""
    df = read_csv(CSV_DIR / "Table_5_4.csv", skip_rows=1)
    df = clean_column_names(df)
    df = drop_footnotes(df, "triage_category")
    df = replace_na_values(df)
    df = drop_total_rows(df, "triage_category")

    remoteness_cols = [c for c in df.columns
                       if c != "triage_category" and not c.startswith("total")]
    df = df.melt(
        id_vars=["triage_category"],
        value_vars=remoteness_cols,
        var_name="remoteness_area",
        value_name="value",
    )
    df["remoteness_area"] = df["remoteness_area"].str.replace("_", " ").str.title().str.strip()
    df["measure"] = "median_time"
    return df


def process_combined_triage_remoteness() -> pd.DataFrame:
    """Combine Tables 3.3 and 5.4 into one triage-by-remoteness table."""
    log.info("Processing: Combined triage-remoteness table (3.3+5.4)")
    df_3_3 = _load_table_3_3_raw()
    df_3_3 = df_3_3.rename(columns={"measure": "measure"})  # already correct
    df_5_4 = _load_table_5_4_raw()

    # Align columns: both need triage_category, remoteness_area, measure, value
    df = pd.concat([df_3_3, df_5_4], ignore_index=True)
    df = finalise(df, "value")
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

PIPELINE = [
    # (process_fn, table_name)
    # 2.x tables
    (lambda: process_2x_file(FILE_CONFIGS_2X[0]),
     "emergency_department_presentations_by_peer_group"),
    (lambda: process_2x_file(FILE_CONFIGS_2X[1]),
     "emergency_department_presentations_by_state_territory"),
    # Specific tables
    (process_table_2_3, "presentations_by_peer_group_and_state"),
    (process_table_3_1, "ed_presentations_by_state_and_demographics"),
    (process_table_3_2, "ed_presentations_by_indigenous_status"),
    (process_table_3_3, "ed_presentations_by_triage_and_remoteness_area"),
    (process_table_3_4, "ed_presentations_by_triage_and_socioeconomic_status"),
    (process_table_4_1, "ed_presentations_by_type_of_visit_and_state"),
    (process_table_4_2, "ed_presentations_by_triage_and_arrival_mode"),
    (process_table_4_3, "ed_presentations_by_age_sex_and_triage"),
    (process_table_4_4, "ed_presentations_by_time_and_day"),
    (process_table_4_5, "ed_presentations_by_diagnosis_and_state"),
    (process_table_4_6, "ed_presentations_by_diagnosis_and_triage"),
    (process_table_4_7, "ed_presentations_by_diagnosis_and_admission_status"),
    (process_table_4_8, "ed_presentations_by_diagnosis_and_age"),
    (process_table_4_9, "ed_top20_diagnoses_by_state"),
    (process_table_4_10, "ed_top20_admitted_diagnoses_by_triage"),
    (process_table_4_11, "ed_top5_diagnoses_by_age_group"),
    (process_table_4_12, "ed_presentations_by_episode_end_status_and_triage"),
    (process_table_4_13, "ed_presentations_by_episode_end_status_and_state"),
    (process_table_4_14, "ed_proportion_admitted_by_triage_and_state"),
    (process_table_4_15, "ed_presentations_by_age_and_episode_end_status"),
    (process_table_4_16, "ed_presentations_by_funding_source_and_state"),
    (process_table_5_1, "ed_waiting_time_statistics_by_state_and_year"),
    (process_table_5_5, "ed_proportion_seen_on_time_by_indigenous_status"),
    (process_table_5_6, "ed_presentations_and_wait_time_by_indigenous_status"),
    (process_table_6_1, "ed_90th_pct_los_by_admission_status_and_year"),
    (process_table_6_2, "ed_90th_pct_los_by_triage_admission_and_state"),
    (process_table_6_3, "ed_proportion_4hr_los_by_state_and_year"),
    # Combined tables
    (process_combined_performance, "ed_performance_by_peer_group_and_state"),
    (process_combined_duration, "ed_duration_of_clinical_care"),
    (process_combined_triage_remoteness, "ed_presentations_by_triage_and_remoteness"),
]


def run():
    log.info("=" * 60)
    log.info("ETL START")
    log.info("=" * 60)

    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("Database connection OK")
    except Exception as e:
        log.error(f"Cannot connect to database: {e}")
        return

    # Build existing-tables set once to avoid per-table inspect() calls
    existing_tables: set = set(inspect(engine).get_table_names())

    for process_fn, table_name in PIPELINE:
        log.info(f"\n--- {table_name} ---")
        try:
            df = process_fn()
            if df is not None and not df.empty:
                load_to_postgres(df, table_name, engine, existing_tables)
            else:
                log.warning(f"  Empty result for '{table_name}' – skipping load")
        except Exception as e:
            log.error(f"  Failed '{table_name}': {e}", exc_info=True)

    log.info("=" * 60)
    log.info("ETL COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    run()
