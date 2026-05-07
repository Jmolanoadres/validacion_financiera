from __future__ import annotations

import re
import unicodedata
import pandas as pd


class SchemaError(Exception):
    pass


def normalize_col_name(col: str) -> str:
    # Normaliza espacios y trims, mantiene tildes para “display”
    if col is None:
        return ""
    col = str(col).strip()
    col = re.sub(r"\s+", " ", col)
    return col


def _fold_for_match(s: str) -> str:
    """
    “Fold” para comparar robustamente: minúsculas + sin tildes + sin dobles espacios.
    """
    s = normalize_col_name(s).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def resolve_required_columns(
    df: pd.DataFrame,
    required: dict[str, list[str]],
    *,
    context: str
) -> pd.DataFrame:
    """
    required: {canonical_name: [synonyms...]}
    - Si encuentra un sinónimo, renombra a canonical_name.
    - Si no encuentra, lanza SchemaError con sugerencias.
    """
    df = df.copy()
    cols = list(df.columns)
    folded_map = {_fold_for_match(c): c for c in cols}

    renames: dict[str, str] = {}
    missing: list[str] = []

    for canonical, synonyms in required.items():
        candidates = [canonical] + (synonyms or [])
        found_original = None
        for cand in candidates:
            key = _fold_for_match(cand)
            if key in folded_map:
                found_original = folded_map[key]
                break
        if found_original is None:
            missing.append(canonical)
        else:
            renames[found_original] = canonical

    if missing:
        # sugerencias (hasta 20) para ayudar al usuario
        suggestions = ", ".join(cols[:20])
        raise SchemaError(
            f"[{context}] Faltan columnas requeridas: {missing}. "
            f"Columnas detectadas (muestra): {suggestions}"
        )

    return df.rename(columns=renames)


def clean_account_key(x) -> str:
    """
    Normaliza cuentas:
    - str, strip
    - elimina sufijo .0 típico de Excel
    - conserva ceros a la izquierda al no convertir a int
    """
    if pd.isna(x):
        return ""
    s = str(x).strip()
    # Eliminar .0 final
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    # También si viene como '110505,0' (raro) no lo tocamos, se manejará luego
    return s


def parse_number(x):
    """
    Convierte importes a float robustamente:
    - Maneja miles '.' y decimal ',': '73.274.131,02' -> 73274131.02
    - Maneja decimal ',' sin miles: '123,45' -> 123.45
    - Maneja miles ',' y decimal '.': '1,234.56' -> 1234.56
    - Si no se puede, devuelve NaN
    """
    if x is None or (isinstance(x, float) and pd.isna(x)) or (isinstance(x, str) and x.strip() == ""):
        return pd.NA

    s = str(x).strip()
    s = s.replace("\u00A0", " ").replace(" ", "")

    # Si tiene . y , asumimos:
    # - si el último separador es ',' => decimal ',' y '.' miles
    # - si el último separador es '.' => decimal '.' y ',' miles
    if "." in s and "," in s:
        if s.rfind(",") > s.rfind("."):
            # '.' miles, ',' decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # ',' miles, '.' decimal
            s = s.replace(",", "")
    else:
        # Solo ',': asumimos decimal ','
        if "," in s and "." not in s:
            s = s.replace(".", "")  # por si hay alguno extraño
            s = s.replace(",", ".")
        # Solo '.': si hay más de uno, probablemente miles
        elif "." in s and "," not in s:
            if s.count(".") > 1:
                s = s.replace(".", "")

    # Quitar caracteres no numéricos salvo signo y punto
    s = re.sub(r"[^0-9\.\-]", "", s)
    try:
        return float(s)
    except Exception:
        return pd.NA


def to_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(parse_number)
    return df


def build_balance_agg(df_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa Balance por CUENTA PRINCIPAL y suma SALDO INICIAL, DÉBITO, CRÉDITO, SALDO FINAL
    """
    df = df_balance.copy()
    df["CUENTA PRINCIPAL"] = df["CUENTA PRINCIPAL"].map(clean_account_key)

    num_cols = ["SALDO INICIAL", "DÉBITO", "CRÉDITO", "SALDO FINAL"]
    df = to_numeric_columns(df, num_cols)

    agg = (
        df.groupby("CUENTA PRINCIPAL", dropna=False)[num_cols]
        .sum(min_count=1)
        .reset_index()
    )
    return agg


def transform_aux_tercero(df_aux: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica reglas:
    - elimina Descripción de Líneas == Total
    - reemplaza resto por 'movimientos' excepto Saldo Inicial/Saldo Final
    - agrega por cuenta: Saldo Inicial Tercero, Débito, Crédito, Saldo Final Tercero
    """
    df = df_aux.copy()
    df["Cuenta Contable"] = df["Cuenta Contable"].map(clean_account_key)
    df["Descripción de Líneas"] = df["Descripción de Líneas"].fillna('').astype(str).str.strip()

    # Eliminar Total (case-insensitive)
    df = df[df["Descripción de Líneas"].str.lower() != "total"].copy()

    def _map_desc(v: str) -> str:
        v2 = v.strip()
        if v2 == "Saldo Inicial":
            return "Saldo Inicial"
        if v2 == "Saldo Final":
            return "Saldo Final"
        return "movimientos"

    df["Descripción de Líneas"] = df["Descripción de Líneas"].map(_map_desc)

    num_cols = ["Saldo Inicial Tercero", "Débito", "Crédito", "Saldo Final Tercero"]
    df = to_numeric_columns(df, num_cols)

    # Agregación por Cuenta y Descripción (como pediste)
    grouped = (
        df.groupby(["Cuenta Contable", "Descripción de Líneas"], dropna=False)[num_cols]
        .sum(min_count=1)
        .reset_index()
    )

    # Consolidado por cuenta para comparar con Balance
    # - saldo inicial: filas etiqueta 'Saldo Inicial' de 'Saldo Inicial Tercero'
    # - saldo final: filas etiqueta 'Saldo Final' de 'Saldo Final Tercero'
    # - débito/crédito: sum de todo (movimientos normalmente)
    by_account = (
        grouped.groupby("Cuenta Contable", dropna=False)
        .apply(lambda g: pd.Series({
            "Saldo Inicial Tercero": g.loc[g["Descripción de Líneas"] == "Saldo Inicial", "Saldo Inicial Tercero"].sum(min_count=1),
            "Débito": g["Débito"].sum(min_count=1),
            "Crédito": g["Crédito"].sum(min_count=1),
            "Saldo Final Tercero": g.loc[g["Descripción de Líneas"] == "Saldo Final", "Saldo Final Tercero"].sum(min_count=1),
        }))
        .reset_index()
    )

    return by_account


def transform_aux_cuenta(
    df_aux: pd.DataFrame,
    *,
    saldo_inicial_hint: float | None = None,
    saldo_final_hint: float | None = None
) -> pd.DataFrame:

    df = df_aux.copy()

    # Normalizar cuenta
    df["Cuenta Contable"] = df["Cuenta Contable"].fillna("").astype(str).str.strip()
    df = df[df["Cuenta Contable"] != ""]

    df["Cuenta Contable"] = df["Cuenta Contable"].map(clean_account_key)

    # Numéricos
    df = to_numeric_columns(df, ["Débito", "Crédito"])

    # Eliminar fila TOTAL
    is_total = df.apply(
        lambda r: any(isinstance(v, str) and _fold_for_match(v) == "total" for v in r.values),
        axis=1
    )

    df_mov = df[~is_total]

    rows = []

    for cuenta, g in df_mov.groupby("Cuenta Contable", dropna=False):

        debitos = g["Débito"].sum(min_count=1)
        creditos = g["Crédito"].sum(min_count=1)

        # USAR SIEMPRE los hints
        saldo_ini = saldo_inicial_hint if saldo_inicial_hint is not None else 0.0

        if saldo_final_hint is not None:
            saldo_fin = saldo_final_hint
        else:
            saldo_fin = saldo_ini + debitos - creditos

        rows.append({
            "Cuenta Contable": cuenta,
            "Saldo Inicial": saldo_ini,
            "Débito": debitos,
            "Crédito": creditos,
            "Saldo Final": saldo_fin,
        })

    return pd.DataFrame(rows)