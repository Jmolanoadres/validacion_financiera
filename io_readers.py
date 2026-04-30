from __future__ import annotations
import os
import pandas as pd

class InputError(Exception):
    pass


def detect_header_by_required_columns(
    df_preview: pd.DataFrame,
    required_columns: list[str],
    max_scan_rows: int = 50
) -> int:
    """
    Detecta la fila de encabezados buscando coincidencia
    con columnas esperadas (Oracle-style reports).
    """
    required_norm = {c.lower().strip() for c in required_columns}

    for idx in range(min(len(df_preview), max_scan_rows)):
        row_vals = (
            df_preview.iloc[idx]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
            .tolist()
        )

        matches = sum(1 for v in row_vals if v in required_norm)

        # si detecta varias columnas clave → es el header correcto
        if matches >= max(2, len(required_norm) // 2):
            return idx

    return -1


def read_table(
    path: str,
    skiprows: int | None = None,
    sheet_name: int | str = 0,
    required_columns_hint: list[str] | None = None
) -> pd.DataFrame:

    if not os.path.exists(path):
        raise InputError(f"No se encontró el archivo: {path}")

    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in (".xlsx", ".xls"):
            df_preview = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=None,
                nrows=60
            )

            if skiprows is not None:
                header_row = skiprows

            elif required_columns_hint:
                header_row = detect_header_by_required_columns(
                    df_preview,
                    required_columns_hint
                )
                if header_row < 0:
                    raise InputError(
                        f"No se pudo detectar encabezado usando columnas esperadas en {path}"
                    )
            else:
                raise InputError(
                    f"Debe indicarse skiprows o required_columns_hint para {path}"
                )

            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                header=header_row
            )

        elif ext == ".csv":
            df = pd.read_csv(path)

        else:
            raise InputError(f"Formato no soportado: {ext}")

    except Exception as e:
        raise InputError(f"Error leyendo {path}: {e}")

    # limpieza
    df = df.loc[:, ~df.columns.isna()]
    df = df.dropna(axis=1, how="all")

    if df.empty:
        raise InputError(f"{path} quedó vacío tras lectura.")

    return df