from __future__ import annotations

import os
import pandas as pd


class InputError(Exception):
    pass


def _try_read_csv(path: str, skiprows: int = 0) -> pd.DataFrame:
    # Intenta ; y luego , como separador (sin inferir demasiado)
    last_exc = None
    for sep in [";", ","]:
        try:
            return pd.read_csv(
                path,
                sep=sep,
                skiprows=skiprows,
                dtype=str,            # leer como texto primero (normalización posterior)
                encoding="utf-8",
                engine="python"
            )
        except Exception as exc:
            last_exc = exc
            continue
    raise InputError(f"No fue posible leer CSV {path}. Error: {last_exc}")


def read_table(
    path: str,
    *,
    skiprows: int = 0,
    sheet: str | int | None = None
) -> pd.DataFrame:
    """
    Lee .xlsx o .csv. Para .xlsx, por defecto lee la primera hoja (sheet=0)
    salvo que se indique 'sheet'.
    """
    if not path:
        raise InputError("Ruta vacía.")
    if not os.path.exists(path):
        raise InputError(f"No existe el archivo: {path}")

    ext = os.path.splitext(path.lower())[1]

    if ext in [".xlsx", ".xlsm", ".xls"]:
        # Por defecto, primera hoja
        sheet_name = 0 if sheet is None else sheet
        try:
            df = pd.read_excel(
                path,
                sheet_name=sheet_name,
                skiprows=skiprows,
                dtype=str,
                engine="openpyxl" if ext in [".xlsx", ".xlsm"] else None
            )
            return df
        except Exception as exc:
            raise InputError(f"No fue posible leer Excel {path}. Error: {exc}") from exc

    if ext == ".csv":
        return _try_read_csv(path, skiprows=skiprows)

    raise InputError(f"Extensión no soportada: {ext}. Use .xlsx o .csv")