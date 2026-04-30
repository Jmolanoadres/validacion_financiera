from __future__ import annotations

import pandas as pd
from transformers import build_balance_agg


def _diff_df(
    merged: pd.DataFrame,
    *,
    key_col_balance: str,
    key_col_aux: str,
    pairs: list[tuple[str, str]],
    tolerance: float
) -> pd.DataFrame:
    """
    Crea tabla de diferencias por cada par (aux vs balance).
    Devuelve un dataframe “largo” con: cuenta, campo, valor_balance, valor_aux, diferencia, abs>tol
    """
    rows = []
    for aux_col, bal_col in pairs:
        sub = merged[[key_col_balance, key_col_aux, bal_col, aux_col]].copy()
        # elegir cuenta preferida
        sub["Cuenta"] = sub[key_col_balance].where(sub[key_col_balance].notna() & (sub[key_col_balance] != ""), sub[key_col_aux])
        sub["Campo"] = f"{aux_col} vs {bal_col}"
        sub["Valor Balance"] = sub[bal_col]
        sub["Valor Auxiliar"] = sub[aux_col]
        sub["Diferencia (Aux - Balance)"] = sub[aux_col] - sub[bal_col]
        sub["Fuera de tolerancia"] = sub["Diferencia (Aux - Balance)"].abs() > tolerance
        rows.append(sub[["Cuenta", "Campo", "Valor Balance", "Valor Auxiliar", "Diferencia (Aux - Balance)", "Fuera de tolerancia"]])

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    # filtrar solo donde haya dato en alguno y/o diferencia
    if not out.empty:
        out = out[(out["Valor Balance"].notna()) | (out["Valor Auxiliar"].notna())].copy()
    return out


def validate_v1_balance_vs_aux(
    df_balance_agg: pd.DataFrame,
    df_aux_agg: pd.DataFrame,
    *,
    aux_type: str,
    tolerance: float
) -> dict[str, pd.DataFrame]:
    """
    aux_type: "tercero" o "cuenta"
    Retorna dict de dataframes:
      - cuentas_sobrantes_auxiliar
      - cuentas_sobrantes_balance
      - diferencias_importes
    """
    # claves
    bal_key = "CUENTA PRINCIPAL"
    aux_key = "Cuenta Contable"

    # Outer merge para sobrantes + comparación
    merged = df_balance_agg.merge(
        df_aux_agg,
        left_on=bal_key,
        right_on=aux_key,
        how="outer",
        indicator=True,
        suffixes=("_bal", "_aux")
    )

    # Sobrantes
    sobrantes_aux = merged[merged["_merge"] == "right_only"][[aux_key]].copy()
    sobrantes_aux = sobrantes_aux.rename(columns={aux_key: "Cuenta Contable sobrante"}).drop_duplicates()

    sobrantes_bal = merged[merged["_merge"] == "left_only"][[bal_key]].copy()
    sobrantes_bal = sobrantes_bal.rename(columns={bal_key: "CUENTA PRINCIPAL sobrante"}).drop_duplicates()

    # Diferencias por importes (solo cuentas en ambos)
    both = merged[merged["_merge"] == "both"].copy()

    if aux_type == "tercero":
        pairs = [
            ("Saldo Inicial Tercero", "SALDO INICIAL"),
            ("Débito", "DÉBITO"),
            ("Crédito", "CRÉDITO"),
            ("Saldo Final Tercero", "SALDO FINAL"),
        ]
    else:
        pairs = [
            ("Saldo Inicial", "SALDO INICIAL"),
            ("Débito", "DÉBITO"),
            ("Crédito", "CRÉDITO"),
            ("Saldo Final", "SALDO FINAL"),
        ]

    diffs = _diff_df(
        both,
        key_col_balance=bal_key,
        key_col_aux=aux_key,
        pairs=pairs,
        tolerance=tolerance
    )

    return {
        "V1_Cuentas_Sobrantes_Auxiliar": sobrantes_aux,
        "V1_Cuentas_Sobrantes_Balance": sobrantes_bal,
        "V1_Diferencias_Importes": diffs,
    }


def validate_v2_transacciones(df_tx: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Reglas:
      1) Excluir Tipo de Comprobante == 'Transacciones Ppto'
      2) Numero de Comprobante: YYYY (1-4) y mm (5-6) debe coincidir con Fecha contable
      3) No duplicar Numero de Comprobante dentro del mismo Tipo de Comprobante
    """
    df = df_tx.copy()

    # Excluir
    df = df[df["Tipo de Comprobante"] != "Transacciones Ppto"].copy()

    # Fecha
    df["Fecha contable_parsed"] = pd.to_datetime(df["Fecha contable"], errors="coerce", dayfirst=True)
    expected_prefix = df["Fecha contable_parsed"].dt.strftime("%Y%m")

    num = df["Numero de Comprobante"].astype(str).fillna("")
    prefix_found = num.str.slice(0, 6)

    invalid_date = df["Fecha contable_parsed"].isna()
    invalid_prefix = (~invalid_date) & (prefix_found != expected_prefix)

    df_invalid_prefix = df[invalid_prefix | invalid_date].copy()
    if not df_invalid_prefix.empty:
        df_invalid_prefix["Año esperado"] = df_invalid_prefix["Fecha contable_parsed"].dt.strftime("%Y")
        df_invalid_prefix["Mes esperado"] = df_invalid_prefix["Fecha contable_parsed"].dt.strftime("%m")
        df_invalid_prefix["Prefijo esperado (YYYYmm)"] = df_invalid_prefix["Fecha contable_parsed"].dt.strftime("%Y%m")
        df_invalid_prefix["Prefijo encontrado"] = df_invalid_prefix["Numero de Comprobante"].astype(str).str.slice(0, 6)
        df_invalid_prefix["Motivo"] = df_invalid_prefix["Fecha contable_parsed"].isna().map(
            lambda x: "Fecha contable inválida" if x else "Prefijo Numero de Comprobante no coincide con Fecha contable"
        )

    # Duplicados por Tipo + Numero
    dup_mask = df.duplicated(subset=["Tipo de Comprobante", "Numero de Comprobante"], keep=False)
    df_dup = df[dup_mask].copy()

    return {
        "V2_NumComp_AnoMes_Invalido": df_invalid_prefix.drop(columns=["Fecha contable_parsed"], errors="ignore"),
        "V2_NumComp_Duplicado": df_dup.drop(columns=["Fecha contable_parsed"], errors="ignore"),
        "V2_Total_Evaluados": df  # para resumen
    }