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
    dup_mask = df.duplicated(subset=["Origen","Tipo de Comprobante", "Numero de Comprobante"], keep=False)
    df_dup = df[dup_mask].copy()

    return {
        "V2_NumComp_AnoMes_Invalido": df_invalid_prefix.drop(columns=["Fecha contable_parsed"], errors="ignore"),
        "V2_NumComp_Duplicado": df_dup.drop(columns=["Fecha contable_parsed"], errors="ignore"),
        "V2_Total_Evaluados": df  # para resumen
    }

def validate_aux_totales(
    df_aux_raw: pd.DataFrame,
    *,
    aux_type: str,
    tolerance: float = 0.01
) -> pd.DataFrame:
    """
    Valida que la sumatoria de Débito/Crédito sea consistente
    con la fila 'Total' del reporte.

    aux_type: "tercero" o "cuenta"
    """

    df = df_aux_raw.copy()

    # -----------------------------
    # 1) Detectar columna descripción
    # -----------------------------
    desc_candidates = [
        "Descripción de Líneas",
        "Descripcion de Linea",
        "Descripción de Linea",
        "Descripcion de Líneas",
    ]

    desc_col = None
    for c in desc_candidates:
        if c in df.columns:
            desc_col = c
            break

    if desc_col is None:
        raise ValueError(f"[{aux_type}] No se encontró columna de descripción")

    # -----------------------------
    # 2) Convertir valores numéricos
    # -----------------------------
    df["Débito"] = pd.to_numeric(df.get("Débito"), errors="coerce")
    df["Crédito"] = pd.to_numeric(df.get("Crédito"), errors="coerce")

    # -----------------------------
    # 3) Identificar fila TOTAL
    # -----------------------------
    is_total = (
        df[desc_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("total")
    )

    df_total = df[is_total]
    df_mov = df[~is_total]

    if df_total.empty:
        return pd.DataFrame([{
            "Tipo": aux_type,
            "Error": "No se encontró fila TOTAL"
        }])

    # Tomar primer total (Oracle suele tener uno)
    total_row = df_total.iloc[0]

    total_debito = total_row["Débito"]
    total_credito = total_row["Crédito"]

    # -----------------------------
    # 4) Sumatorias calculadas
    # -----------------------------
    sum_debito = df_mov["Débito"].sum()
    sum_credito = df_mov["Crédito"].sum()

    # -----------------------------
    # 5) Diferencias
    # -----------------------------
    diff_debito = sum_debito - total_debito
    diff_credito = sum_credito - total_credito

    return pd.DataFrame([
        {
            "Tipo": aux_type,
            "Campo": "Débito Total",
            "Valor Calculado": sum_debito,
            "Valor Reporte": total_debito,
            "Diferencia": diff_debito,
            "Fuera Tolerancia": abs(diff_debito) > tolerance
        },
        {
            "Tipo": aux_type,
            "Campo": "Crédito Total",
            "Valor Calculado": sum_credito,
            "Valor Reporte": total_credito,
            "Diferencia": diff_credito,
            "Fuera Tolerancia": abs(diff_credito) > tolerance
        }
    ])

def validate_aux_tercero_totales_detallado(
    df_aux: pd.DataFrame,
    *,
    tolerance: float = 0.01
) -> pd.DataFrame:
    """
    Valida consistencia entre:
    - Sumatoria de movimientos
    - Sumatoria de filas 'Total' (por tercero)

    Aplica a Libro Auxiliar por Tercero (Oracle Fusion)
    """

    df = df_aux.copy()

    # -----------------------------
    # 1) Detectar columna descripción
    # -----------------------------
    desc_col = None
    for c in [
        "Descripción de Líneas",
        "Descripcion de Lineas",
        "Descripción de Linea",
    ]:
        if c in df.columns:
            desc_col = c
            break

    if desc_col is None:
        raise ValueError("No se encontró columna de descripción")

    # -----------------------------
    # 2) Normalizar valores
    # -----------------------------
    df[desc_col] = df[desc_col].fillna("").astype(str).str.strip().str.lower()

    df["Débito"] = pd.to_numeric(df.get("Débito"), errors="coerce")
    df["Crédito"] = pd.to_numeric(df.get("Crédito"), errors="coerce")

    # -----------------------------
    # 3) Separar movimientos vs totales
    # -----------------------------
    is_total = df[desc_col] == "total"

    df_total = df[is_total]
    df_mov = df[~is_total]

    # -----------------------------
    # 4) Validar existencia de totales
    # -----------------------------
    if df_total.empty:
        return pd.DataFrame([{
            "Validación": "Aux Tercero Totales",
            "Resultado": "ERROR",
            "Detalle": "No existen filas 'Total'"
        }])

    # -----------------------------
    # 5) Sumatorias
    # -----------------------------
    total_debito_reporte = df_total["Débito"].sum()
    total_credito_reporte = df_total["Crédito"].sum()

    total_debito_calc = df_mov["Débito"].sum()
    total_credito_calc = df_mov["Crédito"].sum()

    # -----------------------------
    # 6) Diferencias
    # -----------------------------
    diff_debito = total_debito_calc - total_debito_reporte
    diff_credito = total_credito_calc - total_credito_reporte

    # -----------------------------
    # 7) Resultado estructurado
    # -----------------------------
    return pd.DataFrame([
        {
            "Campo": "Débito",
            "Valor Calculado": total_debito_calc,
            "Valor Reporte (Totales)": total_debito_reporte,
            "Diferencia": diff_debito,
            "Fuera Tolerancia": abs(diff_debito) > tolerance
        },
        {
            "Campo": "Crédito",
            "Valor Calculado": total_credito_calc,
            "Valor Reporte (Totales)": total_credito_reporte,
            "Diferencia": diff_credito,
            "Fuera Tolerancia": abs(diff_credito) > tolerance
        }
    ])

def split_aux_by_cuenta(df: pd.DataFrame) -> list[pd.DataFrame]:

    blocks = []
    current_block = []

    for _, row in df.iterrows():

        texto = " ".join([str(v) for v in row.values if pd.notna(v)])

        # Detecta inicio de nueva cuenta
        if "Cuenta:" in texto and current_block:
            blocks.append(pd.DataFrame(current_block))
            current_block = []

        current_block.append(row)

    if current_block:
        blocks.append(pd.DataFrame(current_block))

    return blocks


def transform_aux_cuenta_por_bloques(df_aux_raw: pd.DataFrame):

    blocks = split_aux_by_cuenta(df_aux_raw)

    rows = []

    for block in blocks:

        # ------------------------
        # Detectar cuenta
        # ------------------------
        cuenta = None
        for _, r in block.iterrows():
            vals = [str(v) for v in r.values if pd.notna(v)]
            for v in vals:
                if v.strip().isdigit():
                    cuenta = v.strip()
                    break
            if cuenta:
                break

        if not cuenta:
            continue

        # ------------------------
        # Identificar columnas
        # ------------------------
        if "Débito" not in block.columns or "Crédito" not in block.columns:
            continue

        block["Débito"] = pd.to_numeric(block["Débito"], errors="coerce")
        block["Crédito"] = pd.to_numeric(block["Crédito"], errors="coerce")

        # ------------------------
        # Detectar totales
        # ------------------------
        desc_col = None
        for c in ["Descripcion de Linea"]:
            if c in block.columns:
                desc_col = c
                break

        if desc_col is None:
            continue

        is_total = (
            block[desc_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .eq("total")
        )

        df_total = block[is_total]
        df_mov = block[~is_total]

        if df_total.empty:
            continue

        total_row = df_total.iloc[0]

        debito_total = total_row["Débito"]
        credito_total = total_row["Crédito"]

        # ------------------------
        # Sumatorias reales
        # ------------------------
        debitos = df_mov["Débito"].sum()
        creditos = df_mov["Crédito"].sum()

        # ------------------------
        # Saldos (del header del bloque)
        # ------------------------
        saldo_ini = None
        saldo_fin = None

        texto_block = " ".join(block.astype(str).values.flatten())

        # muy robusto sin depender de posición exacta
        # puedes mejorar con regex si quieres
        # (opcional)

        rows.append({
            "Cuenta Contable": cuenta,
            "Saldo Inicial": saldo_ini,
            "Débito": debitos,
            "Crédito": creditos,
            "Saldo Final": saldo_fin,
            "Débito Total Reporte": debito_total,
            "Crédito Total Reporte": credito_total,
        })

    return pd.DataFrame(rows)
