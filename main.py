from __future__ import annotations

import argparse
import logging
from datetime import datetime
import os
import pandas as pd

from io_readers import read_table, InputError
from transformers import (
    standardize_columns, resolve_required_columns,
    build_balance_agg, transform_aux_tercero, transform_aux_cuenta,
    to_numeric_columns
)
from validators import validate_v1_balance_vs_aux, validate_v2_transacciones
from report_writer import write_report


VERSION = "1.0.0"


def debug_df(df, name: str):
    cols = list(map(str, df.columns))
    logging.info(f"[DEBUG] {name}")
    logging.info(f"        Filas: {len(df)}")
    logging.info(f"        Columnas ({len(cols)}): {cols}")


def ensure_not_empty(df, context: str):
    if df is None or df.empty:
        raise SystemExit(f"{context}: dataframe vacío. Revisa estructura del archivo o encabezados.")


def setup_logger(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "validacion.log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # limpiar handlers previos (por si se llama desde notebook)
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return log_path


def summarize_counts(rule_name: str, evaluated: int, findings: int) -> dict:
    pct = (findings / evaluated * 100.0) if evaluated else 0.0
    return {
        "Validacion": rule_name.split("_")[0],
        "Regla": rule_name,
        "Registros evaluados": evaluated,
        "Registros con hallazgo": findings,
        "% hallazgos": round(pct, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Validador financiero (Balance/Auxiliares + Transacciones)")
    parser.add_argument("--balance", required=True, help="Ruta Balance_prueba (.xlsx o .csv)")
    parser.add_argument("--aux-cuenta", default=None, help="Ruta Libro_auxiliar_cuenta (.xlsx o .csv) [opcional]")
    parser.add_argument("--aux-tercero", default=None, help="Ruta Libro_auxiliar_tercero (.xlsx o .csv) [opcional]")
    parser.add_argument("--transacciones", required=True, help="Ruta Transacciones (.xlsx o .csv)")
    parser.add_argument("--out", default="resultado_validaciones.xlsx", help="Salida Excel (por defecto resultado_validaciones.xlsx)")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerancia para diferencias (por defecto 0.01)")
    args = parser.parse_args()

    out_path = args.out
    out_dir = os.path.dirname(out_path) if os.path.dirname(out_path) else "."
    log_path = setup_logger(out_dir)

    logging.info("=== Inicio validador ===")
    logging.info(f"Versión: {VERSION}")
    logging.info(f"Balance: {args.balance}")
    logging.info(f"Aux cuenta: {args.aux_cuenta}")
    logging.info(f"Aux tercero: {args.aux_tercero}")
    logging.info(f"Transacciones: {args.transacciones}")
    logging.info(f"Salida Excel: {out_path}")
    logging.info(f"Tolerancia: {args.tolerance}")

    # Regla: para V1 debe venir al menos un auxiliar
    if not args.aux_cuenta and not args.aux_tercero:
        raise SystemExit("Para la validación 1 debes proporcionar --aux-cuenta o --aux-tercero (al menos uno).")

    # 1) Leer Balance
    try:
        df_balance_raw = read_table(args.balance)
        ensure_not_empty(df_balance_raw, "Balance_prueba")
        debug_df(df_balance_raw, "Balance RAW")
    except InputError as e:
        raise SystemExit(str(e))

    df_balance_raw = standardize_columns(df_balance_raw)
    logging.info(f"Balance filas: {len(df_balance_raw)} | columnas: {list(df_balance_raw.columns)}")

    # Equivalencias robustas (tildes/variantes)
    balance_required = {
        "CUENTA PRINCIPAL": ["Cuenta Principal", "CUENTA_PRINCIPAL", "Cuenta principal"],
        "SALDO INICIAL": ["Saldo Inicial", "SALDO_INICIAL", "Saldo inicial"],
        "DÉBITO": ["DEBITO", "Debito", "Débito", "DEBITO "],
        "CRÉDITO": ["CREDITO", "Credito", "Crédito"],
        "SALDO FINAL": ["Saldo Final", "SALDO_FINAL", "Saldo final"],
    }

    df_balance = resolve_required_columns(df_balance_raw, balance_required, context="Balance_prueba")
    df_balance_agg = build_balance_agg(df_balance)
    logging.info(f"Balance agregado filas (cuentas únicas): {len(df_balance_agg)}")
    logging.info(f"Balance agregado primeras 10 filas:\n{df_balance_agg.head(10)}")

    v1_outputs = {}
    v1_eval_count = len(df_balance_agg)

    # 2) Validación 1: Aux tercero
    if args.aux_tercero:
        aux_t_required = {
        "Cuenta Contable": ["CUENTA CONTABLE", "Cuenta Contable"],
        "Descripción de Líneas": ["Descripción de Líneas", "Descripcion de Lineas"],
        "Saldo Inicial Tercero": ["Saldo Inicial Tercero"],
        "Débito": ["Débito", "Debito"],
        "Crédito": ["Crédito", "Credito"],
        "Saldo Final Tercero": ["Saldo Final Tercero"],
        } 
        df_aux_t_raw = read_table(args.aux_tercero,
        required_columns_hint=list(aux_t_required.keys()))
        ensure_not_empty(df_aux_t_raw, "Libro_auxiliar_tercero") 
        df_aux_t_raw = standardize_columns(df_aux_t_raw)
        debug_df(df_aux_t_raw, "Aux Tercero RAW")
        aux_t_required = {
            "Cuenta Contable": ["CUENTA CONTABLE", "Cuenta", "CuentaContable"],
            "Descripción de Líneas": ["Descripcion de Lineas", "Descripción de lineas", "DESCRIPCIÓN DE LÍNEAS"],
            "Saldo Inicial Tercero": ["SALDO INICIAL TERCERO", "Saldo inicial tercero"],
            "Débito": ["DEBITO", "Debito", "Débito"],
            "Crédito": ["CREDITO", "Credito", "Crédito"],
            "Saldo Final Tercero": ["SALDO FINAL TERCERO", "Saldo final tercero"],
        }

        df_aux_t = resolve_required_columns(df_aux_t_raw, aux_t_required, context="Libro_auxiliar_tercero")
        df_aux_t_agg = transform_aux_tercero(df_aux_t)

        out = validate_v1_balance_vs_aux(df_balance_agg, df_aux_t_agg, aux_type="tercero", tolerance=args.tolerance)
        # prefijar hojas para distinguir si también corre aux cuenta
        for k, v in out.items():
            v1_outputs[f"{k}_TERCERO"] = v

    # 3) Validación 1: Aux cuenta
        
    if args.aux_cuenta:
        aux_c_required = {
            "Cuenta Contable": ["Cuenta Contable"],
            "Descripción de Líneas": ["Descripcion de Linea", "Descripción de Líneas"],
            "Débito": ["Débito", "Debito"],
            "Crédito": ["Crédito", "Credito"],
        }

        df_aux_c_raw = read_table(args.aux_cuenta,
            required_columns_hint=list(aux_c_required.keys())        )
        ensure_not_empty(df_aux_c_raw, "Libro_auxiliar_cuenta")
        debug_df(df_aux_c_raw, "Aux Cuenta RAW")

        df_aux_c_raw = standardize_columns(df_aux_c_raw)
        logging.info(f"Aux cuenta filas (raw): {len(df_aux_c_raw)} | columnas: {list(df_aux_c_raw.columns)}")

        aux_c_required = {
            "Cuenta Contable": ["CUENTA CONTABLE", "Cuenta", "CuentaContable"],
            "Descripción de Líneas": ["Descripcion de Lineas", "Descripción de lineas", "DESCRIPCIÓN DE LÍNEAS"],
            "Débito": ["DEBITO", "Debito", "Débito"],
            "Crédito": ["CREDITO", "Credito", "Crédito"],
        }
        # OJO: Saldo Inicial/Final pueden o no venir como columnas; se gestionan en transformer
        df_aux_c = resolve_required_columns(df_aux_c_raw, aux_c_required, context="Libro_auxiliar_cuenta")
        df_aux_c_agg = transform_aux_cuenta(df_aux_c)
        logging.info(f"Aux cuenta agregado (cuentas únicas): {len(df_aux_c_agg)}")
        logging.info(f"Aux cuenta agregado primeras 10 filas:\n{df_aux_c_agg.head(10)}")

        out = validate_v1_balance_vs_aux(df_balance_agg, df_aux_c_agg, aux_type="cuenta", tolerance=args.tolerance)
        for k, v in out.items():
            v1_outputs[f"{k}_CUENTA"] = v

    # 4) Leer Transacciones
    try:
        df_tx_raw = read_table(args.transacciones)
        ensure_not_empty(df_tx_raw, "Transacciones")
        debug_df(df_tx_raw, "Transacciones RAW")

    except InputError as e:
        raise SystemExit(str(e))

    df_tx_raw = standardize_columns(df_tx_raw)
    logging.info(f"Transacciones filas: {len(df_tx_raw)} | columnas: {list(df_tx_raw.columns)}")

    tx_required = {
        "Tipo de Comprobante": ["Tipo Comprobante", "TIPO DE COMPROBANTE", "Tipo_de_Comprobante"],
        "Numero de Comprobante": ["Número de Comprobante", "NUMERO DE COMPROBANTE", "Numero_de_Comprobante"],
        "Fecha contable": ["Fecha Contable", "FECHA CONTABLE", "Fecha_contable"],
    }
    from transformers import resolve_required_columns as _rrc
    df_tx = _rrc(df_tx_raw, tx_required, context="Transacciones")

    v2 = validate_v2_transacciones(df_tx)
    df_tx_eval = v2.pop("V2_Total_Evaluados")
    v2_outputs = {
        "V2_NumComp_AnoMes_Invalido": v2["V2_NumComp_AnoMes_Invalido"],
        "V2_NumComp_Duplicado": v2["V2_NumComp_Duplicado"],
    }

    # 5) Construir Resumen
    resumen_rows = []

    # V1: evaluados aproximados = cuentas en balance agregado
    for sheet_name, df in v1_outputs.items():
        findings = 0 if df is None else len(df)
        resumen_rows.append(summarize_counts(sheet_name, v1_eval_count, findings))

    # V2: evaluados = transacciones tras excluir "Transacciones Ppto"
    v2_eval_count = len(df_tx_eval)
    for sheet_name, df in v2_outputs.items():
        findings = 0 if df is None else len(df)
        resumen_rows.append(summarize_counts(sheet_name, v2_eval_count, findings))

    resumen_df = pd.DataFrame(resumen_rows)

    # 6) README
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    readme_rows = [
        ("Fecha/Hora ejecución", now),
        ("Versión", VERSION),
        ("Balance_prueba", args.balance),
        ("Libro_auxiliar_cuenta", args.aux_cuenta or ""),
        ("Libro_auxiliar_tercero", args.aux_tercero or ""),
        ("Transacciones", args.transacciones),
        ("Salida", out_path),
        ("Tolerancia", str(args.tolerance)),
        ("Skiprows Aux Tercero", "18 (datos desde fila 19)"),
        ("Skiprows Aux Cuenta", "15 (datos desde fila 16)"),
        ("Hojas Excel (si aplica)", "Por defecto se lee la primera hoja (sheet=0)"),
        ("Log", log_path),
    ]

    # 7) Escribir reporte
    all_sheets = {}
    all_sheets.update(v1_outputs)
    all_sheets.update(v2_outputs)

    write_report(
        out_path,
        readme_rows=readme_rows,
        resumen_df=resumen_df,
        sheets=all_sheets
    )

    logging.info("=== Fin validador ===")
    logging.info(f"Archivo generado: {out_path}")
    logging.info(f"Log generado: {log_path}")


if __name__ == "__main__":
    main()
