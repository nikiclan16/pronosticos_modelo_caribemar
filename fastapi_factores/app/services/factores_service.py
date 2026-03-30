from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

from app.db import fetch_all, fetch_one, execute, execute_many, get_connection
from app.utils.sql import build_in_clause


DATE_FMT = "%Y-%m-%d"


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, DATE_FMT)
    except ValueError:
        # Accept dd-mm-yyyy from existing UI
        return datetime.strptime(value, "%d-%m-%Y")


def _period_params(periodos: List[float]) -> Dict[str, Any]:
    if len(periodos) != 24:
        raise ValueError("periodos must have 24 values")
    return {f"p{i+1}": periodos[i] for i in range(24)}


# --- Barras ---

def consultar_barras_index(dsn: Optional[str] = None):
    sql = (
        "SELECT id, barra, descripcion, nivel_tension, observaciones, habilitar, estado, mc "
        "FROM barras WHERE estado = 1 ORDER BY id"
    )
    return fetch_all(sql, dsn=dsn)


def consultar_barras_index_xmc(mc: str, dsn: Optional[str] = None):
    sql = (
        "SELECT id, barra, descripcion, nivel_tension, observaciones, habilitar, estado, mc "
        "FROM barras WHERE mc = %(mc)s AND estado = 1 ORDER BY id"
    )
    return fetch_all(sql, {"mc": mc}, dsn=dsn)


def consultar_barra(barra_id: str):
    sql = (
        "SELECT id, barra, descripcion, nivel_tension, observaciones, habilitar, estado, mc "
        "FROM barras WHERE id = %(id)s AND estado = 1 ORDER BY id"
    )
    return fetch_all(sql, {"id": barra_id})


def consultar_barra_nombre(barra: str, dsn: Optional[str] = None):
    sql = (
        "SELECT codigo_rpm FROM barras b "
        "INNER JOIN agrupaciones a ON b.id=a.barra_id "
        "WHERE barra = %(barra)s AND b.estado='1' AND a.estado='1' GROUP BY codigo_rpm"
    )
    return fetch_all(sql, {"barra": barra}, dsn=dsn)


def consultar_barra_flujo_nombre(barra: str, tipo: str, codigo_rpm: List[str]):
    in_clause, params = build_in_clause(codigo_rpm, "codigo_rpm")
    params.update({"barra": barra, "tipo": tipo})
    sql = (
        "SELECT flujo FROM barras b "
        "INNER JOIN agrupaciones a ON b.id=a.barra_id "
        f"WHERE barra = %(barra)s AND codigo_rpm IN {in_clause} "
        "AND substring(flujo from 1 for 1) = %(tipo)s AND b.estado='1' AND a.estado='1' "
        "GROUP BY flujo"
    )
    return fetch_all(sql, params)


def consultar_barra_factor_nombre(barra: str, tipo: str, codigo_rpm: List[str], dsn: Optional[str] = None):
    in_clause, params = build_in_clause(codigo_rpm, "codigo_rpm")
    params.update({"barra": barra, "tipo": tipo})
    sql = (
        "SELECT factor, codigo_rpm, flujo FROM barras b "
        "INNER JOIN agrupaciones a ON b.id=a.barra_id "
        f"WHERE barra = %(barra)s AND codigo_rpm IN {in_clause} "
        "AND substring(flujo from 1 for 1) = %(tipo)s AND b.estado='1' AND a.estado='1'"
    )
    return fetch_all(sql, params, dsn=dsn)


def consultar_barra_flujo_nombre_inicial(barra: str, tipo: str):
    sql = (
        "SELECT flujo FROM barras b "
        "INNER JOIN agrupaciones a ON b.id=a.barra_id "
        "WHERE barra = %(barra)s AND substring(flujo from 1 for 1) = %(tipo)s "
        "AND b.estado='1' AND a.estado='1' GROUP BY flujo"
    )
    return fetch_all(sql, {"barra": barra, "tipo": tipo})


def consultar_barras_mercado(mc: str):
    sql = (
        "SELECT barra, a.factor, a.codigo_rpm FROM barras b "
        "INNER JOIN ("
        "SELECT barra_id, STRING_AGG(DISTINCT factor,'*') AS factor, "
        "STRING_AGG(DISTINCT CONCAT_WS('','*',codigo_rpm,'*'),',') AS codigo_rpm "
        "FROM agrupaciones WHERE estado='1' GROUP BY barra_id"
        ") AS a ON b.id = a.barra_id "
        "WHERE estado='1' AND b.mc = %(mc)s GROUP BY barra, a.factor, a.codigo_rpm ORDER BY barra ASC"
    )
    return fetch_all(sql, {"mc": mc})


def consultar_barra_x_codigo_rpm(codigo_rpm: str):
    sql = (
        "SELECT BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, "
        "BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, "
        "BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, "
        "AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, "
        "AG.estado As AGestado, AG.factor As AGfactor "
        "FROM barras BA "
        "INNER JOIN agrupaciones AG on (BA.id = AG.barra_id) "
        "INNER JOIN medidas ME on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "WHERE ME.codigo_rpm = %(codigo_rpm)s AND AG.estado = 1 "
        "ORDER BY ME.fecha limit 1;"
    )
    return fetch_all(sql, {"codigo_rpm": codigo_rpm})


# --- Agrupaciones ---

def consultar_agrupaciones_index():
    sql = "SELECT id, barra_id, codigo_rpm, flujo, habilitar, revision, estado, factor FROM agrupaciones WHERE estado = 1 ORDER BY id"
    return fetch_all(sql)


def consultar_agrupaciones_index_x_barra_id(barra_id: str):
    sql = "SELECT id, barra_id, codigo_rpm, flujo, habilitar, revision, estado, factor FROM agrupaciones WHERE barra_id = %(barra_id)s AND estado = 1 ORDER BY id"
    return fetch_all(sql, {"barra_id": barra_id})


def consultar_agrupacion(agrupacion_id: str):
    sql = "SELECT id, barra_id, codigo_rpm, flujo, habilitar, revision, estado, factor FROM agrupaciones WHERE id = %(id)s AND estado = 1 ORDER BY id"
    return fetch_all(sql, {"id": agrupacion_id})


def consultar_factor(codigo_rpm: str, flujo: str):
    sql = "SELECT factor FROM agrupaciones WHERE codigo_rpm = %(codigo_rpm)s AND flujo = %(flujo)s"
    return fetch_all(sql, {"codigo_rpm": codigo_rpm, "flujo": flujo})


def consultar_agrupaciones_index_x_barra_id_order_rpm(barra_id: str):
    sql = "SELECT id, barra_id, codigo_rpm, flujo, habilitar, revision, estado, factor FROM agrupaciones WHERE barra_id = %(barra_id)s ORDER BY codigo_rpm"
    return fetch_all(sql, {"barra_id": barra_id})


def consultar_agrupaciones_index_x_barra_order_rpm(barra: str):
    sql = (
        "SELECT a.* FROM agrupaciones a "
        "INNER JOIN barras b ON a.barra_id=b.id WHERE b.barra = %(barra)s ORDER BY codigo_rpm"
    )
    return fetch_all(sql, {"barra": barra})


# --- Medidas ---

def consultar_medidas_index():
    sql = "SELECT codigo, flujo, fecha, codigo_rpm, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24 FROM medidas ORDER BY codigo"
    return fetch_all(sql)


def consultar_medidas(fecha_inicial: str, fecha_final: str, barra: str, e_ar: str):
    sql = (
        "SELECT ME.codigo As MEcodigo, ME.flujo As MEflujo, ME.fecha As MEfecha, ME.codigo_rpm As MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, (ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, (ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, (ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, (ME.Marcado) As MEMarcado, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, AG.estado As AGestado, AG.factor As AGfactor, "
        "BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s AND BA.barra = %(barra)s "
        "AND substring(ME.flujo from 1 for 1) = %(e_ar)s "
        "ORDER BY ME.fecha;"
    )
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final, "barra": barra, "e_ar": e_ar}
    return fetch_all(sql, params)


def consultar_medidas_calcular(fecha_inicial: str, fecha_final: str, e_ar: str, mc: str):
    sql = (
        "SELECT ME.codigo As MEcodigo, ME.flujo As MEflujo, ME.fecha As MEfecha, ME.codigo_rpm As MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, (ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, (ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, (ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, (ME.Marcado) As MEMarcado, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, AG.estado As AGestado, AG.factor As AGfactor, "
        "BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s "
        "AND substring(ME.flujo from 1 for 1) = %(e_ar)s AND BA.mc = %(mc)s "
        "ORDER BY ME.fecha;"
    )
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final, "e_ar": e_ar, "mc": mc}
    return fetch_all(sql, params)


def consultar_medidas_e_ar(fecha_inicial: str, fecha_final: str, mc: str):
    sql = (
        "SELECT ME.codigo As MEcodigo, ME.flujo As MEflujo, ME.fecha As MEfecha, ME.codigo_rpm As MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, (ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, (ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, (ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, (ME.Marcado) As MEMarcado, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, AG.estado As AGestado, AG.factor As AGfactor, "
        "BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s AND BA.mc = %(mc)s "
        "AND (substring(ME.flujo from 1 for 1) = 'A' OR substring(ME.flujo from 1 for 1) = 'R') "
        "ORDER BY ME.fecha;"
    )
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final, "mc": mc}
    return fetch_all(sql, params)


def consultar_medidas_completo(fecha_inicial: str, fecha_final: str, mc: str, e_ar: str, tipo_dia: str):
    sql = (
        "SELECT ME.codigo As MEcodigo, ME.flujo As MEflujo, ME.fecha As MEfecha, ME.codigo_rpm As MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, (ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, (ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, (ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, (ME.Marcado) As MEMarcado, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, AG.estado As AGestado, AG.factor As AGfactor, "
        "BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s AND BA.mc = %(mc)s "
    )
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final, "mc": mc}

    if not e_ar:
        sql += "AND (substring(ME.flujo from 1 for 1) = 'A' OR substring(ME.flujo from 1 for 1) = 'R') "
    else:
        sql += "AND substring(ME.flujo from 1 for 1) = %(e_ar)s "
        params["e_ar"] = e_ar

    if tipo_dia == "ORDINARIO":
        sql += "AND date_part('dow', ME.fecha) in (1, 2, 3, 4, 5) "
    elif tipo_dia == "SABADO":
        sql += "AND date_part('dow', ME.fecha) in (6) "
    elif tipo_dia == "FESTIVO":
        sql += (
            "AND (ME.fecha in (SELECT fecha FROM festivos WHERE ucp=%(mc)s AND fecha >= %(fecha_inicial)s AND fecha <= %(fecha_final)s) "
            "OR date_part('dow', ME.fecha) in (0)) "
        )

    sql += "ORDER BY ME.fecha;"
    return fetch_all(sql, params)


def consultar_medidas_calcular_completo(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    flujo: List[str],
    tipo_dia: str,
    codigo_rpm: List[str],
    barra: str,
    marcado: bool = False,
    dsn: Optional[str] = None,
):
    flujo_clause, flujo_params = build_in_clause(flujo, "flujo")
    rpm_clause, rpm_params = build_in_clause(codigo_rpm, "codigo_rpm")
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final}
    params.update(flujo_params)
    params.update(rpm_params)

    sql = (
        "SELECT %(barra)s AS BAbarra, ME.flujo AS MEflujo, TO_CHAR(ME.fecha, 'DD-MM-YYYY') AS MEfecha, "
        "ME.codigo_rpm AS MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, (ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, (ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, (ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, (ME.Marcado) As MEMarcado "
        "FROM medidas ME "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s "
        f"AND ME.codigo_rpm IN {rpm_clause} "
        f"AND ME.flujo IN {flujo_clause} "
    )
    params["barra"] = barra

    if tipo_dia == "ORDINARIO":
        sql += (
            "AND date_part('dow', ME.fecha) in (1, 2, 3, 4, 5) "
            "AND (ME.fecha not in (SELECT fecha FROM festivos WHERE ucp=%(mc)s AND fecha >= %(fecha_inicial)s AND fecha <= %(fecha_final)s) "
            "AND date_part('dow', ME.fecha) in (1, 2, 3, 4, 5)) "
        )
        params["mc"] = mc
    elif tipo_dia == "SABADO":
        sql += "AND date_part('dow', ME.fecha) in (6) "
    elif tipo_dia == "FESTIVO":
        sql += (
            "AND (ME.fecha in (SELECT fecha FROM festivos WHERE ucp=%(mc)s AND fecha >= %(fecha_inicial)s AND fecha <= %(fecha_final)s) "
            "OR date_part('dow', ME.fecha) in (0)) "
        )
        params["mc"] = mc

    if marcado:
        sql += "AND ME.Marcado = '1' "

    sql += "GROUP BY ME.flujo, ME.fecha, ME.codigo_rpm, ME.p1, ME.p2, ME.p3, ME.p4, ME.p5, ME.p6, ME.p7, ME.p8, ME.p9, ME.p10, ME.p11, ME.p12, ME.p13, ME.p14, ME.p15, ME.p16, ME.p17, ME.p18, ME.p19, ME.p20, ME.p21, ME.p22, ME.p23, ME.p24, ME.Marcado ORDER BY ME.fecha ASC;"
    return fetch_all(sql, params, dsn=dsn)


def consultar_medidas_calcular_completo_sin_barra(
    fecha_inicial: str,
    fecha_final: str,
    mc: str,
    flujo: List[str],
    tipo_dia: str,
    codigo_rpm: List[str],
    barra: str,
    marcado: bool = False,
):
    flujo_clause, flujo_params = build_in_clause(flujo, "flujo")
    rpm_clause, rpm_params = build_in_clause(codigo_rpm, "codigo_rpm")
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final}
    params.update(flujo_params)
    params.update(rpm_params)

    sql = (
        "SELECT fecha AS MEfecha, flujo AS MEflujo, %(barra)s AS BAbarra, "
        "codigo_rpm AS MEcodigo_rpm, Marcado AS MEMarcado, codigo_rpm, "
        "(p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9 + p10 + p11 + p12 + p13 + p14 + p15 + p16 + p17 + p18 + p19 + p20 + p21 + p22 + p23 + p24) AS MEtotal, "
        "(p1) As MEp1, (p2) As MEp2, (p3) As MEp3, (p4) As MEp4, (p5) As MEp5, (p6) As MEp6, "
        "(p7) As MEp7, (p8) As MEp8, (p9) As MEp9, (p10) As MEp10, (p11) As MEp11, (p12) As MEp12, "
        "(p13) As MEp13, (p14) As MEp14, (p15) As MEp15, (p16) As MEp16, (p17) As MEp17, (p18) As MEp18, "
        "(p19) As MEp19, (p20) As MEp20, (p21) As MEp21, (p22) As MEp22, (p23) As MEp23, (p24) As MEp24 "
        "FROM medidas WHERE codigo_rpm IN " + rpm_clause + " "
        "AND (fecha >= %(fecha_inicial)s AND fecha <= %(fecha_final)s) "
        f"AND flujo IN {flujo_clause} "
    )
    params["barra"] = barra

    if tipo_dia == "ORDINARIO":
        sql += "AND date_part('dow', fecha) in (1, 2, 3, 4, 5) "
    elif tipo_dia == "SABADO":
        sql += "AND date_part('dow', fecha) in (6) "
    elif tipo_dia == "FESTIVO":
        sql += (
            "AND (fecha in (SELECT fecha FROM festivos WHERE ucp=%(mc)s AND fecha >= %(fecha_inicial)s AND fecha <= %(fecha_final)s) "
            "OR date_part('dow', fecha) in (0)) "
        )
        params["mc"] = mc

    if marcado:
        sql += "AND Marcado = '1' "

    sql += "GROUP BY fecha, Marcado, codigo_rpm, flujo, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24;"
    return fetch_all(sql, params)


def consultar_medidas_mc(fecha_inicial: str, fecha_final: str, mc: str):
    sql = (
        "SELECT ME.codigo As MEcodigo, ME.flujo As MEflujo, ME.fecha As MEfecha, ME.codigo_rpm As MEcodigo_rpm, "
        "(ME.p1 + ME.p2 + ME.p3 + ME.p4 + ME.p5 + ME.p6 + ME.p7 + ME.p8 + ME.p9 + ME.p10 + ME.p11 + ME.p12 + ME.p13 + ME.p14 + ME.p15 + ME.p16 + ME.p17 + ME.p18 + ME.p19 + ME.p20 + ME.p21 + ME.p22 + ME.p23 + ME.p24) As MEtotal, "
        "(ME.p1) As MEp1, (ME.p2) As MEp2, (ME.p3) As MEp3, (ME.p4) As MEp4, (ME.p5) As MEp5, (ME.p6) As MEp6, "
        "(ME.p7) As MEp7, (ME.p8) As MEp8, (ME.p9) As MEp9, (ME.p10) As MEp10, (ME.p11) As MEp11, (ME.p12) As MEp12, "
        "(ME.p13) As MEp13, (ME.p14) As MEp14, (ME.p15) As MEp15, (ME.p16) As MEp16, (ME.p17) As MEp17, (ME.p18) As MEp18, "
        "(ME.p19) As MEp19, (ME.p20) As MEp20, (ME.p21) As MEp21, (ME.p22) As MEp22, (ME.p23) As MEp23, (ME.p24) As MEp24, "
        "AG.id As AGid, AG.barra_id As AGbarra_id, AG.codigo_rpm As AGcodigo_rpm, AG.flujo As AGflujo, AG.habilitar As AGhabilitar, AG.revision As AGrevision, AG.estado As AGestado, AG.factor As AGfactor, "
        "BA.id As BAid, BA.barra As BAbarra, BA.descripcion As BAdescripcion, BA.nivel_tension As BAnivel_tension, BA.observaciones As BAobservaciones, BA.habilitar As BAhabilitar, BA.estado As BAestado, BA.mc As BAmc "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE ME.fecha >= %(fecha_inicial)s AND ME.fecha <= %(fecha_final)s AND BA.mc = %(mc)s "
        "ORDER BY ME.fecha;"
    )
    params = {"fecha_inicial": fecha_inicial, "fecha_final": fecha_final, "mc": mc}
    return fetch_all(sql, params)


def consultar_medida(codigo: str):
    sql = (
        "SELECT codigo, flujo, fecha, codigo_rpm, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24 "
        "FROM medidas WHERE codigo = %(codigo)s ORDER BY codigo"
    )
    return fetch_all(sql, {"codigo": codigo})


def consultar_fecha_inicial_medidas(mc: str):
    sql = (
        "SELECT ME.codigo, ME.flujo, ME.fecha, ME.codigo_rpm, ME.p1, ME.p2, ME.p3, ME.p4, ME.p5, ME.p6, ME.p7, ME.p8, ME.p9, ME.p10, ME.p11, ME.p12, ME.p13, ME.p14, ME.p15, ME.p16, ME.p17, ME.p18, ME.p19, ME.p20, ME.p21, ME.p22, ME.p23, ME.p24 "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE BA.mc = %(mc)s ORDER BY fecha ASC LIMIT 1;"
    )
    return fetch_one(sql, {"mc": mc})


def consultar_fecha_final_medidas(mc: str):
    sql = (
        "SELECT ME.codigo, ME.flujo, ME.fecha, ME.codigo_rpm, ME.p1, ME.p2, ME.p3, ME.p4, ME.p5, ME.p6, ME.p7, ME.p8, ME.p9, ME.p10, ME.p11, ME.p12, ME.p13, ME.p14, ME.p15, ME.p16, ME.p17, ME.p18, ME.p19, ME.p20, ME.p21, ME.p22, ME.p23, ME.p24 "
        "FROM medidas ME "
        "INNER JOIN agrupaciones AG on (ME.codigo_rpm = AG.codigo_rpm AND ME.flujo = AG.flujo) "
        "INNER JOIN barras BA on (AG.barra_id = BA.id) "
        "WHERE BA.mc = %(mc)s ORDER BY fecha DESC LIMIT 1;"
    )
    return fetch_one(sql, {"mc": mc})


def guardar_barra(barra_id: Optional[str], barra: str, descripcion: Optional[str], nivel_tension: Optional[str], observaciones: Optional[str], habilitar: Optional[str], mc: Optional[str]):
    if barra_id:
        sql = (
            "UPDATE barras SET barra = %(barra)s, descripcion = %(descripcion)s, nivel_tension = %(nivel_tension)s, "
            "observaciones = %(observaciones)s, habilitar = %(habilitar)s, mc = %(mc)s WHERE id = %(id)s"
        )
        params = {
            "id": barra_id,
            "barra": barra,
            "descripcion": descripcion,
            "nivel_tension": nivel_tension,
            "observaciones": observaciones,
            "habilitar": habilitar,
            "mc": mc,
        }
    else:
        sql = (
            "INSERT INTO barras (barra, descripcion, nivel_tension, observaciones, habilitar, estado, mc) "
            "VALUES (%(barra)s, %(descripcion)s, %(nivel_tension)s, %(observaciones)s, %(habilitar)s, '1', %(mc)s)"
        )
        params = {
            "barra": barra,
            "descripcion": descripcion,
            "nivel_tension": nivel_tension,
            "observaciones": observaciones,
            "habilitar": habilitar,
            "mc": mc,
        }
    return execute(sql, params) > 0


def guardar_agrupacion(agrupacion_id: Optional[str], barra_id: str, codigo_rpm: str, flujo: str, habilitar: Optional[str], revision: Optional[str], factor: Optional[str]):
    if agrupacion_id:
        sql = (
            "UPDATE agrupaciones SET barra_id = %(barra_id)s, codigo_rpm = %(codigo_rpm)s, flujo = %(flujo)s, "
            "habilitar = %(habilitar)s, revision = %(revision)s, factor = %(factor)s WHERE id = %(id)s"
        )
        params = {
            "id": agrupacion_id,
            "barra_id": barra_id,
            "codigo_rpm": codigo_rpm,
            "flujo": flujo,
            "habilitar": habilitar,
            "revision": revision,
            "factor": factor,
        }
    else:
        sql = (
            "INSERT INTO agrupaciones (barra_id, codigo_rpm, flujo, habilitar, revision, estado, factor) "
            "VALUES (%(barra_id)s, %(codigo_rpm)s, %(flujo)s, %(habilitar)s, %(revision)s, '1', %(factor)s)"
        )
        params = {
            "barra_id": barra_id,
            "codigo_rpm": codigo_rpm,
            "flujo": flujo,
            "habilitar": habilitar,
            "revision": revision,
            "factor": factor,
        }
    return execute(sql, params) > 0


def guardar_rango_fecha(fechainicio: str, fechafin: str, ucp: str, barra: Optional[str], tipo_dia: Optional[str], nro_dias: Optional[str]):
    sql = (
        "INSERT INTO fechas_ingresadas (fechainicio, fechafin, ucp, barra, tipo_dia, nro_dias) "
        "VALUES (%(fechainicio)s, %(fechafin)s, %(ucp)s, %(barra)s, %(tipo_dia)s, %(nro_dias)s)"
    )
    params = {
        "fechainicio": fechainicio,
        "fechafin": fechafin,
        "ucp": ucp,
        "barra": barra,
        "tipo_dia": tipo_dia,
        "nro_dias": nro_dias,
    }
    return execute(sql, params) > 0


def buscar_ucp_rango_fechas(ucp: str):
    sql = "SELECT * FROM fechas_ingresadas WHERE ucp=%(ucp)s AND barra IS NOT NULL AND tipo_dia IS NOT NULL AND nro_dias IS NOT NULL"
    return fetch_one(sql, {"ucp": ucp})


def actualizar_rango_fechas(fechainicio: str, fechafin: str, ucp: str, barra: Optional[str], tipo_dia: Optional[str], nro_dias: Optional[str]):
    sql = (
        "UPDATE fechas_ingresadas SET fechainicio=%(fechainicio)s, fechafin=%(fechafin)s, "
        "barra=%(barra)s, tipo_dia=%(tipo_dia)s, nro_dias=%(nro_dias)s "
        "WHERE ucp=%(ucp)s AND barra IS NOT NULL AND tipo_dia IS NOT NULL AND nro_dias IS NOT NULL"
    )
    params = {
        "fechainicio": fechainicio,
        "fechafin": fechafin,
        "ucp": ucp,
        "barra": barra,
        "tipo_dia": tipo_dia,
        "nro_dias": nro_dias,
    }
    return execute(sql, params) > 0


def reiniciar_medidas():
    return execute("UPDATE medidas SET marcado = 0") > 0


def marcar_desmarcar_medida(fecha: str, marcado: bool, codigo_rpm: List[str]):
    marcado_val = 1 if marcado else 0
    in_clause, params = build_in_clause(codigo_rpm, "codigo_rpm")
    params.update({"fecha": fecha, "marcado": marcado_val})
    sql = f"UPDATE medidas SET marcado = %(marcado)s WHERE codigo_rpm in {in_clause} AND fecha=%(fecha)s"
    return execute(sql, params) > 0


def eliminar_barra(barra_id: str):
    return execute("UPDATE barras SET estado = 0 WHERE id = %(id)s", {"id": barra_id}) > 0


def eliminar_agrupacion(agrupacion_id: str):
    return execute("UPDATE agrupaciones SET estado = 0 WHERE id = %(id)s", {"id": agrupacion_id}) > 0


def eliminar_agrupaciones(barra_id: str):
    return execute("UPDATE agrupaciones SET estado = 0 WHERE barra_id = %(id)s", {"id": barra_id}) > 0


def eliminar_barra_agrupaciones(barra_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE agrupaciones SET estado = 0 WHERE barra_id = %(id)s", {"id": barra_id})
            if cur.rowcount < 0:
                return False
            cur.execute("UPDATE barras SET estado = 0 WHERE id = %(id)s", {"id": barra_id})
            return cur.rowcount >= 0


def buscar_medida(flujo: str, fecha: str, codigo_rpm: str):
    sql = "SELECT 1 FROM medidas WHERE flujo=%(flujo)s AND fecha=%(fecha)s AND codigo_rpm=%(codigo_rpm)s"
    return fetch_one(sql, {"flujo": flujo, "fecha": fecha, "codigo_rpm": codigo_rpm})


def actualizar_medida(flujo: str, fecha: str, codigo_rpm: str, periodos: List[float]):
    params = {"flujo": flujo, "fecha": fecha, "codigo_rpm": codigo_rpm}
    params.update(_period_params(periodos))
    sql = (
        "UPDATE medidas SET p1=%(p1)s, p2=%(p2)s, p3=%(p3)s, p4=%(p4)s, p5=%(p5)s, p6=%(p6)s, p7=%(p7)s, p8=%(p8)s, "
        "p9=%(p9)s, p10=%(p10)s, p11=%(p11)s, p12=%(p12)s, p13=%(p13)s, p14=%(p14)s, p15=%(p15)s, p16=%(p16)s, "
        "p17=%(p17)s, p18=%(p18)s, p19=%(p19)s, p20=%(p20)s, p21=%(p21)s, p22=%(p22)s, p23=%(p23)s, p24=%(p24)s "
        "WHERE flujo=%(flujo)s AND fecha=%(fecha)s AND codigo_rpm=%(codigo_rpm)s"
    )
    return execute(sql, params) > 0


def ingresar_medida(flujo: str, fecha: str, codigo_rpm: str, periodos: List[float]):
    params = {"flujo": flujo, "fecha": fecha, "codigo_rpm": codigo_rpm}
    params.update(_period_params(periodos))
    sql = (
        "INSERT INTO medidas (flujo, fecha, codigo_rpm, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, marcado) "
        "VALUES (%(flujo)s, %(fecha)s, %(codigo_rpm)s, %(p1)s, %(p2)s, %(p3)s, %(p4)s, %(p5)s, %(p6)s, %(p7)s, %(p8)s, %(p9)s, %(p10)s, %(p11)s, %(p12)s, %(p13)s, %(p14)s, %(p15)s, %(p16)s, %(p17)s, %(p18)s, %(p19)s, %(p20)s, %(p21)s, %(p22)s, %(p23)s, %(p24)s, 0)"
    )
    return execute(sql, params) > 0


def bulk_medidas(items: List[Dict[str, Any]], mode: str):
    if not items:
        return 0

    if mode == "delete":
        sql = "DELETE FROM medidas WHERE flujo=%(flujo)s AND fecha=%(fecha)s AND codigo_rpm=%(codigo_rpm)s"
        params_list = [
            {"flujo": item["flujo"], "fecha": item["fecha"], "codigo_rpm": item["codigo_rpm"]}
            for item in items
        ]
        return execute_many(sql, params_list)

    results = 0
    for item in items:
        flujo = item["flujo"]
        fecha = item["fecha"]
        codigo_rpm = item["codigo_rpm"]
        periodos = item["periodos"]

        if mode == "insert":
            if ingresar_medida(flujo, fecha, codigo_rpm, periodos):
                results += 1
            continue

        if mode == "update":
            if actualizar_medida(flujo, fecha, codigo_rpm, periodos):
                results += 1
            continue

        existing = buscar_medida(flujo, fecha, codigo_rpm)
        if existing:
            if actualizar_medida(flujo, fecha, codigo_rpm, periodos):
                results += 1
        else:
            if ingresar_medida(flujo, fecha, codigo_rpm, periodos):
                results += 1
    return results


# --- Otros ---

def buscar_mac(mac: str):
    sql = "SELECT 1 FROM con_mac WHERE mac=%(mac)s"
    return fetch_one(sql, {"mac": mac})


def buscar_tipo_dia(nombre: str):
    sql = "SELECT aux FROM ucp WHERE nombre=%(nombre)s AND codpadre='113'"
    return fetch_one(sql, {"nombre": nombre})


def buscar_batch_corriendo():
    sql = "SELECT aux FROM ucp WHERE codigo='117'"
    return fetch_one(sql)


def iniciar_proceso_batch():
    sql = "UPDATE ucp SET aux='1' WHERE codigo='117'"
    return execute(sql) > 0


def buscar_ruta_batch():
    sql = "SELECT * FROM ucp WHERE codigo='90'"
    return fetch_one(sql)


def eliminar_fechas_ingresadas(ucp: str, barra: str, tipo_dia: str, nro_dias: str):
    sql = "DELETE FROM fechas_ingresadas WHERE ucp=%(ucp)s AND barra=%(barra)s AND tipo_dia=%(tipo_dia)s AND nro_dias=%(nro_dias)s"
    params = {"ucp": ucp, "barra": barra, "tipo_dia": tipo_dia, "nro_dias": nro_dias}
    return execute(sql, params) > 0


def eliminar_fechas_ingresadas_todos(ucp: str):
    sql = "DELETE FROM fechas_ingresadas WHERE ucp=%(ucp)s AND barra IS NOT NULL AND tipo_dia IS NOT NULL AND nro_dias IS NOT NULL"
    return execute(sql, {"ucp": ucp}) > 0


def buscar_festivo(fecha: str, ucp: str):
    sql = "SELECT * FROM festivos WHERE ucp=%(ucp)s AND fecha=%(fecha)s"
    return fetch_one(sql, {"ucp": ucp, "fecha": fecha})


# --- Modalinicio / Faltantes ---

def modal_inicio(fechainicio: str, fechafin: str, mc: str, tipodia: str, barra: str, e_ar: str):
    start = _parse_date(fechainicio)
    end = _parse_date(fechafin)
    total_days = (end - start).days

    t_ordinarios = 0
    t_sabados = 0
    t_festivos = 0

    for offset in range(0, total_days + 1):
        current = start + timedelta(days=offset)
        dow = current.weekday()  # Monday=0
        fecha_str = current.strftime(DATE_FMT)

        if dow <= 4:
            if buscar_festivo(fecha_str, mc):
                t_festivos += 1
            else:
                t_ordinarios += 1
        elif dow == 5:
            t_sabados += 1
        else:
            t_festivos += 1

    agrupaciones = consultar_agrupaciones_index_x_barra_order_rpm(barra)
    medidas = consultar_medidas(fechainicio, fechafin, barra, e_ar)

    missing = []
    seen = set()
    html = f"<b>Tipo dia: </b>{tipodia}<br/>"

    if not agrupaciones:
        html += f"Debe existir por lo menos una agrupacion de medidas para la barra: {barra}"
        return {"ok": True, "html": html, "missing": []}

    for row in agrupaciones:
        codrpm = row.get("codigo_rpm")
        flujo = row.get("flujo")

        rows_medida = [m for m in medidas if m.get("mecodigo_rpm") == codrpm]
        total_registros = len(rows_medida)

        if tipodia == "ORDINARIO":
            t_medidas = 0
            for med in rows_medida:
                fecha = med.get("mefecha")
                if isinstance(fecha, datetime):
                    dt = fecha
                else:
                    dt = _parse_date(str(fecha))
                dow = dt.weekday()
                if dow <= 4 and not buscar_festivo(dt.strftime(DATE_FMT), mc):
                    t_medidas += 1
            expected = t_ordinarios
        elif tipodia == "SABADO":
            t_medidas = 0
            for med in rows_medida:
                fecha = med.get("mefecha")
                if isinstance(fecha, datetime):
                    dt = fecha
                else:
                    dt = _parse_date(str(fecha))
                if dt.weekday() == 5:
                    t_medidas += 1
            expected = t_sabados
        else:
            t_medidas = 0
            for med in rows_medida:
                fecha = med.get("mefecha")
                if isinstance(fecha, datetime):
                    dt = fecha
                else:
                    dt = _parse_date(str(fecha))
                if dt.weekday() == 6 or buscar_festivo(dt.strftime(DATE_FMT), mc):
                    t_medidas += 1
            expected = t_festivos

        if total_registros == 0 or t_medidas != expected:
            key = (barra, codrpm)
            if key not in seen:
                seen.add(key)
                missing.append({"barra": barra, "codigo_rpm": codrpm, "flujo": flujo})
                html += f"<br/>Barra: {barra}, Medida: {codrpm}<b>=</b> {flujo}- "
            else:
                html += f"{flujo}- "

    if not missing:
        html += "No hicieron falta medidas."

    return {"ok": True, "html": html, "missing": missing}
