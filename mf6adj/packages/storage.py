"""Storage (STO) terms for the adjoint solution.

Storage is the only term that connects one time step to the next, so it is the
term through which an adjoint solution propagates backward in time. MODFLOW 6
selects between the specific-storage and specific-yield contributions on the
saturation of the cell, and these functions follow that selection rather than
approximating it.
"""

import numpy as np

# default SATOMEGA of the MODFLOW 6 storage package
SATOMEGA = 1.0e-6

from ..utils.utils_modflow import get_ptr_from_gwf


def initial_saturation(gwf, gwf_name: str) -> np.ndarray:
    """Return the saturation the initial heads imply.

    Called before the first solve, where the saturation MODFLOW 6 reports is
    the value it was allocated with rather than one computed from the heads.
    """
    head = gwf.get_value(gwf.get_var_address("X", gwf_name.upper()))
    top = get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
    bot = get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
    head = np.asarray(head)[: top.shape[0]]

    thickness = top - bot
    saturation = np.ones_like(thickness, dtype=float)
    wet = thickness > 0.0
    saturation[wet] = np.clip((head[wet] - bot[wet]) / thickness[wet], 0.0, 1.0)
    # a cell that never converts is full whatever its head
    icelltype = get_ptr_from_gwf(gwf_name, "NPF", "ICELLTYPE", gwf)
    saturation[icelltype == 0] = 1.0
    return saturation


def dresdss_h(
    gwf,
    gwf_name: str,
    head: np.ndarray,
    head_old: np.ndarray,
    dt: float,
    sat: np.ndarray,
    sat_old: np.ndarray,
) -> np.ndarray:
    """Return the specific-storage contribution to the residual derivative.

    The derivative of the transient groundwater flow residual with respect
    to specific storage for one time step, taken over the saturations the
    step starts and ends at. Under SS_CONFINED_ONLY specific storage acts
    only while a cell is full, and the saturation is one at or above the top
    of the cell and zero below it, so the derivative reduces to the terms of
    the full cells. A cell that never converts is treated as full. See
    SsTerms in GwfStorageUtils.f90.

    Parameters
    ----------
    head : ndarray
        Head at the end of the step.
    head_old : ndarray
        Head at the start of the step.
    dt : float
        Length of the current solution step in model time.
    sat : ndarray
        Saturation at the end of the step.
    sat_old : ndarray
        Saturation at the start of the step.

    Returns
    -------
    ndarray
        Per-cell derivative term used later in the adjoint sensitivity
        calculation for specific storage.
    """
    top = get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
    bot = get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
    area = get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
    iconvert = get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)

    # handle iconvert
    sat_mod = sat.copy()
    sat_mod[iconvert == 0] = 1.0
    sat_old_mod = sat_old.copy()
    sat_old_mod[iconvert == 0] = 1.0

    height = top - bot
    dSC1 = area * height

    iconf_ss = int(
        np.asarray(get_ptr_from_gwf(gwf_name, "STO", "ICONF_SS", gwf)).ravel()[0]
    )
    if iconf_ss != 0:
        # specific storage acts only while the cell is full, and the
        # saturation is one at or above the top of the cell and zero below
        full = sat_mod >= 1.0
        full_old = sat_old_mod >= 1.0
        result = (dSC1 / dt) * (
            np.where(full, top - head, 0.0) + np.where(full_old, head_old - top, 0.0)
        )
    else:
        result = (
            (dSC1 / dt) * (sat_old_mod * head_old - sat_mod * head)
            + (dSC1 / dt) * bot * (sat_mod - sat_old_mod)
            + (dSC1 / (2.0 * dt)) * height * (sat_mod**2 - sat_old_mod**2)
        )
    # zero out dry cells
    result[head <= bot] = 0.0
    result[head_old <= bot] = 0.0

    return result


def dresdsy_h(
    gwf,
    gwf_name: str,
    dt: float,
    sat: np.ndarray,
    sat_old: np.ndarray,
) -> np.ndarray:
    """Return the specific-yield contribution to the residual derivative.

    Specific yield releases the water a cell holds between the saturations
    it starts and ends the time step at. See SyTerms in GwfStorageUtils.f90.

    Parameters
    ----------
    dt : float
        Length of the current solution step in model time.
    sat : ndarray
        Saturation at the end of the step.
    sat_old : ndarray
        Saturation at the start of the step.

    Returns
    -------
    ndarray
        Per-cell derivative of the residual with respect to specific yield.
    """
    top = get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
    bot = get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
    area = get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
    iconvert = get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)

    result = area * (top - bot) * (sat_old - sat) / dt
    # a cell that never converts stays full, so specific yield never acts
    result[iconvert == 0] = 0.0
    return result


def smoothed_saturation_slope(head, top, bot, satomega: float = SATOMEGA):
    """Return the thickness times the slope of the smoothed saturation.

    MODFLOW 6 does not let the saturation follow the head linearly all the way
    to the ends of a cell. It rounds both corners over a fraction ``satomega``
    of the thickness, so the slope falls away to nothing as the cell empties or
    fills. This is sQuadraticSaturationDerivative of SmoothingFunctions.f90,
    taken over the thickness, so it is one where the saturation follows the
    head and zero at either end.

    It is formed from the head rather than from the saturation, as MODFLOW
    forms it: the two do not agree inside the rounding, where the saturation is
    quadratic in the head.

    Parameters
    ----------
    head : ndarray
        Head the saturation is taken at.
    top, bot : ndarray
        Top and bottom of each cell.
    satomega : float
        Fraction of the cell the rounding covers.

    Returns
    -------
    ndarray
        Slope over the linear slope, between zero and one.
    """
    thickness = np.asarray(top - bot, dtype=float)
    over = np.divide(
        np.asarray(head, dtype=float) - bot,
        thickness,
        out=np.zeros_like(thickness),
        where=thickness > 0.0,
    )
    fraction = np.clip(over, 0.0, 1.0)
    scale = 1.0 / (1.0 - satomega) if satomega < 1.0 else 1.0
    return np.where(
        fraction < satomega,
        scale * fraction / satomega,
        np.where(
            fraction < 1.0 - satomega,
            scale,
            np.where(fraction < 1.0, scale * (1.0 - fraction) / satomega, 0.0),
        ),
    )


def drhsdh(
    gwf,
    gwf_name: str,
    logger,
    dt: float,
    sat_old: np.ndarray,
    head: np.ndarray,
) -> np.ndarray:
    """Return the previous-head derivative of the storage right-hand side.

    The storage right-hand side depends on the previous head through both
    the head itself and the saturation it sets, and MODFLOW 6 selects
    between the specific-storage and specific-yield terms on that
    saturation. See SsTerms and SyTerms in GwfStorageUtils.f90.

    Parameters
    ----------
    dt : float
        Length of the current solution step in model time.
    sat_old : ndarray
        Saturation from the previous solution step.
    head : ndarray
        Head the saturation is taken at, which the slope of the smoothed
        saturation is formed from.

    Returns
    -------
    ndarray
        Per-cell derivative of the storage-related right-hand side with
        respect to the previous head.
    """
    area = get_ptr_from_gwf(gwf_name, "DIS", "AREA", gwf)
    top = get_ptr_from_gwf(gwf_name, "DIS", "TOP", gwf)
    bot = get_ptr_from_gwf(gwf_name, "DIS", "BOT", gwf)
    storage = get_ptr_from_gwf(gwf_name, "STO", "SS", gwf)
    iconvert = get_ptr_from_gwf(gwf_name, "STO", "ICONVERT", gwf)
    sy = get_ptr_from_gwf(gwf_name, "STO", "SY", gwf)
    iconf_ss = int(
        np.asarray(get_ptr_from_gwf(gwf_name, "STO", "ICONF_SS", gwf)).ravel()[0]
    )
    iorig_ss = int(
        np.asarray(get_ptr_from_gwf(gwf_name, "STO", "IORIG_SS", gwf)).ravel()[0]
    )
    if iorig_ss != 0:
        logger.warning(
            "ORIGINAL_SPECIFIC_STORAGE is not supported; the storage "
            + "sensitivity uses the current specific-storage formulation"
        )

    sat_old_mod = sat_old.copy()
    sat_old_mod[iconvert == 0] = 1.0

    # specific storage, rho1old in SsTerms
    ss_term = area * storage * (top - bot) / dt
    if iconf_ss != 0:
        # SS_CONFINED_ONLY carries a previous-head term only where the cell
        # was full; elsewhere the term is dropped rather than scaled
        ss_scale = np.where(sat_old_mod == 1.0, 1.0, 0.0)
    else:
        # otherwise the term follows the previous saturation
        ss_scale = sat_old_mod
    # a cell that never converts is always full
    ss_scale = np.where(iconvert == 0, 1.0, ss_scale)

    # specific yield, which enters only through the saturation and so
    # vanishes once the cell is full or dry
    # The previous head enters the specific-yield term through the saturation
    # it sets, so the derivative carries the slope of that saturation. MODFLOW
    # rounds the saturation at both ends of the cell and the slope falls away
    # there, so a cell that has emptied passes almost nothing back rather than
    # the whole term: at a saturation of 1e-19 the two differ by a million, and
    # the adjoint state grows by that ratio once per time step.
    sy_term = area * sy / dt
    # a cell that never converts releases nothing through specific yield,
    # whatever its head
    sy_scale = np.where(iconvert == 0, 0.0, smoothed_saturation_slope(head, top, bot))

    return -1.0 * (ss_term * ss_scale + sy_term * sy_scale)
