"""MCP group "space": correlation of the FE model with a vibration test (ECSS-E-ST-32-11C, `fea_core.space.correlation`): import of the test data, MAC / frequency / damping /
response comparison. The limits come ONLY from the requirement set that the user entered and confirmed; these tools have no argument that enters or confirms a limit."""
from __future__ import annotations

import weakref

from .common import out
from .session import FeaError, Session, user_file

_TEST_DATA: "weakref.WeakKeyDictionary[Session, object]" = weakref.WeakKeyDictionary()      # the measured data of a session (not stored in the study)
_TAMS: "weakref.WeakKeyDictionary[Session, object]" = weakref.WeakKeyDictionary()           # the imported mass matrix (TAM) of a session


def register(mcp, sess: Session) -> None:

    @mcp.tool(structured_output=False)
    def fea_space_test_import(path: str | None = None, format: str = "auto", append: bool = False, sensors: list[dict] | None = None, auto_map: bool = True,
                              clear: bool = False, functions: dict[str, dict] | None = None, rotation: str | None = None, rotation_confirmed: bool | None = None,
                              damping_unit: str = "", mass_matrix: str = "") -> str:
        """Imports measured vibration-test data (CSV or UFF 58 functions / 55 mode shapes, real or complex) for the correlation with the FE model.
        Bounded readers; truncated / malformed files are refused. CSV: modes `mode, frequency_hz, [damping, damping_unit ratio|percent], node, dof, value`; functions `function, node, dof, frequency_hz, amplitude [phase_deg] | real, imag, [kind frf|psd, unit, quantity, per]`. UFF: datasets 55 / 58 and the coordinate systems 2411 / 15 / 2420: data at nodes with a fully defined local system are rotated into the global system, nodes with an unknown local system are left out and named. append = add to the data already imported; clear = drop it and the imported mass matrix. sensors = [{node, point [x,y,z] mm | probe | selection, dofs 'xyz'}] maps test nodes to the FE model (stored in the study); auto_map maps nodes named like a probe of the study.
        functions = {function name: {frame absolute|relative (what the measurement is relative to: the base or the ground), per none|base_acceleration|force, quantity, unit, probe, dof, reference_force_n}} states how each measured function is compared (stored in the study): the model curve is built in the quantity of the function's unit (g, m/s^2, mm ...), in the frame and normalisation you give; a function whose unit is unknown or does not fit is reported UNCONFIRMED, never PASS. A controller PSD without node / direction needs probe + dof here. Without arguments: shows the data, what each function needs, and the sensors.
        mass_matrix = path of a CSV mass matrix (TAM, in the GLOBAL axes of the FE model, bound to this study) for the COC / AOC of fea_space_correlate (orth_mass tam): header `,N1:x,N1:y,...`, rows `N1:x,m11,...`, optional `# unit: kg`; symmetric, positive semi-definite, DOFs = the sensor DOFs, at most 2000.
        rotation = columns | rows: how the 3 x 3 part of a UFF dataset 2420 is read (columns = the local sensor axes are the columns, as OpenModal writes them - the default; rows = the rows). The file does not say which, so data at ROTATED sensors stay UNCONFIRMED until the USER confirms the convention for the file: rotation_confirmed=true ONLY when the user asked, in a call of its own (stored as confirmed via MCP, which the dashboard and the report show); changing the rotation withdraws the confirmation. damping_unit = ratio | percent: the unit of the damping of a UFF 55 (a value above 0.2 without a unit is refused). Unit labels: UFF 58 numerator (record 9) and denominator (record 10) units are combined (g and N make g/N); complex-mode eigenvalues are ASSUMED to be rad/s."""
        from ..space import corr_run as CR
        from ..space import correlation as C
        with sess.use():
            st = sess.require_study()
            if clear:
                _TEST_DATA.pop(sess, None)
                _TAMS.pop(sess, None)
                CR.remember_tam(st, None)
            td = _TEST_DATA.get(sess)
            try:
                if mass_matrix:
                    tam = C.load_mass_file(user_file(mass_matrix, max_bytes=C.MAX_MASS_BYTES, what="Mass matrix file"))
                    _TAMS[sess] = tam
                    CR.remember_tam(st, tam, getattr(sess.results, "mesh", None))
                if rotation is not None and rotation not in C.ROTATIONS:
                    raise FeaError(f"rotation must be {' or '.join(C.ROTATIONS)}")
                if path:
                    p = user_file(path, max_bytes=C.MAX_FILE_BYTES, what="Test data file")
                    new = C.load_test_file(p, format, rotation=rotation or C.DEFAULT_ROTATION, damping_unit=damping_unit.strip().lower())
                    if append and isinstance(td, C.TestData):
                        td.merge(new)
                    else:
                        td = new
                    _TEST_DATA[sess] = td
                    CR.adopt_rotation(st, td)
                elif rotation is not None and isinstance(td, C.TestData):
                    CR.set_rotation(st, td, rotation)
                if isinstance(td, C.TestData):
                    CR.remember_test(st, td)
                    if rotation_confirmed is not None:
                        CR.confirm_rotation(st, td, "mcp", bool(rotation_confirmed))
                if sensors is not None:
                    CR.upsert_sensors(st, C.sensors_from_dicts(sensors, st))
                if functions is not None:
                    CR.update_setups(st, functions)
                tam_info = _TAMS[sess].summary() if sess in _TAMS else None
                if not isinstance(td, C.TestData):
                    return out({"dataset": None, "sensors": [s.to_dict() for s in CR.load_sensors(st)], "mass_matrix": tam_info, "hint": "give path to import a test file"})
                if auto_map:
                    have = {s.node for s in CR.load_sensors(st)}
                    new_s = [s for s in C.auto_sensors(td, st) if s.node not in have]
                    if new_s:
                        CR.upsert_sensors(st, new_s)
                sens = CR.load_sensors(st)
            except C.TestDataError as e:
                raise FeaError(str(e)) from None
            except ValueError as e:
                raise FeaError(str(e)) from None
            mapped = {s.node for s in sens}
            return out({"dataset": td.summary(), "mass_matrix": tam_info, "sensors": [s.to_dict() for s in sens], "unmapped_nodes": [n for n in td.nodes() if n not in mapped][:50],
                        "functions": CR.describe_functions(td, CR.load_setups(st)),
                        "next": "map the unmapped nodes with `sensors`, state the frame / normalisation of the functions with `functions`, solve the modal analysis, then fea_space_correlate"})

    @mcp.tool(structured_output=False)
    def fea_space_correlate(method: str = "hungarian", pair_min_mac: float = 0.0, f_min_hz: float | None = None, f_max_hz: float | None = None,
                            fe_modes: list[int] | None = None, mode_set: int = 0, response_band_hz: list[float] | None = None,
                            apply_damping: bool = False, damping_at: str = "fe", matrices: bool = False, limit_rows: bool = False, hidden_peak_db: float = 6.0,
                            orth_mass: str = "serep", tam_global_axes: bool = False) -> str:
        """Correlates the solved FE model with the imported test data (MAC, frequency, modulus difference, CoMAC, COC / AOC, responses).

        Verdict PASS | FAIL | UNCONFIRMED_REQUIREMENTS | PARTIAL_SCOPE | OUTDATED | UNVERIFIABLE. Pairing hungarian | max_mac | order. Response curves (peak shift, peak amplitude error [dB], RMS error) are predicted from the dynamics results at the probes of the sensors in the quantity / frame / normalisation of each function's setup (fea_space_test_import functions); PSD from the random response.
        COC / AOC (ECSS-E-ST-32-11C 3.2.39; mandatory, Table 5-2): orth_mass = serep (default, INDICATIVE only: a SEREP of the compared FE modes under the study's mass, the COC of the PROJECTION of a test shape onto them; needs >= 2 compared FE modes and more sensor DOFs than modes; every SEREP COC / AOC row is UNCONFIRMED - never PASS and never FAIL, a value outside the limit is a warning hint; a PASS / FAIL needs the TAM) | tam (the matrix of fea_space_test_import mass_matrix; with rotated UFF 2420 sensors the rows stay UNCONFIRMED until the USER states the TAM is in global axes: tam_global_axes=true, only on the user's word; a TAM whose FE-FE off-diagonal exceeds 0.1, a singular one with a test mode in its null space, one FE mode only, or another model state is UNCONFIRMED) | none. Not computable -> UNCONFIRMED rows. The modulus difference is the program's own definition (the standard gives no formula).
        PARTIAL_SCOPE = every judged row passed but part of the test was left out (see coverage; hidden_peak_db = detection setting of a resonance outside the band). Data at rotated sensors stay UNCONFIRMED until the user confirms the UFF 2420 rotation convention.
        Limits: ONLY the corr_* entries of the confirmed requirement set (fea_space_requirement_set; limit_rows=true adds the empty rows and withdraws a confirmation; a criterion that does not apply is declared with not_applicable: true + document / revision / clause, and is listed in the result with who confirmed the set and through which interface); no argument enters or confirms a limit. pair_min_mac drops pairs below it; f_min_hz, f_max_hz, fe_modes select modes; mode_set = which modal result; apply_damping writes the measured damping of the pairs into the damping table (damping_at fe | test) and invalidates the dynamics results.
        The result with its provenance (fingerprints of every input) and coverage is stored in the study; it is shown as current only while every input is unchanged (OUTDATED / UNVERIFIABLE otherwise; the provenance is tamper-evident, not a recomputation)."""
        from ..space import corr_run as CR
        from ..space import correlation as C
        with sess.use(op="test correlation"):
            st = sess.require_study()
            td = _TEST_DATA.get(sess)
            if not isinstance(td, C.TestData):
                raise FeaError("No test data are imported - use fea_space_test_import first")
            if limit_rows and st.space is not None and st.space.requirement_set is not None:
                C.install_limit_rows(st.space.requirement_set)
            try:
                opt = C.CorrelationOptions(method=method, pair_min_mac=pair_min_mac, f_min_hz=f_min_hz, f_max_hz=f_max_hz, fe_modes=fe_modes,
                                           response_band_hz=(float(response_band_hz[0]), float(response_band_hz[1])) if response_band_hz else None,
                                           damping_at=damping_at, hidden_peak_db=hidden_peak_db, orth_mass=orth_mass, tam_global_axes=tam_global_axes,
                                           tam_global_axes_via="mcp" if tam_global_axes else "")
                opt.validate()
                res, table = CR.run_correlation(st, sess.results, td, options=opt, mode_set=mode_set, apply_damping=apply_damping, tam=_TAMS.get(sess))
            except (ValueError, IndexError, TypeError, KeyError) as e:
                raise FeaError(str(e)) from None
            d = res.to_dict(matrices=matrices)
            d["text"] = res.text()
            if table is not None:
                d["damping_table_written"] = table
                sess.changed()                                          # the damping of the dynamics analysis changed: its results do not hold (as fea_dyn_damping_set)
                d["note_damping"] = "the damping table of the dynamics analysis was changed: its results were discarded - solve again"
            if limit_rows:
                d["limit_rows"] = "the empty corr_* limit rows were added to the requirement set (its confirmation was withdrawn if it had one)"
            return out(d)
