"""
generate_drilling_data.py

Synthetic Drilling Data Generator — mimics real-world rig sensor data (WITSML-style
surface parameters) for a Non-Productive Time (NPT) reduction case study.

Produces THREE files, deliberately different in shape and update frequency,
mirroring the three O&G data categories this lab is built to teach:

  1. well_header.csv          -> STATIC / STRUCTURED reference data (one row per well)
  2. drilling_sensor_log.csv  -> HIGH-FREQUENCY TIME-SERIES sensor log (5-minute samples)
  3. daily_drilling_report.csv-> DAILY STRUCTURED report (one row per well per day)

Physical/engineering logic used to keep the data realistic:
  - Depth is built up over time using a formation-dependent Rate of Penetration (ROP).
  - WOB, RPM, Torque, Standpipe Pressure and Hook Load are correlated with depth,
    formation hardness and each other (not just random noise).
  - Non-drilling activities (connections, circulating, tripping) periodically stop
    the bit, exactly as happens on a real rig.
  - Two realistic NPT event types are injected per well: STUCK PIPE and
    EQUIPMENT FAILURE (mud pump), each with the parameter signature engineers
    actually look for.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)

# ------------------------------------------------------------------
# 1. WELL HEADER (static, structured, one row per well)
# ------------------------------------------------------------------
wells = ["WELL-D01", "WELL-D02", "WELL-D03"]
well_header_rows = [
    ["WELL-D01", "Rig-12", "Badra Field",   "Al-Waha Drilling", "2025-02-01", "Deviated",    3450, 8.5,  "WBM"],
    ["WELL-D02", "Rig-07", "Badra Field",   "Al-Waha Drilling", "2025-02-10", "Vertical",    2980, 8.5,  "WBM"],
    ["WELL-D03", "Rig-12", "Sindbad North", "Al-Waha Drilling", "2025-02-18", "Horizontal",  4120, 6.125,"OBM"],
]
well_header = pd.DataFrame(well_header_rows, columns=[
    "Well_ID", "Rig_Name", "Field", "Operator", "Spud_Date",
    "Well_Type", "Target_Depth_m", "Bit_Diameter_in", "Mud_Type",
])
well_header.to_csv("well_header.csv", index=False)

# ------------------------------------------------------------------
# Formation model: each well drills through the same 4 generic zones.
# (hardness_factor: higher = harder = slower ROP, higher torque/WOB)
# ------------------------------------------------------------------
FORMATIONS = [
    # name,          top_m, base_m, rop_m_hr(mean), wob_klb(mean), rpm(mean), hardness
    ("Sandstone",       0,   800,   38,  12,  140, 0.9),
    ("Shale",         800,  1900,   16,  20,  120, 1.6),
    ("Limestone",    1900,  2900,    9,  28,  100, 2.3),
    ("Tight_Sand",   2900,  4300,    6,  32,   90, 2.8),
]

def formation_at(depth_m):
    for name, top, base, rop, wob, rpm, hard in FORMATIONS:
        if top <= depth_m < base:
            return rop, wob, rpm, hard
    return FORMATIONS[-1][3], FORMATIONS[-1][4], FORMATIONS[-1][5], FORMATIONS[-1][6]

SAMPLE_MIN = 5                     # sensor sample interval, minutes
SAMPLES_PER_HOUR = 60 // SAMPLE_MIN

sensor_frames = []
daily_frames = []

for w_idx, well in enumerate(wells):
    target_depth = well_header.loc[well_header.Well_ID == well, "Target_Depth_m"].iloc[0]
    bit_dia = well_header.loc[well_header.Well_ID == well, "Bit_Diameter_in"].iloc[0]
    start_time = pd.Timestamp(well_header.loc[well_header.Well_ID == well, "Spud_Date"].iloc[0])

    # Scale NPT event duration to this well's overall length (deeper/longer wells
    # naturally accumulate proportionally more NPT hours too) so the resulting
    # NPT % of total time lands in the operator's flagged 8-12% range for every well.
    depth_scale = target_depth / 3000.0
    depth = 0.0
    t = start_time
    rows = []

    # Pre-decide 2 NPT events for this well: one stuck-pipe, one equipment-failure,
    # triggered once drilling passes a randomly chosen depth milestone.
    stuck_trigger_depth = RNG.uniform(900, 2200)      # stuck pipe more likely in reactive shale
    equip_trigger_depth = RNG.uniform(2200, target_depth - 200)
    stuck_done, equip_done = False, False
    active_npt_type = None
    npt_remaining_steps = 0
    since_last_connection_min = 0

    while depth < target_depth:
        rop_mean, wob_mean, rpm_mean, hardness = formation_at(depth)

        # ---- decide activity for this 5-min step ----
        since_last_connection_min += SAMPLE_MIN
        activity = "Drilling"

        # Routine connection every ~90 minutes of drilling (add a joint of pipe), lasts 15 min
        if since_last_connection_min >= 90 and active_npt_type is None:
            activity = "Connection"
            since_last_connection_min = 0

        # ---- NPT event trigger checks ----
        if active_npt_type is None:
            if not stuck_done and depth >= stuck_trigger_depth:
                active_npt_type = "Stuck_Pipe"
                npt_remaining_steps = int(RNG.uniform(10, 24) * depth_scale * 60 / SAMPLE_MIN)
                stuck_done = True
            elif not equip_done and depth >= equip_trigger_depth:
                active_npt_type = "Equipment_Failure"
                npt_remaining_steps = int(RNG.uniform(4, 10) * depth_scale * 60 / SAMPLE_MIN)
                equip_done = True

        if active_npt_type is not None:
            activity = "NPT"
            npt_remaining_steps -= 1
            if npt_remaining_steps <= 0:
                active_npt_type_finished = active_npt_type
                active_npt_type = None
            else:
                active_npt_type_finished = active_npt_type
        else:
            active_npt_type_finished = None

        # ---- generate sensor values based on activity ----
        if activity == "Drilling":
            wob = max(2, RNG.normal(wob_mean, wob_mean * 0.08))
            rpm = max(20, RNG.normal(rpm_mean, rpm_mean * 0.06))
            rop = max(0.5, RNG.normal(rop_mean, rop_mean * 0.12))
            torque = 0.5 + 0.02 * wob + 0.004 * rpm + 0.00015 * depth * hardness + RNG.normal(0, 0.3)
            hook_load = 120 + 0.018 * depth - 0.15 * wob + RNG.normal(0, 3)
            spp = 800 + 0.35 * depth + 4.0 * (rpm / 100) + RNG.normal(0, 25)
            mud_flow = max(200, RNG.normal(650, 20))
            gas_units = max(5, RNG.normal(20 + 0.002 * depth, 4))
            depth += rop * (SAMPLE_MIN / 60.0)

        elif activity == "Connection":
            wob, rpm, rop = 0.0, 0.0, 0.0
            torque = RNG.normal(1.0, 0.3)
            hook_load = 150 + 0.018 * depth + RNG.normal(0, 4)
            spp = RNG.normal(30, 10)
            mud_flow = RNG.normal(0, 5)
            gas_units = max(3, RNG.normal(12, 3))

        elif activity == "NPT" and active_npt_type_finished == "Stuck_Pipe":
            # Stuck pipe signature: torque & hook load spike, ROP ~0, SPP climbs (pack-off)
            wob, rpm, rop = 0.0, RNG.normal(15, 5), 0.0
            torque = 22 + 0.02 * depth * hardness + RNG.normal(0, 2.5)
            hook_load = 260 + 0.03 * depth + RNG.normal(0, 8)     # overpull trying to free string
            spp = 1600 + 0.35 * depth + RNG.normal(0, 60)          # pressure builds against pack-off
            mud_flow = max(0, RNG.normal(300, 40))
            gas_units = max(5, RNG.normal(25, 5))

        else:  # Equipment_Failure (mud pump down)
            wob, rpm, rop = 0.0, 0.0, 0.0
            torque = RNG.normal(1.0, 0.4)
            hook_load = 150 + 0.018 * depth + RNG.normal(0, 4)
            spp = max(0, RNG.normal(10, 5))         # pump down -> pressure collapses
            mud_flow = max(0, RNG.normal(5, 3))     # no circulation
            gas_units = max(2, RNG.normal(8, 2))

        mud_density = 9.6 + 0.0006 * depth + RNG.normal(0, 0.05)

        rows.append([
            t, well, round(depth, 2), activity,
            round(wob, 2), round(rpm, 1), round(torque, 2), round(rop, 2),
            round(hook_load, 2), round(spp, 1), round(mud_flow, 1), round(mud_density, 2),
            round(gas_units, 1),
            1 if activity == "NPT" else 0,
            active_npt_type_finished if active_npt_type_finished else "No_NPT",
        ])
        t = t + pd.Timedelta(minutes=SAMPLE_MIN)

    df_well = pd.DataFrame(rows, columns=[
        "Timestamp", "Well_ID", "Hole_Depth_m", "Activity_Code",
        "WOB_klb", "RPM", "Torque_kftlb", "ROP_m_hr",
        "Hook_Load_klb", "Standpipe_Pressure_psi", "Mud_Flow_Rate_gpm", "Mud_Density_ppg",
        "Gas_Units", "NPT_Flag", "NPT_Type",
    ])
    sensor_frames.append(df_well)

    # ---- build the daily drilling report from the sensor log ----
    report_date_series = df_well["Timestamp"].dt.date
    for date, day_df in df_well.groupby(report_date_series):
        depth_start = day_df["Hole_Depth_m"].iloc[0]
        depth_end = day_df["Hole_Depth_m"].iloc[-1]
        rotating_hours = (day_df["Activity_Code"] == "Drilling").sum() * SAMPLE_MIN / 60.0
        npt_hours = (day_df["Activity_Code"] == "NPT").sum() * SAMPLE_MIN / 60.0
        npt_types_today = sorted(set(day_df.loc[day_df.NPT_Type != "No_NPT", "NPT_Type"]))
        npt_summary = ", ".join(npt_types_today) if npt_types_today else "No_NPT"
        mud_cost = round(RNG.uniform(4500, 9500), 2)
        summary_txt = (
            f"Drilled {depth_end - depth_start:.0f} m in {well_header.loc[well_header.Well_ID==well,'Well_Type'].iloc[0]} section."
            + (f" NPT recorded: {npt_summary}." if npt_summary != "No_NPT" else " No NPT recorded.")
        )
        daily_frames.append([
            date, well, round(depth_start, 1), round(depth_end, 1),
            round(depth_end - depth_start, 1), round(rotating_hours, 2), round(npt_hours, 2),
            npt_summary, mud_cost, summary_txt,
        ])

sensor_log = pd.concat(sensor_frames, ignore_index=True)
sensor_log.to_csv("drilling_sensor_log.csv", index=False)

daily_report = pd.DataFrame(daily_frames, columns=[
    "Report_Date", "Well_ID", "Depth_Start_m", "Depth_End_m", "Meters_Drilled",
    "Rotating_Hours", "NPT_Hours", "NPT_Type_Summary", "Daily_Mud_Cost_USD", "Operation_Summary",
])
daily_report.to_csv("daily_drilling_report.csv", index=False)

# ------------------------------------------------------------------
# Sanity print
# ------------------------------------------------------------------
print("well_header.csv           :", well_header.shape)
print("drilling_sensor_log.csv   :", sensor_log.shape)
print("daily_drilling_report.csv :", daily_report.shape)
print()
print("NPT summary by well and type:")
print(sensor_log[sensor_log.NPT_Flag == 1].groupby(["Well_ID", "NPT_Type"]).size() * SAMPLE_MIN / 60.0)
