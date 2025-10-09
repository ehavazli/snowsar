#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, OrderedDict
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape as xml_escape, quoteattr as xml_quoteattr
# ---------- Helpers ----------

def _bool_str(v: bool) -> str:
    return "True" if v else "False"

def _validate_ref_date(s: str) -> str:
    # Expect YYMMDD (six digits)
    if not re.fullmatch(r"\d{6}", s):
        raise argparse.ArgumentTypeError("reference date must be YYMMDD (e.g., 211211)")
    return s

# ---------- Config with required & defaulted fields ----------

@dataclass
class StackInsarConfig:
    # REQUIRED (first 5)
    data_directory: str
    dem_for_coregistration: str
    dem_for_geocoding: str
    water_body: str
    reference_date_of_the_stack: str  # YYMMDD

    # EDITABLE (defaults prefilled)
    number_of_subsequent_dates: int = 5
    number_of_subsequent_dates_for_estimating_ionosphere: int = 5

    interferogram_filter_strength: float = 0.5
    interferogram_filter_window_size: int = 32
    interferogram_filter_step_size: int = 4
    remove_magnitude_before_filtering: bool = True

    water_body_mask_starting_step: str = "unwrap"  # None|filt|unwrap

    do_ionospheric_phase_estimation: bool = True
    apply_ionospheric_phase_correction: bool = True

    apply_polynomial_fit_before_filtering_ionosphere_phase: bool = True
    whether_filtering_ionosphere_phase: bool = True
    apply_polynomial_fit_in_adaptive_filtering_window: bool = True
    whether_do_secondary_filtering_of_ionosphere_phase: bool = True
    maximum_window_size_for_filtering_ionosphere_phase: int = 301
    minimum_window_size_for_filtering_ionosphere_phase: int = 101
    window_size_of_secondary_filtering_of_ionosphere_phase: int = 5

    filter_subband_interferogram: bool = True
    subband_interferogram_filter_strength: float = 0.3
    subband_interferogram_filter_window_size: int = 128
    subband_interferogram_filter_step_size: int = 4
    remove_magnitude_before_filtering_subband_interferogram: bool = True

    def to_xml(self) -> str:
        """
        Render to the compact active-options-only XML in the exact property order below.
        """
        # Property order (first 5 required, then defaults)
        ordered_props = [
            ("data directory", self.data_directory),
            ("dem for coregistration", self.dem_for_coregistration),
            ("dem for geocoding", self.dem_for_geocoding),
            ("water body", self.water_body),
            ("reference date of the stack", self.reference_date_of_the_stack),

            ("number of subsequent dates", self.number_of_subsequent_dates),
            ("number of subsequent dates for estimating ionosphere",
             self.number_of_subsequent_dates_for_estimating_ionosphere),

            ("interferogram filter strength", self.interferogram_filter_strength),
            ("interferogram filter window size", self.interferogram_filter_window_size),
            ("interferogram filter step size", self.interferogram_filter_step_size),
            ("remove magnitude before filtering", _bool_str(self.remove_magnitude_before_filtering)),

            ("water body mask starting step", self.water_body_mask_starting_step),

            ("do ionospheric phase estimation", _bool_str(self.do_ionospheric_phase_estimation)),
            ("apply ionospheric phase correction", _bool_str(self.apply_ionospheric_phase_correction)),

            ("apply polynomial fit before filtering ionosphere phase",
             _bool_str(self.apply_polynomial_fit_before_filtering_ionosphere_phase)),
            ("whether filtering ionosphere phase", _bool_str(self.whether_filtering_ionosphere_phase)),
            ("apply polynomial fit in adaptive filtering window",
             _bool_str(self.apply_polynomial_fit_in_adaptive_filtering_window)),
            ("whether do secondary filtering of ionosphere phase",
             _bool_str(self.whether_do_secondary_filtering_of_ionosphere_phase)),
            ("maximum window size for filtering ionosphere phase",
             self.maximum_window_size_for_filtering_ionosphere_phase),
            ("minimum window size for filtering ionosphere phase",
             self.minimum_window_size_for_filtering_ionosphere_phase),
            ("window size of secondary filtering of ionosphere phase",
             self.window_size_of_secondary_filtering_of_ionosphere_phase),

            ("filter subband interferogram", _bool_str(self.filter_subband_interferogram)),
            ("subband interferogram filter strength", self.subband_interferogram_filter_strength),
            ("subband interferogram filter window size", self.subband_interferogram_filter_window_size),
            ("subband interferogram filter step size", self.subband_interferogram_filter_step_size),
            ("remove magnitude before filtering subband interferogram",
             _bool_str(self.remove_magnitude_before_filtering_subband_interferogram)),
        ]
        lines = ['<?xml version="1.0" encoding="UTF-8"?>',
                 '<stack>',
                 '  <component name="stackinsar">']

        for key, val in ordered_props:
            # Escape attribute & text properly
            name_attr = xml_quoteattr(str(key))  # includes surrounding quotes
            text = xml_escape(str(val))
            lines.append(f'    <property name={name_attr}>{text}</property>')

        lines += ['  </component>', '</stack>']
        return "\n".join(lines) + "\n"
        # stack = ET.Element("stack")
        # comp = ET.SubElement(stack, "component", {"name": "stackinsar"})
        # for key, val in ordered_props:
        #     prop = ET.SubElement(comp, "property", {"name": key})
        #     prop.text = str(val)

        # # Pretty print without extra comments
        # xml_bytes = ET.tostring(stack, encoding="utf-8")
        # # Minimal pretty: add header and no extra whitespace
        # return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes.decode("utf-8")

# ---------- API ----------

def generate_stackinsar_xml(
    data_directory: str,
    dem_for_coregistration: str,
    dem_for_geocoding: str,
    water_body: str,
    reference_date_of_the_stack: str,
    **overrides,
) -> str:
    """
    Build XML string. First five args are REQUIRED. Any other field in StackInsarConfig
    can be overridden via kwargs (e.g., number_of_subsequent_dates=7).
    """
    cfg = StackInsarConfig(
        data_directory=data_directory,
        dem_for_coregistration=dem_for_coregistration,
        dem_for_geocoding=dem_for_geocoding,
        water_body=water_body,
        reference_date_of_the_stack=reference_date_of_the_stack,
    )
    # Apply overrides if provided
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise KeyError(f"Unknown option '{k}'")
        setattr(cfg, k, v)
    return cfg.to_xml()

def write_xml(path: str | Path, xml_text: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    if not xml_text.endswith("\n"):
        xml_text += "\n"
    p.write_text(xml_text, encoding="utf-8")
    return p

# def write_xml(path: str | Path, xml_text: str) -> Path:
#     p = Path(path).expanduser().resolve()
#     p.parent.mkdir(parents=True, exist_ok=True)
#     p.write_text(xml_text, encoding="utf-8")
#     return p

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(
        description="Generate stackinsar XML: first 5 options required, rest editable with defaults."
    )

    # Required
    ap.add_argument("--data-directory", required=True, help="Path to 'Dates' (or 'data') directory.")
    ap.add_argument("--dem-coreg", required=True, help="DEM path for coregistration (wgs84).")
    ap.add_argument("--dem-geocode", required=True, help="DEM path for geocoding (wgs84).")
    ap.add_argument("--water-body", required=True, help="WBD path (swbd*.wbd).")
    ap.add_argument("--ref-date", required=True, type=_validate_ref_date, help="Reference date (YYMMDD).")

    # Editable options (optional)
    ap.add_argument("--subseq", type=int, default=5, help="number of subsequent dates (default 5)")
    ap.add_argument("--subseq-ion", type=int, default=5, help="number of subsequent dates for estimating ionosphere (default 5)")

    ap.add_argument("--ifg-filter-strength", type=float, default=0.5)
    ap.add_argument("--ifg-filter-window", type=int, default=32)
    ap.add_argument("--ifg-filter-step", type=int, default=4)
    ap.add_argument("--ifg-remove-mag", action="store_true", default=True)
    ap.add_argument("--ifg-keep-mag", dest="ifg_remove_mag", action="store_false")

    ap.add_argument("--wbd-mask-step", choices=["None", "filt", "unwrap"], default="unwrap")

    ap.add_argument("--iono-est", action="store_true", default=True)
    ap.add_argument("--no-iono-est", dest="iono_est", action="store_false")
    ap.add_argument("--iono-apply", action="store_true", default=True)
    ap.add_argument("--no-iono-apply", dest="iono_apply", action="store_false")

    ap.add_argument("--iono-poly-before", action="store_true", default=True)
    ap.add_argument("--no-iono-poly-before", dest="iono_poly_before", action="store_false")
    ap.add_argument("--iono-filter", action="store_true", default=True)
    ap.add_argument("--no-iono-filter", dest="iono_filter", action="store_false")
    ap.add_argument("--iono-poly-adapt", action="store_true", default=True)
    ap.add_argument("--no-iono-poly-adapt", dest="iono_poly_adapt", action="store_false")
    ap.add_argument("--iono-second", action="store_true", default=True)
    ap.add_argument("--no-iono-second", dest="iono_second", action="store_false")
    ap.add_argument("--iono-win-max", type=int, default=301)
    ap.add_argument("--iono-win-min", type=int, default=101)
    ap.add_argument("--iono-win2", type=int, default=5)

    ap.add_argument("--subband-filter", action="store_true", default=True)
    ap.add_argument("--no-subband-filter", dest="subband_filter", action="store_false")
    ap.add_argument("--subband-strength", type=float, default=0.3)
    ap.add_argument("--subband-window", type=int, default=128)
    ap.add_argument("--subband-step", type=int, default=4)
    ap.add_argument("--subband-remove-mag", action="store_true", default=True)
    ap.add_argument("--subband-keep-mag", dest="subband_remove_mag", action="store_false")

    ap.add_argument("-o", "--out", required=True, help="Output XML path")

    args = ap.parse_args()

    xml_text = generate_stackinsar_xml(
        data_directory=args.data_directory,
        dem_for_coregistration=args.dem_coreg,
        dem_for_geocoding=args.dem_geocode,
        water_body=args.water_body,
        reference_date_of_the_stack=args.ref_date,
        number_of_subsequent_dates=args.subseq,
        number_of_subsequent_dates_for_estimating_ionosphere=args.subseq_ion,
        interferogram_filter_strength=args.ifg_filter_strength,
        interferogram_filter_window_size=args.ifg_filter_window,
        interferogram_filter_step_size=args.ifg_filter_step,
        remove_magnitude_before_filtering=args.ifg_remove_mag,
        water_body_mask_starting_step=args.wbd_mask_step,
        do_ionospheric_phase_estimation=args.iono_est,
        apply_ionospheric_phase_correction=args.iono_apply,
        apply_polynomial_fit_before_filtering_ionosphere_phase=args.iono_poly_before,
        whether_filtering_ionosphere_phase=args.iono_filter,
        apply_polynomial_fit_in_adaptive_filtering_window=args.iono_poly_adapt,
        whether_do_secondary_filtering_of_ionosphere_phase=args.iono_second,
        maximum_window_size_for_filtering_ionosphere_phase=args.iono_win_max,
        minimum_window_size_for_filtering_ionosphere_phase=args.iono_win_min,
        window_size_of_secondary_filtering_of_ionosphere_phase=args.iono_win2,
        filter_subband_interferogram=args.subband_filter,
        subband_interferogram_filter_strength=args.subband_strength,
        subband_interferogram_filter_window_size=args.subband_window,
        subband_interferogram_filter_step_size=args.subband_step,
        remove_magnitude_before_filtering_subband_interferogram=args.subband_remove_mag,
    )

    out_path = write_xml(args.out, xml_text)
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
