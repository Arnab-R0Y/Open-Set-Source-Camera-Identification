#!/usr/bin/env python3
"""
metadata_extract.py

Usage:
  python metadata_extract.py /path/to/images_or_folder --out metadata.csv
  python metadata_extract.py image1.jpg image2.heic --json metadata.json
  python metadata_extract.py /path/to/folder -r --out metadata.csv

Requirements:
 - Recommended: ExifTool installed and in PATH (best coverage). The script will use it if available.
 - Fallback (no ExifTool): pillow and exifread (pip install pillow exifread) — limited format coverage.
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from shutil import which
from datetime import datetime

# --- Utilities --------------------------------------------------------------

def find_files(paths, recursive=True, exts=None):
    """Expand input paths (files or directories) into a list of file paths."""
    out = []
    for p in paths:
        if os.path.isfile(p):
            out.append(os.path.abspath(p))
        elif os.path.isdir(p):
            if recursive:
                for root, _, files in os.walk(p):
                    for f in files:
                        if exts is None or os.path.splitext(f)[1].lower() in exts:
                            out.append(os.path.join(root, f))
            else:
                for f in os.listdir(p):
                    fp = os.path.join(p, f)
                    if os.path.isfile(fp) and (exts is None or os.path.splitext(fp)[1].lower() in exts):
                        out.append(fp)
        else:
            # try shell-style expansion for globs etc.
            import glob
            globbed = glob.glob(p, recursive=recursive)
            for g in globbed:
                if os.path.isfile(g):
                    out.append(os.path.abspath(g))
    # deduplicate and sort
    seen = set()
    uniq = []
    for f in out:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq

def choose_datetime(d):
    """Pick the best available datetime string from ExifTool/PIL keys."""
    candidates = [
        d.get('DateTimeOriginal'),
        d.get('CreateDate'),
        d.get('DateTime'),
        d.get('ModifyDate'),
        d.get('DateTimeDigitized'),
        d.get('FileModifyDate'),
        d.get('FileCreateDate'),
    ]
    for c in candidates:
        if c:
            # ExifTool often returns "YYYY:MM:DD HH:MM:SS" or "YYYY:MM:DD HH:MM:SS+ZZ:ZZ"
            # Try to normalize to ISO8601-ish
            s = str(c)
            # ExifTool's FileModifyDate looks like: "2023:10:23 13:45:12+05:30"
            s = s.replace(':', '-', 2)  # only replace first two colons to get YYYY-MM-DD...
            s = s.replace(' ', '/Time:', 1)
            return s
    return None

# --- Robust float conversion with DMS parsing -------------------------------
import re

def parse_dms_string(s):
    """Parse common DMS strings like '50 deg 49' 8.59\" N' or '50 49 8.59 N' -> decimal degrees."""
    if s is None:
        return None
    s = str(s).strip()
    # Typical formats: "50 deg 49' 8.59\" N", "50 49 8.59 N", "50°49'8.59\"N"
    # Regex: capture deg, min, sec (sec optional fractional), hemisphere
    m = re.search(r'([0-9]{1,3})[^0-9\-+]*([0-9]{1,3})[^0-9\-+]*([0-9]+(?:\.[0-9]+)?)\s*["\']?\s*([NnSsEeWw])', s)
    if m:
        try:
            deg = float(m.group(1))
            minute = float(m.group(2))
            sec = float(m.group(3))
            hemi = m.group(4).upper()
            dec = deg + minute / 60.0 + sec / 3600.0
            if hemi in ('S', 'W'):
                dec = -dec
            return dec
        except Exception:
            return None
    # fallback: try to extract a simple float from string (e.g. "0 m Above Sea Level" -> 0)
    m2 = re.search(r'([-+]?\d+\.\d+|[-+]?\d+)', s)
    if m2:
        try:
            return float(m2.group(0))
        except Exception:
            return None
    return None

def safe_float(x):
    """Robust conversion to float:
       - handles numeric types
       - handles tuples/lists like (num, den)
       - parses DMS strings and unit-bearing strings
    """
    try:
        if x is None:
            return None
        # numbers
        if isinstance(x, (int, float)):
            return float(x)
        # rational tuple/list from some libraries: (num, den)
        if isinstance(x, (list, tuple)):
            if len(x) == 2 and (isinstance(x[0], (int, float, str)) and isinstance(x[1], (int, float, str))):
                try:
                    return float(x[0]) / float(x[1])
                except Exception:
                    # attempt fallback: join and parse
                    return parse_dms_string(str(x))
            # fallback to first element or string parse
            return safe_float(x[0])
        s = str(x).strip()
        # numeric string (simple)
        try:
            return float(s)
        except Exception:
            pass
        # DMS or unit-bearing or other string -> try parse_dms_string
        return parse_dms_string(s)
    except Exception:
        return None


# --- ExifTool backend ------------------------------------------------------

def run_exiftool(files):
    """Run exiftool -j -n -c "%.8f" on a list of files and return parsed JSON objects.
    - -j : JSON output
    - -n : numeric values when possible
    - -c "%.8f" : format coordinates if exiftool still outputs non-numeric strings
    """
    if not files:
        return []
    cmd = ['exiftool', '-j', '-n', '-c', '%.8f'] + files
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        data = json.loads(out.decode('utf-8', errors='ignore'))
        return data
    except FileNotFoundError:
        raise
    except subprocess.CalledProcessError as e:
        try:
            out = e.output or b''
            if out:
                return json.loads(out.decode('utf-8', errors='ignore'))
        except Exception:
            pass
        raise


def extract_from_exiftool_record(rec):
    """Normalize exiftool JSON record into our output schema (robust GPS/key detection)."""
    model_keys = ['Model', 'CameraModelName', 'DeviceModel', 'LensModel']
    make_keys = ['Make', 'Manufacturer']

    dt = choose_datetime(rec)

    lat = None
    lon = None

    # 1) try direct GPSLatitude/GPSLongitude first (numeric or strings)
    lat = safe_float(rec.get('GPSLatitude'))
    lon = safe_float(rec.get('GPSLongitude'))

    # 2) try GPSPosition or Composite:GPSPosition (text like "50 deg ...")
    if (lat is None or lon is None) and 'GPSPosition' in rec:
        s = str(rec.get('GPSPosition'))
        parts = [p.strip().replace('N', '').replace('S', '').replace('E', '').replace('W', '') for p in s.replace(',', ' ').split()]
        if len(parts) >= 2:
            lat = safe_float(parts[0]) if lat is None else lat
            lon = safe_float(parts[1]) if lon is None else lon

    if (lat is None or lon is None) and 'Composite:GPSPosition' in rec:
        s = str(rec.get('Composite:GPSPosition'))
        parts = s.split(',')
        if len(parts) >= 2:
            lat = safe_float(parts[0].strip()) if lat is None else lat
            lon = safe_float(parts[1].strip()) if lon is None else lon

    # 3) defensive scan of keys (catch XMP:GPSLatitude, etc)
    if lat is None or lon is None:
        for k, v in rec.items():
            lk = str(k).lower()
            if lat is None and 'gpslatitude' in lk:
                lat = safe_float(v)
            if lon is None and 'gpslongitude' in lk:
                lon = safe_float(v)
            # sometimes keys are 'latitude' / 'longitude' under other namespaces
            if lat is None and (lk == 'latitude' or lk.endswith(':latitude')):
                lat = safe_float(v)
            if lon is None and (lk == 'longitude' or lk.endswith(':longitude')):
                lon = safe_float(v)
            if lat is not None and lon is not None:
                break

    # altitude: try common keys and robust parsing (handles "0 m Above Sea Level", tuples, numbers)
    alt = None
    for alt_key in ('GPSAltitude', 'Altitude', 'GPS Altitude', 'GPSAltitudeRef'):
        if alt_key in rec:
            alt = safe_float(rec.get(alt_key))
            if alt is not None:
                break
    # as fallback, scan keys
    if alt is None:
        for k, v in rec.items():
            if 'altitude' in str(k).lower():
                alt = safe_float(v)
                if alt is not None:
                    break

    # camera make/model selection
    model = None
    for k in model_keys:
        if k in rec:
            model = rec.get(k)
            break
    make = None
    for k in make_keys:
        if k in rec:
            make = rec.get(k)
            break

    def norm(v):
        if v is None:
            return None
        if isinstance(v, list):
            return ', '.join(map(str, v))
        return str(v)

    return {
        'source_file': norm(rec.get('SourceFile') or rec.get('FileName') or rec.get('SourceFile')),
        'datetime': dt,
        'camera_make': norm(make),
        'camera_model': norm(model),
        'latitude': lat,
        'longitude': lon,
        'altitude': alt,
        'raw': rec
    }


# --- Pure-Python fallback (Pillow + exifread) -------------------------------

def dms_to_decimal(dms, ref):
    # dms is a tuple like ((deg_num,deg_den),(min_num,min_den),(sec_num,sec_den))
    try:
        import fractions
        deg = fractions.Fraction(dms[0][0], dms[0][1])
        minute = fractions.Fraction(dms[1][0], dms[1][1])
        sec = fractions.Fraction(dms[2][0], dms[2][1])
        dec = float(deg + minute / 60 + sec / 3600)
        if ref in ('S', 'W', 's', 'w'):
            dec = -dec
        return dec
    except Exception:
        return None

def extract_with_pillow(filepath):
    """Fallback extraction for JPEG/TIFF using Pillow and/or exifread."""
    res = {
        'source_file': filepath,
        'datetime': None,
        'camera_make': None,
        'camera_model': None,
        'latitude': None,
        'longitude': None,
        'altitude': None,
        'raw': {}
    }
    try:
        from PIL import Image, ExifTags
    except Exception:
        return res

    try:
        img = Image.open(filepath)
        exif_raw = getattr(img, '_getexif', lambda: None)()
        if not exif_raw:
            return res
        exif = {}
        for tag_id, value in exif_raw.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            exif[tag] = value
        res['raw'].update(exif)
        # datetime
        res['datetime'] = exif.get('DateTimeOriginal') or exif.get('DateTime') or exif.get('DateTimeDigitized')
        # camera
        res['camera_make'] = exif.get('Make')
        res['camera_model'] = exif.get('Model')
        # GPS
        gps_info = exif.get('GPSInfo')
        if gps_info:
            # convert keys
            gps = {}
            for t, val in gps_info.items():
                name = ExifTags.GPSTAGS.get(t, t)
                gps[name] = val
            lat = gps.get('GPSLatitude')
            lat_ref = gps.get('GPSLatitudeRef')
            lon = gps.get('GPSLongitude')
            lon_ref = gps.get('GPSLongitudeRef')
            if lat and lat_ref and lon and lon_ref:
                res['latitude'] = dms_to_decimal(lat, lat_ref)
                res['longitude'] = dms_to_decimal(lon, lon_ref)
            # altitude
            alt = gps.get('GPSAltitude')
            if alt:
                try:
                    # often a ratio
                    if isinstance(alt, tuple):
                        res['altitude'] = float(alt[0]) / float(alt[1])
                    else:
                        res['altitude'] = float(alt)
                except Exception:
                    res['altitude'] = None
    except Exception:
        # swallow errors - return whatever we got
        pass
    return res

# --- Main extraction wrapper ------------------------------------------------

def extract_metadata(filepaths):
    """Try ExifTool first, otherwise fallback to Pillow/exifread for each file."""
    use_exiftool = which('exiftool') is not None
    results = []
    if use_exiftool:
        # ExifTool handles directories too, but we passed explicit files
        # To avoid very long command lines, call exiftool in chunks if needed
        CHUNK = 200
        for i in range(0, len(filepaths), CHUNK):
            chunk = filepaths[i:i+CHUNK]
            try:
                recs = run_exiftool(chunk)
            except FileNotFoundError:
                use_exiftool = False
                break
            except Exception as e:
                # fallback to per-file approach on failure
                recs = []
                try:
                    for f in chunk:
                        recs += run_exiftool([f])
                except Exception:
                    recs = []
            for r in recs:
                results.append(extract_from_exiftool_record(r))
    if not use_exiftool:
        # fallback: per-file Pillow extraction (limited to common types)
        for f in filepaths:
            results.append(extract_with_pillow(f))
    return results

# --- CLI / Output -----------------------------------------------------------

def write_csv(records, outpath):
    fieldnames = ['source_file', 'datetime', 'camera_make', 'camera_model', 'latitude', 'longitude', 'altitude']
    with open(outpath, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: ('' if r.get(k) is None else r.get(k)) for k in fieldnames}
            writer.writerow(row)
    print(f"CSV written to: {outpath}")

def write_json(records, outpath):
    # Remove 'raw' if you don't want the big raw dict dumped
    with open(outpath, 'w', encoding='utf-8') as jf:
        json.dump(records, jf, indent=2, ensure_ascii=False, default=str)
    print(f"JSON written to: {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Extract time/date, camera model, and GPS from images.")
    parser.add_argument('paths', nargs='+', help='Files or folders (supports globs).')
    parser.add_argument('--out', '-o', help='Output CSV file (or path). If omitted, prints JSON to stdout.')
    parser.add_argument('--json', help='Write JSON to this path instead of CSV.')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recurse into directories.')
    parser.add_argument('--exts', help='Comma-separated list of allowed extensions (e.g. .jpg,.png,.heic). Default: common image formats.')
    parser.add_argument('--no-raw', action='store_true', help='Do not include raw metadata block in JSON output.')
    args = parser.parse_args()

    exts = None
    if args.exts:
        exts = set(e.strip().lower() for e in args.exts.split(',') if e.strip().startswith('.'))
    else:
        exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.heif', '.cr2', '.nef', '.arw', '.raf', '.raw', '.dng'}

    files = find_files(args.paths, recursive=args.recursive, exts=exts)
    if not files:
        print("No files found. Check your paths or extension filter.", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(files)} files. Using ExifTool: {which('exiftool') is not None}")
    records = extract_metadata(files)

    # Optionally remove raw block to keep JSON small
    if args.no_raw:
        for r in records:
            if 'raw' in r:
                r.pop('raw', None)

    if args.out:
        out = args.out
        if out.lower().endswith('.csv'):
            write_csv(records, out)
        elif out.lower().endswith('.json'):
            write_json(records, out)
        else:
            # default to CSV
            write_csv(records, out)
    elif args.json:
        write_json(records, args.json)
    else:
        # print JSON to stdout
        print(json.dumps(records, indent=2, ensure_ascii=False, default=str))

if __name__ == '__main__':
    # If no CLI args provided, open a minimal Tk GUI for selecting a file.
    # If CLI args present, run normal CLI main().
    if len(sys.argv) == 1:
        try:
            import tkinter as tk
            from tkinter import ttk, filedialog, messagebox, scrolledtext
        except Exception as e:
            print("Tkinter not available. Install tkinter for GUI or run via CLI.")
            sys.exit(1)

        class SimpleMetadataViewer:
            def __init__(self, root):
                self.root = root
                root.title("Metadata Extractor - Select image")
                root.geometry("720x480")
                frm = ttk.Frame(root, padding=8)
                frm.pack(fill=tk.BOTH, expand=True)

                top = ttk.Frame(frm)
                top.pack(fill=tk.X, pady=(0,8))

                self.btn = ttk.Button(top, text="Select Image...", command=self.select_file)
                self.btn.pack(side=tk.LEFT)

                self.path_label = ttk.Label(top, text="No file selected", anchor='w')
                self.path_label.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

                self.show_raw_btn = ttk.Button(top, text="Show Raw JSON", command=self.show_raw, state=tk.DISABLED)
                self.show_raw_btn.pack(side=tk.RIGHT)

                cols = ("Field", "Value")
                self.tree = ttk.Treeview(frm, columns=cols, show='headings')
                self.tree.heading("Field", text="Field")
                self.tree.heading("Value", text="Value")
                self.tree.column("Field", width=200, anchor='w')
                self.tree.column("Value", width=480, anchor='w')
                self.tree.pack(fill=tk.BOTH, expand=True)

                self.status = ttk.Label(frm, text="Ready", anchor='w')
                self.status.pack(fill=tk.X, pady=(6,0))

                self.last_raw = None
                self.exiftool_ok = which('exiftool') is not None

            def select_file(self):
                filetypes = [("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.heic *.heif *.cr2 *.nef *.arw *.dng"), ("All files","*.*")]
                path = filedialog.askopenfilename(title="Select image", filetypes=filetypes)
                if not path:
                    return
                self.path_label.config(text=path)
                self.status.config(text="Reading metadata...")
                self.root.update_idletasks()

                raw = {}
                try:
                    if self.exiftool_ok:
                        # run exiftool for this single file
                        recs = run_exiftool([path])
                        if recs:
                            raw = recs[0]
                        else:
                            raw = {}
                    else:
                        # fallback to pillow
                        raw = extract_with_pillow(path).get('raw', {}) if isinstance(extract_with_pillow(path), dict) else {}
                except Exception as e:
                    # fallback: try pillow once more
                    try:
                        raw = extract_with_pillow(path).get('raw', {}) if isinstance(extract_with_pillow(path), dict) else {}
                    except Exception:
                        raw = {}

                # Normalize a display dict using extract_from_exiftool_record if ExifTool record available
                display = {}
                try:
                    # if raw looks like exiftool's JSON (i.e., has tags), use extractor
                    if 'SourceFile' in raw or 'FileName' in raw or isinstance(raw, dict):
                        rec_obj = extract_from_exiftool_record(raw if 'SourceFile' in raw else raw)
                        display = {
                            'File': rec_obj.get('source_file'),
                            'Date/Time': rec_obj.get('datetime') or 'N/A',
                            'Camera Make': rec_obj.get('camera_make') or 'N/A',
                            'Camera Model': rec_obj.get('camera_model') or 'N/A',
                            'Latitude': rec_obj.get('latitude') if rec_obj.get('latitude') is not None else 'N/A',
                            'Longitude': rec_obj.get('longitude') if rec_obj.get('longitude') is not None else 'N/A',
                            'Altitude': rec_obj.get('altitude') if rec_obj.get('altitude') is not None else 'N/A'
                        }
                        self.last_raw = raw
                        self.show_raw_btn.config(state=tk.NORMAL)
                    else:
                        display = {'File': path, 'Date/Time': 'N/A', 'Camera Make': 'N/A', 'Camera Model': 'N/A', 'Latitude': 'N/A', 'Longitude': 'N/A'}
                        self.last_raw = raw
                        self.show_raw_btn.config(state=tk.NORMAL if raw else tk.DISABLED)
                except Exception:
                    display = {'File': path, 'Date/Time': 'N/A', 'Camera Make': 'N/A', 'Camera Model': 'N/A', 'Latitude': 'N/A', 'Longitude': 'N/A'}

                # populate tree
                self.tree.delete(*self.tree.get_children())
                for k, v in display.items():
                    self.tree.insert("", tk.END, values=(k, str(v)))
                self.status.config(text="Done")

            def show_raw(self):
                if not self.last_raw:
                    return
                top = tk.Toplevel(self.root)
                top.title("Raw metadata JSON")
                top.geometry("800x600")
                st = scrolledtext.ScrolledText(top, wrap=tk.WORD)
                st.pack(fill=tk.BOTH, expand=True)
                try:
                    pretty = json.dumps(self.last_raw, indent=2, ensure_ascii=False, default=str)
                except Exception:
                    pretty = str(self.last_raw)
                st.insert("1.0", pretty)
                st.configure(state='disabled')

        root = tk.Tk()
        app = SimpleMetadataViewer(root)
        root.mainloop()

    else:
        # CLI mode: keep original behavior
        main()
