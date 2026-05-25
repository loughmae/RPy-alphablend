"""
DICOM Image Series Comparison Tool (v4)
----------------------------------------

"""

import sys
import os
import gc
import numpy as np
import pydicom
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QWidget, QMessageBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QInputDialog, QCheckBox, QGroupBox,
    QComboBox, QRadioButton, QButtonGroup, QSizePolicy
)
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor
import pyqtgraph as pg
import csv
from skimage.metrics import structural_similarity as ssim_func
from skimage.transform import resize as sk_resize
from scipy.ndimage import sobel
from scipy.spatial.distance import braycurtis

# ── Optional dependency: SimpleITK (rigid registration only) ─────────────────
try:
    import SimpleITK as sitk
    HAVE_SITK = True
except ImportError:
    HAVE_SITK = False

# ── Constants ─────────────────────────────────────────────────────────────────
HIST_BINS = 256
F32       = np.float32
_BITS_TO_MAX   = {8: 255, 10: 1023, 12: 4095, 14: 16383, 16: 65535}
_DEFAULT_PSNR_MAX = 4095


# ═══════════════════════════════════════════════════════════════════════════════
#  Low-level pixel / metric helpers
# ═══════════════════════════════════════════════════════════════════════════════

def read_pixels_f32(filepath: str):
    """Return (pixel_array_float32, bits_stored). Dataset released immediately."""
    ds   = pydicom.dcmread(filepath)
    arr  = ds.pixel_array.astype(F32)
    bits = int(getattr(ds, 'BitsStored', 12))
    del ds
    gc.collect()
    return arr, bits


def resample_to_match(source: np.ndarray, target_shape: tuple) -> np.ndarray:
    if source.shape == target_shape:
        return source
    return sk_resize(source, target_shape, order=1,
                     mode='reflect', anti_aliasing=True,
                     preserve_range=True).astype(F32)


def register_rigid(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    if not HAVE_SITK:
        return resample_to_match(moving, fixed.shape)
    fixed_2d  = np.squeeze(fixed).astype(np.float64)
    moving_2d = np.squeeze(resample_to_match(moving, fixed_2d.shape)).astype(np.float64)
    if fixed_2d.ndim != 2 or moving_2d.ndim != 2:
        return resample_to_match(moving, fixed_2d.shape)
    fixed_sitk  = sitk.GetImageFromArray(fixed_2d)
    moving_sitk = sitk.GetImageFromArray(moving_2d)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.10)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=100,
        convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    tx_init = sitk.CenteredTransformInitializer(
        fixed_sitk, moving_sitk, sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg.SetInitialTransform(tx_init, inPlace=False)
    result = None
    try:
        final_tx = reg.Execute(fixed_sitk, moving_sitk)
        rs = sitk.Resample(moving_sitk, fixed_sitk, final_tx,
                           sitk.sitkLinear, 0.0, moving_sitk.GetPixelID())
        result = sitk.GetArrayFromImage(rs).astype(F32)
        del rs
    except Exception:
        result = resample_to_match(moving, fixed_2d.shape)
    finally:
        del fixed_2d, moving_2d, fixed_sitk, moving_sitk
        gc.collect()
    return result if result is not None else resample_to_match(moving, fixed.shape)


def _shared_histogram(arr1, arr2):
    gmin = min(arr1.min(), arr2.min())
    gmax = max(arr1.max(), arr2.max())
    if gmax == gmin:
        gmax = gmin + 1.0
    h1, _ = np.histogram(arr1, bins=HIST_BINS, range=(gmin, gmax))
    h2, _ = np.histogram(arr2, bins=HIST_BINS, range=(gmin, gmax))
    h1 = h1.astype(F32) / (h1.sum() + 1e-12)
    h2 = h2.astype(F32) / (h2.sum() + 1e-12)
    return h1, h2


# ─── Paired metrics ───────────────────────────────────────────────────────────

def metric_mse(a, b):
    return float(np.mean((a - b) ** 2, dtype=np.float64))

def metric_mad(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def metric_maxad(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))

def metric_psnr(a, b, bits_stored=12):
    mse = metric_mse(a, b)
    if mse == 0:
        return float('inf')
    data_max = float(_BITS_TO_MAX.get(bits_stored, _DEFAULT_PSNR_MAX))
    return float(10.0 * np.log10((data_max ** 2) / mse))

def metric_ssim(a, b):
    a = np.squeeze(a); b = np.squeeze(b)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    min_dim  = int(min(a.shape))
    win_size = min(7, min_dim)
    if win_size % 2 == 0: win_size -= 1
    win_size = max(win_size, 3)
    idx, img = ssim_func(a, b, data_range=data_range, win_size=win_size, full=True)
    return float(idx), img.astype(F32)

def metric_sam(a, b):
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    na = np.linalg.norm(af); nb = np.linalg.norm(bf)
    if na == 0 or nb == 0: return float('nan')
    return float(np.degrees(np.arccos(np.clip(np.dot(af, bf) / (na * nb), -1., 1.))))

def metric_frobenius(a, b):
    return float(np.linalg.norm((a.astype(np.float64) - b.astype(np.float64)).ravel()))

def metric_bray_curtis(a, b):
    return float(braycurtis(a.flatten().astype(np.float64),
                            b.flatten().astype(np.float64)))

def metrics_from_histograms(h1, h2):
    eps = 1e-12
    h1 = h1.astype(np.float64); h2 = h2.astype(np.float64)
    bc   = float(np.clip(np.sum(np.sqrt(h1 * h2)), eps, 1.0 - eps))
    bd   = float(-np.log(bc))
    den  = h1 + h2
    chi2 = float(0.5 * np.sum(np.where(den > eps, (h1 - h2) ** 2 / den, 0.0)))
    wjs  = float(np.sum(np.minimum(h1, h2)) / (np.sum(np.maximum(h1, h2)) + eps))
    return {"BD": bd, "BC": bc, "Chi2": chi2, "WJS": wjs}

def metric_vif(a, b):
    from scipy.ndimage import uniform_filter
    EPS = 1e-10; SIGMA = 2.0; N = 3
    ref = a.astype(np.float64); dist = b.astype(np.float64)
    num = den = 0.0
    for _ in range(2):
        mu_r = uniform_filter(ref,  size=N, mode='reflect')
        mu_d = uniform_filter(dist, size=N, mode='reflect')
        srr  = np.maximum(uniform_filter(ref*ref,   size=N, mode='reflect') - mu_r**2, 0)
        sdd  = np.maximum(uniform_filter(dist*dist, size=N, mode='reflect') - mu_d**2, 0)
        srd  = uniform_filter(ref*dist, size=N, mode='reflect') - mu_r*mu_d
        g    = srd / (srr + EPS)
        sv   = np.maximum(sdd - g*srd, 0)
        mask = srr > EPS
        num += np.sum(np.log2(1 + g[mask]**2 * srr[mask] / (sv[mask] + SIGMA)))
        den += np.sum(np.log2(1 + srr[mask] / SIGMA))
        ref = ref[::2, ::2]; dist = dist[::2, ::2]
    return float(num / den) if den > EPS else float('nan')

# ─── Unpaired metrics ─────────────────────────────────────────────────────────

def metric_entropy(arr):
    hist, _ = np.histogram(arr, bins=HIST_BINS)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0: return 0.0
    pk = hist / total; pk = pk[pk > 0]
    return float(-np.sum(pk * np.log2(pk)))

def metric_tenengrad(arr):
    a64 = arr.astype(np.float64)
    gx = sobel(a64, axis=1); gy = sobel(a64, axis=0)
    tg = float(np.mean(gx**2 + gy**2))
    del a64, gx, gy
    return tg

# ─── ROI-based metrics ────────────────────────────────────────────────────────

def compute_roi_metrics(signal_vals, noise_vals):
    mean_sig = float(np.mean(signal_vals)); std_sig = float(np.std(signal_vals))
    mean_noi = float(np.mean(noise_vals));  std_noi = float(np.std(noise_vals))
    if std_noi == 0:
        snr = sdnr = cnr = float('nan')
    else:
        snr  = mean_sig / std_noi
        sdnr = (mean_sig - mean_noi) / std_noi
        cnr  = abs(mean_sig - mean_noi) / std_noi
    return {"mean_sig": mean_sig, "std_sig": std_sig,
            "mean_noi": mean_noi, "std_noi": std_noi,
            "SNR": snr, "SDNR": sdnr, "CNR": cnr}

def _pct_change(current, reference):
    try:
        ref = float(reference); cur = float(current)
    except (TypeError, ValueError):
        return float('nan')
    if ref == 0 or np.isnan(ref) or np.isnan(cur): return float('nan')
    return (cur - ref) / abs(ref) * 100.0

def _safe_metric(name, func, a, b, errors):
    try:
        return func(a, b)
    except Exception as e:
        errors[name] = str(e); return float('nan')


def compute_all_metrics(arr1, arr2, bits_stored=12):
    """All IQMs. Arrays squeezed to (H,W) at entry."""
    arr1 = np.squeeze(arr1); arr2 = np.squeeze(arr2)
    if arr1.shape != arr2.shape:
        raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")
    errors = {}
    mean1 = float(np.mean(arr1, dtype=np.float64))
    sd1   = float(np.std(arr1,  dtype=np.float64))
    mean2 = float(np.mean(arr2, dtype=np.float64))
    sd2   = float(np.std(arr2,  dtype=np.float64))
    mse_val   = _safe_metric("MSE",   metric_mse,         arr1, arr2, errors)
    mad_val   = _safe_metric("MAD",   metric_mad,         arr1, arr2, errors)
    maxad_val = _safe_metric("MaxAD", metric_maxad,       arr1, arr2, errors)
    sam_val   = _safe_metric("SAM",   metric_sam,         arr1, arr2, errors)
    fn_val    = _safe_metric("FN",    metric_frobenius,   arr1, arr2, errors)
    bcd_val   = _safe_metric("BCD",   metric_bray_curtis, arr1, arr2, errors)
    vif_val   = _safe_metric("VIF",   metric_vif,         arr1, arr2, errors)
    try:
        psnr_val = metric_psnr(arr1, arr2, bits_stored)
    except Exception as e:
        errors["pSNR"] = str(e); psnr_val = float('nan')
    try:
        ssim_val, ssim_map = metric_ssim(arr1, arr2)
    except Exception as e:
        errors["SSIM"] = str(e); ssim_val = float('nan'); ssim_map = None
    try:
        h1, h2 = _shared_histogram(arr1, arr2)
        hm = metrics_from_histograms(h1, h2)
        bd_val = hm["BD"]; bc_val = hm["BC"]
        chi2_val = hm["Chi2"]; wjs_val = hm["WJS"]
    except Exception as e:
        errors["Hist"] = str(e)
        bd_val = bc_val = chi2_val = wjs_val = float('nan')
    try:
        ent1 = metric_entropy(arr1); ten1 = metric_tenengrad(arr1)
    except Exception as e:
        errors["Unpaired1"] = str(e); ent1 = ten1 = float('nan')
    try:
        ent2 = metric_entropy(arr2); ten2 = metric_tenengrad(arr2)
    except Exception as e:
        errors["Unpaired2"] = str(e); ent2 = ten2 = float('nan')
    diff_map = (arr2.astype(np.float64) - arr1.astype(np.float64)).astype(F32)
    return {
        "MSE": mse_val, "MAD": mad_val, "MaxAD": maxad_val,
        "pSNR": psnr_val, "SSIM": ssim_val,
        "SAM": sam_val, "FN": fn_val, "BCD": bcd_val, "VIF": vif_val,
        "BD": bd_val, "BC": bc_val, "Chi2": chi2_val, "WJS": wjs_val,
        "Mean1": mean1, "SD1": sd1, "Mean2": mean2, "SD2": sd2,
        "Entropy1": ent1, "Tenengrad1": ten1,
        "Entropy2": ent2, "Tenengrad2": ten2,
        "_ssim_map": ssim_map, "_diff_map": diff_map, "_errors": errors,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Column definitions
# ═══════════════════════════════════════════════════════════════════════════════

COLUMNS = [
    "Series", "Slice", "ROI_Mode",
    "SNR1", "SDNR1", "CNR1",
    "SNR2", "SDNR2", "CNR2",
    "SNR%Δ", "SDNR%Δ", "CNR%Δ",
    "Mean1", "SD1", "Mean2", "SD2",
    "SSIM", "SSIM%Δ", "pSNR",
    "MSE", "MAD", "MaxAD",
    "SAM", "FN", "BCD", "VIF",
    "BD", "BC", "Chi2", "WJS",
    "Entropy1", "Tenengrad1",
    "Entropy2", "Tenengrad2",
    "Errors",
]
# Numeric columns available for charting (excludes text columns)
CHART_COLS = [c for c in COLUMNS if c not in ("Series", "Slice", "ROI_Mode", "Errors")]


# ═══════════════════════════════════════════════════════════════════════════════
#  Chart window
# ═══════════════════════════════════════════════════════════════════════════════

class ChartWindow(QMainWindow):
    """Live metric chart — updates each time the viewer adds a new row.

    Modes
    -----
    Running  : line + scatter of selected Y column vs selected X column.
    Tolerance: same plot plus ±5 % and ±10 % horizontal bands around the
               mean of all currently plotted Y values.
    """

    _TOLERANCE_PCTS = [5.0, 10.0]           # % bands to draw
    _BAND_COLOURS   = [(255,165,0,80),       # ±5 % — amber, semi-transparent
                       (255,  0,0,50)]       # ±10% — red,   semi-transparent

    def __init__(self, measurements_ref: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metric Chart")
        self.resize(800, 500)
        # Live reference to viewer's measurement list (not a copy)
        self._data = measurements_ref
        self._tol_items = []   # tolerance band PlotItems currently on the plot
        self._initUI()

    def _initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout()

        # ── Controls row ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        ctrl.addWidget(QLabel("X axis:"))
        self.x_combo = QComboBox()
        self.x_combo.addItem("Index")
        self.x_combo.addItems(CHART_COLS)
        self.x_combo.setCurrentIndex(0)
        self.x_combo.currentIndexChanged.connect(self.refresh)
        ctrl.addWidget(self.x_combo)

        ctrl.addWidget(QLabel("Y axis:"))
        self.y_combo = QComboBox()
        self.y_combo.addItems(CHART_COLS)
        self.y_combo.setCurrentIndex(CHART_COLS.index("SSIM") if "SSIM" in CHART_COLS else 0)
        self.y_combo.currentIndexChanged.connect(self.refresh)
        ctrl.addWidget(self.y_combo)

        # Mode radio buttons
        self.mode_running  = QRadioButton("Running chart")
        self.mode_toleranc = QRadioButton("Tolerance bands")
        self.mode_running.setChecked(True)
        self.mode_running.setToolTip("Line plot of selected metric vs analysis order.")
        self.mode_toleranc.setToolTip(
            "Running chart plus ±5 % and ±10 % horizontal bands around the mean."
        )
        mode_grp = QButtonGroup(self)
        mode_grp.addButton(self.mode_running)
        mode_grp.addButton(self.mode_toleranc)
        self.mode_running.toggled.connect(self.refresh)
        ctrl.addWidget(self.mode_running)
        ctrl.addWidget(self.mode_toleranc)

        ctrl.addStretch()
        layout.addLayout(ctrl)

        # ── Plot widget ───────────────────────────────────────────────────────
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setLabel('left',   '')
        self.plot_widget.setLabel('bottom', 'Index')
        layout.addWidget(self.plot_widget)

        central.setLayout(layout)

    def _get_col_values(self, col_name: str):
        """Extract numeric values for *col_name* from current measurements.
        Returns list of (row_index, float_value) for finite values only."""
        if col_name == "Index":
            return [(i, float(i)) for i in range(len(self._data))]
        col_idx = COLUMNS.index(col_name) if col_name in COLUMNS else -1
        if col_idx < 0:
            return []
        out = []
        for i, row in enumerate(self._data):
            try:
                v = float(row[col_idx])
                if np.isfinite(v):
                    out.append((i, v))
            except (ValueError, TypeError, IndexError):
                pass
        return out

    def refresh(self):
        """Redraw the chart with current data."""
        self.plot_widget.clear()
        self._tol_items.clear()

        x_name = self.x_combo.currentText()
        y_name = self.y_combo.currentText()

        x_pairs = self._get_col_values(x_name)
        y_pairs = self._get_col_values(y_name)

        # Match indices that exist in both x and y
        x_dict = dict(x_pairs)
        y_dict = dict(y_pairs)
        common = sorted(set(x_dict.keys()) & set(y_dict.keys()))
        if not common:
            return

        xs = np.array([x_dict[i] for i in common])
        ys = np.array([y_dict[i] for i in common])

        # Line
        self.plot_widget.plot(xs, ys,
                              pen=pg.mkPen(color=(70, 130, 180), width=2),
                              name=y_name)
        # Scatter
        scatter = pg.ScatterPlotItem(
            x=xs, y=ys,
            symbol='o', size=8,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(70, 130, 180, 200)
        )
        self.plot_widget.addItem(scatter)

        self.plot_widget.setLabel('left',   y_name)
        self.plot_widget.setLabel('bottom', x_name)
        self.setWindowTitle(f"Chart — {y_name} vs {x_name}")

        # Tolerance bands
        if self.mode_toleranc.isChecked() and len(ys) > 0:
            mean_y = float(np.mean(ys))
            x_min, x_max = float(xs.min()), float(xs.max())
            pad = (x_max - x_min) * 0.02 if x_max != x_min else 0.5

            # Draw mean line
            self.plot_widget.addLine(
                y=mean_y,
                pen=pg.mkPen(color=(200, 200, 200), width=1, style=Qt.DashLine)
            )

            for pct, (r, g, b, a) in zip(
                    self._TOLERANCE_PCTS, self._BAND_COLOURS):
                delta = abs(mean_y) * pct / 100.0
                upper = mean_y + delta
                lower = mean_y - delta
                # Fill region as a polygon
                fill_x = np.array([x_min - pad, x_max + pad,
                                   x_max + pad, x_min - pad])
                fill_y = np.array([upper, upper, lower, lower])
                fill = pg.PlotDataItem(
                    x=fill_x, y=fill_y,
                    pen=pg.mkPen(None),
                    fillLevel=lower,
                    brush=pg.mkBrush(r, g, b, a)
                )
                # Use FillBetweenItem for a clean band
                upper_curve = pg.PlotDataItem(
                    x=[x_min - pad, x_max + pad], y=[upper, upper],
                    pen=pg.mkPen(color=(r, g, b), width=1, style=Qt.DotLine)
                )
                lower_curve = pg.PlotDataItem(
                    x=[x_min - pad, x_max + pad], y=[lower, lower],
                    pen=pg.mkPen(color=(r, g, b), width=1, style=Qt.DotLine)
                )
                band = pg.FillBetweenItem(
                    upper_curve, lower_curve,
                    brush=pg.mkBrush(r, g, b, a)
                )
                self.plot_widget.addItem(upper_curve)
                self.plot_widget.addItem(lower_curve)
                self.plot_widget.addItem(band)
                self._tol_items.extend([upper_curve, lower_curve, band])

                # Annotate band label at right edge
                label_x = x_max + pad * 0.3
                for val, sign in [(upper, "+"), (lower, "-")]:
                    text = pg.TextItem(
                        f"{sign}{pct:.0f}%",
                        color=(r, g, b), anchor=(0, 0.5)
                    )
                    text.setPos(label_x, val)
                    self.plot_widget.addItem(text)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main viewer
# ═══════════════════════════════════════════════════════════════════════════════

class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer and Analyzer v4")

        self.series1_paths = []; self.series2_paths = []
        self.series_label1 = ""; self.series_label2 = ""
        self.bits_stored1  = 12; self.bits_stored2  = 12

        self.current_slice1 = 0; self.current_slice2 = 0
        self._reg_cache: dict = {}

        self.measurements: list = []
        self.roi_signal = None; self.roi_noise = None
        self.roi_signal_mirror_item = None
        self.roi_noise_mirror_item  = None

        self.ssim_window = None
        self.diff_window = None
        self.chart_window: ChartWindow | None = None

        self.measurements_window = None
        self._table_widget: QTableWidget | None = None

        self.initUI()

    # ─────────────────────────────────────────────────────────────────────────
    #  UI
    # ─────────────────────────────────────────────────────────────────────────

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout()

        desc = QLabel(
            "Left = Reference (Series 1).  "
            "Blue ROI = Signal,  Red ROI = Noise.  "
            "Press Enter or Analyze."
        )
        desc.setWordWrap(True)
        main_layout.addWidget(desc)

        dep_label = QLabel(
            "SimpleITK (rigid registration): " +
            ("✓ available" if HAVE_SITK else "✗ not installed — pip install SimpleITK")
        )
        dep_label.setStyleSheet("color: grey; font-size: 10px;")
        main_layout.addWidget(dep_label)

        # Image viewers
        display_layout = QHBoxLayout()
        self.graphics_view1 = pg.ImageView()
        self.graphics_view2 = pg.ImageView()
        display_layout.addWidget(self.graphics_view1)
        display_layout.addWidget(self.graphics_view2)
        main_layout.addLayout(display_layout)

        # Sliders
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.valueChanged.connect(self.update_images_from_slider1)
        self.slider1.setToolTip("Series 1 slice")
        main_layout.addWidget(self.slider1)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.valueChanged.connect(self.update_images_from_slider2)
        self.slider2.setToolTip("Series 2 slice")
        main_layout.addWidget(self.slider2)

        # ── Buttons row ───────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        load_btn1 = QPushButton("Load Series 1 (Reference)")
        load_btn1.clicked.connect(lambda: self.load_dicom_series(1))
        load_btn2 = QPushButton("Load Series 2 (Current)")
        load_btn2.clicked.connect(lambda: self.load_dicom_series(2))
        analyse_btn  = QPushButton("Analyze")
        analyse_btn.clicked.connect(self.analyze)
        summary_btn  = QPushButton("Add Summary Row")
        summary_btn.clicked.connect(self._append_summary_row)
        chart_btn    = QPushButton("Open Chart")
        chart_btn.setToolTip("Open live metric chart window.")
        chart_btn.clicked.connect(self._open_chart)

        self.reg_checkbox = QCheckBox("Rigid registration")
        if not HAVE_SITK:
            self.reg_checkbox.setEnabled(False)

        for w in (load_btn1, load_btn2, analyse_btn,
                  summary_btn, chart_btn, self.reg_checkbox):
            btn_layout.addWidget(w)
        main_layout.addLayout(btn_layout)

        # ── Options group ─────────────────────────────────────────────────────
        opt_group  = QGroupBox("Options")
        opt_layout = QHBoxLayout()

        self.show_ssim_cb = QCheckBox("Show SSIM map")
        self.show_ssim_cb.setChecked(False)
        self.show_ssim_cb.setToolTip(
            "Show per-pixel SSIM map after each analysis.\n"
            "Range [-1,1]: bright = high similarity."
        )

        self.show_diff_cb = QCheckBox("Show difference image (Series 2 − 1)")
        self.show_diff_cb.setChecked(False)
        self.show_diff_cb.setToolTip(
            "Signed difference image: positive = current brighter."
        )

        self.roi_only_cb = QCheckBox("Restrict metrics to signal ROI")
        self.roi_only_cb.setChecked(False)
        self.roi_only_cb.setToolTip(
            "Crop both images to the blue signal ROI bounding box before\n"
            "computing all IQMs, SSIM map and difference image.\n"
            "ROI-based metrics (SNR/SDNR/CNR) are unaffected."
        )

        for w in (self.show_ssim_cb, self.show_diff_cb, self.roi_only_cb):
            opt_layout.addWidget(w)
        opt_group.setLayout(opt_layout)
        main_layout.addWidget(opt_group)

        central.setLayout(main_layout)
        self.initROIs()
        self.installEventFilter(self)

    # ─────────────────────────────────────────────────────────────────────────
    #  DICOM loading
    # ─────────────────────────────────────────────────────────────────────────

    def load_dicom_series(self, series_num: int):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select DICOM Files", "",
            "DICOM Files (*.dcm *.ima);;All Files (*)", options=options
        )
        if not files:
            return
        path_instance_pairs = []
        series_label = ""; bits_stored = 12
        for filepath in files:
            try:
                ds_meta = pydicom.dcmread(filepath, defer_size=256)
                instance = int(getattr(ds_meta, 'InstanceNumber', 0))
                path_instance_pairs.append((filepath, instance))
                if not series_label:
                    series_label = getattr(ds_meta, 'SeriesDescription', "") or ""
                if bits_stored == 12:
                    bits_stored = int(getattr(ds_meta, 'BitsStored', 12))
                del ds_meta
            except Exception as e:
                QMessageBox.critical(self, "Error",
                                     f"Error reading {os.path.basename(filepath)}: {e}")
        if not path_instance_pairs:
            QMessageBox.warning(self, "No Files", "No valid DICOM files found.")
            return
        if not series_label:
            series_label, ok = QInputDialog.getText(
                self, "Series Description", "Enter series description:")
            if not ok:
                return
        path_instance_pairs.sort(key=lambda x: x[1])
        sorted_paths = [p for p, _ in path_instance_pairs]
        if series_num == 1:
            self.series1_paths = sorted_paths
            self.series_label1 = series_label
            self.bits_stored1  = bits_stored
        else:
            self.series2_paths = sorted_paths
            self.series_label2 = series_label
            self.bits_stored2  = bits_stored
        self._reg_cache.clear()
        self.update_slider_range()
        self.update_images()

    # ─────────────────────────────────────────────────────────────────────────
    #  Slider / display
    # ─────────────────────────────────────────────────────────────────────────

    def update_slider_range(self):
        if self.series1_paths:
            self.slider1.setMaximum(len(self.series1_paths) - 1)
        if self.series2_paths:
            self.slider2.setMaximum(len(self.series2_paths) - 1)

    def update_images(self):
        self._display_slice(self.graphics_view1, self.series1_paths, self.current_slice1)
        self._display_slice(self.graphics_view2, self.series2_paths, self.current_slice2)

    def update_images_from_slider1(self):
        self.current_slice1 = self.slider1.value()
        self._display_slice(self.graphics_view1, self.series1_paths, self.current_slice1)
        if self.current_slice1 <= self.slider2.maximum():
            self.slider2.blockSignals(True)
            self.slider2.setValue(self.current_slice1)
            self.slider2.blockSignals(False)
            self.current_slice2 = self.current_slice1
        self._display_slice(self.graphics_view2, self.series2_paths, self.current_slice2)
        self._reg_cache.clear()

    def update_images_from_slider2(self):
        self.current_slice2 = self.slider2.value()
        self._display_slice(self.graphics_view2, self.series2_paths, self.current_slice2)
        self._reg_cache.clear()

    def _display_slice(self, view: pg.ImageView, paths: list, idx: int):
        """Load one slice, display it, release the array.

        Fix vs v3: view.clear() called first so pyqtgraph releases its
        internal reference to the previous array before the new one is loaded.
        Without this, both the old and new arrays are alive simultaneously.
        """
        if not paths or idx >= len(paths):
            return
        try:
            view.clear()                      # ← release previous array reference
            arr, _ = read_pixels_f32(paths[idx])
            arr = np.squeeze(arr)
            view.setImage(arr.T, autoLevels=False, levels=(0, arr.max()))
            view.ui.histogram.gradient.loadPreset('grey')
            view.ui.histogram.setImageItem(view.imageItem)
            del arr
        except Exception as e:
            QMessageBox.critical(self, "Display Error", str(e))

    # ─────────────────────────────────────────────────────────────────────────
    #  ROI management
    # ─────────────────────────────────────────────────────────────────────────

    def initROIs(self):
        self.roi_signal = pg.RectROI([20, 20], [20, 20], pen='b')
        self.roi_noise  = pg.RectROI([60, 60], [20, 20], pen='r')
        self.roi_signal.setZValue(10); self.roi_noise.setZValue(10)
        self.graphics_view1.addItem(self.roi_signal)
        self.graphics_view1.addItem(self.roi_noise)
        self.roi_signal.sigRegionChanged.connect(self.update_mirrored_rois)
        self.roi_noise.sigRegionChanged.connect(self.update_mirrored_rois)

    def update_mirrored_rois(self):
        for item in (self.roi_signal_mirror_item, self.roi_noise_mirror_item):
            if item:
                try:
                    self.graphics_view2.removeItem(item)
                except Exception:
                    pass
        ps = self.roi_signal.pos(); ss = self.roi_signal.size()
        pn = self.roi_noise.pos();  sn = self.roi_noise.size()
        self.roi_signal_mirror_item = pg.RectROI(ps, ss, pen='b')
        self.roi_noise_mirror_item  = pg.RectROI(pn, sn, pen='r')
        self.graphics_view2.addItem(self.roi_signal_mirror_item)
        self.graphics_view2.addItem(self.roi_noise_mirror_item)
        self.setWindowTitle(
            f"DICOM Viewer v4 — Signal ROI: {ss.x()*ss.y():.0f} px²  "
            f"Noise ROI: {sn.x()*sn.y():.0f} px²"
        )

    def _extract_roi_values(self, arr: np.ndarray, roi) -> np.ndarray:
        """Flattened pixel values inside *roi* from a (H,W) array."""
        arr2d = np.squeeze(arr)
        x, y  = map(int, roi.pos())
        w, h  = map(int, roi.size())
        y0, y1 = max(0, y), min(arr2d.shape[0], y + h)
        x0, x1 = max(0, x), min(arr2d.shape[1], x + w)
        return arr2d[y0:y1, x0:x1].flatten()

    def _crop_to_signal_roi(self, arr: np.ndarray) -> np.ndarray:
        """Return the 2D sub-array corresponding to the signal ROI bounding box.
        Used when 'Restrict to signal ROI' is checked.
        Guarantees at least a 3×3 patch (minimum viable for SSIM win_size=3).
        """
        arr2d = np.squeeze(arr)
        x, y  = map(int, self.roi_signal.pos())
        w, h  = map(int, self.roi_signal.size())
        y0, y1 = max(0, y), min(arr2d.shape[0], y + h)
        x0, x1 = max(0, x), min(arr2d.shape[1], x + w)
        patch = arr2d[y0:y1, x0:x1]
        # Safety guard: ensure minimum 3×3
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            raise ValueError(
                f"Signal ROI patch too small for metric computation: {patch.shape}. "
                "Enlarge the blue signal ROI."
            )
        return patch

    # ─────────────────────────────────────────────────────────────────────────
    #  Event filter
    # ─────────────────────────────────────────────────────────────────────────

    def eventFilter(self, source, event):
        if (event.type() == QEvent.KeyPress and
                event.key() in (Qt.Key_Return, Qt.Key_Enter)):
            self.analyze()
            return True
        return super().eventFilter(source, event)

    # ─────────────────────────────────────────────────────────────────────────
    #  Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self):
        if not self.series1_paths or not self.series2_paths:
            QMessageBox.warning(self, "Missing Data",
                                "Please load both DICOM series.")
            return
        if self.current_slice1 >= len(self.series1_paths):
            QMessageBox.warning(self, "Index Error", "Series 1 slice index out of range.")
            return
        if self.current_slice2 >= len(self.series2_paths):
            QMessageBox.warning(self, "Index Error", "Series 2 slice index out of range.")
            return

        p1 = self.series1_paths[self.current_slice1]
        p2 = self.series2_paths[self.current_slice2]

        arr1, bits1 = read_pixels_f32(p1)
        bits_stored  = bits1

        use_registration = self.reg_checkbox.isChecked()
        cache_key = (p1, p2, use_registration)

        if cache_key in self._reg_cache:
            arr2_aligned = self._reg_cache[cache_key]
        else:
            arr2, _ = read_pixels_f32(p2)
            if use_registration:
                if not HAVE_SITK:
                    QMessageBox.warning(
                        self, "SimpleITK Missing",
                        "SimpleITK not installed — falling back to bilinear resampling.\n"
                        "Install with: pip install SimpleITK"
                    )
                arr2_aligned = register_rigid(arr1, arr2)
            elif arr1.shape != arr2.shape:
                QMessageBox.information(
                    self, "Size Mismatch",
                    f"Image sizes differ ({np.squeeze(arr1).shape} vs "
                    f"{np.squeeze(arr2).shape}).\nSeries 2 resampled to match Series 1."
                )
                arr2_aligned = resample_to_match(arr2, np.squeeze(arr1).shape)
            else:
                arr2_aligned = arr2
            self._reg_cache.clear()
            self._reg_cache[cache_key] = arr2_aligned
            if arr2_aligned is not arr2:
                del arr2

        # ── ROI metrics (always full image ROI extraction) ────────────────────
        sig1 = self._extract_roi_values(arr1,         self.roi_signal)
        noi1 = self._extract_roi_values(arr1,         self.roi_noise)
        sig2 = self._extract_roi_values(arr2_aligned, self.roi_signal)
        noi2 = self._extract_roi_values(arr2_aligned, self.roi_noise)
        roi1 = compute_roi_metrics(sig1, noi1)
        roi2 = compute_roi_metrics(sig2, noi2)

        # ── Optionally crop to signal ROI before IQM computation ─────────────
        roi_only = self.roi_only_cb.isChecked()
        roi_label = "SignalROI" if roi_only else "Full"

        if roi_only:
            try:
                comp_arr1 = self._crop_to_signal_roi(arr1)
                comp_arr2 = self._crop_to_signal_roi(arr2_aligned)
            except ValueError as e:
                QMessageBox.warning(self, "ROI too small", str(e))
                return
        else:
            comp_arr1 = np.squeeze(arr1)
            comp_arr2 = np.squeeze(arr2_aligned)

        # ── IQMs ──────────────────────────────────────────────────────────────
        metrics = compute_all_metrics(comp_arr1, comp_arr2, bits_stored)

        # ── Window title ───────────────────────────────────────────────────────
        roi_tag = " [ROI]" if roi_only else ""
        self.setWindowTitle(
            f"SSIM {metrics['SSIM']:.4f}{roi_tag}  "
            f"pSNR {metrics['pSNR']:.2f} dB  "
            f"MAD {metrics['MAD']:.1f}  "
            f"SNR1 {roi1['SNR']:.2f}  SNR2 {roi2['SNR']:.2f}"
        )

        # ── Optional overlays ──────────────────────────────────────────────────
        if self.show_ssim_cb.isChecked():
            ssim_map = metrics.get("_ssim_map")
            if ssim_map is not None:
                self._show_ssim_image(ssim_map, metrics["SSIM"], roi_only)

        if self.show_diff_cb.isChecked():
            diff_map = metrics.get("_diff_map")
            if diff_map is not None:
                self._show_diff_image(diff_map, roi_only)

        # ── Record, update table, refresh chart ───────────────────────────────
        self._save_results(roi1, roi2, metrics, roi_label)

        del arr1, arr2_aligned, comp_arr1, comp_arr2
        gc.collect()

    # ─────────────────────────────────────────────────────────────────────────
    #  Overlays
    # ─────────────────────────────────────────────────────────────────────────

    def _show_ssim_image(self, ssim_map: np.ndarray, ssim_index: float,
                         roi_only: bool):
        if self.ssim_window is not None:
            try: self.ssim_window.close()
            except Exception: pass
        roi_tag = "  [Signal ROI crop]" if roi_only else ""
        self.ssim_window = QMainWindow()
        self.ssim_window.setWindowTitle(
            f"SSIM Map{roi_tag}  (index = {ssim_index:.5f})  "
            "[bright = high similarity]"
        )
        view = pg.ImageView()
        view.setImage(ssim_map.T, autoLevels=False, levels=(-1.0, 1.0))
        view.ui.histogram.gradient.loadPreset('bipolar')
        self.ssim_window.setCentralWidget(view)
        self.ssim_window.resize(600, 650)
        self.ssim_window.show()

    def _show_diff_image(self, diff_map: np.ndarray, roi_only: bool):
        if self.diff_window is not None:
            try: self.diff_window.close()
            except Exception: pass
        abs_max = float(np.max(np.abs(diff_map))) or 1.0
        roi_tag = "  [Signal ROI crop]" if roi_only else ""
        self.diff_window = QMainWindow()
        self.diff_window.setWindowTitle(
            f"Difference Image{roi_tag}  (Series 2 − Series 1)  "
            f"[±{abs_max:.1f}  MAD={float(np.mean(np.abs(diff_map))):.2f}  "
            f"MaxAD={abs_max:.2f}]"
        )
        view = pg.ImageView()
        view.setImage(diff_map.T, autoLevels=False, levels=(-abs_max, abs_max))
        view.ui.histogram.gradient.loadPreset('bipolar')
        self.diff_window.setCentralWidget(view)
        self.diff_window.resize(600, 650)
        self.diff_window.show()

    # ─────────────────────────────────────────────────────────────────────────
    #  Chart
    # ─────────────────────────────────────────────────────────────────────────

    def _open_chart(self):
        """Open (or raise) the live chart window."""
        if self.chart_window is None or not self.chart_window.isVisible():
            self.chart_window = ChartWindow(
                measurements_ref=self.measurements, parent=None
            )
        self.chart_window.show()
        self.chart_window.raise_()
        self.chart_window.refresh()

    # ─────────────────────────────────────────────────────────────────────────
    #  Measurements table
    # ─────────────────────────────────────────────────────────────────────────

    def _save_results(self, roi1, roi2, metrics, roi_label):
        def _fmt(v):
            if v is None: return "N/A"
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return str(v)
            if isinstance(v, float): return f"{v:.4f}"
            return str(v)

        snr1 = roi1["SNR"];  sdnr1 = roi1["SDNR"]; cnr1 = roi1["CNR"]
        snr2 = roi2["SNR"];  sdnr2 = roi2["SDNR"]; cnr2 = roi2["CNR"]

        row = [
            f"{self.series_label1} vs {self.series_label2}",
            f"{self.current_slice1}/{self.current_slice2}",
            roi_label,
            _fmt(snr1), _fmt(sdnr1), _fmt(cnr1),
            _fmt(snr2), _fmt(sdnr2), _fmt(cnr2),
            _fmt(_pct_change(snr2,  snr1)),
            _fmt(_pct_change(sdnr2, sdnr1)),
            _fmt(_pct_change(cnr2,  cnr1)),
            _fmt(metrics.get("Mean1")), _fmt(metrics.get("SD1")),
            _fmt(metrics.get("Mean2")), _fmt(metrics.get("SD2")),
            _fmt(metrics.get("SSIM")),
            _fmt(_pct_change(metrics.get("SSIM"), 1.0)),
            _fmt(metrics.get("pSNR")),
            _fmt(metrics.get("MSE")),
            _fmt(metrics.get("MAD")),
            _fmt(metrics.get("MaxAD")),
            _fmt(metrics.get("SAM")),
            _fmt(metrics.get("FN")),
            _fmt(metrics.get("BCD")),
            _fmt(metrics.get("VIF")),
            _fmt(metrics.get("BD")),  _fmt(metrics.get("BC")),
            _fmt(metrics.get("Chi2")),_fmt(metrics.get("WJS")),
            _fmt(metrics.get("Entropy1")), _fmt(metrics.get("Tenengrad1")),
            _fmt(metrics.get("Entropy2")), _fmt(metrics.get("Tenengrad2")),
            ",".join(metrics.get("_errors", {}).keys()) or "OK",
        ]
        self.measurements.append(row)
        self._update_table()
        # Refresh chart if open
        if self.chart_window is not None and self.chart_window.isVisible():
            self.chart_window.refresh()

    def _ensure_measurements_window(self):
        if self.measurements_window is not None and \
                self.measurements_window.isVisible():
            return
        self.measurements_window = QMainWindow()
        self.measurements_window.setWindowTitle("Measurements")
        self.measurements_window.resize(1400, 450)
        self._table_widget = QTableWidget()
        self._table_widget.setColumnCount(len(COLUMNS))
        self._table_widget.setHorizontalHeaderLabels(COLUMNS)
        self._table_widget.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        save_btn    = QPushButton("Save to CSV")
        save_btn.clicked.connect(self.save_measurements_to_csv)
        copy_btn    = QPushButton("Copy to Clipboard")
        copy_btn.clicked.connect(self._copy_to_clipboard)
        summary_btn = QPushButton("Add Summary Row")
        summary_btn.clicked.connect(self._append_summary_row)
        chart_btn2  = QPushButton("Open Chart")
        chart_btn2.clicked.connect(self._open_chart)
        btn_row = QHBoxLayout()
        for w in (save_btn, copy_btn, summary_btn, chart_btn2):
            btn_row.addWidget(w)
        layout = QVBoxLayout()
        layout.addWidget(self._table_widget)
        layout.addLayout(btn_row)
        container = QWidget()
        container.setLayout(layout)
        self.measurements_window.setCentralWidget(container)
        self.measurements_window.show()

    def _update_table(self):
        self._ensure_measurements_window()
        table = self._table_widget
        total = len(self.measurements)
        table.setRowCount(total)
        r = total - 1
        for c, val in enumerate(self.measurements[r]):
            table.setItem(r, c, QTableWidgetItem(val))
        table.scrollToBottom()

    def _append_summary_row(self):
        if not self.measurements:
            QMessageBox.information(self, "No Data",
                                    "Analyse at least one slice first.")
            return
        self._ensure_measurements_window()
        table  = self._table_widget
        col_idx = {col: i for i, col in enumerate(COLUMNS)}
        summary = ["SUMMARY", f"n={len(self.measurements)}", "—"]
        for col in COLUMNS[3:]:
            ci   = col_idx[col]
            vals = []
            for row in self.measurements:
                try:
                    v = float(row[ci])
                    if np.isfinite(v):
                        vals.append(v)
                except (ValueError, TypeError):
                    pass
            summary.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "N/A")
        r = table.rowCount()
        table.setRowCount(r + 1)
        for c, val in enumerate(summary):
            item = QTableWidgetItem(val)
            item.setBackground(QColor(220, 235, 255))
            table.setItem(r, c, item)
        table.scrollToBottom()

    def _copy_to_clipboard(self):
        lines = ["\t".join(COLUMNS)]
        for row in self.measurements:
            lines.append("\t".join(row))
        QApplication.clipboard().setText("\n".join(lines))
        QMessageBox.information(self, "Copied", "Measurements copied to clipboard.")

    def save_measurements_to_csv(self):
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Measurements", "", "CSV Files (*.csv)")
        if not save_path:
            return
        try:
            with open(save_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(COLUMNS)
                for row in self.measurements:
                    writer.writerow(row)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())
