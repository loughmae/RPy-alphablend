"""
Radiation Protection Scatter Map Generator
-------------------------------------------
A PyQt6 + Matplotlib GUI for overlaying radiation intensity data
onto floor-plan images or PDFs.

Features:
  - Load images (PNG, JPG, BMP, HEIC/HEIF) and PDFs as background
  - Crop, rotate, and annotate images (numbered markers, lines, free-draw)
  - Import intensity grids from CSV, Excel, or clipboard
  - Blended heatmap / contour overlays with draggable positioning
  - Highlight specific dose levels with contour lines + labels
  - Grid overlay for measurement planning
  - Configurable axis scales, units, colour maps, and export settings

"""

import sys
import gc
import io
import os
import json
import math
import threading

# ---------------------------------------------------------------------------
# PyQt6 – always imported (needed for the event loop)
# ---------------------------------------------------------------------------
from PyQt6.QtCore import Qt, QRect, QPointF, QRectF, QSizeF, QTimer
from PyQt6.QtGui import (
    QPixmap, QImage, QTransform, QPainter, QPen, QColor, QFont, QBrush,
    QKeySequence, QShortcut, QAction, QIcon,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QLabel, QMessageBox,
    QComboBox, QLineEdit, QGroupBox, QDoubleSpinBox, QSlider, QCheckBox, QSpinBox,
    QDialog, QFormLayout, QSizePolicy, QButtonGroup, QRadioButton, QColorDialog,
    QInputDialog, QScrollArea, QToolBar, QGridLayout,
)

# ---------------------------------------------------------------------------
# Matplotlib – imported eagerly (needed for DraggableCanvas widgets at init)
# ---------------------------------------------------------------------------
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Background pre-loader for heavy optional libraries
# ---------------------------------------------------------------------------
# These libraries are NOT needed at window creation time but WILL be needed
# when the user loads CSV/Excel, images, or PDFs.  By importing them in a
# background thread right after the window appears, they are ready by the
# time the user actually clicks a button.

_preload_ready = threading.Event()
_pd = None
_scipy_zoom = None
_fitz = None
_PILImage = None


def _preload_heavy_libs():
    """Import pandas, scipy, fitz, PIL in a background thread."""
    global _pd, _scipy_zoom, _fitz, _PILImage
    try:
        import pandas
        _pd = pandas
    except ImportError:
        pass
    try:
        from scipy.ndimage import zoom
        _scipy_zoom = zoom
    except ImportError:
        pass
    try:
        import fitz
        _fitz = fitz
    except ImportError:
        pass
    try:
        from PIL import Image
        _PILImage = Image
    except ImportError:
        pass
    _preload_ready.set()


def _wait_for_preload():
    """Block (briefly) until the background preload has finished.

    Called at the start of any function that needs pandas/scipy/fitz/PIL.
    If the preload already finished (typical), this returns instantly.
    """
    _preload_ready.wait()


def start_background_preload():
    """Kick off the background import thread.  Call once after window.show()."""
    t = threading.Thread(target=_preload_heavy_libs, daemon=True)
    t.start()


# Convenience accessors (call _wait_for_preload first)
def _get_pandas():
    _wait_for_preload()
    if _pd is None:
        raise ImportError("pandas is not installed.  pip install pandas")
    return _pd


def _get_scipy_zoom():
    _wait_for_preload()
    if _scipy_zoom is None:
        raise ImportError("scipy is not installed.  pip install scipy")
    return _scipy_zoom


def _get_fitz():
    _wait_for_preload()
    if _fitz is None:
        raise ImportError("PyMuPDF is not installed.  pip install pymupdf")
    return _fitz


def _get_pil():
    _wait_for_preload()
    if _PILImage is None:
        raise ImportError("Pillow is not installed.  pip install Pillow")
    return _PILImage


# ---------------------------------------------------------------------------
# Safe image size limit
# ---------------------------------------------------------------------------
# Images larger than this (on either axis) are downscaled on load to prevent
# out-of-memory crashes, especially with HEIC photos from modern phones
# (which can exceed 8000x6000).  The original aspect ratio is preserved.

MAX_IMAGE_DIMENSION = 4096  # pixels


def _safe_downscale_pixmap(pixmap: 'QPixmap', limit: int = MAX_IMAGE_DIMENSION) -> 'QPixmap':
    """Downscale *pixmap* so neither dimension exceeds *limit*.

    Returns the original pixmap unchanged if it is already within bounds.
    Uses smooth (bilinear) scaling.
    """
    w, h = pixmap.width(), pixmap.height()
    if w <= limit and h <= limit:
        return pixmap
    return pixmap.scaled(
        limit, limit,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


# ---------------------------------------------------------------------------
# Named colour palette  (plain-language selection)
# ---------------------------------------------------------------------------

NAMED_COLOURS = [
    ("Red", "#FF3232"),
    ("Blue", "#3264FF"),
    ("Green", "#32CD32"),
    ("Orange", "#FFA500"),
    ("Yellow", "#FFD700"),
    ("Cyan", "#00C8FF"),
    ("Magenta", "#FF00FF"),
    ("White", "#FFFFFF"),
    ("Black", "#000000"),
    ("Purple", "#9B59B6"),
    ("Pink", "#FF69B4"),
    ("Lime", "#00FF00"),
]

# Cache for colour-swatch QIcons so we build them only once
_colour_icon_cache: dict[str, QIcon] = {}


def _get_colour_icon(hex_val: str) -> QIcon:
    """Return a cached 14x14 colour-swatch QIcon."""
    if hex_val not in _colour_icon_cache:
        px = QPixmap(14, 14)
        px.fill(QColor(hex_val))
        _colour_icon_cache[hex_val] = QIcon(px)
    return _colour_icon_cache[hex_val]


def _build_colour_combo(default_name="Red") -> QComboBox:
    """Create a QComboBox populated with named colours and a colour swatch icon."""
    combo = QComboBox()
    for name, hex_val in NAMED_COLOURS:
        combo.addItem(_get_colour_icon(hex_val), name, hex_val)
    combo.addItem("Custom...", "custom")
    idx = next((i for i, (n, _) in enumerate(NAMED_COLOURS) if n == default_name), 0)
    combo.setCurrentIndex(idx)
    return combo


def _resolve_colour_combo(combo: QComboBox, parent=None) -> QColor:
    """Read the selected colour from a named-colour combo.

    If 'Custom...' is chosen, opens a QColorDialog.
    """
    data = combo.currentData()
    if data == "custom":
        col = QColorDialog.getColor(QColor("#FF0000"), parent, "Choose Colour")
        if col.isValid():
            return col
        return QColor("#FF0000")
    return QColor(data)


# ---------------------------------------------------------------------------
# Colour-map helpers
# ---------------------------------------------------------------------------

def get_threat_zone_cmap():
    """Seven-colour discrete colour map for threat-zone visualisation."""
    colors = ['#0033CC', '#00CCCC', '#00CC44', '#FFDD00', '#FF8800', '#FF2222', '#AA00CC']
    return ListedColormap(colors, name='threat_zones')


def get_continuous_dose_cmap():
    """Seven-colour continuous-style dose-field colour map."""
    colors = ['#0022CC', '#0099CC', '#00CC66', '#CCFF33', '#FFDD00', '#FF8800', '#FF2222']
    return ListedColormap(colors, name='dose_field')


# ---------------------------------------------------------------------------
# Utility helpers (shared by DraggableCanvas and MainWindow)
# ---------------------------------------------------------------------------

def _resolve_cmap(name_or_obj):
    """Return a Matplotlib colormap, resolving custom names and strings."""
    if isinstance(name_or_obj, str):
        if name_or_obj == "Threat Zones (7)":
            return get_threat_zone_cmap()
        elif name_or_obj == "Dose Field (7)":
            return get_continuous_dose_cmap()
        return plt.get_cmap(name_or_obj).copy()
    return name_or_obj.copy() if hasattr(name_or_obj, 'copy') else name_or_obj


def _pixmap_to_rgb_array(pixmap: QPixmap) -> np.ndarray:
    """Convert a QPixmap to an (H, W, 3) uint8 NumPy array (RGB).

    The returned array owns its own memory  so it
    does not depend on the QImage staying alive. 
    """
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = image.width(), image.height()
    if w == 0 or h == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    bpl = image.bytesPerLine()  # may include row padding
    # -- Critical: copy the raw bytes OUT of Qt's buffer immediately --
    # image.constBits() returns a sip.voidptr that becomes dangling if
    # `image` is garbage-collected.  Using image.bits().asstring()
    # copies the data into a Python bytes object, decoupling lifetimes.
    ptr = image.constBits()
    ptr.setsize(bpl * h)
    raw = bytes(ptr)  # copy Qt's buffer into a Python bytes object
    arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, bpl))
    # Strip any row padding (bpl may be > w*3) and return a contiguous copy
    return arr[:, :w * 3].reshape((h, w, 3)).copy()


def _safe_load_raster_image(file_path: str) -> QPixmap | None:
    """Load a image (PNG/JPG/BMP, etc.) via Pillow and downscale
    before converting to QPixmap, to avoid large intermediate buffers.

    The PIL image is downscaled before extracting raw bytes, and the
    QImage is copyed to decouple it from the Python before returning. 
    Both data and pil_img are kept
    alive until after the copy to prevent an access-violation on
    Windows when the GC races the copy.
    """
    try:
        Image = _get_pil()
        pil_img = Image.open(file_path)

        w, h = pil_img.size
        limit = MAX_IMAGE_DIMENSION
        if w > limit or h > limit:
            pil_img.thumbnail((limit, limit), Image.LANCZOS)

        pil_img = pil_img.convert("RGB")
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(
            data,
            pil_img.width,
            pil_img.height,
            pil_img.width * 3,
            QImage.Format.Format_RGB888,
        )
        # .copy() decouples the QImage from the Python `data` buffer.
        result = QPixmap.fromImage(qimg.copy())
        del qimg, data, pil_img
        return result
    except Exception:
        return None



def _try_load_heic(file_path: str):
    """Attempt to open a HEIC / HEIF file via pillow-heif.

    Returns a QPixmap on success or None if the library is unavailable
    or the file cannot be read.

    Large images are downscaled before creating the QImage to
    avoid a memory crash on Windows that can
    occur when converting multi-megapixel HEIC photos to raw RGBA bytes.
    """
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        return None

    try:
        Image = _get_pil()
        pil_img = Image.open(file_path)

        # Downscale in PIL *before* creating the expensive raw-bytes buffer.
        w, h = pil_img.size
        limit = MAX_IMAGE_DIMENSION
        if w > limit or h > limit:
            pil_img.thumbnail((limit, limit), Image.LANCZOS)

        # Convert to RGB (not RGBA) — saves 25 % memory.
        pil_img = pil_img.convert("RGB")
        data = pil_img.tobytes("raw", "RGB")
        qimg = QImage(data, pil_img.width, pil_img.height,
                      pil_img.width * 3,
                      QImage.Format.Format_RGB888)
        # .copy() decouples the QImage from the Python `data` buffer.
        result = QPixmap.fromImage(qimg.copy())
        del qimg, data, pil_img
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# AnnotatedImageWidget – custom drawing canvas for the Image tab
# ---------------------------------------------------------------------------

class AnnotatedImageWidget(QWidget):
    """Widget that displays a QPixmap and supports interactive annotations.

    Supported annotation modes (set via ``set_mode``):
      * MODE_MARKER   – numbered point markers (with fill toggle)
      * MODE_LINE     – straight line segments (click start, click end)
      * MODE_FREEDRAW – free-hand polylines (press-drag-release)
      * MODE_TEXT     – click to place a text label
      * MODE_SCALE    – draw a line, then enter known distance (ImageJ-style)
      * MODE_ICON     – click to place a loaded icon image

    All annotation coordinates are stored in image space so they
    survive widget resizing and can later be mapped to Matplotlib data
    coordinates for overlay integration.
    """

    MODE_NONE = "none"
    MODE_MARKER = "marker"
    MODE_LINE = "line"
    MODE_FREEDRAW = "freedraw"
    MODE_TEXT = "text"
    MODE_SCALE = "scale"
    MODE_ICON = "icon"
    MODE_CROP = "crop"

    __slots__ = (
        '_pixmap', '_mode',
        '_markers', '_lines', '_free_draw_strokes', '_texts', '_icons',
        '_undo_stack',
        '_scale_line_start', '_scale_line', '_pixels_per_unit', '_scale_unit',
        '_icon_to_place',
        '_line_start', '_current_stroke',
        '_marker_color', '_line_color', '_freedraw_color', '_text_color',
        '_line_width', '_marker_filled',
        '_crop_origin', '_crop_rect_preview', '_crop_callback',
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._mode = self.MODE_NONE

        # Annotation storage (image coordinates)
        self._markers: list[QPointF] = []
        self._lines: list[tuple[QPointF, QPointF]] = []
        self._free_draw_strokes: list[list[QPointF]] = []
        self._texts: list[tuple[QPointF, str]] = []       # (pos, text)
        self._icons: list[tuple[QPointF, QPixmap]] = []   # (centre, pixmap)

        # Undo stack: each entry is (type_str, index)
        self._undo_stack: list[tuple[str, int]] = []

        # Scale calibration state
        self._scale_line_start = None
        self._scale_line: tuple[QPointF, QPointF] | None = None
        self._pixels_per_unit: float | None = None
        self._scale_unit: str = "cm"

        # Icon to place (loaded externally)
        self._icon_to_place: QPixmap | None = None

        # Transient state for in-progress drawing
        self._line_start = None
        self._current_stroke = None

        # Crop ROI state
        self._crop_origin = None       # QPointF in image coords (press start)
        self._crop_rect_preview = None  # QRect in image coords (live preview)
        self._crop_callback = None     # callable(QRect) set by MainWindow

        # Style settings (read live from MainWindow controls)
        self._marker_color = QColor(255, 50, 50)
        self._line_color = QColor(0, 200, 255)
        self._freedraw_color = QColor(255, 165, 0)
        self._text_color = QColor(255, 255, 255)
        self._line_width = 2
        self._marker_filled = True  # fill toggle

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
        self.setMouseTracking(True)

    # --- Public--------------------------------------------------------

    def set_pixmap(self, pixmap: QPixmap):
        """Set the background image (does not clear annotations)."""
        self._pixmap = pixmap
        self.update()

    def pixmap(self):
        return self._pixmap

    def set_mode(self, mode: str):
        """Change the current interaction mode."""
        # Clean up crop state when leaving crop mode
        if self._mode == self.MODE_CROP and mode != self.MODE_CROP:
            self._crop_origin = None
            # Note: _crop_rect_preview is left intact so MainWindow can
            # read it in _apply_visual_crop.  MainWindow clears it.
        self._mode = mode
        self._line_start = None
        self._current_stroke = None
        self._scale_line_start = None

    def set_colors(self, marker=None, line=None, freedraw=None, text=None):
        """Update drawing colours from external controls."""
        if marker:
            self._marker_color = marker
        if line:
            self._line_color = line
        if freedraw:
            self._freedraw_color = freedraw
        if text:
            self._text_color = text
        self.update()

    def set_line_width(self, w: int):
        self._line_width = w
        self.update()  # live update on canvas

    def set_marker_filled(self, filled: bool):
        self._marker_filled = filled
        self.update()

    def set_icon_to_place(self, pixmap: QPixmap):
        self._icon_to_place = pixmap

    def clear_annotations(self):
        """Remove all annotations."""
        self._markers.clear()
        self._lines.clear()
        self._free_draw_strokes.clear()
        self._texts.clear()
        self._icons.clear()
        self._undo_stack.clear()
        self._line_start = None
        self._current_stroke = None
        self._scale_line_start = None
        self._scale_line = None
        self.update()

    def get_markers(self) -> list[QPointF]:
        return list(self._markers)

    def get_lines(self) -> list[tuple[QPointF, QPointF]]:
        return list(self._lines)

    def get_free_draw(self) -> list[list[QPointF]]:
        return [list(s) for s in self._free_draw_strokes]

    def get_texts(self) -> list[tuple[QPointF, str]]:
        return list(self._texts)

    def get_icons(self) -> list[tuple[QPointF, QPixmap]]:
        return list(self._icons)

    def has_annotations(self) -> bool:
        return bool(self._markers or self._lines or self._free_draw_strokes
                    or self._texts or self._icons)

    def annotation_count(self) -> dict:
        return {
            'markers': len(self._markers),
            'lines': len(self._lines),
            'strokes': len(self._free_draw_strokes),
            'texts': len(self._texts),
            'icons': len(self._icons),
        }

    # --- Scale calibration & distance measurement --------------------------

    def get_pixels_per_unit(self) -> float | None:
        return self._pixels_per_unit

    def get_scale_unit(self) -> str:
        return self._scale_unit

    def get_line_distances(self) -> list[float]:
        """Return the real-world distance of each line segment.

        Returns an empty list if the scale has not been calibrated.
        """
        if self._pixels_per_unit is None or self._pixels_per_unit == 0:
            return []
        distances = []
        for start, end in self._lines:
            px = math.hypot(end.x() - start.x(), end.y() - start.y())
            distances.append(px / self._pixels_per_unit)
        return distances

    def get_total_line_distance(self) -> float | None:
        """Return the sum of all line segment distances in real-world units.

        Returns None if scale has not been calibrated, or 0.0 if no lines.
        """
        dists = self.get_line_distances()
        if not dists and self._pixels_per_unit is None:
            return None
        return sum(dists)

    # --- Undo (Ctrl+Z aware) -----------------------------------------------

    def undo_last(self):
        """Remove the most recently added annotation using the undo stack."""
        if not self._undo_stack:
            return
        kind, idx = self._undo_stack.pop()
        try:
            if kind == "marker" and idx < len(self._markers):
                self._markers.pop(idx)
            elif kind == "line" and idx < len(self._lines):
                self._lines.pop(idx)
            elif kind == "stroke" and idx < len(self._free_draw_strokes):
                self._free_draw_strokes.pop(idx)
            elif kind == "text" and idx < len(self._texts):
                self._texts.pop(idx)
            elif kind == "icon" and idx < len(self._icons):
                self._icons.pop(idx)
        except (IndexError, TypeError):
            pass
        self.update()

    def _push_undo(self, kind: str, idx: int):
        self._undo_stack.append((kind, idx))

    # --- Burn annotations onto pixmap --------------------------------------

    def burn_annotations_to_pixmap(self,
                                   marker_color=None,
                                   line_color=None,
                                   freedraw_color=None,
                                   text_color=None,
                                   line_width=2) -> QPixmap:
        """Render all annotations onto a *copy* of the pixmap."""
        if self._pixmap is None:
            return QPixmap()

        result = self._pixmap.copy()
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        m_col = marker_color or self._marker_color
        l_col = line_color or self._line_color
        f_col = freedraw_color or self._freedraw_color
        t_col = text_color or self._text_color

        radius = self._marker_radius_img()
        font_size = max(8, int(radius * 0.9))

        # Markers
        marker_pen = QPen(m_col, line_width)
        marker_font = QFont("Arial", font_size, QFont.Weight.Bold)
        painter.setFont(marker_font)
        for idx, pt in enumerate(self._markers, start=1):
            painter.setPen(marker_pen)
            if self._marker_filled:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(pt, radius, radius)
            painter.setPen(QColor(0, 0, 0) if self._marker_filled else m_col)
            text_rect = QRectF(pt.x() - radius, pt.y() - radius,
                               radius * 2, radius * 2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(idx))

        # Line segments (with distance labels when scale is set)
        dist_font = QFont("Arial", max(8, font_size - 2), QFont.Weight.Bold)
        for start, end in self._lines:
            painter.setPen(QPen(l_col, line_width, Qt.PenStyle.SolidLine))
            painter.drawLine(start, end)
            if self._pixels_per_unit is not None:
                px_dist = math.hypot(end.x() - start.x(), end.y() - start.y())
                real_dist = px_dist / self._pixels_per_unit
                mid = QPointF((start.x() + end.x()) / 2,
                              (start.y() + end.y()) / 2 - radius)
                painter.setFont(dist_font)
                painter.setPen(l_col)
                painter.drawText(mid, f"{real_dist:.2f} {self._scale_unit}")

        # Free-draw strokes
        painter.setPen(QPen(f_col, line_width))
        for stroke in self._free_draw_strokes:
            for i in range(len(stroke) - 1):
                painter.drawLine(stroke[i], stroke[i + 1])

        # Text labels
        text_font = QFont("Arial", max(10, font_size), QFont.Weight.Bold)
        painter.setFont(text_font)
        painter.setPen(t_col)
        for pt, txt in self._texts:
            painter.drawText(pt, txt)

        # Icons
        for pt, icon_pm in self._icons:
            target = QRectF(pt.x() - icon_pm.width() / 2,
                            pt.y() - icon_pm.height() / 2,
                            icon_pm.width(), icon_pm.height())
            painter.drawPixmap(target.toRect(), icon_pm)

        # Scale line (always red)
        if self._scale_line is not None:
            painter.setPen(QPen(QColor(255, 50, 50), max(2, line_width), Qt.PenStyle.DashLine))
            painter.drawLine(self._scale_line[0], self._scale_line[1])

        painter.end()
        return result

    def export_annotations_layer(self) -> QPixmap | None:
        """Export annotations as a transparent PNG layer (no background)."""
        if self._pixmap is None:
            return None
        layer = QPixmap(self._pixmap.size())
        layer.fill(QColor(0, 0, 0, 0))  # fully transparent
        painter = QPainter(layer)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        lw = self._line_width
        radius = self._marker_radius_img()
        font_size = max(8, int(radius * 0.9))

        # Markers
        marker_pen = QPen(self._marker_color, lw)
        painter.setFont(QFont("Arial", font_size, QFont.Weight.Bold))
        for idx, pt in enumerate(self._markers, start=1):
            painter.setPen(marker_pen)
            if self._marker_filled:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(pt, radius, radius)
            painter.setPen(QColor(0, 0, 0) if self._marker_filled else self._marker_color)
            text_rect = QRectF(pt.x() - radius, pt.y() - radius, radius * 2, radius * 2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(idx))

        # Lines (with distance labels when scale is set)
        dist_font = QFont("Arial", max(8, font_size - 2), QFont.Weight.Bold)
        for s, e in self._lines:
            painter.setPen(QPen(self._line_color, lw))
            painter.drawLine(s, e)
            if self._pixels_per_unit is not None:
                px_dist = math.hypot(e.x() - s.x(), e.y() - s.y())
                real_dist = px_dist / self._pixels_per_unit
                mid = QPointF((s.x() + e.x()) / 2,
                              (s.y() + e.y()) / 2 - radius)
                painter.setFont(dist_font)
                painter.setPen(self._line_color)
                painter.drawText(mid, f"{real_dist:.2f} {self._scale_unit}")

        # Free-draw
        painter.setPen(QPen(self._freedraw_color, lw))
        for stroke in self._free_draw_strokes:
            for i in range(len(stroke) - 1):
                painter.drawLine(stroke[i], stroke[i + 1])

        # Text
        painter.setFont(QFont("Arial", max(10, font_size), QFont.Weight.Bold))
        painter.setPen(self._text_color)
        for pt, txt in self._texts:
            painter.drawText(pt, txt)

        # Icons
        for pt, icon_pm in self._icons:
            target = QRectF(pt.x() - icon_pm.width() / 2,
                            pt.y() - icon_pm.height() / 2,
                            icon_pm.width(), icon_pm.height())
            painter.drawPixmap(target.toRect(), icon_pm)

        painter.end()
        return layer

    def import_annotations_layer(self, layer_pixmap: 'QPixmap') -> bool:
        """Composite a transparent annotation layer onto the current image.

        *layer_pixmap* should be a RGBA PNG (the same size as the original
        image, as produced by export_annotations_layer).  If it has a
        different size it is scaled to fit.

        Returns True on success.
        """
        if self._pixmap is None:
            return False
        if layer_pixmap.isNull():
            return False

        # Scale the layer to match the current image if sizes differ
        if layer_pixmap.size() != self._pixmap.size():
            layer_pixmap = layer_pixmap.scaled(
                self._pixmap.size(),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

        # Paint the layer on top of the current pixmap
        painter = QPainter(self._pixmap)
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_SourceOver
        )
        painter.drawPixmap(0, 0, layer_pixmap)
        painter.end()
        self.update()
        return True

    # --- Coordinate mapping ------------------------------------------------

    def _display_rect(self) -> QRectF:
        if self._pixmap is None:
            return QRectF(0, 0, self.width(), self.height())
        pw, ph = self._pixmap.width(), self._pixmap.height()
        ww, wh = self.width(), self.height()
        scale = min(ww / pw, wh / ph)
        dw, dh = pw * scale, ph * scale
        x0 = (ww - dw) / 2
        y0 = (wh - dh) / 2
        return QRectF(x0, y0, dw, dh)

    def _widget_to_image(self, pos):
        if self._pixmap is None:
            return None
        r = self._display_rect()
        if not r.contains(QPointF(pos.x(), pos.y())):
            return None
        ix = (pos.x() - r.x()) / r.width() * self._pixmap.width()
        iy = (pos.y() - r.y()) / r.height() * self._pixmap.height()
        return QPointF(ix, iy)

    def _image_to_widget(self, pt: QPointF) -> QPointF:
        r = self._display_rect()
        wx = r.x() + pt.x() / self._pixmap.width() * r.width()
        wy = r.y() + pt.y() / self._pixmap.height() * r.height()
        return QPointF(wx, wy)

    def _widget_scale(self) -> float:
        """Current widget-to-image scale factor."""
        if self._pixmap is None:
            return 1.0
        r = self._display_rect()
        return self._pixmap.width() / r.width() if r.width() > 0 else 1.0

    def _marker_radius_img(self) -> int:
        """Marker radius in *image* pixels — consistent across paint & burn.

        Uses a fixed 8-pixel widget-space radius converted to image space
        via the current display scale, but clamps to a sensible range so
        markers are always visible and never absurdly large.
        """
        ws = self._widget_scale()
        return max(6, min(int(8 * ws), 40))

    # --- Mouse events ------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or self._pixmap is None:
            return
        img_pt = self._widget_to_image(event.pos())
        if img_pt is None:
            return

        if self._mode == self.MODE_MARKER:
            self._markers.append(img_pt)
            self._push_undo("marker", len(self._markers) - 1)
            self.update()
        elif self._mode == self.MODE_LINE:
            if self._line_start is None:
                self._line_start = img_pt
            else:
                self._lines.append((self._line_start, img_pt))
                self._push_undo("line", len(self._lines) - 1)
                self._line_start = None
                self.update()
        elif self._mode == self.MODE_FREEDRAW:
            self._current_stroke = [img_pt]
        elif self._mode == self.MODE_TEXT:
            text, ok = QInputDialog.getText(self, "Text Annotation", "Enter label text:")
            if ok and text.strip():
                self._texts.append((img_pt, text.strip()))
                self._push_undo("text", len(self._texts) - 1)
                self.update()
        elif self._mode == self.MODE_SCALE:
            if self._scale_line_start is None:
                self._scale_line_start = img_pt
            else:
                self._scale_line = (self._scale_line_start, img_pt)
                self._scale_line_start = None
                self._prompt_scale_distance()
                self.update()
        elif self._mode == self.MODE_ICON:
            if self._icon_to_place is not None:
                self._icons.append((img_pt, self._icon_to_place))
                self._push_undo("icon", len(self._icons) - 1)
                self.update()
        elif self._mode == self.MODE_CROP:
            self._crop_origin = img_pt
            self._crop_rect_preview = None
            self.update()

    def mouseMoveEvent(self, event):
        if self._mode == self.MODE_FREEDRAW and self._current_stroke is not None:
            img_pt = self._widget_to_image(event.pos())
            if img_pt is not None:
                self._current_stroke.append(img_pt)
                self.update()
        elif self._mode == self.MODE_CROP and self._crop_origin is not None:
            img_pt = self._widget_to_image(event.pos())
            if img_pt is not None and self._pixmap is not None:
                # Build a rect clamped to image bounds
                x1 = max(0, min(self._crop_origin.x(), img_pt.x()))
                y1 = max(0, min(self._crop_origin.y(), img_pt.y()))
                x2 = min(self._pixmap.width(), max(self._crop_origin.x(), img_pt.x()))
                y2 = min(self._pixmap.height(), max(self._crop_origin.y(), img_pt.y()))
                self._crop_rect_preview = QRect(
                    int(x1), int(y1), int(x2 - x1), int(y2 - y1)
                )
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._mode == self.MODE_FREEDRAW and self._current_stroke is not None:
                if len(self._current_stroke) > 1:
                    self._free_draw_strokes.append(self._current_stroke)
                    self._push_undo("stroke", len(self._free_draw_strokes) - 1)
                self._current_stroke = None
                self.update()
            elif self._mode == self.MODE_CROP and self._crop_origin is not None:
                self._crop_origin = None
                # Only accept if the rectangle has meaningful size
                if (self._crop_rect_preview is not None
                        and self._crop_rect_preview.width() > 10
                        and self._crop_rect_preview.height() > 10):
                    # Notify MainWindow via callback
                    if self._crop_callback is not None:
                        self._crop_callback(self._crop_rect_preview)
                else:
                    # Too small — treat as accidental click, clear preview
                    self._crop_rect_preview = None
                    self.update()

    # --- Scale helper ------------------------------------------------------

    def _prompt_scale_distance(self):
        """After drawing a scale line, prompt for the known distance (ImageJ-style)."""
        if self._scale_line is None:
            return
        p1, p2 = self._scale_line
        pixel_dist = math.hypot(p2.x() - p1.x(), p2.y() - p1.y())
        if pixel_dist < 1:
            return

        text, ok = QInputDialog.getText(
            self, "Set Scale",
            f"Line length: {pixel_dist:.1f} pixels\n\n"
            f"Enter the known distance and unit (e.g. '100 cm' or '1.5 m'):"
        )
        if ok and text.strip():
            parts = text.strip().split()
            try:
                known_dist = float(parts[0])
                unit = parts[1] if len(parts) > 1 else "cm"
                self._pixels_per_unit = pixel_dist / known_dist
                self._scale_unit = unit
            except (ValueError, IndexError):
                QMessageBox.warning(self, "Invalid input",
                                    "Enter a number followed by a unit, e.g. '100 cm'.")

    # --- Paint -------------------------------------------------------------

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        if self._pixmap is not None:
            r = self._display_rect()
            painter.drawPixmap(r.toRect(), self._pixmap)
        else:
            painter.setPen(QColor(128, 128, 128))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Load an image or PDF to begin")
            painter.end()
            return

        lw = self._line_width

        # Markers (numbered circles)
        radius_img = self._marker_radius_img()
        w_scale = self._widget_scale()
        radius = max(4, int(radius_img / w_scale)) if w_scale > 0 else 8
        font_size_w = max(6, int(radius * 0.9))
        marker_pen = QPen(self._marker_color, lw)
        marker_font = QFont("Arial", font_size_w, QFont.Weight.Bold)
        painter.setFont(marker_font)
        for idx, pt in enumerate(self._markers, start=1):
            wpt = self._image_to_widget(pt)
            painter.setPen(marker_pen)
            if self._marker_filled:
                painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(wpt, radius, radius)
            painter.setPen(QColor(0, 0, 0) if self._marker_filled else self._marker_color)
            text_rect = QRectF(wpt.x() - radius, wpt.y() - radius,
                               radius * 2, radius * 2)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, str(idx))

        # Line segments (with distance labels when scale is set)
        painter.setPen(QPen(self._line_color, lw, Qt.PenStyle.SolidLine))
        dist_font = QFont("Arial", 9, QFont.Weight.Bold)
        for start, end in self._lines:
            ws = self._image_to_widget(start)
            we = self._image_to_widget(end)
            painter.setPen(QPen(self._line_color, lw, Qt.PenStyle.SolidLine))
            painter.drawLine(ws, we)
            # Show distance label at midpoint if scale is calibrated
            if self._pixels_per_unit is not None:
                px_dist = math.hypot(end.x() - start.x(), end.y() - start.y())
                real_dist = px_dist / self._pixels_per_unit
                mid = QPointF((ws.x() + we.x()) / 2,
                              (ws.y() + we.y()) / 2 - radius)
                painter.setFont(dist_font)
                painter.setPen(self._line_color)
                painter.drawText(mid, f"{real_dist:.2f} {self._scale_unit}")

        # In-progress line start indicator
        if self._line_start is not None:
            painter.setPen(QPen(self._line_color.lighter(150), lw, Qt.PenStyle.DashLine))
            ws = self._image_to_widget(self._line_start)
            painter.drawEllipse(ws, 4, 4)

        # Free-draw strokes
        painter.setPen(QPen(self._freedraw_color, lw))
        for stroke in self._free_draw_strokes:
            self._paint_stroke(painter, stroke)
        if self._current_stroke and len(self._current_stroke) > 1:
            self._paint_stroke(painter, self._current_stroke)

        # Text labels
        text_font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(text_font)
        painter.setPen(self._text_color)
        for pt, txt in self._texts:
            wpt = self._image_to_widget(pt)
            painter.drawText(wpt, txt)

        # Icon overlays
        for pt, icon_pm in self._icons:
            wpt = self._image_to_widget(pt)
            ws = self._widget_scale()
            disp_w = icon_pm.width() / ws if ws > 0 else icon_pm.width()
            disp_h = icon_pm.height() / ws if ws > 0 else icon_pm.height()
            target = QRectF(wpt.x() - disp_w / 2, wpt.y() - disp_h / 2, disp_w, disp_h)
            painter.drawPixmap(target.toRect(), icon_pm)

        # Scale line (always red)
        if self._scale_line is not None:
            painter.setPen(QPen(QColor(255, 50, 50), max(2, lw), Qt.PenStyle.DashLine))
            ws = self._image_to_widget(self._scale_line[0])
            we = self._image_to_widget(self._scale_line[1])
            painter.drawLine(ws, we)
            # Show pixel/unit info
            if self._pixels_per_unit is not None:
                mid = QPointF((ws.x() + we.x()) / 2, (ws.y() + we.y()) / 2 - 10)
                ppu = self._pixels_per_unit
                painter.setPen(QColor(255, 50, 50))
                painter.setFont(QFont("Arial", 9))
                painter.drawText(mid, f"{ppu:.1f} px/{self._scale_unit}")

        # Scale-in-progress start indicator (red)
        if self._scale_line_start is not None:
            painter.setPen(QPen(QColor(255, 50, 50, 180), 2, Qt.PenStyle.DotLine))
            ws = self._image_to_widget(self._scale_line_start)
            painter.drawEllipse(ws, 5, 5)

        # Crop ROI preview (dimmed outside, bright inside)
        if self._crop_rect_preview is not None and self._pixmap is not None:
            # Map image-space crop rect corners to widget coords
            tl = self._image_to_widget(
                QPointF(self._crop_rect_preview.x(),
                        self._crop_rect_preview.y()))
            br = self._image_to_widget(
                QPointF(self._crop_rect_preview.x() + self._crop_rect_preview.width(),
                        self._crop_rect_preview.y() + self._crop_rect_preview.height()))
            crop_w = QRectF(tl.x(), tl.y(), br.x() - tl.x(), br.y() - tl.y())

            # Dim everything outside the selection
            dim = QColor(0, 0, 0, 120)
            disp = self._display_rect()
            # Top strip
            painter.fillRect(QRectF(disp.x(), disp.y(),
                                    disp.width(), crop_w.y() - disp.y()), dim)
            # Bottom strip
            painter.fillRect(QRectF(disp.x(), crop_w.bottom(),
                                    disp.width(), disp.bottom() - crop_w.bottom()), dim)
            # Left strip (between top and bottom)
            painter.fillRect(QRectF(disp.x(), crop_w.y(),
                                    crop_w.x() - disp.x(), crop_w.height()), dim)
            # Right strip
            painter.fillRect(QRectF(crop_w.right(), crop_w.y(),
                                    disp.right() - crop_w.right(), crop_w.height()), dim)

            # Draw the selection border
            painter.setPen(QPen(QColor(0, 200, 255), 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(crop_w)

            # Size label
            painter.setPen(QColor(0, 200, 255))
            painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
            size_txt = (f"{self._crop_rect_preview.width()} \u00d7 "
                        f"{self._crop_rect_preview.height()} px")
            painter.drawText(crop_w.bottomRight() + QPointF(-80, 16), size_txt)

        painter.end()

    def _paint_stroke(self, painter: QPainter, stroke: list[QPointF]):
        for i in range(len(stroke) - 1):
            painter.drawLine(self._image_to_widget(stroke[i]),
                             self._image_to_widget(stroke[i + 1]))


# ---------------------------------------------------------------------------
# DraggableCanvas – Matplotlib canvas supporting overlay drag
# ---------------------------------------------------------------------------

class DraggableCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas that supports dragging the intensity overlay
    (heatmap / contours) relative to the background image."""

    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)
        self.parent_window = parent
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.dragging = False
        self.last_mouse_pos = None
        self.cbar = None
        self.intensity_offset_x = 0
        self.intensity_offset_y = 0
        self.original_extent = None

    # --- Mouse interaction for dragging ------------------------------------

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True

            if self.ax.get_xlim() and self.ax.get_ylim():
                x_pixel = event.pos().x()
                y_pixel = self.figure.bbox.height - event.pos().y()  # Flip Y

                try:
                    x_data, y_data = self.ax.transData.inverted().transform([x_pixel, y_pixel])
                    self.last_mouse_pos = (x_data, y_data)
                except Exception:
                    self.last_mouse_pos = None
                    self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos:
            try:
                x_pixel = event.pos().x()
                y_pixel = self.figure.bbox.height - event.pos().y()  # Flip Y
                x_data, y_data = self.ax.transData.inverted().transform([x_pixel, y_pixel])

                dx = x_data - self.last_mouse_pos[0]
                dy = y_data - self.last_mouse_pos[1]

                # Update offset
                self.intensity_offset_x += dx
                self.intensity_offset_y += dy

                moved_something = False

                for img in self.ax.images:
                    arr = img.get_array()
                    if arr is None:
                        continue
                    if hasattr(arr, 'dtype') and arr.dtype == np.uint8:
                        # Skip the base image drawn from pixmap
                        continue

                    extent = img.get_extent()
                    new_extent = (extent[0] + dx, extent[1] + dx, extent[2] + dy, extent[3] + dy)
                    img.set_extent(new_extent)
                    moved_something = True

                # Handle collection objects (contours)
                for collection in self.ax.collections:
                    for path in collection.get_paths():
                        vertices = path.vertices
                        vertices[:, 0] += dx
                        vertices[:, 1] += dy
                    moved_something = True

                if moved_something:
                    self.last_mouse_pos = (x_data, y_data)
                    self.draw()

            except Exception as e:
                print(f"Drag error: {e}")
                self.dragging = False

    def mouseReleaseEvent(self, event):
        """After drag, re-apply highlights so contour lines + labels
        stay consistent with the moved overlay."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None

            # Re-apply highlights after drag if any were set 
            try:
                if (
                    hasattr(self, "parent_window")
                    and self.parent_window is not None
                    and hasattr(self.parent_window, "last_highlight_values")
                    and self.parent_window.last_highlight_values
                ):
                    self.parent_window.apply_highlights()
            except Exception as e:
                print(f"Error reapplying highlights after drag: {e}")

    # --- Helper: prepare axes with background image ------------------------

    def _prepare_axes(self, base_img: QPixmap):
        """Clear figure, create fresh axes, draw background image.

        Returns (extent, width, height) for the background.
        """
        self.figure.clf()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.ax = self.figure.add_subplot(111)

        arr_rgb = _pixmap_to_rgb_array(base_img)
        height, width = arr_rgb.shape[:2]
        extent = [-width / 2, width / 2, -height / 2, height / 2]
        self.original_extent = extent
        self.ax.imshow(arr_rgb, aspect='equal', extent=extent, origin='upper')
        return extent, width, height

    def _intensity_extent(self, extent):
        """Return the overlay extent shifted by the current drag offset."""
        return [
            extent[0] + self.intensity_offset_x,
            extent[1] + self.intensity_offset_x,
            extent[2] + self.intensity_offset_y,
            extent[3] + self.intensity_offset_y,
        ]

    # --- Drawing methods ---------------------------------------------------

    def draw_heatmap(
        self,
        base_img: QPixmap,
        intensity_array,
        alpha=0.6,
        cmap='jet',
        interpolation='bilinear',
        vmin=None,
        vmax=None,
        units='',
        scale_x=1.0,
        scale_y=1.0,
        distance_units='pixels'
    ):
        """Draw (or redraw) a heatmap overlay on the background image."""
        extent, width, height = self._prepare_axes(base_img)
        intensity_extent = self._intensity_extent(extent)

        cmap_obj = _resolve_cmap(cmap)
        cmap_obj.set_over(cmap_obj(1.0))
        top_color = list(cmap_obj(1.0))
        top_color[3] = 1.0
        cmap_obj.set_over(tuple(top_color))
        cmap_obj.set_under((0, 0, 0, 0))

        im = self.ax.imshow(
            intensity_array,
            cmap=cmap_obj,
            alpha=alpha,
            interpolation=interpolation,
            extent=intensity_extent,
            origin='upper',
            vmin=vmin,
            vmax=vmax
        )

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self._apply_axis_scaling(scale_x, scale_y, distance_units)

        self.cbar = self.figure.colorbar(im, ax=self.ax, orientation='vertical', pad=0.05, extend='max')
        cbar_label = f'Intensity ({units})' if units else 'Intensity'
        self.cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=15, fontweight='bold')
        self.draw()

    def _apply_axis_scaling(self, scale_x, scale_y, distance_units):
        """Scale axis tick labels and set axis labels."""
        if scale_x != 1.0 or scale_y != 1.0:
            x_ticks = self.ax.get_xticks()
            y_ticks = self.ax.get_yticks()

            x_labels = [f'{tick * scale_x:.0f}' for tick in x_ticks]
            y_labels = [f'{tick * scale_y:.0f}' for tick in y_ticks]

            self.ax.set_xticklabels(x_labels)
            self.ax.set_yticklabels(y_labels)

        self.ax.set_xlabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')
        self.ax.set_ylabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')

    def draw_contours(
        self,
        base_img: QPixmap,
        intensity_array,
        alpha=0.6,
        cmap='jet',
        levels=6,
        interpolation='bilinear',
        units='',
        extend='max',
        scale_x=1.0,
        scale_y=1.0,
        distance_units='pixels'
    ):
        """Draw (or redraw) filled contour overlay on the background image."""
        extent, width, height = self._prepare_axes(base_img)

        rows, cols = intensity_array.shape

        ie = self._intensity_extent(extent)
        X = np.linspace(ie[0], ie[1], cols)
        Y = np.linspace(ie[2], ie[3], rows)
        xx, yy = np.meshgrid(X, Y)

        if isinstance(levels, int):
            valid_data = intensity_array[~np.isnan(intensity_array)]
            if valid_data.size == 0:
                return
            vmin, vmax = np.min(valid_data), np.max(valid_data)
            levels = np.linspace(vmin, vmax, levels)

        intensity_array = np.flipud(intensity_array)

        cmap_obj = _resolve_cmap(cmap)
        top_color = list(cmap_obj(1.0))
        top_color[3] = 1.0
        cmap_obj.set_over(tuple(top_color))
        cmap_obj.set_under((0, 0, 0, 0))

        cs = self.ax.contourf(xx, yy, intensity_array, levels=levels, cmap=cmap_obj, alpha=alpha, extend=extend)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])
        self._apply_axis_scaling(scale_x, scale_y, distance_units)

        self.cbar = self.figure.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05)
        cbar_label = f'Intensity ({units})' if units else 'Intensity'
        self.cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=14, fontweight='bold')
        self.cbar.ax.tick_params(labelsize=13)
        self.draw()

    def reset_intensity_position(self):
        self.intensity_offset_x = 0
        self.intensity_offset_y = 0

    def closeEvent(self, event):
        try:
            if self.cbar is not None:
                try:
                    self.cbar.remove()
                except Exception:
                    pass
            self.figure.clear()
            plt.close(self.figure)
        except Exception:
            pass
        gc.collect()  # Cleanup on close only
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# CropDialog
# ---------------------------------------------------------------------------

class CropDialog(QDialog):
    """Simple dialog to define a rectangular crop area on the original image."""

    def __init__(self, parent=None, max_width=800, max_height=600):
        super().__init__(parent)
        self.setWindowTitle("Set Crop Area on Original Image")
        layout = QFormLayout(self)

        self.x_spin = QSpinBox()
        self.x_spin.setRange(0, max_width)
        self.y_spin = QSpinBox()
        self.y_spin.setRange(0, max_height)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(1, max_width)
        self.w_spin.setValue(max_width)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(1, max_height)
        self.h_spin.setValue(max_height)

        layout.addRow("X:", self.x_spin)
        layout.addRow("Y:", self.y_spin)
        layout.addRow("Width:", self.w_spin)
        layout.addRow("Height:", self.h_spin)

        buttons = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        buttons.addWidget(self.ok_button)
        buttons.addWidget(self.cancel_button)
        layout.addRow(buttons)

    def get_values(self):
        return (self.x_spin.value(), self.y_spin.value(), self.w_spin.value(), self.h_spin.value())


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        # Attributes
        self.original_pixmap = None   # Single authoritative full-resolution image
        self.current_pixmap = None    # View-only copy (rotated / cropped)
        self.crop_rect = None
        self.intensity_data = None
        self.alpha = 0.6
        self.cmap = "jet"
        self._current_vis_mode = "heatmap"
        self.cached_intensity_array = None
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None
        self.current_rotation_angle = 0
        self.source_type = None  # 'pdf', 'image', 'heic', etc.
        self.source_path = None  # filesystem path to original
        self.screen_dpi = None  # effective dpi of background for PDFs

        # Cached rotated background pixmap  (invalidated on image/crop/rotation change)
        self._bg_cache = None          # QPixmap or None
        self._bg_cache_key = None      # (_bg_version, crop_rect, rotation_angle)
        self._bg_version = 0           # bumped on each new image load

        # Highlight memory for re-application after drag
        self.last_highlight_values = []
        self.last_highlight_color_mode = None

        self.setWindowTitle("Radiation Protection Scatter Map Generator")
        self._dark_theme_on = False

        # Screen-aware initial sizing: fit within 90% of available screen
        screen = QApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = min(1400, int(avail.width() * 0.90))
            target_h = min(900, int(avail.height() * 0.90))
            self._screen_w = avail.width()
            self.resize(target_w, target_h)
            # Centre on screen
            self.move(
                avail.x() + (avail.width() - target_w) // 2,
                avail.y() + (avail.height() - target_h) // 2,
            )
        else:
            self._screen_w = 1400
            self.resize(1400, 900)

        # --- Menu bar ---
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("Save Session...", self.save_session)
        file_menu.addAction("Load Session...", self.load_session)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        view_menu = menu_bar.addMenu("View")
        self._theme_action = view_menu.addAction("Toggle Dark Theme", self._toggle_theme)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Image ---
        img_tab = QWidget()
        img_tab_layout = QHBoxLayout(img_tab)
        img_tab_left = QVBoxLayout()

        # Replace QLabel with AnnotatedImageWidget for annotation support
        self.image_canvas = AnnotatedImageWidget()
        img_tab_left.addWidget(self.image_canvas)
        img_tab_layout.addLayout(img_tab_left, stretch=3)

        img_tab_right_inner = QWidget()
        img_tab_right = QVBoxLayout(img_tab_right_inner)
        img_tab_right.setContentsMargins(4, 4, 4, 4)
        btn_load_img = QPushButton("Load Image/PDF")
        btn_load_img.clicked.connect(self.load_image)
        img_tab_right.addWidget(btn_load_img)

        img_tab_right.addWidget(QLabel("Rotation (applied to view):"))
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.valueChanged.connect(self.rotate_image_view)
        self.rotation_label = QLabel("0\u00b0")
        self.rotation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_tab_right.addWidget(self.rotation_slider)
        img_tab_right.addWidget(self.rotation_label)

        # Crop controls: Enter crop mode, then Apply / Cancel
        crop_row = QHBoxLayout()
        self._btn_start_crop = QPushButton("Select Crop Area")
        self._btn_start_crop.setToolTip("Draw a rectangle on the image to define the crop region.")
        self._btn_start_crop.clicked.connect(self.define_crop_area)
        crop_row.addWidget(self._btn_start_crop)

        self._btn_apply_crop = QPushButton("Apply Crop")
        self._btn_apply_crop.setToolTip("Crop the image to the selected rectangle.")
        self._btn_apply_crop.clicked.connect(self._apply_visual_crop)
        self._btn_apply_crop.setEnabled(False)
        self._btn_apply_crop.setVisible(False)
        crop_row.addWidget(self._btn_apply_crop)

        self._btn_cancel_crop = QPushButton("Cancel")
        self._btn_cancel_crop.setToolTip("Cancel the crop selection.")
        self._btn_cancel_crop.clicked.connect(self._cancel_visual_crop)
        self._btn_cancel_crop.setEnabled(False)
        self._btn_cancel_crop.setVisible(False)
        crop_row.addWidget(self._btn_cancel_crop)
        img_tab_right.addLayout(crop_row)

        btn_reset_img = QPushButton("Reset View and Crop")
        btn_reset_img.clicked.connect(self.reset_image)
        img_tab_right.addWidget(btn_reset_img)

        # --- Drawing-mode controls ---
        draw_group = QGroupBox("Annotation Tools")
        draw_layout = QVBoxLayout()

        self._draw_mode_group = QButtonGroup(self)
        self._draw_mode_group.setExclusive(True)

        modes = [
            ("None (navigate)", AnnotatedImageWidget.MODE_NONE),
            ("Point Markers", AnnotatedImageWidget.MODE_MARKER),
            ("Line Segments", AnnotatedImageWidget.MODE_LINE),
            ("Free Draw", AnnotatedImageWidget.MODE_FREEDRAW),
            ("Text Label", AnnotatedImageWidget.MODE_TEXT),
            ("Set Scale", AnnotatedImageWidget.MODE_SCALE),
            ("Place Icon", AnnotatedImageWidget.MODE_ICON),
        ]
        for label, mode_val in modes:
            rb = QRadioButton(label)
            rb.mode_value = mode_val
            self._draw_mode_group.addButton(rb)
            draw_layout.addWidget(rb)
            if mode_val == AnnotatedImageWidget.MODE_NONE:
                rb.setChecked(True)

        self._draw_mode_group.buttonClicked.connect(self._on_draw_mode_changed)

        # Annotation colour controls (plain-language named colours)
        colour_grid = QHBoxLayout()
        colour_grid.addWidget(QLabel("Marker:"))
        self.marker_color_combo = _build_colour_combo("Red")
        self.marker_color_combo.currentIndexChanged.connect(self._sync_annotation_colors)
        colour_grid.addWidget(self.marker_color_combo)

        colour_grid.addWidget(QLabel("Line:"))
        self.line_color_combo = _build_colour_combo("Cyan")
        self.line_color_combo.currentIndexChanged.connect(self._sync_annotation_colors)
        colour_grid.addWidget(self.line_color_combo)
        draw_layout.addLayout(colour_grid)

        colour_grid2 = QHBoxLayout()
        colour_grid2.addWidget(QLabel("Draw:"))
        self.freedraw_color_combo = _build_colour_combo("Orange")
        self.freedraw_color_combo.currentIndexChanged.connect(self._sync_annotation_colors)
        colour_grid2.addWidget(self.freedraw_color_combo)

        colour_grid2.addWidget(QLabel("Text:"))
        self.text_color_combo = _build_colour_combo("White")
        self.text_color_combo.currentIndexChanged.connect(self._sync_annotation_colors)
        colour_grid2.addWidget(self.text_color_combo)
        draw_layout.addLayout(colour_grid2)

        # Line width (live update) and marker fill toggle
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Line width:"))
        self.annot_line_width_spin = QSpinBox()
        self.annot_line_width_spin.setRange(1, 20)
        self.annot_line_width_spin.setValue(2)
        self.annot_line_width_spin.valueChanged.connect(
            lambda v: self.image_canvas.set_line_width(v))
        style_row.addWidget(self.annot_line_width_spin)

        self.marker_fill_check = QCheckBox("Filled markers")
        self.marker_fill_check.setChecked(True)
        self.marker_fill_check.toggled.connect(
            lambda on: self.image_canvas.set_marker_filled(on))
        style_row.addWidget(self.marker_fill_check)
        draw_layout.addLayout(style_row)

        # Icon import button
        btn_load_icon = QPushButton("Load Icon Image...")
        btn_load_icon.setToolTip(
            "Load a PNG or other image to use as an overlay icon.\n"
            "After loading, select 'Place Icon' mode and click on the image."
        )
        btn_load_icon.clicked.connect(self._load_icon_for_overlay)
        draw_layout.addWidget(btn_load_icon)

        # Undo / Clear / Export buttons
        btn_row_1 = QHBoxLayout()
        btn_undo_annot = QPushButton("Undo (Ctrl+Z)")
        btn_undo_annot.clicked.connect(self._undo_last_annotation)
        btn_row_1.addWidget(btn_undo_annot)

        btn_clear_annot = QPushButton("Clear All")
        btn_clear_annot.clicked.connect(self._clear_and_update_status)
        btn_row_1.addWidget(btn_clear_annot)
        draw_layout.addLayout(btn_row_1)

        btn_row_2 = QHBoxLayout()
        btn_export_layer = QPushButton("Export Layer")
        btn_export_layer.setToolTip("Save annotations as a transparent PNG (no background).")
        btn_export_layer.clicked.connect(self._export_annotation_layer)
        btn_row_2.addWidget(btn_export_layer)

        btn_import_layer = QPushButton("Import Layer")
        btn_import_layer.setToolTip("Load a previously exported annotation layer PNG\nand composite it onto the current image.")
        btn_import_layer.clicked.connect(self._import_annotation_layer)
        btn_row_2.addWidget(btn_import_layer)
        draw_layout.addLayout(btn_row_2)

        # Annotation status label
        self.annot_status_label = QLabel("No annotations")
        self.annot_status_label.setStyleSheet("color: grey; font-style: italic;")
        draw_layout.addWidget(self.annot_status_label)

        # Scale calibration readout
        self.scale_readout_label = QLabel("Scale: not set")
        self.scale_readout_label.setStyleSheet("color: grey; font-style: italic;")
        draw_layout.addWidget(self.scale_readout_label)

        # Total line distance readout
        self.total_distance_label = QLabel("")
        self.total_distance_label.setStyleSheet("color: grey; font-style: italic;")
        draw_layout.addWidget(self.total_distance_label)

        # Finalise button
        btn_finalise = QPushButton("\U0001F4CC Finalise Annotations onto Image")
        btn_finalise.setToolTip(
            "Permanently paint annotations onto the image so they\n"
            "appear in the Blended, Grid, and Preview tabs.\n"
            "This cannot be undone (but you can reload the image)."
        )
        btn_finalise.clicked.connect(self.finalise_annotations)
        draw_layout.addWidget(btn_finalise)

        draw_group.setLayout(draw_layout)
        img_tab_right.addWidget(draw_group)

        # Ctrl+Z shortcut
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self._undo_last_annotation)

        img_tab_right.addStretch(1)

        img_scroll = QScrollArea()
        img_scroll.setWidgetResizable(True)
        img_scroll.setWidget(img_tab_right_inner)
        img_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        img_tab_layout.addWidget(img_scroll, stretch=1)
        self.tab_widget.addTab(img_tab, "Image")

        # --- Tab 2: Intensity Data ---
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.table_widget.cellChanged.connect(self.invalidate_cache)

        intensity_layout = QHBoxLayout()
        intensity_left = QVBoxLayout()
        intensity_left.addWidget(self.table_widget)
        intensity_layout.addLayout(intensity_left, stretch=3)

        intensity_right = QVBoxLayout()
        btn_load_csv = QPushButton("Load CSV/Excel")
        btn_load_csv.clicked.connect(self.load_csv_excel)
        intensity_right.addWidget(btn_load_csv)

        btn_paste_clipboard = QPushButton("Paste from Clipboard (Excel)")
        btn_paste_clipboard.clicked.connect(self.paste_clipboard_data)
        intensity_right.addWidget(btn_paste_clipboard)

        btn_add_row = QPushButton("Add Row")
        btn_add_row.clicked.connect(self.add_row)
        intensity_right.addWidget(btn_add_row)

        btn_remove_row = QPushButton("Remove Row")
        btn_remove_row.clicked.connect(self.remove_row)
        intensity_right.addWidget(btn_remove_row)

        btn_add_column = QPushButton("Add Column")
        btn_add_column.clicked.connect(self.add_column)
        intensity_right.addWidget(btn_add_column)

        btn_remove_column = QPushButton("Remove Column")
        btn_remove_column.clicked.connect(self.remove_column)
        intensity_right.addWidget(btn_remove_column)

        intensity_right.addStretch(1)

        # Preview Mode and Box  (canvas created lazily on first use)
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Heatmap", "Contour Map"])
        intensity_right.addWidget(QLabel("Preview Mode:"))
        intensity_right.addWidget(self.preview_mode_combo)

        # Placeholder for lazy preview canvas
        self._preview_canvas_holder = QWidget()
        self._preview_canvas_holder.setMinimumSize(150, 150)
        self._preview_canvas_layout = QVBoxLayout(self._preview_canvas_holder)
        self._preview_canvas_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_canvas = None  # created lazily
        intensity_right.addWidget(self._preview_canvas_holder)
        self.preview_mode_combo.currentTextChanged.connect(self.update_intensity_preview)

        intensity_tab = QWidget()
        intensity_tab.setLayout(intensity_layout)
        intensity_layout.addLayout(intensity_right, stretch=1)
        self.tab_widget.addTab(intensity_tab, "Intensity Data")

        # Tab 3: Blended
        self.plot_canvas = DraggableCanvas(self)
        self.plot_canvas.setMinimumSize(400, 300)

        blend_layout = QHBoxLayout()
        blend_left = QVBoxLayout()
        blend_left.addWidget(self.plot_canvas)
        blend_layout.addLayout(blend_left, stretch=2)

        # --- Right panel wrapped in a scroll area for smaller screens ---
        blend_right_inner = QWidget()
        blend_right = QVBoxLayout(blend_right_inner)
        blend_right.setContentsMargins(4, 4, 4, 4)

        # DPI, width and height controls for export (two rows for compact layout)
        export_row1 = QHBoxLayout()
        export_row1.addWidget(QLabel("DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(72)
        self.dpi_spin.setMaximum(1200)
        self.dpi_spin.setValue(300)
        export_row1.addWidget(self.dpi_spin)
        export_row1.addWidget(QLabel("W (in):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 40.0)
        self.width_spin.setDecimals(2)
        self.width_spin.setValue(8.0)
        export_row1.addWidget(self.width_spin)
        export_row1.addWidget(QLabel("H (in):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 40.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setValue(6.0)
        export_row1.addWidget(self.height_spin)
        blend_right.addLayout(export_row1)

        # Overlay buttons (2x2 grid so they don't need extreme width)
        overlay_grid = QGridLayout()
        btn_heatmap = QPushButton("Heatmap")
        btn_heatmap.clicked.connect(self.show_heatmap)
        overlay_grid.addWidget(btn_heatmap, 0, 0)

        btn_contour = QPushButton("Contours")
        btn_contour.clicked.connect(self.show_contours)
        overlay_grid.addWidget(btn_contour, 0, 1)

        btn_save_blend = QPushButton("Save Blended")
        btn_save_blend.clicked.connect(self.save_blended_image)
        overlay_grid.addWidget(btn_save_blend, 1, 0)

        btn_reset_pos = QPushButton("Reset Position")
        btn_reset_pos.setToolTip("Reset the drag offset so the intensity overlay\nsnaps back to centre over the background image.")
        btn_reset_pos.clicked.connect(self._reset_overlay_position)
        overlay_grid.addWidget(btn_reset_pos, 1, 1)

        blend_right.addLayout(overlay_grid)

        # Alpha slider
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(1)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(60)
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.alpha_slider.valueChanged.connect(self.update_alpha)

        blend_right.addWidget(QLabel("Overlay Transparency"))
        blend_right.addWidget(self.alpha_slider)

        # Color map selection
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(
            ["jet", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "YlGnBu", "Reds", "Accent"]
        )
        self.cmap_combo.addItem("Threat Zones (7)")
        self.cmap_combo.addItem("Dose Field (7)")
        self.cmap_combo.setCurrentText("jet")
        self.cmap_combo.currentTextChanged.connect(self.update_cmap)

        blend_right.addWidget(QLabel("Colormap"))
        blend_right.addWidget(self.cmap_combo)

        # Highlight controls
        highlight_group = QGroupBox("Highlight Levels")
        highlight_layout = QVBoxLayout()

        self.highlight_input = QLineEdit("1, 6, 15")
        self.highlight_input.setPlaceholderText("Enter comma-separated values (e.g., 1, 6, 15)")
        highlight_layout.addWidget(QLabel("Custom highlight values:"))
        highlight_layout.addWidget(self.highlight_input)

        self.color_combo = QComboBox()
        self.color_combo.addItems(["Default Colors", "White Only"])
        highlight_layout.addWidget(QLabel("Highlight colors:"))
        highlight_layout.addWidget(self.color_combo)

        btn_apply_highlights = QPushButton("Apply Highlights")
        btn_apply_highlights.clicked.connect(self.apply_highlights)
        highlight_layout.addWidget(btn_apply_highlights)

        highlight_group.setLayout(highlight_layout)
        blend_right.addWidget(highlight_group)

        # Custom colour scale
        custom_scale_group = QGroupBox("Custom Color Scale")
        custom_layout = QVBoxLayout()

        self.custom_levels_input = QLineEdit()
        self.custom_levels_input.setPlaceholderText("Enter 2-7 comma-separated values (e.g., 1000,3000,5000)")
        custom_layout.addWidget(QLabel("Threshold Values:"))
        custom_layout.addWidget(self.custom_levels_input)

        btn_apply_custom = QPushButton("Apply Custom Scale")
        btn_apply_custom.clicked.connect(self.apply_custom_scale)
        custom_layout.addWidget(btn_apply_custom)

        custom_scale_group.setLayout(custom_layout)
        blend_right.addWidget(custom_scale_group)

        # Colormap scale controls (spinboxes only)
        scale_controls = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(3)
        self.vmin_spin.setRange(-1e9, 1e9)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(3)
        self.vmax_spin.setRange(-1e9, 1e9)

        scale_controls.addWidget(QLabel("Min:"))
        scale_controls.addWidget(self.vmin_spin)
        scale_controls.addWidget(QLabel("Max:"))
        scale_controls.addWidget(self.vmax_spin)

        blend_right.addLayout(scale_controls)
        self.vmin_spin.valueChanged.connect(self.update_colormap_scale)
        self.vmax_spin.valueChanged.connect(self.update_colormap_scale)

        # Axis scale and units
        axis_group = QGroupBox("Axis Scale and Units")
        axis_layout = QVBoxLayout()

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("X Scale:"))
        self.scale_x_input = QLineEdit("1.0")
        scale_layout.addWidget(self.scale_x_input)

        scale_layout.addWidget(QLabel("Y Scale:"))
        self.scale_y_input = QLineEdit("1.0")
        scale_layout.addWidget(self.scale_y_input)

        scale_layout.addWidget(QLabel("Units:"))
        self.scale_unit_input = QLineEdit("cm")
        scale_layout.addWidget(self.scale_unit_input)

        axis_layout.addLayout(scale_layout)

        intensity_layout_axis = QHBoxLayout()
        intensity_layout_axis.addWidget(QLabel("Intensity Units:"))
        self.intensity_unit_input = QLineEdit("uGy/h")
        intensity_layout_axis.addWidget(self.intensity_unit_input)

        axis_layout.addLayout(intensity_layout_axis)

        btn_apply_scale = QPushButton("Apply Scale and Units")
        btn_apply_scale.clicked.connect(self.apply_units_and_scale)
        axis_layout.addWidget(btn_apply_scale)

        axis_group.setLayout(axis_layout)
        blend_right.addWidget(axis_group)

        formatting_group = QGroupBox("Axis Formatting")
        formatting_layout = QVBoxLayout()

        # Font size controls
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Label Font Size:"))
        self.label_fontsize_spin = QDoubleSpinBox()
        self.label_fontsize_spin.setRange(6, 24)
        self.label_fontsize_spin.setValue(12)
        font_layout.addWidget(self.label_fontsize_spin)

        font_layout.addWidget(QLabel("Tick Font Size:"))
        self.tick_fontsize_spin = QDoubleSpinBox()
        self.tick_fontsize_spin.setRange(6, 20)
        self.tick_fontsize_spin.setValue(10)
        font_layout.addWidget(self.tick_fontsize_spin)

        formatting_layout.addLayout(font_layout)

        # Font style controls
        style_layout = QHBoxLayout()
        self.bold_checkbox = QCheckBox("Bold Labels")
        self.italic_checkbox = QCheckBox("Italic Labels")
        style_layout.addWidget(self.bold_checkbox)
        style_layout.addWidget(self.italic_checkbox)
        formatting_layout.addLayout(style_layout)

        # Tick controls
        tick_layout = QHBoxLayout()
        tick_layout.addWidget(QLabel("X Ticks:"))
        self.x_ticks_spin = QSpinBox()
        self.x_ticks_spin.setRange(2, 20)
        self.x_ticks_spin.setValue(5)
        tick_layout.addWidget(self.x_ticks_spin)

        tick_layout.addWidget(QLabel("Y Ticks:"))
        self.y_ticks_spin = QSpinBox()
        self.y_ticks_spin.setRange(2, 20)
        self.y_ticks_spin.setValue(5)
        tick_layout.addWidget(self.y_ticks_spin)

        formatting_layout.addLayout(tick_layout)

        btn_apply_formatting = QPushButton("Apply Formatting")
        btn_apply_formatting.clicked.connect(self.apply_formatting)
        formatting_layout.addWidget(btn_apply_formatting)

        formatting_group.setLayout(formatting_layout)
        blend_right.addWidget(formatting_group)

        textbox_group = QGroupBox("Annotation Box")
        textbox_layout = QVBoxLayout()

        self.textbox_text_input = QLineEdit("Room: X, Date: 2026-03-13")
        textbox_layout.addWidget(QLabel("Text:"))
        textbox_layout.addWidget(self.textbox_text_input)

        self.textbox_fontsize_spin = QDoubleSpinBox()
        self.textbox_fontsize_spin.setRange(6, 32)
        self.textbox_fontsize_spin.setValue(10)
        textbox_layout.addWidget(QLabel("Font size:"))
        textbox_layout.addWidget(self.textbox_fontsize_spin)

        self.textbox_color_input = QLineEdit("white")
        textbox_layout.addWidget(QLabel("Font color (name or #RRGGBB):"))
        textbox_layout.addWidget(self.textbox_color_input)

        btn_apply_textbox = QPushButton("Apply Text Box")
        btn_apply_textbox.clicked.connect(self.apply_textbox)
        textbox_layout.addWidget(btn_apply_textbox)

        textbox_group.setLayout(textbox_layout)
        blend_right.addWidget(textbox_group)

        blend_right.addStretch(1)

        # Wrap the right panel in a scroll area
        blend_scroll = QScrollArea()
        blend_scroll.setWidgetResizable(True)
        blend_scroll.setWidget(blend_right_inner)
        blend_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        blend_layout.addWidget(blend_scroll, stretch=1)

        self.measuring = False
        self.measurement_points = []
        self.measurement_lines = []
        self.measurement_annotation = None

        blend_tab = QWidget()
        blend_tab.setLayout(blend_layout)
        self.tab_widget.addTab(blend_tab, "Blended")

        # Tab 4: Grid Overlay
        grid_tab = QWidget()
        grid_layout = QHBoxLayout()
        grid_left = QVBoxLayout()

        self.grid_canvas = DraggableCanvas(self)
        self.grid_canvas.setMinimumSize(400, 300)
        grid_left.addWidget(self.grid_canvas)
        grid_layout.addLayout(grid_left, stretch=3)

        grid_right = QVBoxLayout()

        self.grid_type_combo = QComboBox()
        self.grid_type_combo.addItems(["Points", "Dotted Lines"])
        grid_right.addWidget(QLabel("Grid Type:"))
        grid_right.addWidget(self.grid_type_combo)

        self.grid_nx_spin = QSpinBox()
        self.grid_nx_spin.setRange(1, 200)
        self.grid_nx_spin.setValue(20)
        self.grid_nx_spin.setPrefix("Columns: ")

        self.grid_ny_spin = QSpinBox()
        self.grid_ny_spin.setRange(1, 200)
        self.grid_ny_spin.setValue(20)
        self.grid_ny_spin.setPrefix("Rows: ")

        grid_right.addWidget(self.grid_nx_spin)
        grid_right.addWidget(self.grid_ny_spin)

        btn_show_grid = QPushButton("Show Grid Overlay")
        btn_show_grid.clicked.connect(self.show_grid_overlay)
        grid_right.addWidget(btn_show_grid)

        btn_save_grid = QPushButton("Save Overlay Image")
        btn_save_grid.clicked.connect(self.save_grid_image)
        grid_right.addWidget(btn_save_grid)

        grid_right.addStretch(1)
        grid_layout.addLayout(grid_right, stretch=1)

        grid_tab.setLayout(grid_layout)
        self.grid_nx_spin.valueChanged.connect(self.show_grid_overlay)
        self.grid_ny_spin.valueChanged.connect(self.show_grid_overlay)
        self.tab_widget.addTab(grid_tab, "Grid Overlay")

    # -------------------------------------------------------------------
    # Lazy preview canvas creation
    # -------------------------------------------------------------------

    def _ensure_preview_canvas(self):
        """Create the preview FigureCanvas on first use."""
        if self.preview_canvas is not None:
            return
        self.preview_canvas = FigureCanvasQTAgg(Figure(figsize=(2, 2)))
        self.preview_canvas.setMinimumSize(150, 150)
        self._preview_canvas_layout.addWidget(self.preview_canvas)

    # -------------------------------------------------------------------
    # Image-tab annotation helpers
    # -------------------------------------------------------------------

    def _on_draw_mode_changed(self, button):
        """Sync the annotation widget's mode with the selected radio button."""
        self.image_canvas.set_mode(button.mode_value)

    def _sync_annotation_colors(self):
        """Push the current colour-combo selections to the canvas widget."""
        self.image_canvas.set_colors(
            marker=_resolve_colour_combo(self.marker_color_combo, self),
            line=_resolve_colour_combo(self.line_color_combo, self),
            freedraw=_resolve_colour_combo(self.freedraw_color_combo, self),
            text=_resolve_colour_combo(self.text_color_combo, self),
        )
        self._update_scale_readout()

    def _update_annotation_status(self):
        """Refresh the annotation count label on the Image tab."""
        counts = self.image_canvas.annotation_count()
        total = sum(counts.values())
        if total == 0:
            self.annot_status_label.setText("No annotations")
            self.annot_status_label.setStyleSheet("color: grey; font-style: italic;")
        else:
            parts = []
            for key, label in [('markers', 'marker'), ('lines', 'line'),
                               ('strokes', 'stroke'), ('texts', 'text'),
                               ('icons', 'icon')]:
                n = counts.get(key, 0)
                if n:
                    parts.append(f"{n} {label}{'s' if n != 1 else ''}")
            self.annot_status_label.setText(", ".join(parts))
            self.annot_status_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self._update_scale_readout()

    def _update_scale_readout(self):
        ppu = self.image_canvas.get_pixels_per_unit()
        if ppu is not None:
            unit = self.image_canvas.get_scale_unit()
            self.scale_readout_label.setText(f"Scale: {ppu:.2f} px/{unit}")
            self.scale_readout_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.scale_readout_label.setText("Scale: not set")
            self.scale_readout_label.setStyleSheet("color: grey; font-style: italic;")
        self._update_total_distance()

    def _update_total_distance(self):
        """Refresh the total line-segment distance readout."""
        total = self.image_canvas.get_total_line_distance()
        if total is None:
            # Scale not set – hide or show hint
            n_lines = len(self.image_canvas.get_lines())
            if n_lines > 0:
                self.total_distance_label.setText(
                    f"{n_lines} line{'s' if n_lines != 1 else ''} "
                    f"(set scale to see distances)")
                self.total_distance_label.setStyleSheet(
                    "color: grey; font-style: italic;")
            else:
                self.total_distance_label.setText("")
        elif total > 0:
            unit = self.image_canvas.get_scale_unit()
            dists = self.image_canvas.get_line_distances()
            n = len(dists)
            self.total_distance_label.setText(
                f"Total: {total:.2f} {unit} ({n} segment{'s' if n != 1 else ''})")
            self.total_distance_label.setStyleSheet(
                "color: #FF9800; font-weight: bold;")
        else:
            self.total_distance_label.setText("")

    def _undo_last_annotation(self):
        self.image_canvas.undo_last()
        self._update_annotation_status()

    def _clear_and_update_status(self):
        self.image_canvas.clear_annotations()
        self._update_annotation_status()

    def _get_annotation_colors(self):
        """Read the current colour choices from the named-colour combos."""
        return (
            _resolve_colour_combo(self.marker_color_combo, self),
            _resolve_colour_combo(self.line_color_combo, self),
            _resolve_colour_combo(self.freedraw_color_combo, self),
            _resolve_colour_combo(self.text_color_combo, self),
        )

    def _load_icon_for_overlay(self):
        """Let the user pick an image file to use as a placeable icon."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Icon Image", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.svg *.gif);;All Files (*)"
        )
        if not path:
            return
        pm = QPixmap(path)
        if pm.isNull():
            QMessageBox.warning(self, "Error", "Could not load image.")
            return
        size, ok = QInputDialog.getInt(
            self, "Icon Size",
            "Enter the icon width in pixels (height will scale proportionally):",
            value=64, min=16, max=1024
        )
        if ok:
            pm = pm.scaledToWidth(size, Qt.TransformationMode.SmoothTransformation)
        self.image_canvas.set_icon_to_place(pm)
        QMessageBox.information(
            self, "Icon Loaded",
            f"Icon loaded ({pm.width()}x{pm.height()}).\n"
            "Select 'Place Icon' mode and click on the image to place it."
        )

    def _export_annotation_layer(self):
        """Save current annotations as a transparent PNG (no background)."""
        layer = self.image_canvas.export_annotations_layer()
        if layer is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Annotation Layer", "annotations_layer.png",
            "PNG Files (*.png)"
        )
        if path:
            layer.save(path, "PNG")
            QMessageBox.information(self, "Saved", f"Annotation layer saved to:\n{path}")

    def _import_annotation_layer(self):
        """Load a transparent PNG and composite it onto the current image."""
        if self.image_canvas._pixmap is None:
            QMessageBox.warning(self, "No Image", "Load an image first.")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Import Annotation Layer", "",
            "PNG Files (*.png);;All Files (*)"
        )
        if not path:
            return

        layer = QPixmap(path)
        if layer.isNull():
            QMessageBox.warning(self, "Error", f"Could not load image:\n{path}")
            return

        ok = self.image_canvas.import_annotations_layer(layer)
        if ok:
            # Update the shared original_pixmap so Blended/Grid tabs see it
            self.original_pixmap = self.image_canvas._pixmap.copy()
            self.invalidate_cache()
            QMessageBox.information(
                self, "Imported",
                "Annotation layer composited onto the image.\n"
                "It is now part of the base image visible in all tabs."
            )
        else:
            QMessageBox.warning(self, "Error", "Failed to composite the layer.")

    def finalise_annotations(self):
        """Burn annotations permanently into ``original_pixmap``."""
        if not self.image_canvas.has_annotations():
            QMessageBox.information(self, "Nothing to finalise",
                                   "Add some annotations first.")
            return

        reply = QMessageBox.question(
            self, "Finalise Annotations?",
            "This will permanently paint the annotations onto the image.\n"
            "They will then appear in all tabs (Blended, Grid, Preview).\n\n"
            "This cannot be undone \u2014 you would need to reload the original\n"
            "image file to remove them.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        m_col, l_col, f_col, t_col = self._get_annotation_colors()
        lw = self.annot_line_width_spin.value()

        burned = self.image_canvas.burn_annotations_to_pixmap(
            marker_color=m_col, line_color=l_col,
            freedraw_color=f_col, text_color=t_col, line_width=lw,
        )

        self.original_pixmap = burned
        self.image_canvas.clear_annotations()
        self._update_annotation_status()

        if self.crop_rect:
            base = self.original_pixmap.copy(self.crop_rect)
        else:
            base = self.original_pixmap
        if self.current_rotation_angle:
            transform = QTransform().rotate(self.current_rotation_angle)
            self.current_pixmap = base.transformed(
                transform, Qt.TransformationMode.SmoothTransformation)
        else:
            self.current_pixmap = base

        self.image_canvas.set_pixmap(self.current_pixmap)
        self._invalidate_bg_cache()
        self.invalidate_cache()
        self.update_intensity_preview()

        QMessageBox.information(
            self, "Done",
            "Annotations burned onto image.\n"
            "They will now appear in Blended, Grid, and Preview tabs."
        )

    def _reset_overlay_position(self):
        self.plot_canvas.reset_intensity_position()
        self.update_display()

    # -------------------------------------------------------------------
    # Background-pixmap cache  (avoids re-rotating on every draw call)
    # -------------------------------------------------------------------

    def _invalidate_bg_cache(self):
        """Clear the cached rotated background pixmap."""
        self._bg_cache = None
        self._bg_cache_key = None
        self._bg_version += 1

    def get_rotated_background_pixmap(self):
        """Return the background pixmap with current crop + rotation applied.

        The result is cached until the source image, crop rect, or rotation
        angle changes.
        """
        key = (self._bg_version, self.crop_rect, self.current_rotation_angle)
        if self._bg_cache is not None and self._bg_cache_key == key:
            return self._bg_cache

        base_pixmap = self.original_pixmap.copy(self.crop_rect) if self.crop_rect else self.original_pixmap
        transform = QTransform().rotate(self.current_rotation_angle)
        result = base_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        self._bg_cache = result
        self._bg_cache_key = key
        return result

    # -------------------------------------------------------------------
    # Image loading  (PNG, JPG, BMP, HEIC/HEIF, PDF)
    # -------------------------------------------------------------------

    def load_image(self):
        """Load an image or PDF file as the background."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image or PDF", "",
            "All Supported (*.png *.jpg *.jpeg *.bmp *.heic *.heif *.pdf);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp *.heic *.heif);;"
            "PDF Files (*.pdf)"
        )
        if not file_name:
            return

        self.source_path = file_name
        self.screen_dpi = None
        self.source_type = None

        ext = os.path.splitext(file_name)[1].lower()

        # --- PDF path ---
        if ext == '.pdf':
            self._load_pdf_background(file_name)
            self.source_type = "pdf"
            self._on_new_image_loaded()
            return

        # --- HEIC / HEIF path ---
        if ext in ('.heic', '.heif'):
            pm = _try_load_heic(file_name)
            if pm is None or pm.isNull():
                QMessageBox.warning(self, "Error",
                                    "Could not load HEIC/HEIF file.\n"
                                    "Make sure pillow-heif is installed.")
                return
        else:
            # --- Standard raster path (PNG, JPG, BMP, etc.) ---
            pm = _safe_load_raster_image(file_name)
            if pm is None or pm.isNull():
                # Fallback: let Qt try natively
                pm = QPixmap(file_name)
            if pm is None or pm.isNull():
                QMessageBox.warning(self, "Error", "Could not load image file.")
                return

        # --- Common post-load for all raster images ---
        raw_w, raw_h = pm.width(), pm.height()
        pm = _safe_downscale_pixmap(pm, MAX_IMAGE_DIMENSION)
        if pm.width() < raw_w or pm.height() < raw_h:
            QMessageBox.information(
                self, "Image Resized",
                f"Image was {raw_w}\u00d7{raw_h} pixels and has been\n"
                f"auto-resized to {pm.width()}\u00d7{pm.height()} to prevent "
                f"memory issues."
            )

        self.original_pixmap = pm
        self.current_pixmap = pm
        self.source_type = "image"
        self.image_canvas.set_pixmap(pm)
        self.current_rotation_angle = 0
        self._on_new_image_loaded()

    def _on_new_image_loaded(self):
        """Shared post-load actions for standard images and HEIC files.

        Performs gc.collect() here 
        instead of on every cache invalidation.
        """
        self._invalidate_bg_cache()
        self.reset_image()
        gc.collect()

    def _load_pdf_background(self, file_path: str):
        fitz = _get_fitz()
        doc = fitz.open(file_path)
        if doc.page_count == 0:
            QMessageBox.warning(self, "Error", "PDF has no pages.")
            doc.close()
            return

        page = doc.load_page(0)
        page_rect = page.rect  # points (1/72 inch)

        page_width_in = page_rect.width / 72.0
        page_height_in = page_rect.height / 72.0

        # Choose scale so long side ~ MAX_IMAGE_DIMENSION
        long_side_pts = max(page_rect.width, page_rect.height)
        target_long_px = float(MAX_IMAGE_DIMENSION)
        scale = target_long_px / long_side_pts

        # Render at this scale
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

        # Effective on-screen dpi (approx)
        screen_dpi_x = pix.width / page_width_in if page_width_in > 0 else 72.0
        screen_dpi_y = pix.height / page_height_in if page_height_in > 0 else 72.0
        self.screen_dpi = min(screen_dpi_x, screen_dpi_y)

        # Convert pixmap -> QImage -> QPixmap (RGB or RGBA)
        if pix.alpha:
            fmt = QImage.Format.Format_RGBA8888
        else:
            fmt = QImage.Format.Format_RGB888

        qimg = QImage(
            pix.samples, pix.width, pix.height, pix.stride, fmt
        )
        # Copy to detach from pix.samples buffer
        qimg = qimg.copy()
        pm = QPixmap.fromImage(qimg)

        doc.close()

        # Honour global size cap (usually already within limit)
        pm = _safe_downscale_pixmap(pm, MAX_IMAGE_DIMENSION)

        self.original_pixmap = pm
        self.current_pixmap = pm
        self.image_canvas.set_pixmap(pm)
        self.current_rotation_angle = 0

    def reset_image(self):
        if not self.original_pixmap:
            return
        # If the user is mid-crop, exit that mode cleanly
        if self.image_canvas._mode == AnnotatedImageWidget.MODE_CROP:
            self._exit_crop_mode()
        self.crop_rect = None
        self.current_pixmap = self.original_pixmap
        self.rotation_slider.setValue(0)
        self.rotation_label.setText("0\u00b0")
        self.image_canvas.set_pixmap(self.current_pixmap)
        self._invalidate_bg_cache()
        self.invalidate_cache()
        self.update_intensity_preview()
        self.set_grid_spinboxes_from_data()

    def define_crop_area(self):
        """Enter visual crop mode — the user draws a rectangle on the image."""
        if not self.original_pixmap:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return

        # Store the previous draw mode so we can restore on cancel
        self._prev_draw_mode = self.image_canvas._mode

        # Enter crop mode on the canvas
        self.image_canvas.set_mode(AnnotatedImageWidget.MODE_CROP)
        self.image_canvas._crop_rect_preview = None
        self.image_canvas._crop_callback = self._on_crop_selected

        # Show Apply/Cancel, hide Select
        self._btn_start_crop.setVisible(False)
        self._btn_apply_crop.setVisible(True)
        self._btn_apply_crop.setEnabled(False)  # enabled once a rect is drawn
        self._btn_cancel_crop.setVisible(True)
        self._btn_cancel_crop.setEnabled(True)

    def _on_crop_selected(self, rect: QRect):
        """Callback from AnnotatedImageWidget when the user finishes dragging
        a crop rectangle.  Enables the Apply button."""
        self._btn_apply_crop.setEnabled(True)

    def _apply_visual_crop(self):
        """Apply the drawn crop rectangle to the image."""
        rect = self.image_canvas._crop_rect_preview
        if rect is None or rect.width() < 2 or rect.height() < 2:
            return

        self.crop_rect = rect
        self.current_pixmap = self.original_pixmap.copy(self.crop_rect)
        self.rotation_slider.setValue(0)
        self.rotation_label.setText("0\u00b0")
        self.image_canvas.set_pixmap(self.current_pixmap)
        self._invalidate_bg_cache()
        self.invalidate_cache()
        self.update_intensity_preview()

        # Clean up crop mode
        self._exit_crop_mode()

    def _cancel_visual_crop(self):
        """Cancel the crop selection and return to the previous draw mode."""
        self.image_canvas._crop_rect_preview = None
        self.image_canvas._crop_origin = None
        self.image_canvas.update()
        self._exit_crop_mode()

    def _exit_crop_mode(self):
        """Restore UI state after crop is applied or cancelled."""
        prev = getattr(self, '_prev_draw_mode', AnnotatedImageWidget.MODE_NONE)
        self.image_canvas.set_mode(prev)
        self.image_canvas._crop_callback = None
        self.image_canvas._crop_rect_preview = None

        self._btn_start_crop.setVisible(True)
        self._btn_apply_crop.setVisible(False)
        self._btn_apply_crop.setEnabled(False)
        self._btn_cancel_crop.setVisible(False)
        self._btn_cancel_crop.setEnabled(False)

    def rotate_image_view(self, angle):
        if not self.original_pixmap:
            return
        self.current_rotation_angle = angle
        base_pixmap = self.original_pixmap.copy(self.crop_rect) if self.crop_rect else self.original_pixmap
        transform = QTransform().rotate(angle)
        self.current_pixmap = base_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        self.image_canvas.set_pixmap(self.current_pixmap)
        self.rotation_label.setText(f"{angle}\u00b0")
        self._invalidate_bg_cache()

    # -------------------------------------------------------------------
    # Intensity data helpers
    # -------------------------------------------------------------------

    def get_raw_intensity_data(self):
        """Read the table widget contents into a NumPy array."""
        rows = self.table_widget.rowCount()
        cols = self.table_widget.columnCount()
        if rows == 0 or cols == 0:
            return None

        data = np.zeros((rows, cols))
        for r in range(rows):
            for c in range(cols):
                item = self.table_widget.item(r, c)
                try:
                    data[r, c] = float(item.text()) if item and item.text() else 0.0
                except (ValueError, TypeError):
                    data[r, c] = 0.0
        return data

    def get_final_intensity_array(self):
        """Resize the raw grid to match the image (or place in crop region)."""
        from scipy.ndimage import zoom
        raw_data = self.get_raw_intensity_data()
        if raw_data is None or not self.original_pixmap:
            return None

        if self.crop_rect:
            full_h, full_w = self.original_pixmap.height(), self.original_pixmap.width()
            composite_array = np.full((full_h, full_w), np.nan, dtype=float)
            crop_h, crop_w = self.crop_rect.height(), self.crop_rect.width()
            zoom_y, zoom_x = crop_h / raw_data.shape[0], crop_w / raw_data.shape[1]

            try:
                resized_intensity = zoom(raw_data, (zoom_y, zoom_x), order=1)
            except Exception:
                rep_y = max(1, int(np.ceil(zoom_y)))
                rep_x = max(1, int(np.ceil(zoom_x)))
                resized_intensity = np.repeat(np.repeat(raw_data, rep_y, axis=0), rep_x, axis=1)
                resized_intensity = resized_intensity[:crop_h, :crop_w]

            x, y = self.crop_rect.x(), self.crop_rect.y()
            composite_array[y: y + crop_h, x: x + crop_w] = resized_intensity
            return composite_array
        else:
            full_h, full_w = self.original_pixmap.height(), self.original_pixmap.width()
            zoom_y, zoom_x = full_h / raw_data.shape[0], full_w / raw_data.shape[1]
            try:
                return zoom(raw_data, (zoom_y, zoom_x), order=1)
            except Exception:
                rep_y = max(1, int(np.ceil(zoom_y)))
                rep_x = max(1, int(np.ceil(zoom_x)))
                resized = np.repeat(np.repeat(raw_data, rep_y, axis=0), rep_x, axis=1)
                return resized[:full_h, :full_w]

    # -------------------------------------------------------------------
    # Blended view: heatmap / contours
    # -------------------------------------------------------------------

    def show_heatmap(self):
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and intensity data first.")
            return

        final_intensity = self.get_final_intensity_array()
        if final_intensity is None:
            QMessageBox.warning(self, "Warning", "Could not generate intensity data.")
            return

        valid_data = final_intensity[~np.isnan(final_intensity)]
        if valid_data.size == 0:
            QMessageBox.warning(self, "Warning", "No valid data to plot.")
            return

        vmin, vmax = float(np.min(valid_data)), float(np.max(valid_data))

        # Block signals while bulk-updating range and value to prevent
        # cascading redraws.
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setRange(vmin, vmax)
        self.vmax_spin.setRange(vmin, vmax)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        self.plot_canvas.draw_heatmap(
            self.get_rotated_background_pixmap(), final_intensity,
            alpha=self.alpha, cmap=self.cmap,
            vmin=vmin, vmax=vmax, units=intensity_units,
            scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
        )
        self._current_vis_mode = "heatmap"

    def show_contours(self):
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and intensity data first.")
            return

        final_intensity = self.get_final_intensity_array()
        if final_intensity is None:
            QMessageBox.warning(self, "Warning", "Could not generate intensity data.")
            return

        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        self.plot_canvas.draw_contours(
            self.get_rotated_background_pixmap(), final_intensity,
            alpha=self.alpha, cmap=self.cmap, levels=7,
            units=intensity_units, scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
        )
        self._current_vis_mode = "contour"

    def _redraw_intensity_overlay(self, ax, bg_extent):
            """
            Redraw the intensity overlay (heatmap or contours) on the given
            Matplotlib axes, using the supplied background extent.

            Used by _export_with_high_res_pdf to reproduce the blended tab state.
            """
            # Recompute final intensity as in show_heatmap/show_contours
            final_intensity = self.get_final_intensity_array()
            if final_intensity is None:
                return

            # Units and scaling
            intensity_units = self.intensity_unit_input.text()
            distance_units = self.scale_unit_input.text()
            scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
            scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

            # Apply same vmin/vmax logic as show_heatmap
            valid_data = final_intensity[~np.isnan(final_intensity)]
            if valid_data.size == 0:
                return

            # Use the current spinbox limits if they are set; otherwise recompute
            try:
                vmin = float(self.vmin_spin.value())
                vmax = float(self.vmax_spin.value())
            except Exception:
                vmin = float(np.min(valid_data))
                vmax = float(np.max(valid_data))

            # Base extent is bg_extent; apply current drag offset from plot_canvas
            # to keep alignment consistent with the UI.
            if hasattr(self, "plot_canvas"):
                dx = getattr(self.plot_canvas, "intensity_offset_x", 0.0)
                dy = getattr(self.plot_canvas, "intensity_offset_y", 0.0)
            else:
                dx = dy = 0.0

            ie = [
                bg_extent[0] + dx,
                bg_extent[1] + dx,
                bg_extent[2] + dy,
                bg_extent[3] + dy,
            ]

            mode = getattr(self, "_current_vis_mode", "heatmap")
            cmap_obj = _resolve_cmap(self.cmap)

            if mode == "contour":
                # Contour mode
                rows, cols = final_intensity.shape
                X = np.linspace(ie[0], ie[1], cols)
                Y = np.linspace(ie[2], ie[3], rows)
                xx, yy = np.meshgrid(X, Y)

                # Flip vertical like draw_contours does
                intensity_flipped = np.flipud(final_intensity)

                cs = ax.contourf(
                    xx,
                    yy,
                    intensity_flipped,
                    levels=7,
                    cmap=cmap_obj,
                    alpha=self.alpha,
                    extend="max",
                )

                # Colorbar and labels
                cbar = ax.figure.colorbar(cs, ax=ax, orientation="vertical", pad=0.05)
                cbar_label = f"Intensity ({intensity_units})" if intensity_units else "Intensity"
                cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=14, fontweight="bold")
                cbar.ax.tick_params(labelsize=13)

            else:
                # Heatmap mode (default)
                cmap_obj.set_over(cmap_obj(1.0))
                top_color = list(cmap_obj(1.0))
                top_color[3] = 1.0
                cmap_obj.set_over(tuple(top_color))
                cmap_obj.set_under((0, 0, 0, 0))

                im = ax.imshow(
                    final_intensity,
                    cmap=cmap_obj,
                    alpha=self.alpha,
                    interpolation="bilinear",
                    extent=ie,
                    origin="upper",
                    vmin=vmin,
                    vmax=vmax,
                )

                cbar = ax.figure.colorbar(im, ax=ax, orientation="vertical", pad=0.05, extend="max")
                cbar_label = f"Intensity ({intensity_units})" if intensity_units else "Intensity"
                cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=15, fontweight="bold")

            # Apply axis scaling (x/y labels and tick scaling) to this axes
            if scale_x != 1.0 or scale_y != 1.0:
                x_ticks = ax.get_xticks()
                y_ticks = ax.get_yticks()
                x_labels = [f"{tick * scale_x:.0f}" for tick in x_ticks]
                y_labels = [f"{tick * scale_y:.0f}" for tick in y_ticks]
                ax.set_xticklabels(x_labels)
                ax.set_yticklabels(y_labels)

            ax.set_xlabel(f"Distance ({distance_units})", fontsize=15, fontweight="bold")
            ax.set_ylabel(f"Distance ({distance_units})", fontsize=15, fontweight="bold")

    # -------------------------------------------------------------------
    # Cache invalidation  (no gc.collect here – only lightweight)
    # -------------------------------------------------------------------

    def invalidate_cache(self):
        """Mark cached intensity array as stale."""
        self.cached_intensity_array = None
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None

    # -------------------------------------------------------------------
    # Intensity preview  (enhanced with image background)
    # -------------------------------------------------------------------

    def update_intensity_preview(self):
        """Update the small preview plot on the Intensity Data tab."""
        self._ensure_preview_canvas()

        intensity = self.get_raw_intensity_data()
        fig = self.preview_canvas.figure
        fig.clf()
        ax = fig.add_subplot(111)

        if intensity is None or intensity.size == 0:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.axis('off')
            self.preview_canvas.draw()
            return

        mode = self.preview_mode_combo.currentText()
        rows, cols = intensity.shape
        cmap = 'jet'

        if self.original_pixmap is None:
            try:
                if mode == "Contour Map":
                    ax.contourf(intensity, levels=7, cmap=cmap)
                else:
                    ax.imshow(intensity, cmap=cmap, aspect='auto')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
                ax.axis('off')
        else:
            bg_pixmap = self.current_pixmap if self.current_pixmap else self.original_pixmap
            bg_arr = _pixmap_to_rgb_array(bg_pixmap)

            ax.imshow(bg_arr, extent=[0, cols, rows, 0], aspect='auto',
                      origin='upper', alpha=0.6)

            try:
                if mode == "Contour Map":
                    X = np.linspace(0, cols, cols)
                    Y = np.linspace(0, rows, rows)
                    ax.contourf(X, Y, intensity, levels=7, cmap=cmap, alpha=0.75)
                else:
                    ax.imshow(intensity, cmap=cmap, aspect='auto', origin='upper',
                              extent=[0, cols, rows, 0], alpha=0.75)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')

            ax.text(0.02, 0.02, "Row 0 / Col 0", transform=ax.transAxes,
                    fontsize=6, color='white', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.6))

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        self.preview_canvas.draw()

    # -------------------------------------------------------------------
    # Grid spinbox sync
    # -------------------------------------------------------------------

    def set_grid_spinboxes_from_data(self):
        """Sync grid overlay spinboxes with the current table dimensions."""
        current_rows = self.table_widget.rowCount()
        current_cols = self.table_widget.columnCount()

        if hasattr(self, "grid_nx_spin") and hasattr(self, "grid_ny_spin"):
            if current_rows > 0 and current_cols > 0:
                self.grid_nx_spin.setValue(current_cols)
                self.grid_ny_spin.setValue(current_rows)
            else:
                self.grid_nx_spin.setValue(100)
                self.grid_ny_spin.setValue(100)

    # -------------------------------------------------------------------
    # CSV / Excel / Clipboard loading
    # -------------------------------------------------------------------

    def load_csv_excel(self):
        pd = _get_pandas()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xls *.xlsx)"
        )
        if file_name:
            try:
                if file_name.endswith(".csv"):
                    df = pd.read_csv(file_name, header=None)
                else:
                    df = pd.read_excel(file_name, header=None)

                data = df.values
                rows, cols = data.shape

                self.table_widget.blockSignals(True)
                self.table_widget.setRowCount(rows)
                self.table_widget.setColumnCount(cols)

                for r in range(rows):
                    for c in range(cols):
                        self.table_widget.setItem(r, c, QTableWidgetItem(str(data[r, c])))

                self.table_widget.blockSignals(False)
                self.intensity_data = data
                self.invalidate_cache()
                self.update_intensity_preview()
                self.set_grid_spinboxes_from_data()

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file.\n{str(e)}")

    def paste_clipboard_data(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()

        if text:
            rows = [r for r in text.split('\n') if r.strip()]
            data = [r.split('\t') for r in rows]
            row_count = len(data)
            col_count = max(len(row) for row in data) if row_count > 0 else 0

            self.table_widget.blockSignals(True)
            self.table_widget.setRowCount(row_count)
            self.table_widget.setColumnCount(col_count)

            for r, row in enumerate(data):
                for c, val in enumerate(row):
                    try:
                        num_val = float(val)
                    except ValueError:
                        num_val = 0.0
                    self.table_widget.setItem(r, c, QTableWidgetItem(str(num_val)))

            self.table_widget.blockSignals(False)
            self.invalidate_cache()
            self.update_intensity_preview()
            self.set_grid_spinboxes_from_data()

    # -------------------------------------------------------------------
    # Table row / column management
    # -------------------------------------------------------------------

    def add_row(self):
        self.table_widget.insertRow(self.table_widget.rowCount())
        self.invalidate_cache()
        self.update_intensity_preview()
        self.set_grid_spinboxes_from_data()

    def remove_row(self):
        if self.table_widget.rowCount() > 0:
            self.table_widget.removeRow(self.table_widget.rowCount() - 1)
            self.invalidate_cache()
            self.update_intensity_preview()
            self.set_grid_spinboxes_from_data()

    def add_column(self):
        self.table_widget.insertColumn(self.table_widget.columnCount())
        self.invalidate_cache()
        self.update_intensity_preview()
        self.set_grid_spinboxes_from_data()

    def remove_column(self):
        if self.table_widget.columnCount() > 0:
            self.table_widget.removeColumn(self.table_widget.columnCount() - 1)
            self.invalidate_cache()
            self.update_intensity_preview()
            self.set_grid_spinboxes_from_data()

    # --- Visualization Controls ---
    def update_alpha(self):
        value = self.alpha_slider.value()
        self.alpha = value / 100.0
        self.update_display()

    def update_cmap(self):
        cmap_name = self.cmap_combo.currentText()
        if cmap_name == "Threat Zones (7)":
            self.cmap = get_threat_zone_cmap()
        elif cmap_name == "Dose Field (7)":
            self.cmap = get_continuous_dose_cmap()
        else:
            self.cmap = cmap_name
        self.update_display()

    def update_colormap_scale(self):
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            return

        final_intensity = self.get_final_intensity_array()
        if final_intensity is None:
            return

        valid_data = final_intensity[~np.isnan(final_intensity)]
        if valid_data.size == 0:
            return

        data_min = float(np.min(valid_data))
        data_max = float(np.max(valid_data))

        # Block signals while adjusting range so that clamping doesn't
        # re-trigger this method in a feedback loop.
        self.vmin_spin.blockSignals(True)
        self.vmax_spin.blockSignals(True)
        self.vmin_spin.setRange(data_min, data_max)
        self.vmax_spin.setRange(data_min, data_max)
        self.vmin_spin.blockSignals(False)
        self.vmax_spin.blockSignals(False)

        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
            self.plot_canvas.draw_contours(
                self.get_rotated_background_pixmap(), final_intensity,
                alpha=self.alpha, cmap=self.cmap,
                levels=np.linspace(vmin, vmax, 20),
                units=intensity_units, scale_x=scale_x, scale_y=scale_y,
                distance_units=distance_units
            )
        else:
            self.plot_canvas.draw_heatmap(
                self.get_rotated_background_pixmap(), final_intensity,
                alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax,
                units=intensity_units, scale_x=scale_x, scale_y=scale_y,
                distance_units=distance_units
            )

    def apply_formatting(self):
        if not hasattr(self.plot_canvas, 'ax') or not self.plot_canvas.ax.get_children():
            QMessageBox.warning(self, "Warning", "Create a plot first")
            return

        label_fontsize = self.label_fontsize_spin.value()
        tick_fontsize = self.tick_fontsize_spin.value()
        is_bold = self.bold_checkbox.isChecked()
        is_italic = self.italic_checkbox.isChecked()
        x_ticks = self.x_ticks_spin.value()
        y_ticks = self.y_ticks_spin.value()

        fontweight = 'bold' if is_bold else 'normal'
        fontstyle = 'italic' if is_italic else 'normal'

        self.plot_canvas.ax.xaxis.label.set_fontsize(label_fontsize)
        self.plot_canvas.ax.xaxis.label.set_fontweight(fontweight)
        self.plot_canvas.ax.xaxis.label.set_fontstyle(fontstyle)
        self.plot_canvas.ax.yaxis.label.set_fontsize(label_fontsize)
        self.plot_canvas.ax.yaxis.label.set_fontweight(fontweight)
        self.plot_canvas.ax.yaxis.label.set_fontstyle(fontstyle)

        self.plot_canvas.ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        if hasattr(self.plot_canvas, 'cbar') and self.plot_canvas.cbar is not None:
            self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

        xlim = self.plot_canvas.ax.get_xlim()
        ylim = self.plot_canvas.ax.get_ylim()
        x_tick_positions1 = np.linspace(xlim[0], xlim[1], x_ticks)
        y_tick_positions1 = np.linspace(ylim[0], ylim[1], y_ticks)

        x_tick_positions = np.round(x_tick_positions1 / 50) * 50
        y_tick_positions = np.round(y_tick_positions1 / 50) * 50

        if xlim[0] <= 0 <= xlim[1] and 0 not in x_tick_positions:
            x_tick_positions = np.append(x_tick_positions, 0)
            x_tick_positions = np.sort(x_tick_positions)

        if ylim[0] <= 0 <= ylim[1] and 0 not in y_tick_positions:
            y_tick_positions = np.append(y_tick_positions, 0)
            y_tick_positions = np.sort(y_tick_positions)

        self.plot_canvas.ax.set_xticks(x_tick_positions)
        self.plot_canvas.ax.set_yticks(y_tick_positions)

        self.plot_canvas.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
        self.plot_canvas.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

        if hasattr(self.plot_canvas, 'cbar') and self.plot_canvas.cbar is not None:
            self.plot_canvas.cbar.ax.yaxis.label.set_fontsize(label_fontsize)
            self.plot_canvas.cbar.ax.yaxis.label.set_fontweight(fontweight)
            self.plot_canvas.cbar.ax.yaxis.label.set_fontstyle(fontstyle)
            self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

        self.plot_canvas.draw()

    def apply_textbox(self):
        if not hasattr(self.plot_canvas, "ax") or self.plot_canvas.ax is None:
            QMessageBox.warning(self, "Warning", "Create a plot first.")
            return

        text = self.textbox_text_input.text()
        if not text:
            return

        fontsize = self.textbox_fontsize_spin.value()
        color = self.textbox_color_input.text().strip() or "white"

        if hasattr(self, "corner_annotation") and self.corner_annotation is not None:
            try:
                self.corner_annotation.remove()
            except Exception:
                pass

        self.corner_annotation = self.plot_canvas.ax.text(
            0.98, 0.98, text,
            transform=self.plot_canvas.ax.transAxes,
            ha="right", va="top", fontsize=fontsize, color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5, edgecolor="none"),
        )
        self.plot_canvas.draw()

    def apply_custom_scale(self):
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load image and data first")
            return

        try:
            levels = [float(x.strip()) for x in self.custom_levels_input.text().split(',')]
            if not 2 <= len(levels) <= 7:
                raise ValueError("Enter 2-7 values")

            levels = sorted(levels)
            final_intensity = self.get_final_intensity_array()
            distance_units = self.scale_unit_input.text()
            scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
            scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

            base_cmap = plt.get_cmap(self.cmap)
            colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
            custom_cmap = ListedColormap(colors)
            custom_cmap.set_under((0, 0, 0, 0))

            self.plot_canvas.draw_contours(
                self.get_rotated_background_pixmap(), final_intensity,
                alpha=self.alpha, cmap=custom_cmap, levels=levels,
                units=self.intensity_unit_input.text(), extend='min',
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

    # -------------------------------------------------------------------
    # Highlight levels  
    # -------------------------------------------------------------------

    def apply_highlights(self):
        """Draw highlight contour lines and labels at specified dose values.

        Called from the UI button and also from DraggableCanvas.mouseReleaseEvent
        after a drag .  The base heatmap/contour is
        re-drawn first, then the highlight contours are added on top so
        both contour lines and labels are recomputed at the correct
        (potentially dragged) position.
        """
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return

        highlight_text = self.highlight_input.text().strip()
        if not highlight_text:
            if self._current_vis_mode == "contour":
                self.show_contours()
            else:
                self.show_heatmap()
            return

        try:
            highlight_values = [float(x.strip()) for x in highlight_text.split(",") if x.strip()]
        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Invalid highlight values. Please enter comma-separated numbers.")
            return

        final_intensity = self.get_final_intensity_array()
        if final_intensity is None:
            QMessageBox.warning(self, "Warning", "Could not generate intensity data.")
            return

        # Store highlight settings for re-application after drag
        self.last_highlight_values = highlight_values
        self.last_highlight_color_mode = self.color_combo.currentText()

        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        base_img = self.get_rotated_background_pixmap()

        if self._current_vis_mode == "contour":
            self.plot_canvas.draw_contours(
                base_img, final_intensity, alpha=self.alpha, cmap=self.cmap,
                levels=20, units=intensity_units,
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units,
            )
        else:
            self.plot_canvas.draw_heatmap(
                base_img, final_intensity, alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax, units=intensity_units,
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units,
            )

        image = base_img.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = image.width(), image.height()
        extent = [-width / 2, width / 2, -height / 2, height / 2]

        rows, cols = final_intensity.shape
        X = np.linspace(extent[0] + self.plot_canvas.intensity_offset_x,
                        extent[1] + self.plot_canvas.intensity_offset_x, cols)
        Y = np.linspace(extent[2] + self.plot_canvas.intensity_offset_y,
                        extent[3] + self.plot_canvas.intensity_offset_y, rows)
        xx, yy = np.meshgrid(X, Y)

        intensity_for_contour = np.flipud(final_intensity)

        if self.color_combo.currentText() == "White Only":
            colors = ["white"] * len(highlight_values)
        else:
            base_colors = ["white", "black", "red", "yellow", "green", "cyan"]
            colors = [base_colors[i % len(base_colors)] for i in range(len(highlight_values))]

        legend_handles = []

        for i, value in enumerate(highlight_values):
            cs = self.plot_canvas.ax.contour(
                xx, yy, intensity_for_contour,
                levels=[value], colors=[colors[i]], linewidths=2,
            )
            self.plot_canvas.ax.clabel(cs, inline=True, fmt=f"{value}", fontsize=10)
            line = mlines.Line2D([], [], color=colors[i], linewidth=2, label=f"{value}")
            legend_handles.append(line)

        if legend_handles:
            self.plot_canvas.ax.legend(handles=legend_handles, loc="best")

        self.plot_canvas.draw()

    # -------------------------------------------------------------------
    # Axis scale / units
    # -------------------------------------------------------------------

    def apply_units_and_scale(self):
        try:
            scale_x = float(self.scale_x_input.text())
            scale_y = float(self.scale_y_input.text())
            distance_units = self.scale_unit_input.text()
            intensity_units = self.intensity_unit_input.text()

            self.real_scale_x = scale_x
            self.real_scale_y = scale_y

            if not self.original_pixmap or self.table_widget.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "Load image and data first")
                return

            final_intensity = self.get_final_intensity_array()

            if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
                self.plot_canvas.draw_contours(
                    self.get_rotated_background_pixmap(), final_intensity,
                    alpha=self.alpha, cmap=self.cmap,
                    levels=np.linspace(self.vmin_spin.value(), self.vmax_spin.value(), 20),
                    units=intensity_units, scale_x=scale_x, scale_y=scale_y,
                    distance_units=distance_units
                )
            else:
                self.plot_canvas.draw_heatmap(
                    self.get_rotated_background_pixmap(), final_intensity,
                    alpha=self.alpha, cmap=self.cmap,
                    vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value(),
                    units=intensity_units, scale_x=scale_x, scale_y=scale_y,
                    distance_units=distance_units
                )

        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numeric values for scales.")

    def update_display(self):
        """Re-draw the current visualisation after alpha/cmap change."""
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            return

        if self.tab_widget.currentIndex() == 2:
            if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
                self.show_contours()
            else:
                self.show_heatmap()
        elif self.tab_widget.currentIndex() == 3:
            self.show_grid_overlay()

    # -------------------------------------------------------------------
    # Save blended image
    # -------------------------------------------------------------------

    def save_blended_image(self):
        if not self.plot_canvas.figure:
            return

        dpi = self.dpi_spin.value()
        width_inches = self.width_spin.value()
        height_inches = self.height_spin.value()

        if (
                getattr(self, "source_type", None) == "pdf"
                and getattr(self, "screen_dpi", None) is not None
                and dpi > self.screen_dpi
        ):
            try:
                self._export_with_high_res_pdf(dpi)
            except Exception as e:
                QMessageBox.warning(self, "Export error", f"High-res export failed:\n{e}")
            return

        # Ask for a file path FIRST — avoid resizing the live figure (and
        # causing a visible flash) if the user is going to cancel.
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Blended Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;PDF Files (*.pdf);;All Files (*)"
        )
        if not file_name:
            return

        # Temporarily resize the figure for export, then restore
        fig = self.plot_canvas.figure
        orig_size = fig.get_size_inches().copy()
        orig_dpi = fig.get_dpi()

        fig.set_size_inches(width_inches, height_inches)
        fig.set_dpi(dpi)

        try:
            ext = os.path.splitext(file_name)[1].lower().lstrip('.')
            if ext in ('svg', 'pdf'):
                fig.savefig(file_name, bbox_inches=None, pad_inches=0, format=ext)
            else:
                fig.savefig(
                    file_name, dpi=dpi, bbox_inches=None, pad_inches=0,
                    transparent=(ext == 'png')
                )
        finally:
            fig.set_size_inches(orig_size)
            fig.set_dpi(orig_dpi)
            self.plot_canvas.draw()

    # -------------------------------------------------------------------
    # Grid overlay
    # -------------------------------------------------------------------

    def save_grid_image(self):
        if not self.grid_canvas.figure:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Grid Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        if file_name:
            # Use the rotated/cropped background size (matches what's drawn)
            bg = self.get_rotated_background_pixmap()
            dpi = self.grid_canvas.figure.get_dpi()
            width = bg.width() / dpi
            height = bg.height() / dpi
            self.grid_canvas.figure.set_size_inches(width, height)

            use_transparency = not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'))

            self.grid_canvas.figure.savefig(
                file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent=use_transparency
            )

    def show_grid_overlay(self):
        if not self.original_pixmap:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return

        try:
            arr_rgb = _pixmap_to_rgb_array(self.get_rotated_background_pixmap())
            height, width = arr_rgb.shape[:2]

            self.grid_canvas.ax.clear()
            self.grid_canvas.ax.imshow(arr_rgb, aspect='equal', extent=[0, width, 0, height], origin='upper')

            cols = self.grid_nx_spin.value()
            rows = self.grid_ny_spin.value()

            x = np.linspace(0, width, cols + 1)
            y = np.linspace(0, height, rows + 1)

            if self.grid_type_combo.currentText() == "Points":
                xx, yy = np.meshgrid(x, y)
                self.grid_canvas.ax.plot(xx, yy, 'r.', markersize=2)
            else:
                for xi in x:
                    self.grid_canvas.ax.plot([xi, xi], [0, height], 'r:', linewidth=0.5)
                for yi in y:
                    self.grid_canvas.ax.plot([0, width], [yi, yi], 'r:', linewidth=0.5)

            self.grid_canvas.ax.set_xlim([0, width])
            self.grid_canvas.ax.set_ylim([0, height])
            self.grid_canvas.draw()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show grid overlay: {str(e)}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------------
    # Integration hooks
    # -------------------------------------------------------------------

    def get_image_annotations(self):
        """Retrieve annotations from the Image tab for overlay integration."""
        markers = [(pt.x(), pt.y()) for pt in self.image_canvas.get_markers()]
        lines = [((s.x(), s.y()), (e.x(), e.y())) for s, e in self.image_canvas.get_lines()]
        free_draw = [[(pt.x(), pt.y()) for pt in stroke] for stroke in self.image_canvas.get_free_draw()]
        texts = [(pt.x(), pt.y(), txt) for pt, txt in self.image_canvas.get_texts()]
        return {'markers': markers, 'lines': lines, 'free_draw': free_draw, 'texts': texts}

    _MAX_EXPORT_PIXELS = 8192  # hard cap per axis for high-res PDF export

    def _export_with_high_res_pdf(self, dpi: int):
        """Re-open the source PDF and render at the requested *dpi*, then
        export via a **temporary** Matplotlib Figure so the on-screen
        Blended-tab plot is not destroyed.

        The render is capped at ``_MAX_EXPORT_PIXELS`` per axis to avoid
        out-of-memory on very large pages at very high DPI.
        """
        if not self.source_path or self.source_type != "pdf":
            raise RuntimeError("No PDF source available for high-res export.")

        fitz = _get_fitz()
        doc = fitz.open(self.source_path)
        if doc.page_count == 0:
            doc.close()
            raise RuntimeError("PDF has no pages.")

        page = doc.load_page(0)
        page_rect = page.rect  # points (1/72 in)

        page_width_in = page_rect.width / 72.0
        page_height_in = page_rect.height / 72.0
        if page_width_in <= 0 or page_height_in <= 0:
            page_width_in = page_height_in = 8.0  # fallback

        # Background pixels required for this dpi, capped
        target_px_w = min(int(page_width_in * dpi), self._MAX_EXPORT_PIXELS)
        target_px_h = min(int(page_height_in * dpi), self._MAX_EXPORT_PIXELS)

        # Compute scale from points to pixels
        scale_x = target_px_w / page_rect.width
        scale_y = target_px_h / page_rect.height
        scale = min(scale_x, scale_y)

        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))

        if pix.alpha:
            fmt = QImage.Format.Format_RGBA8888
        else:
            fmt = QImage.Format.Format_RGB888

        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, fmt).copy()
        bg_pixmap = QPixmap.fromImage(qimg)
        doc.close()
        del pix, qimg  # free intermediate buffers

        # Prompt for save path first — bail out before heavy work if cancelled
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save High-Resolution Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;TIFF Image (*.tif);;PDF (*.pdf)"
        )
        if not save_path:
            return

        # Build a *temporary* figure so we don't destroy the on-screen plot
        export_fig = Figure(dpi=dpi)
        export_fig.set_size_inches(page_width_in, page_height_in)
        ax = export_fig.add_subplot(111)

        arr_rgb = _pixmap_to_rgb_array(bg_pixmap)
        h, w = arr_rgb.shape[:2]
        extent = [-w / 2, w / 2, -h / 2, h / 2]
        ax.imshow(arr_rgb, extent=extent, origin="upper", aspect="equal")
        del arr_rgb  # free the large array early

        # Re-plot intensity overlay
        self._redraw_intensity_overlay(ax, extent)

        ext = os.path.splitext(save_path)[1].lower()
        if ext in ('.svg', '.pdf'):
            export_fig.savefig(save_path, bbox_inches="tight", format=ext.lstrip('.'))
        else:
            export_fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        plt.close(export_fig)  # release memory
        QMessageBox.information(self, "Saved", f"High-resolution export saved to:\n{save_path}")

    # -------------------------------------------------------------------
    # Session save / restore  (JSON project file)
    # -------------------------------------------------------------------

    def save_session(self):
        """Save the current session to a JSON project file."""
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "scatter_session.json", "JSON Files (*.json)")
        if not path:
            return

        session = {}
        session['_note'] = 'Reload the image manually; this file stores settings only.'

        rows = self.table_widget.rowCount()
        cols = self.table_widget.columnCount()
        table_data = []
        for r in range(rows):
            row = []
            for c in range(cols):
                item = self.table_widget.item(r, c)
                row.append(item.text() if item else "")
            table_data.append(row)
        session['table_data'] = table_data

        if self.crop_rect:
            session['crop_rect'] = [self.crop_rect.x(), self.crop_rect.y(),
                                    self.crop_rect.width(), self.crop_rect.height()]
        else:
            session['crop_rect'] = None

        session['rotation_angle'] = self.current_rotation_angle
        session['alpha'] = self.alpha
        session['cmap'] = self.cmap_combo.currentText()
        session['vis_mode'] = getattr(self, '_current_vis_mode', 'heatmap')
        session['highlight_input'] = self.highlight_input.text()
        session['highlight_color_mode'] = self.color_combo.currentText()
        session['scale_x'] = self.scale_x_input.text()
        session['scale_y'] = self.scale_y_input.text()
        session['scale_unit'] = self.scale_unit_input.text()
        session['intensity_unit'] = self.intensity_unit_input.text()

        ppu = self.image_canvas.get_pixels_per_unit()
        session['pixels_per_unit'] = ppu
        session['scale_calibration_unit'] = self.image_canvas.get_scale_unit()

        annotations = self.get_image_annotations()
        session['annotations'] = annotations

        session['overlay_offset_x'] = self.plot_canvas.intensity_offset_x
        session['overlay_offset_y'] = self.plot_canvas.intensity_offset_y
        session['dark_theme'] = self._dark_theme_on

        try:
            with open(path, 'w') as f:
                json.dump(session, f, indent=2)
            QMessageBox.information(self, "Saved", f"Session saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save session:\n{e}")

    def load_session(self):
        """Load a session from a JSON project file."""
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", "JSON Files (*.json);;All Files (*)")
        if not path:
            return

        try:
            with open(path) as f:
                session = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load session:\n{e}")
            return

        table_data = session.get('table_data', [])
        if table_data:
            rows = len(table_data)
            cols = max(len(row) for row in table_data) if rows else 0
            self.table_widget.blockSignals(True)
            self.table_widget.setRowCount(rows)
            self.table_widget.setColumnCount(cols)
            for r, row in enumerate(table_data):
                for c, val in enumerate(row):
                    self.table_widget.setItem(r, c, QTableWidgetItem(val))
            self.table_widget.blockSignals(False)

        angle = session.get('rotation_angle', 0)
        self.rotation_slider.setValue(angle)

        cr = session.get('crop_rect')
        if cr:
            self.crop_rect = QRect(*cr)
        else:
            self.crop_rect = None

        self.alpha = session.get('alpha', 0.6)
        self.alpha_slider.setValue(int(self.alpha * 100))
        cmap_text = session.get('cmap', 'jet')
        idx = self.cmap_combo.findText(cmap_text)
        if idx >= 0:
            self.cmap_combo.setCurrentIndex(idx)

        self.highlight_input.setText(session.get('highlight_input', '1, 6, 15'))
        hc = session.get('highlight_color_mode', 'Default Colors')
        idx = self.color_combo.findText(hc)
        if idx >= 0:
            self.color_combo.setCurrentIndex(idx)

        self.scale_x_input.setText(session.get('scale_x', '1.0'))
        self.scale_y_input.setText(session.get('scale_y', '1.0'))
        self.scale_unit_input.setText(session.get('scale_unit', 'cm'))
        self.intensity_unit_input.setText(session.get('intensity_unit', 'uGy/h'))

        ppu = session.get('pixels_per_unit')
        if ppu is not None:
            self.image_canvas._pixels_per_unit = ppu
            self.image_canvas._scale_unit = session.get('scale_calibration_unit', 'cm')
        self._update_scale_readout()

        self.plot_canvas.intensity_offset_x = session.get('overlay_offset_x', 0)
        self.plot_canvas.intensity_offset_y = session.get('overlay_offset_y', 0)

        if session.get('dark_theme', False):
            if not self._dark_theme_on:
                self._toggle_theme()
        else:
            if self._dark_theme_on:
                self._toggle_theme()

        self._invalidate_bg_cache()
        self.invalidate_cache()
        self.update_intensity_preview()
        self.set_grid_spinboxes_from_data()

        QMessageBox.information(self, "Loaded",
                                "Session restored.\nIf you had an image loaded previously, please reload it now.")

    # -------------------------------------------------------------------
    # Dark / Light theme toggle
    # -------------------------------------------------------------------

    def _toggle_theme(self):
        self._dark_theme_on = not self._dark_theme_on
        if self._dark_theme_on:
            self.setStyleSheet(_DARK_STYLESHEET)
        else:
            self.setStyleSheet("")  # revert to system default

    # -------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------

    def closeEvent(self, event):
        """Release heavy resources on window close."""
        try:
            if hasattr(self, 'plot_canvas') and self.plot_canvas.figure:
                self.plot_canvas.figure.clear()
                plt.close(self.plot_canvas.figure)

            if hasattr(self, 'grid_canvas') and self.grid_canvas.figure:
                self.grid_canvas.figure.clear()
                plt.close(self.grid_canvas.figure)
        except Exception:
            pass

        self.original_pixmap = None
        self.current_pixmap = None
        self.cached_intensity_array = None
        self._bg_cache = None

        gc.collect()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Dark theme stylesheet
# ---------------------------------------------------------------------------

_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
}
QTabWidget::pane {
    border: 1px solid #444;
    background: #2b2b2b;
}
QTabBar::tab {
    background: #3c3c3c;
    color: #e0e0e0;
    padding: 6px 14px;
    border: 1px solid #555;
    border-bottom: none;
}
QTabBar::tab:selected {
    background: #2b2b2b;
    border-bottom: 2px solid #4fc3f7;
}
QGroupBox {
    border: 1px solid #555;
    margin-top: 6px;
    padding-top: 10px;
    color: #e0e0e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 3px;
}
QPushButton {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 5px 12px;
    border-radius: 3px;
}
QPushButton:hover { background-color: #4a4a4a; }
QPushButton:pressed { background-color: #555; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
    padding: 3px;
}
QTableWidget {
    background-color: #333;
    color: #e0e0e0;
    gridline-color: #555;
}
QHeaderView::section {
    background-color: #3c3c3c;
    color: #e0e0e0;
    border: 1px solid #555;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #555;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #4fc3f7;
    width: 14px;
    margin: -4px 0;
    border-radius: 7px;
}
QLabel { color: #e0e0e0; }
QCheckBox { color: #e0e0e0; }
QRadioButton { color: #e0e0e0; }
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    # Start background preloading of pandas/scipy/fitz/PIL now that the
    # window is visible.  By the time the user clicks "Load CSV" or
    # "Load Image (PDF)", these will already be imported and ready.
    QTimer.singleShot(0, start_background_preload)
    sys.exit(app.exec())
