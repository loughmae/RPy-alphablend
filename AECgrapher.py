import sys, os, pathlib, logging, re
import numpy as np
import pydicom
from typing import List, Dict, Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QColorDialog
import pyqtgraph as pg
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


def mpl_canvas_to_qpixmap(canvas):
    buf, (w, h) = canvas.print_to_buffer()
    img = QtGui.QImage(buf, w, h, QtGui.QImage.Format.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(img)


class TubeCanvas(FigureCanvasQTAgg):
    def __init__(self, width_px, height_px, parent=None):
        fig = Figure(figsize=(width_px/100, height_px/100), dpi=100)
        super().__init__(fig)
        self.ax = fig.add_subplot(111)
        fig.patch.set_alpha(0)
        self.ax.set_facecolor("none")
        self.setStyleSheet("background: transparent;")

    def draw_curve_autoscaled(
        self, xs, ys, scout_shape, ser_label="",
        line_color='tab:blue', line_width=2,
        show_axes=False, title_fontsize=10, axes_color="#000000"
    ):
        h, w = scout_shape
        xs = np.asarray(xs)
        ys = np.asarray(ys)
        self.ax.cla()
        if np.all(np.isnan(ys)) or len(xs) == 0:
            self.ax.set_title("No valid mA data", fontsize=title_fontsize)
            self.draw()
            return
        is_landscape = w >= h
        if is_landscape:
            x_plot = np.linspace(0, w-1, len(xs))
            y_plot = np.interp(ys, [np.nanmin(ys), np.nanmax(ys)], [h-1, 0])
            self.ax.plot(x_plot, y_plot, color=line_color, lw=line_width)
            self.ax.set_xlim(0, w-1)
            self.ax.set_ylim(h-1, 0)
        else:
            y_plot = np.linspace(0, h-1, len(xs))
            x_plot = np.interp(ys, [np.nanmin(ys), np.nanmax(ys)], [0, w-1])
            self.ax.plot(x_plot, y_plot, color=line_color, lw=line_width)
            self.ax.set_xlim(0, w-1)
            self.ax.set_ylim(h-1, 0)
        self.ax.axis('on' if show_axes else 'off')
        self.ax.set_title(ser_label, fontsize=title_fontsize, color=axes_color)
        for spine in self.ax.spines.values():
            spine.set_color(axes_color)
        self.ax.xaxis.label.set_color(axes_color)
        self.ax.yaxis.label.set_color(axes_color)
        self.ax.tick_params(axis='both', colors=axes_color)
        self.draw()


class ResizableOverlay(QtWidgets.QGraphicsPixmapItem):
    def __init__(self, xs, ys, scout_shape, ser_label="",
                 color='tab:blue', linewidth=2, axes=True, alpha=0.9, axes_color="#000000", parent=None):
        self.xs = xs
        self.ys = ys
        self.scout_shape = scout_shape
        self.ser_label = ser_label
        self.color = color
        self.linewidth = linewidth
        self.axes = axes
        self.alpha = alpha
        self.axes_color = axes_color
        h, w = scout_shape
        width = w
        height = max(h // 4, 40)
        self.current_width = width
        self.current_height = height
        pixmap = self.make_plot_pixmap(width, height)
        super().__init__(pixmap, parent)
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setOpacity(self.alpha)
        self.resizing = False
        self.handle_size = 14

    def make_plot_pixmap(self, width, height):
        canvas = TubeCanvas(width, height)
        canvas.draw_curve_autoscaled(
            self.xs, self.ys, (height, width),
            self.ser_label, line_color=self.color, line_width=self.linewidth,
            show_axes=self.axes, axes_color=self.axes_color
        )
        pm = mpl_canvas_to_qpixmap(canvas)
        return pm

    def update_style(self, color, linewidth, axes, label, alpha, axes_color=None):
        if axes_color is not None:
            self.axes_color = axes_color
        self.color = color
        self.linewidth = linewidth
        self.axes = axes
        self.ser_label = label
        self.alpha = alpha
        pm = self.make_plot_pixmap(self.current_width, self.current_height)
        self.setPixmap(pm)
        self.setOpacity(self.alpha)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addRect(self.boundingRect())
        return path

    def hoverMoveEvent(self, event):
        rect = self.boundingRect()
        pos = event.pos()
        sz = self.handle_size
        if (rect.right() - sz < pos.x() < rect.right() and
            rect.bottom() - sz < pos.y() < rect.bottom()):
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)
        rect = self.boundingRect()
        sz = self.handle_size
        painter.setBrush(QtGui.QColor(255, 255, 255, 180))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        painter.drawRect(
            int(rect.right() - sz),
            int(rect.bottom() - sz),
            int(sz),
            int(sz)
        )

    def mousePressEvent(self, event):
        rect = self.boundingRect()
        pos = event.pos()
        sz = self.handle_size
        if (rect.right() - sz < pos.x() < rect.right() and
            rect.bottom() - sz < pos.y() < rect.bottom()):
            self.resizing = True
            self._start_scene_pos = event.scenePos()
            self._orig_width = self.current_width
            self._orig_height = self.current_height
            event.accept()
        else:
            self.resizing = False
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.resizing:
            diff = event.scenePos() - self._start_scene_pos
            new_width = max(40, int(self._orig_width + diff.x()))
            new_height = max(20, int(self._orig_height + diff.y()))
            self.current_width = new_width
            self.current_height = new_height
            pm = self.make_plot_pixmap(new_width, new_height)
            self.setPixmap(pm)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.resizing = False
        super().mouseReleaseEvent(event)


class OverlayControlPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Overlay Style", parent)
        layout = QtWidgets.QFormLayout(self)
        self.colorBtn = QtWidgets.QPushButton("Pick Plot Color")
        self.lineWidthSpin = QtWidgets.QSpinBox()
        self.lineWidthSpin.setRange(1, 10)
        self.lineWidthSpin.setValue(3)
        self.alphaSlider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.alphaSlider.setRange(10, 100)
        self.alphaSlider.setValue(90)
        self.axesCheck = QtWidgets.QCheckBox("Show Axes")
        self.axesCheck.setChecked(True)
        self.titleEdit = QtWidgets.QLineEdit("mA Curve")
        self.axesColorBtn = QtWidgets.QPushButton("Pick Axes Color")
        self.axesColorBtn.setStyleSheet("background-color: black;")
        self.applyBtn = QtWidgets.QPushButton("Apply to Selected Overlay")
        layout.addRow("Plot Color:", self.colorBtn)
        layout.addRow("Line Width:", self.lineWidthSpin)
        layout.addRow("Opacity (%):", self.alphaSlider)
        layout.addRow(self.axesCheck)
        layout.addRow("Title:", self.titleEdit)
        layout.addRow("Axes Color:", self.axesColorBtn)
        layout.addRow(self.applyBtn)


class Series:
    def __init__(self, uid: str):
        self.uid = uid
        self.instances: List[pydicom.dataset.FileDataset] = []
    def add(self, ds): self.instances.append(ds)
    @property
    def desc(self): return self.instances[0].get("SeriesDescription", "N/A") if self.instances else "N/A"
    @property
    def num(self):
        try: return int(self.instances[0].get("SeriesNumber", 9999))
        except: return 9999
    def is_scout(self) -> bool:
        if not self.instances: return False
        ds0 = self.instances[0]
        SCOUT_RX = re.compile(r"(scout|topogram|localiz)", re.I)
        if SCOUT_RX.search(ds0.get("SeriesDescription", "")) or SCOUT_RX.search(",".join(ds0.get("ImageType", []))):
            return True
        if any(tok.upper() == "LOCALIZER" for tok in ds0.get("ImageType", [])):
            return True
        if len(self.instances) == 1 and ds0.get("InstanceNumber", 1) == 0:
            return True
        return False
    def pixel(self): return self.instances[0].pixel_array.astype(np.int16)
    def ma_curve(self) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for ds in sorted(self.instances, key=lambda d: int(d.get("InstanceNumber", 0))):
            xs.append(int(ds.get("InstanceNumber", len(xs))))
            mA = None
            for tag in [(0x0018, 0x1151), (0x0018, 0x9330)]:
                if tag in ds and ds[tag].value not in ("", None):
                    mA = float(ds[tag].value)
                    break
            ys.append(mA if mA is not None else np.nan)
        return np.asarray(xs), np.asarray(ys)


class Scanner:
    def __init__(self, root: pathlib.Path):
        self.root = root
        self.studies: Dict[str, Dict[str, Series]] = {}
    def scan(self):
        self.studies.clear()
        for f in self.root.rglob("*"):
            if not f.is_file() or not is_dicom(f): continue
            ds = safe_read(f)
            if ds is None: continue
            study = ds.get("StudyInstanceUID", "UNK_STUDY")
            ser = ds.get("SeriesInstanceUID", "UNK_SER")
            self.studies.setdefault(study, {}).setdefault(ser, Series(ser)).add(ds)
        for ser in self.iter_series():
            ser.instances.sort(key=lambda d: int(d.get("InstanceNumber", 0)))
    def iter_series(self):
        for smap in self.studies.values():
            for ser in smap.values():
                yield ser

def is_dicom(path: pathlib.Path) -> bool:
    try:
        with path.open("rb") as fh:
            fh.seek(128)
            if fh.read(4) == b"DICM":
                return True
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False

def safe_read(path: pathlib.Path) -> Optional[pydicom.dataset.FileDataset]:
    try:
        return pydicom.dcmread(str(path), force=True)
    except Exception:
        return None

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Integrated CT-AEC Viewer")
        self.resize(1400, 900)
        splitter = QtWidgets.QSplitter()
        self.setCentralWidget(splitter)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Study / Series"])
        self.tree.setMaximumWidth(350)
        splitter.addWidget(self.tree)
        self.scout_imv = pg.ImageView(view=pg.PlotItem())
        splitter.addWidget(self.scout_imv)
        splitter.setSizes([300, 1200])
        self.overlays = []
        self.scanner = None
        self.ser_scout = None
        self.ser_axial = None
        self.selected_overlay = None

        # Toolbar
        tb = self.addToolBar("Main")
        tb.addAction(QtGui.QAction("Open Folder", self, triggered=self.open_folder))
        tb.addAction(QtGui.QAction("⟳ 90°", self, triggered=lambda: self.rotate_scout(90)))
        tb.addAction(QtGui.QAction("⟲ 90°", self, triggered=lambda: self.rotate_scout(-90)))
        tb.addAction(QtGui.QAction("Add Overlays", self, triggered=self.add_overlay_dialog))
        tb.addAction(QtGui.QAction("Save View", self, triggered=self.save_view))

        # Overlay style dock/panel
        self.ctrl_panel = OverlayControlPanel()
        dock = QtWidgets.QDockWidget('Overlay Controls')
        dock.setWidget(self.ctrl_panel)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self.ctrl_panel.colorBtn.clicked.connect(self.pick_color)
        self.ctrl_panel.applyBtn.clicked.connect(self.apply_overlay_style)
        self.ctrl_panel.lineWidthSpin.valueChanged.connect(self.apply_overlay_style)
        self.ctrl_panel.axesCheck.clicked.connect(self.apply_overlay_style)
        self.ctrl_panel.alphaSlider.sliderMoved.connect(self.apply_overlay_style)
        self.ctrl_panel.titleEdit.editingFinished.connect(self.apply_overlay_style)
        self.ctrl_panel.axesColorBtn.clicked.connect(self.pick_axes_color)

        self.tree.itemSelectionChanged.connect(self._on_tree_select)

    def patient_coord_to_scout_pixel(self, patient_pos, scout_dcm):
        # patient_pos: (x, y, z) world mm coordinate
        # scout_dcm: pydicom Dataset for scout
        img_ori = np.array(scout_dcm.ImageOrientationPatient, dtype=float)
        row_cos = img_ori[:3]
        col_cos = img_ori[3:]
        origin = np.array(scout_dcm.ImagePositionPatient, dtype=float)
        spacing = np.array(scout_dcm.PixelSpacing, dtype=float)
        # Find projection of vector from origin to patient_pos onto row & col axes
        v = np.array(patient_pos) - origin
        col_px = np.dot(v, col_cos) / spacing[1]
        row_px = np.dot(v, row_cos) / spacing[0]
        return (col_px, row_px)

    def pick_color(self):
        color = QColorDialog.getColor(QtGui.QColor(self.ctrl_panel.colorBtn.palette().button().color()), self)
        if color.isValid():
            self.ctrl_panel.colorBtn.setStyleSheet(f"background-color: {color.name()};")
            self.apply_overlay_style()

    def pick_axes_color(self):
        color = QColorDialog.getColor(QtGui.QColor(self.ctrl_panel.axesColorBtn.palette().button().color()), self)
        if color.isValid():
            self.ctrl_panel.axesColorBtn.setStyleSheet(f"background-color: {color.name()};")
            self.apply_overlay_style()

    def apply_overlay_style(self):
        ov = self.selected_overlay
        if ov is None:
            return
        color = self.ctrl_panel.colorBtn.palette().button().color().name()
        linewidth = self.ctrl_panel.lineWidthSpin.value()
        axes = self.ctrl_panel.axesCheck.isChecked()
        label = self.ctrl_panel.titleEdit.text()
        alpha = self.ctrl_panel.alphaSlider.value() / 100.0
        axes_color = self.ctrl_panel.axesColorBtn.palette().button().color().name()
        ov.update_style(color, linewidth, axes, label, alpha, axes_color)

    def select_overlay(self, overlay):
        for ov in self.overlays:
            ov.setZValue(10)
        overlay.setZValue(20)
        self.selected_overlay = overlay
        # Fill panel with this overlay's styles
        self.ctrl_panel.colorBtn.setStyleSheet(f"background-color: {overlay.color};")
        self.ctrl_panel.lineWidthSpin.setValue(overlay.linewidth)
        self.ctrl_panel.axesCheck.setChecked(overlay.axes)
        self.ctrl_panel.titleEdit.setText(overlay.ser_label)
        self.ctrl_panel.alphaSlider.setValue(int(overlay.alpha * 100))
        self.ctrl_panel.axesColorBtn.setStyleSheet(f"background-color: {overlay.axes_color};")

    def open_folder(self):
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder", os.getcwd())
        if not root: return
        self.scanner = Scanner(pathlib.Path(root))
        self.scanner.scan()
        self._populate_tree()

    def _populate_tree(self):
        self.tree.clear()
        if not self.scanner: return
        for study, ser_map in self.scanner.studies.items():
            n_study = QtWidgets.QTreeWidgetItem([f"Study {study}"])
            self.tree.addTopLevelItem(n_study)
            for ser in sorted(ser_map.values(), key=lambda s: (not s.is_scout(), s.num)):
                label = ("📐 " if ser.is_scout() else "") + f"{ser.num:03d} | {ser.desc}"
                n_ser = QtWidgets.QTreeWidgetItem([label])
                n_ser.setData(0, QtCore.Qt.ItemDataRole.UserRole, ser)
                n_study.addChild(n_ser)
            n_study.setExpanded(True)
        self.tree.expandAll()

    def _remove_all_overlays(self):
        scene = self.scout_imv.getView().scene()
        for overlay in self.overlays:
            scene.removeItem(overlay)
        self.overlays.clear()
        self.selected_overlay = None
        # Remove all old slice lines
        for item in list(self.scout_imv.getView().allChildItems()):
            if isinstance(item, pg.InfiniteLine):
                self.scout_imv.getView().removeItem(item)

    def _on_tree_select(self):
        sel = self.tree.selectedItems()
        if not sel:
            return
        ser = sel[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        if ser.is_scout():
            self.ser_scout = ser
            self.ser_axial = None
            self._load_scout()
            self._remove_all_overlays()
        else:
            self.ser_axial = ser
            if self.ser_scout:
                self._load_scout()
                self.add_overlay_curve(self.ser_axial.ma_curve(), self.ser_axial.desc, self.ser_axial)

    def _load_scout(self):
        arr = np.stack([ds.pixel_array for ds in self.ser_scout.instances])
        self.scout_imv.setImage(arr, levels=(arr.min(), arr.max()))

    def add_overlay_curve(self, curve, label, series):
        xs, ys = curve
        arr = np.stack([ds.pixel_array for ds in self.ser_scout.instances])
        shape = arr.shape[-2:]
        h, w = shape
        overlay_width = w
        overlay_height = h // 3
        axes_color = self.ctrl_panel.axesColorBtn.palette().button().color().name()
        overlay = ResizableOverlay(xs, ys, shape, ser_label=label,
                                  color=self.ctrl_panel.colorBtn.palette().button().color().name(),
                                  linewidth=self.ctrl_panel.lineWidthSpin.value(),
                                  axes=self.ctrl_panel.axesCheck.isChecked(),
                                  axes_color=axes_color)
        overlay.setZValue(10)
        overlay.setOpacity(self.ctrl_panel.alphaSlider.value() / 100.0)
        overlay.setPos(0, 0)
        self.scout_imv.getView().scene().addItem(overlay)
        self.overlays.append(overlay)
        orig_mouse_press = overlay.mousePressEvent
        def new_mouse_press(event, ov=overlay):
            self.select_overlay(ov)
            orig_mouse_press(event)
        overlay.mousePressEvent = new_mouse_press
        self.select_overlay(overlay)
        self.draw_slice_position_lines(xs, ys, shape, series=series, color="#FF6600")

    def add_overlay_dialog(self):
        if not self.ser_scout or not self.scanner:
            QtWidgets.QMessageBox.information(self, "No scout", "Select a scout in the tree first.")
            return
        self._remove_all_overlays()
        for study, ser_map in self.scanner.studies.items():
            for ser in ser_map.values():
                if not ser.is_scout():
                    self.add_overlay_curve(ser.ma_curve(), ser.desc, ser)

    def draw_slice_position_lines(self, xs, ys, scout_shape, series, color="#FF6600"):
        # Find the DICOM for scout (series.instances[0]) and axial (series)
        if not series.instances:
            return
        # Find beginning and end slice DICOMs
        first_dcm = series.instances[0]
        last_dcm = series.instances[-1]

        if hasattr(self.ser_scout, "instances") and self.ser_scout.instances:
            scout_dcm = self.ser_scout.instances[0]
        else:
            return

        for dcm in [first_dcm, last_dcm]:
            patient_pos = [float(x) for x in dcm.ImagePositionPatient]
            try:
                col_px, row_px = self.patient_coord_to_scout_pixel(patient_pos, scout_dcm)
            except Exception:
                continue  # fallback if bad DICOM geometry

            # For most orientation, show vertical line at col_px (along Y)
            line = pg.InfiniteLine(
                pos=col_px, angle=90,
                pen=pg.mkPen(color=color, style=QtCore.Qt.PenStyle.DashLine, width=2)
            )
            line.setZValue(9)
            self.scout_imv.getView().addItem(line)
            text = pg.TextItem("Begin" if dcm is first_dcm else "End", anchor=(0.5, 0.9), color=color)
            text.setPos(col_px, 5)  # y=5 keeps text near image top
            text.setZValue(10)
            self.scout_imv.getView().addItem(text)

    def rotate_scout(self, deg):
        img_item = self.scout_imv.imageItem
        if img_item.image is None:
            return
        rotated = np.rot90(img_item.image, k=(-deg // 90) % 4)
        self.scout_imv.setImage(rotated, levels=(rotated.min(), rotated.max()))

    def save_view(self):
        scene = self.scout_imv.getView().scene()
        rect = scene.itemsBoundingRect()
        image = QtGui.QImage(rect.size().toSize(), QtGui.QImage.Format.Format_ARGB32)
        image.fill(QtCore.Qt.GlobalColor.black)
        painter = QtGui.QPainter(image)
        scene.render(painter, QtCore.QRectF(image.rect()), rect)
        painter.end()
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PNG", "scout_with_overlay.png", "PNG (*.png)")
        if fn:
            image.save(fn, "PNG")
            QtWidgets.QMessageBox.information(self, "Saved", f"Exported {fn}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()