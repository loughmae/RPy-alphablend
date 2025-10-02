import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QPixmap, QImage, QTransform
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem, QFileDialog, QLabel, QMessageBox,
    QComboBox, QLineEdit, QGroupBox, QDoubleSpinBox, QSlider, QCheckBox, QSpinBox,
    QDialog, QFormLayout
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure

from scipy.ndimage import zoom
import fitz  # PyMuPDF
import io
from PIL import Image


def get_threat_zone_cmap():
    colors = ['#0033CC', '#00CCCC', '#00CC44', '#FFDD00', '#FF8800', '#FF2222', '#AA00CC']
    return ListedColormap(colors, name='threat_zones')

def get_continuous_dose_cmap():
    colors = ['#0022CC', '#0099CC', '#00CC66', '#CCFF33', '#FFDD00', '#FF8800', '#FF2222']
    return ListedColormap(colors, name='dose_field')


class DraggableCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        self.figure = Figure()
        super().__init__(self.figure)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.dragging = False
        self.last_mouse_pos = None
        self.cbar = None
        self.intensity_offset_x = 0
        self.intensity_offset_y = 0
        self.original_extent = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            # FIXED: Convert to data coordinates properly
            if self.ax.get_xlim() and self.ax.get_ylim():
                bbox = self.ax.bbox
                x_pixel = event.pos().x()
                y_pixel = self.figure.bbox.height - event.pos().y()  # Flip Y

                try:
                    x_data, y_data = self.ax.transData.inverted().transform([x_pixel, y_pixel])
                    self.last_mouse_pos = (x_data, y_data)
                except:
                    self.last_mouse_pos = None
                    self.dragging = False

    def mouseMoveEvent(self, event):
        if self.dragging and self.last_mouse_pos:
            try:
                # Convert current position to data coordinates
                x_pixel = event.pos().x()
                y_pixel = self.figure.bbox.height - event.pos().y()  # Flip Y
                x_data, y_data = self.ax.transData.inverted().transform([x_pixel, y_pixel])

                # FIXED: Calculate movement in data coordinates
                dx = x_data - self.last_mouse_pos[0]  # Fixed: use index [0]
                dy = y_data - self.last_mouse_pos[1]  # Fixed: use index [1]

                # Update offset
                self.intensity_offset_x += dx
                self.intensity_offset_y += dy

                # Handle both images (heatmaps) and collections (contours)
                moved_something = False

                # Handle image objects (heatmaps)
                for img in self.ax.images:
                    arr = img.get_array()
                    if arr is None:
                        continue
                    if hasattr(arr, 'dtype') and arr.dtype == np.uint8:
                        # likely the background RGB image
                        continue

                    extent = img.get_extent()
                    # FIXED: Element-wise addition instead of tuple + float
                    new_extent = (extent[0] + dx, extent[1] + dx, extent[2] + dy, extent[3] + dy)
                    img.set_extent(new_extent)
                    moved_something = True

                # Handle collection objects (contours)
                for collection in self.ax.collections:
                    for path in collection.get_paths():
                        vertices = path.vertices
                        vertices[:, 0] += dx  # X offset
                        vertices[:, 1] += dy  # Y offset
                    moved_something = True

                if moved_something:
                    self.last_mouse_pos = (x_data, y_data)
                    self.draw()

            except Exception as e:
                print(f"Drag error: {e}")
                self.dragging = False

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.last_mouse_pos = None

    def draw_heatmap(self, base_img: QPixmap, intensity_array, alpha=0.6, cmap='jet', interpolation='bilinear',
                     vmin=None, vmax=None, units='', scale_x=1.0, scale_y=1.0, distance_units='pixels'):
        self.figure.clf()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.ax = self.figure.add_subplot(111)

        image = base_img.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = image.width(), image.height()
        buffer = image.constBits()
        buffer.setsize(image.sizeInBytes())
        arr = np.frombuffer(buffer, np.uint8).copy().reshape((height, width, 4))
        arr_rgb = arr[..., :3]

        # FIXED: Use pixel coordinates for extent
        extent = [-width / 2, width / 2, -height / 2, height / 2]
        self.original_extent = extent

        self.ax.imshow(arr_rgb, aspect='equal', extent=extent, origin='lower')

        # FIXED: Apply accumulated offset to intensity overlay with proper syntax
        intensity_extent = [
            extent[0] + self.intensity_offset_x,  # Fixed: element-wise addition
            extent[1] + self.intensity_offset_x,
            extent[2] + self.intensity_offset_y,
            extent[3] + self.intensity_offset_y
        ]

        im = self.ax.imshow(intensity_array, cmap=cmap, alpha=alpha, interpolation=interpolation,
                            extent=intensity_extent, origin='lower', vmin=vmin, vmax=vmax)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])  # FIXED: Not flipped for 'lower' origin
        self._apply_axis_scaling(scale_x, scale_y, distance_units)

        self.cbar = self.figure.colorbar(im, ax=self.ax, orientation='vertical', pad=0.05)
        cbar_label = f'Intensity ({units})' if units else 'Intensity'
        self.cbar.set_label(cbar_label, rotation=270, labelpad=20, fontsize=15, fontweight='bold')
        self.draw()

    def _apply_axis_scaling(self, scale_x, scale_y, distance_units):
        """Apply scaling to axis labels without affecting the visual data"""
        if scale_x != 1.0 or scale_y != 1.0:
            # Get current tick positions
            x_ticks = self.ax.get_xticks()
            y_ticks = self.ax.get_yticks()

            # Create scaled labels
            x_labels = [f'{tick * scale_x:.0f}' for tick in x_ticks]
            y_labels = [f'{tick * scale_y:.0f}' for tick in y_ticks]

            self.ax.set_xticklabels(x_labels)
            self.ax.set_yticklabels(y_labels)

        self.ax.set_xlabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')
        self.ax.set_ylabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')

    def draw_contours(self, base_img: QPixmap, intensity_array, alpha=0.6, cmap='jet', levels=6,
                      interpolation='bilinear', units='', extend='neither', scale_x=1.0, scale_y=1.0,
                      distance_units='pixels'):
        self.figure.clf()
        if self.cbar is not None:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        self.ax = self.figure.add_subplot(111)

        image = base_img.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        width, height = image.width(), image.height()
        buffer = image.constBits()
        buffer.setsize(image.sizeInBytes())
        arr = np.frombuffer(buffer, np.uint8).copy().reshape((height, width, 4))
        arr_rgb = arr[..., :3]

        # FIXED: Use pixel coordinates for extent
        extent = [-width / 2, width / 2, -height / 2, height / 2]
        self.original_extent = extent

        self.ax.imshow(arr_rgb, aspect='equal', extent=extent, origin='lower')

        rows, cols = intensity_array.shape
        # FIXED: Create coordinate arrays for contour with proper element access
        X = np.linspace(extent[0] + self.intensity_offset_x, extent[1] + self.intensity_offset_x, cols)
        Y = np.linspace(extent[2] + self.intensity_offset_y, extent[3] + self.intensity_offset_y, rows)
        xx, yy = np.meshgrid(X, Y)

        if isinstance(levels, int):
            valid_data = intensity_array[~np.isnan(intensity_array)]
            if valid_data.size == 0:
                return  # No data to plot
            vmin, vmax = np.min(valid_data), np.max(valid_data)
            levels = np.linspace(vmin, vmax, levels)

        cs = self.ax.contourf(xx, yy, intensity_array, levels=levels, cmap=cmap, alpha=alpha, extend=extend)

        self.ax.set_xlim(extent[0], extent[1])
        self.ax.set_ylim(extent[2], extent[3])  # FIXED: Not flipped
        self._apply_axis_scaling(scale_x, scale_y, distance_units)

        self.cbar = self.figure.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05)
        cbar_label = f'Intensity ({units})' if units else 'Intensity'
        self.cbar.set_label(cbar_label, rotation=270, labelpad=15, fontsize=14, fontweight='bold')
        self.cbar.ax.tick_params(labelsize=13)
        self.draw()

    def reset_intensity_position(self):
        """Reset the intensity overlay to its original position"""
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
        except:
            pass
        super().closeEvent(event)


class CropDialog(QDialog):
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # Attributes
        self.original_pixmap = None
        self.current_pixmap = None
        self.crop_rect = None  # NEW: Store crop area coordinates
        self.intensity_data = None
        self.alpha = 0.6
        self.cmap = "jet"
        self._current_vis_mode = "heatmap"
        self.cached_intensity_array = None
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None
        self.current_rotation_angle = 0

        self.setWindowTitle("Radiation Protection Scatter Map Generator")
        self.resize(1400, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Image ---
        img_tab = QWidget()
        img_tab_layout = QHBoxLayout(img_tab)
        img_tab_left = QVBoxLayout()
        self.image_label = QLabel("Load an image or PDF to begin")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_tab_left.addWidget(self.image_label)
        img_tab_layout.addLayout(img_tab_left, stretch=3)

        img_tab_right = QVBoxLayout()
        btn_load_img = QPushButton("Load Image/PDF")
        btn_load_img.clicked.connect(self.load_image)
        img_tab_right.addWidget(btn_load_img)

        img_tab_right.addWidget(QLabel("Rotation (applied to view):"))
        self.rotation_slider = QSlider(Qt.Orientation.Horizontal)
        self.rotation_slider.setRange(0, 360)
        self.rotation_slider.valueChanged.connect(self.rotate_image_view)
        self.rotation_label = QLabel("0°")
        self.rotation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_tab_right.addWidget(self.rotation_slider)
        img_tab_right.addWidget(self.rotation_label)

        btn_crop_img = QPushButton("Define Crop Area")
        btn_crop_img.clicked.connect(self.define_crop_area)
        img_tab_right.addWidget(btn_crop_img)

        btn_reset_img = QPushButton("Reset View and Crop")
        btn_reset_img.clicked.connect(self.reset_image)
        img_tab_right.addWidget(btn_reset_img)

        img_tab_right.addStretch(1)
        img_tab_layout.addLayout(img_tab_right, stretch=1)
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

        # Preview Mode and Box
        self.preview_mode_combo = QComboBox()
        self.preview_mode_combo.addItems(["Heatmap", "Contour Map"])
        intensity_right.addWidget(QLabel("Preview Mode:"))
        intensity_right.addWidget(self.preview_mode_combo)

        self.preview_canvas = FigureCanvasQTAgg(Figure(figsize=(2, 2)))
        self.preview_canvas.setMinimumSize(150, 150)
        intensity_right.addWidget(self.preview_canvas)
        self.preview_mode_combo.currentTextChanged.connect(self.update_intensity_preview)

        intensity_tab = QWidget()
        intensity_tab.setLayout(intensity_layout)
        intensity_layout.addLayout(intensity_right, stretch=1)
        self.tab_widget.addTab(intensity_tab, "Intensity Data")

        # Tab 3: Blended
        self.plot_canvas = DraggableCanvas(self)
        self.plot_canvas.setMinimumSize(600, 400)

        blend_layout = QHBoxLayout()
        blend_left = QVBoxLayout()
        blend_left.addWidget(self.plot_canvas)
        blend_layout.addLayout(blend_left, stretch=3)

        blend_right = QVBoxLayout()

        # DPI, width and height controls for export
        export_layout = QHBoxLayout()
        export_layout.addWidget(QLabel("Export DPI:"))
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(72)
        self.dpi_spin.setMaximum(1200)
        self.dpi_spin.setValue(300)
        export_layout.addWidget(self.dpi_spin)

        export_layout.addWidget(QLabel("Width (inches):"))
        self.width_spin = QDoubleSpinBox()
        self.width_spin.setRange(1.0, 40.0)
        self.width_spin.setDecimals(2)
        self.width_spin.setValue(8.0)
        export_layout.addWidget(self.width_spin)

        export_layout.addWidget(QLabel("Height (inches):"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1.0, 40.0)
        self.height_spin.setDecimals(2)
        self.height_spin.setValue(6.0)
        export_layout.addWidget(self.height_spin)

        blend_right.addLayout(export_layout)

        overlay_buttons_layout = QHBoxLayout()
        btn_heatmap = QPushButton("Show Heatmap Overlay")
        btn_heatmap.clicked.connect(self.show_heatmap)
        overlay_buttons_layout.addWidget(btn_heatmap)

        btn_contour = QPushButton("Show Contours Overlay")
        btn_contour.clicked.connect(self.show_contours)
        overlay_buttons_layout.addWidget(btn_contour)

        btn_save_blend = QPushButton("Save Blended Image")
        btn_save_blend.clicked.connect(self.save_blended_image)
        overlay_buttons_layout.addWidget(btn_save_blend)

        blend_right.addLayout(overlay_buttons_layout)

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
            ["jet", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "YlGnBu", "Reds", "Accent"])
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

        # Colormap scale controls (spinboxes only, no sliders)
        scale_controls = QHBoxLayout()
        self.vmin_spin = QDoubleSpinBox()
        self.vmin_spin.setDecimals(3)
        self.vmax_spin = QDoubleSpinBox()
        self.vmax_spin.setDecimals(3)

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

        blend_right.addStretch(1)
        blend_layout.addLayout(blend_right, stretch=1)

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
        self.grid_canvas.setMinimumSize(600, 400)
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

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Image or PDF", "",
            "Image and PDF Files (*.png *.jpg *.bmp *.pdf);;Image Files (*.png *.jpg *.bmp);;PDF Files (*.pdf)")
        if not file_name:
            return

        if file_name.lower().endswith('.pdf'):
            try:

                # Open PDF and load first page
                doc = fitz.open(file_name)
                if len(doc) == 0:
                    QMessageBox.warning(self, "PDF Error", "No pages found in PDF file.")
                    doc.close()
                    return

                # Get first page and convert to pixmap at high DPI
                page = doc.load_page(0)  # First page (0-indexed)
                pix = page.get_pixmap(dpi=300)  # 300 DPI for high quality

                # Convert to PIL Image
                img_data = pix.tobytes("ppm")
                pil_img = Image.open(io.BytesIO(img_data))

                # Convert PIL image to QPixmap
                if pil_img.mode != "RGBA":
                    pil_img = pil_img.convert("RGBA")

                data = pil_img.tobytes("raw", "RGBA")
                qimg = QImage(data, pil_img.width, pil_img.height, QImage.Format.Format_RGBA8888)
                self.original_pixmap = QPixmap.fromImage(qimg)

                # Clean up
                doc.close()

                self.reset_image()  # Reset to default state with new image
                QMessageBox.information(self, "PDF Loaded",
                                        f"First page of PDF loaded successfully.\nSize: {pil_img.width}x{pil_img.height} pixels")

            except ImportError:
                QMessageBox.critical(self, "Missing Dependencies",
                                     "PDF support requires PyMuPDF library.\n\n"
                                     "Install with: pip install pymupdf\n\n"
                                     "No external dependencies required!")
            except Exception as e:
                QMessageBox.critical(self, "PDF Error",
                                     f"Failed to load PDF page as image:\n{str(e)}")
            return

        # Handle regular image files
        try:
            self.original_pixmap = QPixmap(file_name)
            if self.original_pixmap.isNull():
                QMessageBox.warning(self, "Image Error", "Could not load image file.")
                return
            self.reset_image()  # Reset to default state with new image
        except Exception as e:
            QMessageBox.critical(self, "Image Error", f"Failed to load image:\n{str(e)}")

    def reset_image(self):

        if not self.original_pixmap:
            return
        self.crop_rect = None
        self.current_pixmap = self.original_pixmap
        self.rotation_slider.setValue(0)
        self.rotation_label.setText("0°")
        self.image_label.setPixmap(self.current_pixmap)
        self.image_label.setScaledContents(True)
        self.invalidate_cache()
        self.update_intensity_preview()
        self.set_grid_spinboxes_from_data()

    def define_crop_area(self):

        if not self.original_pixmap:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return

        dialog = CropDialog(self, self.original_pixmap.width(), self.original_pixmap.height())
        if dialog.exec():
            x, y, w, h = dialog.get_values()
            self.crop_rect = QRect(x, y, w, h)

            self.current_pixmap = self.original_pixmap.copy(self.crop_rect)
            self.rotation_slider.setValue(0)  # Reset rotation on new crop
            self.rotation_label.setText("0°")
            self.image_label.setPixmap(self.current_pixmap)
            self.invalidate_cache()
            self.update_intensity_preview()

    def rotate_image_view(self, angle):
        if not self.original_pixmap:
            return
        self.current_rotation_angle = angle
        base_pixmap = self.original_pixmap.copy(self.crop_rect) if self.crop_rect else self.original_pixmap
        transform = QTransform().rotate(angle)
        self.current_pixmap = base_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)
        self.image_label.setPixmap(self.current_pixmap)
        self.rotation_label.setText(f"{angle}°")

    def get_rotated_background_pixmap(self):

        base_pixmap = self.original_pixmap.copy(self.crop_rect) if self.crop_rect else self.original_pixmap
        transform = QTransform().rotate(self.current_rotation_angle)
        return base_pixmap.transformed(transform, Qt.TransformationMode.SmoothTransformation)

    def get_raw_intensity_data(self):

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

        raw_data = self.get_raw_intensity_data()
        if raw_data is None or not self.original_pixmap:
            return None

        if self.crop_rect:
            # --- Cropped Logic ---
            # 1. Create a full-size transparent array (NaN for transparency)
            full_h, full_w = self.original_pixmap.height(), self.original_pixmap.width()
            composite_array = np.full((full_h, full_w), np.nan, dtype=float)

            # 2. Resize raw data to fit the crop rectangle
            crop_h, crop_w = self.crop_rect.height(), self.crop_rect.width()
            zoom_y, zoom_x = crop_h / raw_data.shape[0], crop_w / raw_data.shape[1]

            try:
                resized_intensity = zoom(raw_data, (zoom_y, zoom_x), order=1)
            except Exception:
                # Fallback method if zoom fails
                rep_y = max(1, int(np.ceil(zoom_y)))
                rep_x = max(1, int(np.ceil(zoom_x)))
                resized_intensity = np.repeat(np.repeat(raw_data, rep_y, axis=0), rep_x, axis=1)
                resized_intensity = resized_intensity[:crop_h, :crop_w]

            # 3. Paste the resized data into the correct spot in the composite
            x, y = self.crop_rect.x(), self.crop_rect.y()
            composite_array[y : y + crop_h, x : x + crop_w] = resized_intensity
            return composite_array
        else:
            # --- Full Image Logic ---
            full_h, full_w = self.original_pixmap.height(), self.original_pixmap.width()
            zoom_y, zoom_x = full_h / raw_data.shape[0], full_w / raw_data.shape[1]
            try:
                return zoom(raw_data, (zoom_y, zoom_x), order=1)
            except Exception:
                rep_y = max(1, int(np.ceil(zoom_y)))
                rep_x = max(1, int(np.ceil(zoom_x)))
                resized = np.repeat(np.repeat(raw_data, rep_y, axis=0), rep_x, axis=1)
                return resized[:full_h, :full_w]


    def show_heatmap(self):

        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and intensity data first.")
            return

        final_intensity = self.get_final_intensity_array()
        if final_intensity is None:
            QMessageBox.warning(self, "Warning", "Could not generate intensity data.")
            return

        # Get only the valid (non-NaN) data for setting color limits
        valid_data = final_intensity[~np.isnan(final_intensity)]
        if valid_data.size == 0:
            QMessageBox.warning(self, "Warning", "No valid data to plot.")
            return

        vmin, vmax = np.min(valid_data), np.max(valid_data)
        self.vmin_spin.setValue(vmin)
        self.vmax_spin.setValue(vmax)

        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        # Always use the full original_pixmap for blending
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

        # Always use the full original_pixmap for blending
        self.plot_canvas.draw_contours(
            self.get_rotated_background_pixmap(), final_intensity,
            alpha=self.alpha, cmap=self.cmap, levels=7,
            units=intensity_units, scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
        )
        self._current_vis_mode = "contour"


    def invalidate_cache(self):
        self.cached_intensity_array = None
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None
        import gc
        gc.collect()

    def update_intensity_preview(self):
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
        cmap = 'jet'

        try:
            if mode == "Contour Map":
                levels = 7
                ax.contourf(intensity, levels=levels, cmap=cmap)
            else:
                ax.imshow(intensity, cmap=cmap, aspect='auto')
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', color='red')
            ax.axis('off')

        fig.tight_layout()
        self.preview_canvas.draw()

    def set_grid_spinboxes_from_data(self):

        current_rows = self.table_widget.rowCount()
        current_cols = self.table_widget.columnCount()

        if hasattr(self, "grid_nx_spin") and hasattr(self, "grid_ny_spin"):
            if current_rows > 0 and current_cols > 0:
                self.grid_nx_spin.setValue(current_cols)
                self.grid_ny_spin.setValue(current_rows)
            else:
                self.grid_nx_spin.setValue(100)
                self.grid_ny_spin.setValue(100)

    def load_csv_excel(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "CSV Files (*.csv);;Excel Files (*.xls *.xlsx)")
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

        data_min = np.min(valid_data)
        data_max = np.max(valid_data)

        # Set spinbox ranges
        self.vmin_spin.setRange(data_min, data_max)
        self.vmax_spin.setRange(data_min, data_max)

        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
            self.plot_canvas.draw_contours(
                self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                final_intensity,
                alpha=self.alpha, cmap=self.cmap,
                levels=np.linspace(vmin, vmax, 20), units=intensity_units,
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
            )
        else:
            self.plot_canvas.draw_heatmap(
                self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                final_intensity,
                alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax, units=intensity_units,
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
            )

    def apply_formatting(self):

        if not hasattr(self.plot_canvas, 'ax') or not self.plot_canvas.ax.get_children():
            QMessageBox.warning(self, "Warning", "Create a plot first")
            return

        # Get formatting parameters
        label_fontsize = self.label_fontsize_spin.value()
        tick_fontsize = self.tick_fontsize_spin.value()
        is_bold = self.bold_checkbox.isChecked()
        is_italic = self.italic_checkbox.isChecked()
        x_ticks = self.x_ticks_spin.value()
        y_ticks = self.y_ticks_spin.value()

        # Determine font weight and style
        fontweight = 'bold' if is_bold else 'normal'
        fontstyle = 'italic' if is_italic else 'normal'

        # Apply to axis labels
        self.plot_canvas.ax.xaxis.label.set_fontsize(label_fontsize)
        self.plot_canvas.ax.xaxis.label.set_fontweight(fontweight)
        self.plot_canvas.ax.xaxis.label.set_fontstyle(fontstyle)
        self.plot_canvas.ax.yaxis.label.set_fontsize(label_fontsize)
        self.plot_canvas.ax.yaxis.label.set_fontweight(fontweight)
        self.plot_canvas.ax.yaxis.label.set_fontstyle(fontstyle)

        # Apply to tick labels
        self.plot_canvas.ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

        if hasattr(self.plot_canvas, 'cbar') and self.plot_canvas.cbar is not None:
            self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

        # Set number of ticks
        xlim = self.plot_canvas.ax.get_xlim()
        ylim = self.plot_canvas.ax.get_ylim()
        x_tick_positions1 = np.linspace(xlim[0], xlim[1], x_ticks)
        y_tick_positions1 = np.linspace(ylim[0], ylim[1], y_ticks)

        # Round each position to nearest multiple of 50 using NumPy
        x_tick_positions = np.round(x_tick_positions1 / 50) * 50
        y_tick_positions = np.round(y_tick_positions1 / 50) * 50

        # Ensure 0 is included if it falls within the axis range
        if xlim[0] <= 0 <= xlim[1] and 0 not in x_tick_positions:
            x_tick_positions = np.append(x_tick_positions, 0)
            x_tick_positions = np.sort(x_tick_positions)

        if ylim[0] <= 0 <= ylim[1] and 0 not in y_tick_positions:
            y_tick_positions = np.append(y_tick_positions, 0)
            y_tick_positions = np.sort(y_tick_positions)

        self.plot_canvas.ax.set_xticks(x_tick_positions)
        self.plot_canvas.ax.set_yticks(y_tick_positions)

        # Format tick labels to show reasonable precision
        self.plot_canvas.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))
        self.plot_canvas.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

        # Apply formatting to colorbar if it exists
        if hasattr(self.plot_canvas, 'cbar') and self.plot_canvas.cbar is not None:
            self.plot_canvas.cbar.ax.yaxis.label.set_fontsize(label_fontsize)
            self.plot_canvas.cbar.ax.yaxis.label.set_fontweight(fontweight)
            self.plot_canvas.cbar.ax.yaxis.label.set_fontstyle(fontstyle)
            self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

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

            # Create custom colormap with transparent under-layer
            base_cmap = plt.get_cmap(self.cmap)
            colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
            custom_cmap = ListedColormap(colors)
            custom_cmap.set_under((0, 0, 0, 0))

            # Draw contours with extend='min' handling
            self.plot_canvas.draw_contours(
                self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                final_intensity,
                alpha=self.alpha,
                cmap=custom_cmap,
                levels=levels,
                units=self.intensity_unit_input.text(),
                extend='min',
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

    def apply_highlights(self):

        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return

        try:
            highlight_text = self.highlight_input.text().strip()
            if not highlight_text:
                self.show_heatmap()
                return

            highlight_values = [float(x.strip()) for x in highlight_text.split(',')]
            final_intensity = self.get_final_intensity_array()
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            intensity_units = self.intensity_unit_input.text()
            distance_units = self.scale_unit_input.text()
            scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
            scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

            self.plot_canvas.draw_heatmap(
                self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                final_intensity,
                alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax, units=intensity_units,
                scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
            )

            if self.color_combo.currentText() == "White Only":
                colors = ['white'] * len(highlight_values)
            else:
                highlight_colors = ['white', 'black', 'red', 'yellow', 'green', 'cyan']
                colors = [highlight_colors[i % len(highlight_colors)] for i in range(len(highlight_values))]

            legend_handles = []
            for i, value in enumerate(highlight_values):
                contour = self.plot_canvas.ax.contour(
                    final_intensity, levels=[value], colors=[colors[i]], linewidths=2
                )
                self.plot_canvas.ax.clabel(contour, inline=True, fmt=f'{value}', fontsize=10)
                line = mlines.Line2D([], [], color=colors[i], linewidth=2, label=f'{value}')
                legend_handles.append(line)

            if legend_handles:
                self.plot_canvas.ax.legend(handles=legend_handles, loc='best')

            self.plot_canvas.draw()

        except ValueError as e:
            QMessageBox.warning(self, "Warning",
                                f"Invalid highlight values: {str(e)}\nPlease enter comma-separated numbers.")

    def apply_units_and_scale(self):

        try:
            scale_x = float(self.scale_x_input.text())
            scale_y = float(self.scale_y_input.text())
            distance_units = self.scale_unit_input.text()
            intensity_units = self.intensity_unit_input.text()

            self.real_scale_x = scale_x
            self.real_scale_y = scale_y

            # Update the current visualization with new scaling
            if not self.original_pixmap or self.table_widget.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "Load image and data first")
                return

            final_intensity = self.get_final_intensity_array()

            if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
                self.plot_canvas.draw_contours(
                    self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                    final_intensity,
                    alpha=self.alpha, cmap=self.cmap,
                    levels=np.linspace(self.vmin_spin.value(), self.vmax_spin.value(), 20),
                    units=intensity_units,
                    scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
                )
            else:
                self.plot_canvas.draw_heatmap(
                    self.get_rotated_background_pixmap(),  # FIXED: was self.original_pixmap
                    final_intensity,
                    alpha=self.alpha, cmap=self.cmap,
                    vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value(),
                    units=intensity_units,
                    scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
                )

        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numeric values for scales.")

    def update_display(self):
        if not self.original_pixmap or self.table_widget.rowCount() == 0:
            return

        if self.tab_widget.currentIndex() == 2:
            if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
                self.show_contours()
            else:
                self.show_heatmap()
        elif self.tab_widget.currentIndex() == 3:
            self.show_grid_overlay()

    def save_blended_image(self):
        if not self.plot_canvas.figure:
            return

        # Get target DPI and size from UI
        dpi = self.dpi_spin.value()
        width_inches = self.width_spin.value()
        height_inches = self.height_spin.value()

        # Save original figure size and DPI for later restore
        orig_size = self.plot_canvas.figure.get_size_inches()
        orig_dpi = self.plot_canvas.figure.get_dpi()

        # Set figure size & DPI for export (WYSIWYG, just more pixels)
        self.plot_canvas.figure.set_size_inches(width_inches, height_inches)
        self.plot_canvas.figure.set_dpi(dpi)
        self.plot_canvas.draw()  # Redraw at new size for layout

        # Ask user for file name and format
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Blended Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;SVG Files (*.svg);;PDF Files (*.pdf);;All Files (*)"
        )

        if file_name:
            ext = file_name.lower().split('.')[-1]
            if ext in ['svg', 'pdf']:
                # SVG/PDF: Vector, dpi is ignored, just use new size/layout
                self.plot_canvas.figure.savefig(
                    file_name,
                    bbox_inches=None,  # Use same as on-screen display
                    pad_inches=0,
                    format=ext
                )
            else:
                # PNG/JPEG: Raster, dpi is crucial!
                self.plot_canvas.figure.savefig(
                    file_name,
                    dpi=dpi,
                    bbox_inches=None,
                    pad_inches=0,
                    transparent=file_name.lower().endswith('.png')
                )

        # Restore figure for interactive use in the GUI
        self.plot_canvas.figure.set_size_inches(orig_size)
        self.plot_canvas.figure.set_dpi(orig_dpi)
        self.plot_canvas.draw()

    def save_grid_image(self):
        if not self.grid_canvas.figure:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Grid Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        if file_name:
            dpi = self.grid_canvas.figure.get_dpi()
            width = self.original_pixmap.width() / dpi
            height = self.original_pixmap.height() / dpi
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
            image = self.get_rotated_background_pixmap().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            width = image.width()
            height = image.height()
            buffer = image.constBits()
            buffer.setsize(image.sizeInBytes())
            arr = np.frombuffer(buffer, np.uint8).copy().reshape((height, width, 4))
            arr_rgb = arr[..., :3]

            self.grid_canvas.ax.clear()
            self.grid_canvas.ax.imshow(arr_rgb, aspect='equal', extent=[0, width, 0, height], origin='lower')

            # Always read spinbox values!
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

    def closeEvent(self, event):
        """Proper cleanup on window close"""
        try:
            if hasattr(self, 'plot_canvas') and self.plot_canvas.figure:
                self.plot_canvas.figure.clear()
                plt.close(self.plot_canvas.figure)

            if hasattr(self, 'grid_canvas') and self.grid_canvas.figure:
                self.grid_canvas.figure.clear()
                plt.close(self.grid_canvas.figure)

        except:
            pass  # Ignore cleanup errors

        # Force garbage collection
        import gc
        gc.collect()

        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
