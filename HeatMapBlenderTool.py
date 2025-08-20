
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QLabel, QMessageBox, QComboBox,
    QLineEdit, QGroupBox, QDoubleSpinBox, QSlider, QCheckBox, QSpinBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from matplotlib.colors import ListedColormap

def get_threat_zone_cmap():
    """Discrete 7-level threat zone colormap (categorical)."""
    colors = [
        "#0033CC",  # Deep Blue
        "#00CCCC",  # Cyan
        "#00CC44",  # Green
        "#FFDD00",  # Yellow
        "#FF8800",  # Orange
        "#FF2222",  # Red
        "#AA00CC",  # Purple
    ]
    return ListedColormap(colors, name='threat_zones')

def get_continuous_dose_cmap():
    """Continuous 7-level sequential colormap (blue→red)."""
    colors = [
        "#0022CC",  # Deep Blue
        "#0099CC",  # Blue-Cyan
        "#00CC66",  # Green
        "#CCFF33",  # Yellow-Green
        "#FFDD00",  # Yellow
        "#FF8800",  # Orange
        "#FF2222",  # Red
    ]
    return ListedColormap(colors, name='dose_field')

class DraggableCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.dragging = False
        self.last_mouse_pos = None
        self.cbar = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.dragging:
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            for img in self.ax.images:
                if img.get_array().dtype == np.uint8:
                    extent = img.get_extent()
                    new_extent = [extent[0] + dx, extent[1] + dx,
                                  extent[2] + dy, extent[3] + dy]
                    img.set_extent(new_extent)
            self.last_mouse_pos = event.pos()
            self.draw()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def draw_heatmap(self, base_img: QPixmap, intensity_array,
                     alpha=0.6, cmap='jet', interpolation='bilinear',
                     vmin=None, vmax=None, units='', scale_x=1.0, scale_y=1.0, distance_units='pixels'):
        self.ax.clear()
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

        image = base_img.toImage()
        width = image.width()
        height = image.height()
        try:
            ptr = image.bits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        except Exception as e:
            print(f"Image processing error: {e}")
            return

        arr_rgb = arr[..., :3]

        # Convert pixel coordinates to real coordinates
        real_width = width * scale_x
        real_height = height * scale_y

        self.ax.imshow(arr_rgb, aspect='equal', extent=[-real_width/2, real_width/2, real_height/2, -real_width/2], origin='lower')
        im = self.ax.imshow(intensity_array, cmap=cmap, alpha=alpha,
                            interpolation=interpolation,
                            extent=[-real_width/2, real_width/2, real_height/2, -real_width/2],
                            origin='lower',
                            vmin=vmin, vmax=vmax)

        #self.ax.set_xlim([0, real_width])
        #self.ax.set_ylim([real_height, 0])
        self.ax.set_xlim([-real_width/2, real_width/2])
        self.ax.set_ylim([real_height/2, -real_width/2])


        # Set axis labels with units
        self.ax.set_xlabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')
        self.ax.set_ylabel(f"Distance ({distance_units})", fontsize=15, fontweight='bold')

        self.cbar = self.figure.colorbar(im, ax=self.ax, orientation='vertical', pad=0.05)
        #self.cbar.ax.tick_params(labelsize=11)
        if units:
            self.cbar.set_label(f'Intensity ({units})', rotation=270, labelpad=20, fontsize=15, fontweight='bold')
            #self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

        else:
            self.cbar.set_label('Intensity', rotation=270, labelpad=20, fontsize=15, fontweight='bold')

        self.draw()

    def draw_contours(self, base_img: QPixmap, intensity_array,
                      alpha=0.6, cmap='jet', levels=6,
                      interpolation='bilinear', units='', extend='neither',
                      scale_x=1.0, scale_y=1.0, distance_units='pixels'):
        self.ax.clear()
        if self.cbar is not None:
            self.cbar.remove()
            self.cbar = None

        image = base_img.toImage()
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        arr_rgb = arr[..., :3]

        # Convert pixel coordinates to real coordinates
        real_width = width * scale_x
        real_height = height * scale_y

        self.ax.imshow(arr_rgb, aspect='equal', extent=[-real_width/2, real_width/2, -real_height/2, real_width/2], origin='lower')

        rows, cols = intensity_array.shape
        X = np.linspace(-real_width/2, real_width/2, cols)
        Y = np.linspace(-real_height/2, real_height/2, rows)
        xx, yy = np.meshgrid(X, Y)

        if isinstance(levels, int):
            vmin = np.min(intensity_array)
            vmax = np.max(intensity_array)
            levels = np.linspace(vmin, vmax, levels)

        cs = self.ax.contourf(
            xx, yy, intensity_array,
            levels=levels,
            cmap=cmap,
            alpha=alpha,
            extend='max'
        )

        self.ax.set_xlim([-real_width/2, real_width/2])
        self.ax.set_ylim([real_height/2, -real_width/2])

        # Set axis labels with units
        self.ax.set_xlabel(f"Distance ({distance_units})", fontsize=17, fontweight='bold')
        self.ax.set_ylabel(f"Distance ({distance_units})", fontsize=17, fontweight='bold')

        self.cbar = self.figure.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05)
        if units:
            self.cbar.set_label(f'Intensity ({units})', rotation=270, labelpad=15, fontsize=14, fontweight='bold')
            self.cbar.ax.tick_params(labelsize=13)

        else:
            self.cbar.set_label('Intensity', rotation=270, labelpad=15, fontsize=14, fontweight='bold')
            self.cbar.ax.tick_params(labelsize=13)

        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiation Protection Scatter Map Generator")
        self.resize(100, 100)  # Set your preferred width and height here

        # --- Caching for efficiency ---
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None
        self.cached_intensity_array = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Image
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_tab_layout = QHBoxLayout()
        img_tab_left = QVBoxLayout()
        img_tab_left.addWidget(self.image_label)
        img_tab_layout.addLayout(img_tab_left, stretch=3)
        img_tab_right = QVBoxLayout()
        btn_load_img = QPushButton("Load Image")
        btn_load_img.clicked.connect(self.load_image)
        img_tab_right.addWidget(btn_load_img)
        img_tab_right.addStretch(1)
        img_tab_layout.addLayout(img_tab_right, stretch=1)
        img_tab = QWidget()
        img_tab.setLayout(img_tab_layout)
        self.tab_widget.addTab(img_tab, "Image")

        # Tab 2: Intensity Data
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)
        self.table_widget.cellChanged.connect(self.invalidate_cache)  # Efficient cache invalidation
        intensity_layout = QHBoxLayout()
        intensity_left = QVBoxLayout()
        intensity_left.addWidget(self.table_widget)
        intensity_layout.addLayout(intensity_left, stretch=3)
        intensity_right = QVBoxLayout()
        btn_load_csv = QPushButton("Load CSV/Excel")
        btn_load_csv.clicked.connect(self.load_csv_excel)
        intensity_right.addWidget(btn_load_csv)
        btn_add_row = QPushButton("Add Row")
        btn_add_row.clicked.connect(self.add_row)
        intensity_right.addWidget(btn_add_row)
        btn_remove_row = QPushButton("Remove Row")
        btn_remove_row.clicked.connect(self.remove_row)
        intensity_right.addWidget(btn_remove_row)
        intensity_right.addStretch(1)
        intensity_layout.addLayout(intensity_right, stretch=1)
        intensity_tab = QWidget()
        intensity_tab.setLayout(intensity_layout)
        self.tab_widget.addTab(intensity_tab, "Intensity Data")

        # Tab 3: Blended
        self.plot_canvas = DraggableCanvas(self)
        self.plot_canvas.setMinimumSize(600, 400)
        blend_layout = QHBoxLayout()
        blend_left = QVBoxLayout()
        blend_left.addWidget(self.plot_canvas)
        blend_layout.addLayout(blend_left, stretch=3)
        blend_right = QVBoxLayout()

        # Overlay buttons
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
        self.cmap_combo.addItems(["jet", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "YlGnBu", "Reds", "Accent"])
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

        #custom colour scale
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
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity Units:"))
        self.intensity_unit_input = QLineEdit("uGy/h")
        intensity_layout.addWidget(self.intensity_unit_input)
        axis_layout.addLayout(intensity_layout)
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
        btn_show_grid = QPushButton("Show Grid Overlay")
        btn_show_grid.clicked.connect(self.show_grid_overlay)
        grid_right.addWidget(btn_show_grid)
        btn_save_grid = QPushButton("Save Overlay Image")
        btn_save_grid.clicked.connect(self.save_grid_image)
        grid_right.addWidget(btn_save_grid)
        grid_right.addStretch(1)
        grid_layout.addLayout(grid_right, stretch=1)
        grid_tab.setLayout(grid_layout)
        self.tab_widget.addTab(grid_tab, "Grid Overlay")

        self.current_pixmap = None
        self.intensity_data = None
        self.alpha = 0.6
        self.cmap = "jet"
        self._current_vis_mode = "heatmap"

    # --- Data and Image Loading ---
    def apply_formatting(self):
        """Apply advanced formatting to the current plot"""
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
        # self.cbar.ax.tick_params(labelsize=tick_fontsize)
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
            # self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize - 1)
            self.plot_canvas.cbar.ax.tick_params(labelsize=tick_fontsize)

        self.plot_canvas.draw()

    def apply_custom_scale(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load image and data first")
            return

        try:
            levels = [float(x.strip()) for x in self.custom_levels_input.text().split(',')]
            #levels = [0, 200, 500, 1000, 6000, 15000]
            if not 2 <= len(levels) <= 7:
                raise ValueError("Enter 2-7 values")

            levels = sorted(levels)
            intens = self.get_intensity_array()

            # Create custom colormap with transparent under-layer
            base_cmap = plt.get_cmap(self.cmap)
            colors = base_cmap(np.linspace(0, 1, len(levels) - 1))
            custom_cmap = ListedColormap(colors)
            custom_cmap.set_under((0, 0, 0, 0))

            # Draw contours with extend='min' handling
            self.plot_canvas.draw_contours(
                self.current_pixmap, intens,
                alpha=self.alpha,
                cmap=custom_cmap,
                levels=levels,
                units=self.intensity_unit_input.text(),
                extend='min'
            )

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid input: {str(e)}")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.current_pixmap = QPixmap(file_name)
            self.image_label.setPixmap(self.current_pixmap)
            self.image_label.setScaledContents(True)
            self.invalidate_cache()

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
                self.table_widget.setRowCount(rows)
                self.table_widget.setColumnCount(cols)
                for r in range(rows):
                    for c in range(cols):
                        item = QTableWidgetItem(str(data[r, c]))
                        self.table_widget.setItem(r, c, item)
                self.intensity_data = data
                self.invalidate_cache()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file.\n{str(e)}")

    def add_row(self):
        current_rows = self.table_widget.rowCount()
        self.table_widget.insertRow(current_rows)
        self.invalidate_cache()

    def remove_row(self):
        current_rows = self.table_widget.rowCount()
        if current_rows > 0:
            self.table_widget.removeRow(current_rows - 1)
            self.invalidate_cache()

    def invalidate_cache(self):
        self.cached_intensity_array = None
        self.last_csv_data = None
        self.last_csv_shape = None
        self.last_pixmap_size = None

    def get_intensity_array(self):
        # Check if cache is valid
        rows = self.table_widget.rowCount()
        cols = self.table_widget.columnCount()
        current_data = []
        for r in range(rows):
            row = []
            for c in range(cols):
                item = self.table_widget.item(r, c)
                row.append(float(item.text()) if item and item.text() else 0.0)
            current_data.append(row)
        pixmap_size = (self.current_pixmap.width(), self.current_pixmap.height()) if self.current_pixmap else None
        if (self.cached_intensity_array is not None and
            self.last_csv_data == current_data and
                self.last_pixmap_size == pixmap_size):
            return self.cached_intensity_array
        arr = np.array(current_data, dtype=float)
        if self.current_pixmap:
            target_height = self.current_pixmap.height()
            target_width = self.current_pixmap.width()
            zoom_y = target_height / arr.shape[0]
            zoom_x = target_width / arr.shape[1]
            arr_resized = zoom(arr, (zoom_y, zoom_x), order=1)
            self.cached_intensity_array = arr_resized
        else:
            self.cached_intensity_array = arr
        self.last_csv_data = current_data
        self.last_pixmap_size = pixmap_size
        return self.cached_intensity_array

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
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            return
        intens = self.get_intensity_array()
        data_min = np.min(intens)
        data_max = np.max(intens)
        # Set spinbox ranges
        self.vmin_spin.setRange(data_min, data_max)
        self.vmax_spin.setRange(data_min, data_max)
        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()
        intensity_units = self.intensity_unit_input.text()
        if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
            self.plot_canvas.draw_contours(
                self.current_pixmap, intens,
                alpha=self.alpha, cmap=self.cmap,
                levels=np.linspace(vmin, vmax, 20), units=intensity_units
            )
        else:
            self.plot_canvas.draw_heatmap(
                self.current_pixmap, intens,
                alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax, units=intensity_units
            )

    def show_heatmap(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return

        intens = self.get_intensity_array()
        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        self._current_vis_mode = "heatmap"
        self.vmin_spin.setValue(np.min(intens))
        self.vmax_spin.setValue(np.max(intens))

        self.plot_canvas.draw_heatmap(
            self.current_pixmap, intens,
            alpha=self.alpha, cmap=self.cmap,
            vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value(),
            units=intensity_units,
            scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
        )

    def show_contours(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return

        intens = self.get_intensity_array()
        intensity_units = self.intensity_unit_input.text()
        distance_units = self.scale_unit_input.text()
        scale_x = float(self.scale_x_input.text()) if self.scale_x_input.text() else 1.0
        scale_y = float(self.scale_y_input.text()) if self.scale_y_input.text() else 1.0

        vmin = self.vmin_spin.value()
        vmax = self.vmax_spin.value()

        if isinstance(self.cmap, ListedColormap) and self.cmap.name in ['threat_zones', 'dose_field']:
            num_levels = 7
        else:
            num_levels = 20
        levels = np.linspace(vmin, vmax, num_levels)
        self._current_vis_mode = "contour"

        self.plot_canvas.draw_contours(
            self.current_pixmap, intens,
            alpha=self.alpha, cmap=self.cmap,
            levels=levels, units=intensity_units,
            scale_x=scale_x, scale_y=scale_y, distance_units=distance_units,
        )


    def apply_highlights(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return
        try:
            highlight_text = self.highlight_input.text().strip()
            if not highlight_text:
                self.show_heatmap()
                return
            highlight_values = [float(x.strip()) for x in highlight_text.split(',')]
            intens = self.get_intensity_array()
            vmin = self.vmin_spin.value()
            vmax = self.vmax_spin.value()
            intensity_units = self.intensity_unit_input.text()
            self.plot_canvas.draw_heatmap(
                self.current_pixmap, intens,
                alpha=self.alpha, cmap=self.cmap,
                vmin=vmin, vmax=vmax, units=intensity_units
            )
            if self.color_combo.currentText() == "White Only":
                colors = ['white'] * len(highlight_values)
            else:
                highlight_colors = ['white', 'black', 'red', 'yellow', 'green', 'cyan']
                colors = [highlight_colors[i % len(highlight_colors)] for i in range(len(highlight_values))]
            legend_handles = []
            for i, value in enumerate(highlight_values):
                contour = self.plot_canvas.ax.contour(
                    intens, levels=[value], colors=[colors[i]], linewidths=2
                )
                self.plot_canvas.ax.clabel(contour, inline=True, fmt=f'{value}', fontsize=10)
                line = mlines.Line2D([], [], color=colors[i], linewidth=2, label=f'{value}')
                legend_handles.append(line)
            if legend_handles:
                self.plot_canvas.ax.legend(handles=legend_handles, loc='best')
            self.plot_canvas.draw()
        except ValueError as e:
            QMessageBox.warning(self, "Warning", f"Invalid highlight values: {str(e)}\nPlease enter comma-separated numbers.")

    def apply_units_and_scale(self):
        try:
            scale_x = float(self.scale_x_input.text())
            scale_y = float(self.scale_y_input.text())
            distance_units = self.scale_unit_input.text()
            intensity_units = self.intensity_unit_input.text()

            self.real_scale_x = scale_x
            self.real_scale_y = scale_y

            # Update the current visualization with new scaling
            if not self.current_pixmap or self.table_widget.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "Load image and data first")
                return

            intens = self.get_intensity_array()

            if hasattr(self, '_current_vis_mode') and self._current_vis_mode == "contour":
                self.plot_canvas.draw_contours(
                    self.current_pixmap, intens,
                    alpha=self.alpha, cmap=self.cmap,
                    levels=np.linspace(self.vmin_spin.value(), self.vmax_spin.value(), 20),
                    units=intensity_units,
                    scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
                )
            else:
                self.plot_canvas.draw_heatmap(
                    self.current_pixmap, intens,
                    alpha=self.alpha, cmap=self.cmap,
                    vmin=self.vmin_spin.value(), vmax=self.vmax_spin.value(),
                    units=intensity_units,
                    scale_x=scale_x, scale_y=scale_y, distance_units=distance_units
                )

        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid numeric values for scales.")

    def update_display(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
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
            width = self.current_pixmap.width() / dpi
            height = self.current_pixmap.height() / dpi
            self.grid_canvas.figure.set_size_inches(width, height)
            use_transparency = not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'))
            self.grid_canvas.figure.savefig(
                file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent=use_transparency
            )

    def show_grid_overlay(self):
        if not self.current_pixmap:
            QMessageBox.warning(self, "Warning", "Load an image first.")
            return
        image = self.current_pixmap.toImage()
        width = image.width()
        height = image.height()
        ptr = image.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
        arr_rgb = arr[..., :3]
        self.grid_canvas.ax.clear()
        self.grid_canvas.ax.imshow(arr_rgb, aspect='equal', extent=[0, width, 0, height], origin='lower')
        rows, cols = 20, 20
        x = np.linspace(0, width, cols + 1)
        y = np.linspace(0, height, rows + 1)
        if self.grid_type_combo.currentText() == "Points":
            xx, yy = np.meshgrid(x, y)
            self.grid_canvas.ax.plot(xx, yy, 'r.', markersize=2)
        else:  # Dotted Lines
            for xi in x:
                self.grid_canvas.ax.plot([xi, xi], [0, height], 'r:', linewidth=0.5)
            for yi in y:
                self.grid_canvas.ax.plot([0, width], [yi, yi], 'r:', linewidth=0.5)
        self.grid_canvas.ax.set_xlim([0, width])
        self.grid_canvas.ax.set_ylim([height, 0])
        self.grid_canvas.draw()

    def closeEvent(self, event):
        if hasattr(self, 'plot_canvas') and self.plot_canvas.figure:
            plt.close(self.plot_canvas.figure)
        if hasattr(self, 'grid_canvas') and self.grid_canvas.figure:
            plt.close(self.grid_canvas.figure)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
