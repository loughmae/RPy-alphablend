import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QLabel, QMessageBox, QSlider, QComboBox,
    QLineEdit, QGroupBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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
            # Only move the base image layer
            for img in self.ax.images:
                if img.get_array().dtype == np.uint8:  # Base image
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
                     vmin=None, vmax=None, units=''):
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
        self.ax.imshow(arr_rgb, aspect='equal', extent=[0, width, 0, height], origin='lower')
        im = self.ax.imshow(intensity_array, cmap=cmap, alpha=alpha,
                            interpolation=interpolation,
                            extent=[0, width, 0, height],
                            origin='lower',
                            vmin=vmin, vmax=vmax)
        self.ax.set_xlim([0, width])
        self.ax.set_ylim([height, 0])
        self.cbar = self.figure.colorbar(im, ax=self.ax, orientation='vertical', pad=0.05)
        if units:
            self.cbar.set_label(f'Intensity ({units})', rotation=270, labelpad=15)
        else:
            self.cbar.set_label('Intensity', rotation=270, labelpad=15)
        self.draw()

    def draw_contours(self, base_img: QPixmap, intensity_array,
                      alpha=0.6, cmap='jet', levels=6, interpolation='bilinear', units=''):
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
        self.ax.imshow(arr_rgb, aspect='equal', extent=[0, width, 0, height], origin='lower')
        rows, cols = intensity_array.shape
        X = np.linspace(0, width, cols)
        Y = np.linspace(0, height, rows)
        xx, yy = np.meshgrid(X, Y)
        if isinstance(levels, int):
            vmin = np.min(intensity_array)
            vmax = np.max(intensity_array)
            levels = np.linspace(vmin, vmax, levels)
        cs = self.ax.contourf(xx, yy, intensity_array, levels=levels, cmap=cmap, alpha=alpha)
        self.ax.set_xlim([0, width])
        self.ax.set_ylim([height, 0])
        self.cbar = self.figure.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05)
        if units:
            self.cbar.set_label(f'Intensity ({units})', rotation=270, labelpad=15)
        else:
            self.cbar.set_label('Intensity', rotation=270, labelpad=15)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiation Protection Scatter Map Generator")
        self.resize(100, 100)  # <--- Set your preferred width and height here
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
        self.cmap_combo.addItems(["jet", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm", "YlGnBu"])
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

        # Colormap scale sliders
        self.vmin_slider = QSlider(Qt.Orientation.Horizontal)
        self.vmin_slider.setMinimum(0)
        self.vmin_slider.setMaximum(100)
        self.vmin_slider.setValue(0)
        self.vmin_slider.valueChanged.connect(self.update_colormap_scale)
        self.vmax_slider = QSlider(Qt.Orientation.Horizontal)
        self.vmax_slider.setMinimum(0)
        self.vmax_slider.setMaximum(100)
        self.vmax_slider.setValue(100)
        self.vmax_slider.valueChanged.connect(self.update_colormap_scale)
        blend_right.addWidget(QLabel("Min Scale"))
        blend_right.addWidget(self.vmin_slider)
        blend_right.addWidget(QLabel("Max Scale"))
        blend_right.addWidget(self.vmax_slider)

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
        self.scale_unit_input = QLineEdit("mm")
        scale_layout.addWidget(self.scale_unit_input)
        axis_layout.addLayout(scale_layout)
        intensity_layout = QHBoxLayout()
        intensity_layout.addWidget(QLabel("Intensity Units:"))
        self.intensity_unit_input = QLineEdit("counts")
        intensity_layout.addWidget(self.intensity_unit_input)
        axis_layout.addLayout(intensity_layout)
        btn_apply_scale = QPushButton("Apply Scale and Units")
        btn_apply_scale.clicked.connect(self.apply_units_and_scale)
        axis_layout.addWidget(btn_apply_scale)
        axis_group.setLayout(axis_layout)
        blend_right.addWidget(axis_group)

        # Measurement tool
        measurement_group = QGroupBox("Measurement Tools")
        measurement_layout = QVBoxLayout()
        self.measure_btn = QPushButton("NOT CURRENTLY WORKING:Measure Distance")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self.toggle_measurement_mode)
        measurement_layout.addWidget(self.measure_btn)
        self.status_label = QLabel("Ready")
        measurement_layout.addWidget(self.status_label)
        measurement_group.setLayout(measurement_layout)
        blend_right.addWidget(measurement_group)
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
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "",
                                                   "Image Files (*.png *.jpg *.bmp)")
        if file_name:
            self.current_pixmap = QPixmap(file_name)
            self.image_label.setPixmap(self.current_pixmap)
            self.image_label.setScaledContents(True)

    def load_csv_excel(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "",
                                                   "CSV Files (*.csv);;Excel Files (*.xls *.xlsx)")
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
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not load file.\n{str(e)}")

    def add_row(self):
        current_rows = self.table_widget.rowCount()
        self.table_widget.insertRow(current_rows)

    def remove_row(self):
        current_rows = self.table_widget.rowCount()
        if current_rows > 0:
            self.table_widget.removeRow(current_rows - 1)

    def get_intensity_array(self):
        rows = self.table_widget.rowCount()
        cols = self.table_widget.columnCount()
        arr = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                item = self.table_widget.item(r, c)
                if item is not None:
                    try:
                        arr[r, c] = float(item.text())
                    except ValueError:
                        arr[r, c] = 0.0
        if self.current_pixmap:
            target_height = self.current_pixmap.height()
            target_width = self.current_pixmap.width()
            zoom_y = target_height / rows
            zoom_x = target_width / cols
            arr_resized = zoom(arr, (zoom_y, zoom_x), order=1)
            return arr_resized
        return arr

    # --- Visualization Controls ---
    def update_alpha(self):
        value = self.alpha_slider.value()
        self.alpha = value / 100.0
        self.update_display()

    def update_cmap(self):
        self.cmap = self.cmap_combo.currentText()
        self.update_display()

    def update_colormap_scale(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            return
        intens = self.get_intensity_array()
        data_min = np.min(intens)
        data_max = np.max(intens)
        vmin = data_min + (data_max - data_min) * (self.vmin_slider.value() / 100)
        vmax = data_min + (data_max - data_min) * (self.vmax_slider.value() / 100)
        intensity_units = self.intensity_unit_input.text()
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
        self._current_vis_mode = "heatmap"
        self.plot_canvas.draw_heatmap(
            self.current_pixmap, intens,
            alpha=self.alpha, cmap=self.cmap,
            units=intensity_units
        )

    def show_contours(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return
        intens = self.get_intensity_array()
        data_min = np.min(intens)
        data_max = np.max(intens)
        vmin = data_min + (data_max - data_min) * (self.vmin_slider.value() / 100)
        vmax = data_min + (data_max - data_min) * (self.vmax_slider.value() / 100)
        num_levels = 20
        levels = np.linspace(vmin, vmax, num_levels)
        self._current_vis_mode = "contour"
        intensity_units = self.intensity_unit_input.text()
        self.plot_canvas.draw_contours(
            self.current_pixmap, intens,
            alpha=self.alpha, cmap=self.cmap,
            levels=levels, units=intensity_units
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
            data_min = np.min(intens)
            data_max = np.max(intens)
            vmin = data_min + (data_max - data_min) * (self.vmin_slider.value() / 100)
            vmax = data_min + (data_max - data_min) * (self.vmax_slider.value() / 100)
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
            for canvas in [self.plot_canvas, self.grid_canvas]:
                canvas.ax.set_xlabel(f"Distance ({distance_units})")
                canvas.ax.set_ylabel(f"Distance ({distance_units})")
                if hasattr(canvas, 'cbar') and canvas.cbar is not None:
                    canvas.cbar.set_label(f'Intensity ({intensity_units})', rotation=270, labelpad=15)
                canvas.draw()
            self.update_display()
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

    # --- Measurement Tool ---
    def toggle_measurement_mode(self, checked):
        if checked:
            self.measuring = True
            self.measurement_points = []
            self.measurement_annotation = None
            self.measurement_lines = []
            self.measurement_cid = self.plot_canvas.mpl_connect('button_press_event', self.on_canvas_click)
            self.status_label.setText("NOT CURRENTLY WORKING:Measurement Mode: Click two points to measure distance")
        else:
            if hasattr(self, 'measurement_cid'):
                self.plot_canvas.mpl_disconnect(self.measurement_cid)
            self.clear_measurements()
            self.status_label.setText("NOT CURRENTLY WORKING:Ready")

    def clear_measurements(self):
        """Remove all measurement artifacts from the plot"""
        if hasattr(self, 'measurement_annotation') and self.measurement_annotation:
            self.measurement_annotation.remove()
            self.measurement_annotation = None
        for line in self.measurement_lines:
            if line in self.plot_canvas.ax.lines:
                line.remove()
        self.measurement_lines = []
        self.measurement_points = []
        self.plot_canvas.draw()

    def on_canvas_click(self, event):
        if not self.measuring or not event.inaxes or event.inaxes != self.plot_canvas.ax:
            return

        self.measurement_points.append((event.xdata, event.ydata))

        if len(self.measurement_points) == 2:
            # Remove previous measurements
            self.clear_measurements()

            # Get both points
            p1, p2 = self.measurement_points

            # Draw measurement line
            line, = self.plot_canvas.ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color='red', linewidth=2, linestyle='--'
            )
            self.measurement_lines.append(line)

            # Calculate midpoint
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

            # Calculate distance
            dx_pixels = p2[0] - p1[0]
            dy_pixels = p2[1] - p1[1]
            pixel_dist = np.sqrt(dx_pixels ** 2 + dy_pixels ** 2)

            # Convert to real units if available
            if hasattr(self, 'real_scale_x') and hasattr(self, 'real_scale_y'):
                dx_real = dx_pixels * self.real_scale_x
                dy_real = dy_pixels * self.real_scale_y
                real_dist = np.sqrt(dx_real ** 2 + dy_real ** 2)
                units = self.scale_unit_input.text().strip() or 'units'
                distance_text = f"{real_dist:.2f} {units}"
            else:
                distance_text = f"{pixel_dist:.2f} pixels"

            # Create annotation
            self.measurement_annotation = self.plot_canvas.ax.annotate(
                distance_text,
                xy=midpoint,
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=10,
                color='black',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    fc='white',
                    ec='red',
                    alpha=0.8
                ),
                annotation_clip=False  # Ensure text is always visible
            )

            # Refresh the display
            self.plot_canvas.draw()

            # Reset for next measurement
            self.measurement_points = []

    def on_canvas_click(self, event):
        if not self.measuring or not event.inaxes or event.inaxes != self.plot_canvas.ax:
            return
        self.measurement_points.append((event.xdata, event.ydata))
        if len(self.measurement_points) == 2:
            p1, p2 = self.measurement_points
            for line in self.measurement_lines:
                if line in self.plot_canvas.ax.lines:
                    line.remove()
            line, = self.plot_canvas.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
            self.measurement_lines.append(line)
            pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            if hasattr(self, 'real_scale_x') and hasattr(self, 'real_scale_y'):
                x_dist = abs(p2[0] - p1[0]) * self.real_scale_x
                y_dist = abs(p2[1] - p1[1]) * self.real_scale_y
                real_dist = np.sqrt(x_dist**2 + y_dist**2)
                units = self.scale_unit_input.text() if hasattr(self, 'scale_unit_input') else "units"
                distance_text = f"{real_dist:.2f} {units}"
            else:
                distance_text = f"{pixel_dist:.2f} pixels"
            if hasattr(self, 'measurement_annotation') and self.measurement_annotation:
                self.measurement_annotation.remove()
            midpoint = ((p1[0] + p2[0])/2, (p1[1] + p2[1])/2)
            self.measurement_annotation = self.plot_canvas.ax.annotate(
                distance_text, xy=midpoint, xytext=(10, 10), textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.7)
            )
            self.measurement_points = []
            self.plot_canvas.draw()

    def save_blended_image(self):
        if not self.plot_canvas.figure:
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Blended Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )
        if file_name:
            dpi = self.plot_canvas.figure.get_dpi()
            width = self.current_pixmap.width() / dpi
            height = self.current_pixmap.height() / dpi
            self.plot_canvas.figure.set_size_inches(width, height)
            use_transparency = not (file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'))
            self.plot_canvas.figure.savefig(
                file_name, dpi=dpi, bbox_inches='tight',
                pad_inches=0, transparent=use_transparency
            )

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
        # Clean up matplotlib figures to prevent memory leaks
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
