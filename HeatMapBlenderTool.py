import sys
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget,
    QPushButton, QTableWidget, QTableWidgetItem,
    QFileDialog, QLabel, QMessageBox, QSlider, QComboBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

#class PlotCanvas(FigureCanvasQTAgg):
 #   def __init__(self, parent=None):
  #      fig = Figure()
   #     super().__init__(fig)
    #    self.setParent(parent)
     #   self.ax = self.figure.add_subplot(111)
      #  self.figure.tight_layout()

        # Keep a reference to the current colorbar (initially none)
       # self.cbar = None
# Enable dragging of the base image
class DraggableCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None):
        fig = Figure()
        super().__init__(fig)
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)
        self.figure.tight_layout()
        self.dragging = False
        self.last_mouse_pos = None
        self.cbar = None  # Keep a reference to the current colorbar

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
                     alpha=0.6, cmap='jet', interpolation='bilinear',vmin=None, vmax=None):
        self.ax.clear()

        # Remove old colorbar if it exists
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
                            vmin=vmin,
                            vmax=vmax)

        self.ax.set_xlim([0, width])
        self.ax.set_ylim([height, 0])

        # Add new colorbar
        self.cbar = self.figure.colorbar(im, ax=self.ax, orientation='vertical', pad=0.05)
        self.cbar.set_label('Intensity', rotation=270, labelpad=15)

        self.draw()

    def draw_contours(self, base_img: QPixmap, intensity_array,
                      alpha=0.6, cmap='jet', levels=6, interpolation='bilinear'):
        self.ax.clear()

        # Remove old colorbar if it exists
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

        cs = self.ax.contourf(xx, yy, intensity_array, levels=levels, cmap=cmap, alpha=alpha)
        self.ax.set_xlim([0, width])
        self.ax.set_ylim([height,0])

        # Add new colorbar
        self.cbar = self.figure.colorbar(cs, ax=self.ax, orientation='vertical', pad=0.05)
        self.cbar.set_label('Intensity', rotation=270, labelpad=15)

        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radiation Protection Scatter Map Generator")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Image
        self.image_label = QLabel("No Image Loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_tab_layout = QVBoxLayout()
        img_tab_layout.addWidget(self.image_label)

        btn_load_img = QPushButton("Load Image")
        btn_load_img.clicked.connect(self.load_image)
        img_tab_layout.addWidget(btn_load_img)

        img_tab = QWidget()
        img_tab.setLayout(img_tab_layout)
        self.tab_widget.addTab(img_tab, "Image")

        # Tab 2: Intensity Data
        self.table_widget = QTableWidget()
        self.table_widget.setEditTriggers(QTableWidget.EditTrigger.AllEditTriggers)

        intensity_layout = QVBoxLayout()
        intensity_layout.addWidget(self.table_widget)

        button_layout = QHBoxLayout()
        btn_load_csv = QPushButton("Load CSV/Excel")
        btn_load_csv.clicked.connect(self.load_csv_excel)
        button_layout.addWidget(btn_load_csv)

        btn_add_row = QPushButton("Add Row")
        btn_add_row.clicked.connect(self.add_row)
        button_layout.addWidget(btn_add_row)

        btn_remove_row = QPushButton("Remove Row")
        btn_remove_row.clicked.connect(self.remove_row)
        button_layout.addWidget(btn_remove_row)

        intensity_layout.addLayout(button_layout)
        intensity_tab = QWidget()
        intensity_tab.setLayout(intensity_layout)
        self.tab_widget.addTab(intensity_tab, "Intensity Data")

        # Tab 3: Blended
        self.plot_canvas = DraggableCanvas(self)
        self.plot_canvas.setMinimumSize(600, 400)  # Adjust these values as needed
        blend_layout = QVBoxLayout()
        blend_layout.addWidget(self.plot_canvas)

        overlay_buttons_layout = QHBoxLayout()
        btn_heatmap = QPushButton("Show Heatmap Overlay")
        btn_heatmap.clicked.connect(self.show_heatmap)
        overlay_buttons_layout.addWidget(btn_heatmap)

        btn_contour = QPushButton("Show Contours Overlay")
        btn_contour.clicked.connect(self.show_contours)
        overlay_buttons_layout.addWidget(btn_contour)
        blend_layout.addLayout(overlay_buttons_layout)

        btn_save_blend = QPushButton("Save Blended Image")
        btn_save_blend.clicked.connect(self.save_blended_image)
        overlay_buttons_layout.addWidget(btn_save_blend)

        # Slider for alpha
        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setMinimum(1)
        self.alpha_slider.setMaximum(100)
        self.alpha_slider.setValue(60)  # Default to 60% transparency
        self.alpha_slider.setTickInterval(10)
        self.alpha_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.alpha_slider.valueChanged.connect(self.update_alpha)
        blend_layout.addWidget(self.alpha_slider)

        # Combo box for choosing color map
        self.cmap_combo = QComboBox()
        # Add some standard color maps
        # add in american,european and lower reference colour map?Do this by scaling outputs?Or create colour maps?
        self.cmap_combo.addItems(["jet", "viridis", "plasma", "inferno", "magma", "cividis","coolwarm", "YlGnBu"])
        self.cmap_combo.setCurrentText("jet")
        self.cmap_combo.currentTextChanged.connect(self.update_cmap)
        blend_layout.addWidget(self.cmap_combo)

        blend_tab = QWidget()
        blend_tab.setLayout(blend_layout)
        self.tab_widget.addTab(blend_tab, "Blended")

        self.current_pixmap = None
        self.intensity_data = None
        self.alpha = 0.6
        self.cmap = "jet"

        # Combo box for choosing highlight value
        self.highlight_combo = QComboBox()
        self.highlight_combo.addItems(["No Highlight", "0.15", "0.3", "1.0"])
        self.highlight_combo.currentTextChanged.connect(self.update_highlight)
        blend_layout.addWidget(self.highlight_combo)

        # Add sliders for vmin and vmax
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

        blend_layout.addWidget(QLabel("Min Scale"))
        blend_layout.addWidget(self.vmin_slider)
        blend_layout.addWidget(QLabel("Max Scale"))
        blend_layout.addWidget(self.vmax_slider)

        # Update scale in heatmap rendering
        def update_colormap_scale(self):
            vmin = self.vmin_slider.value()
            vmax = self.vmax_slider.value()
            intens = self.get_intensity_array()
            self.plot_canvas.draw_heatmap(self.current_pixmap, intens,
                                          alpha=self.alpha,
                                          cmap=self.cmap,
                                          interpolation='bilinear',
                                          vmin=vmin, vmax=vmax)

        # Tab 4: Grid Overlay
        grid_tab = QWidget()
        grid_layout = QVBoxLayout()
        grid_tab.setLayout(grid_layout)

        self.grid_canvas = DraggableCanvas(self)
        self.grid_canvas.setMinimumSize(600, 400)
        grid_layout.addWidget(self.grid_canvas)

        grid_options_layout = QHBoxLayout()
        self.grid_type_combo = QComboBox()
        self.grid_type_combo.addItems(["Points", "Dotted Lines"])
        grid_options_layout.addWidget(QLabel("Grid Type:"))
        grid_options_layout.addWidget(self.grid_type_combo)

        btn_show_grid = QPushButton("Show Grid Overlay")
        btn_show_grid.clicked.connect(self.show_grid_overlay)
        grid_options_layout.addWidget(btn_show_grid)

        btn_save_grid = QPushButton("Save Overlay Image")
        btn_save_grid.clicked.connect(self.save_grid_image)
        grid_options_layout.addWidget(btn_save_grid)

        grid_layout.addLayout(grid_options_layout)

        self.tab_widget.addTab(grid_tab, "Grid Overlay")

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

            # Calculate zoom factors
            zoom_y = target_height / rows
            zoom_x = target_width / cols

            # Use scipy's zoom function with order=1 for bilinear interpolation
            arr_resized = zoom(arr, (zoom_y, zoom_x), order=1)

            return arr_resized
        return arr

    def update_alpha(self):
        value = self.alpha_slider.value()
        self.alpha = value / 100.0

    def update_cmap(self):
        self.cmap = self.cmap_combo.currentText()

    def update_colormap_scale(self):
        intens = self.get_intensity_array()
        data_min = np.min(intens)
        data_max = np.max(intens)

        # Convert slider percentages to actual data values
        vmin = data_min + (data_max - data_min) * (self.vmin_slider.value() / 100)
        vmax = data_min + (data_max - data_min) * (self.vmax_slider.value() / 100)

        self.plot_canvas.draw_heatmap(self.current_pixmap, intens,
                                      alpha=self.alpha,
                                      cmap=self.cmap,
                                      vmin=vmin, vmax=vmax)

    def show_heatmap(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return
        intens = self.get_intensity_array()
        self.plot_canvas.draw_heatmap(self.current_pixmap, intens,
                                      alpha=self.alpha,
                                      cmap=self.cmap,
                                      interpolation='bilinear')

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
            self.plot_canvas.figure.savefig(
                file_name,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True
            )


    def save_grid_image(self):
        if not self.grid_canvas.figure:
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "PNG Files (*.png);;JPEG Files (*.jpg *.jpeg);;All Files (*)"
        )

        if file_name:
            # Maintain original DPI and resolution
            dpi = self.grid_canvas.figure.get_dpi()
            width = self.current_pixmap.width() / dpi
            height = self.current_pixmap.height() / dpi

            self.grid_canvas.figure.set_size_inches(width, height)
            self.grid_canvas.figure.savefig(
                file_name,
                dpi=dpi,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True
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

    def update_highlight(self):
        value = self.highlight_combo.currentText()
        if value == "No Highlight":
            self.show_heatmap()  # or self.show_contours()
        else:
            highlight_value = float(value)
            intens = self.get_intensity_array()

            # Create a mask for the highlighted contour
            mask = np.abs(intens - highlight_value) < 0.01  # Tolerance for floating-point comparison

            # Draw the heatmap or contours with the highlight
            self.plot_canvas.draw_heatmap(self.current_pixmap, intens,
                                          alpha=self.alpha,
                                          cmap=self.cmap,
                                          interpolation='bilinear')

            # Overlay the highlighted contour
            self.plot_canvas.ax.contour(intens, levels=[highlight_value],
                                        colors=['white'], linewidths=2)
            self.plot_canvas.draw()

    def show_contours(self):
        if not self.current_pixmap or self.table_widget.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "Load an image and some intensity data first.")
            return
        intens = self.get_intensity_array()
        self.plot_canvas.draw_contours(self.current_pixmap, intens,
                                       alpha=self.alpha,
                                       cmap=self.cmap,
                                       levels=6,
                                       interpolation='bilinear')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

