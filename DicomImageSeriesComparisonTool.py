import sys
import os
import numpy as np
import pydicom
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                             QSlider, QWidget, QMessageBox, QTableWidget,
                             QTableWidgetItem, QHeaderView, QInputDialog)
from PyQt5.QtCore import Qt, QEvent
import pyqtgraph as pg
import csv
from skimage.metrics import structural_similarity as ssim


class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Viewer and Analyzer")
        self.dicom_series1 = []
        self.dicom_series2 = []
        self.series_label1 = ""
        self.series_label2 = ""
        self.current_slice1 = 0
        self.current_slice2 = 0
        self.measurements = []
        self.roi_signal = None
        self.roi_noise = None
        self.roi_signal_mirror_item = None
        self.roi_noise_mirror_item = None
        self.ssim_window = None
        self.measurements_window = None
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        description_label = QLabel(
            "Note: Left Image is SSIM reference Image. Use the blue box for Signal ROI and the red box for Noise ROI. Hit Enter to finalize the ROI positions.")
        main_layout.addWidget(description_label)

        display_layout = QHBoxLayout()

        self.graphics_view1 = pg.ImageView()
        self.graphics_view2 = pg.ImageView()

        display_layout.addWidget(self.graphics_view1)
        display_layout.addWidget(self.graphics_view2)

        main_layout.addLayout(display_layout)

        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1.valueChanged.connect(self.update_images_from_slider1)
        self.slider1.setToolTip("Slide to navigate through slices of Series 1")
        main_layout.addWidget(self.slider1)

        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2.valueChanged.connect(self.update_images_from_slider2)
        self.slider2.setToolTip("Slide to navigate through slices of Series 2")
        main_layout.addWidget(self.slider2)

        load_button1 = QPushButton("Load DICOM Series 1")
        load_button1.clicked.connect(lambda: self.load_dicom_series(1))
        load_button1.setToolTip("Load the first DICOM series")
        load_button2 = QPushButton("Load DICOM Series 2")
        load_button2.clicked.connect(lambda: self.load_dicom_series(2))
        load_button2.setToolTip("Load the second DICOM series")
        analyse_button = QPushButton("Analyze")
        analyse_button.clicked.connect(self.analyze)
        analyse_button.setToolTip("Analyze the loaded DICOM series and display SSIM image")

        button_layout = QHBoxLayout()
        button_layout.addWidget(load_button1)
        button_layout.addWidget(load_button2)
        button_layout.addWidget(analyse_button)

        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

        self.initROIs()
        self.installEventFilter(self)

    def load_dicom_series(self, series_num):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "Select DICOM Files", "",
                                                "DICOM Files (*.dcm *.ima);;All Files (*)", options=options)
        if not files:
            return

        dicom_series = []
        series_label = ""
        for filepath in files:
            try:
                dicom_file = pydicom.dcmread(filepath)
                dicom_series.append(dicom_file)
                if not series_label:
                    series_label = getattr(dicom_file, 'SeriesDescription', None)
                    if not series_label:
                        series_label, ok = QInputDialog.getText(self, "Input Series Description",
                                                                "Enter series description:")
                        if not ok:
                            return
                print(f"Loaded file: {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error loading file {filepath}: {e}")
                print(f"Error loading file {filepath}: {e}")

        if not dicom_series:
            QMessageBox.warning(self, "No Files", "No valid DICOM files found.")
            return

        dicom_series.sort(key=lambda x: int(x.InstanceNumber))

        if series_num == 1:
            self.dicom_series1 = dicom_series
            self.series_label1 = series_label
        else:
            self.dicom_series2 = dicom_series
            self.series_label2 = series_label

        self.update_slider_range()
        self.update_images()

    def update_slider_range(self):
        if self.dicom_series1:
            self.slider1.setMaximum(len(self.dicom_series1) - 1)
        if self.dicom_series2:
            self.slider2.setMaximum(len(self.dicom_series2) - 1)

    def update_images(self):
        self.update_image(self.graphics_view1, self.dicom_series1, self.current_slice1)
        self.update_image(self.graphics_view2, self.dicom_series2, self.current_slice2)

    def update_images_from_slider1(self):
        self.current_slice1 = self.slider1.value()
        self.update_image(self.graphics_view1, self.dicom_series1, self.current_slice1)
        if self.current_slice1 < self.slider2.maximum():
            self.slider2.setValue(self.slider1.value())
        self.update_images_from_slider2()

    def update_images_from_slider2(self):
        self.current_slice2 = self.slider2.value()
        self.update_image(self.graphics_view2, self.dicom_series2, self.current_slice2)

    def update_image(self, graphics_view, dicom_series, current_slice):
        if dicom_series:
            try:
                image = self.get_image(dicom_series[current_slice])
                graphics_view.setImage(image, autoLevels=False, levels=(0, np.max(image)))
                self.apply_lut(graphics_view)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error updating image: {e}")
                print(f"Error updating image: {e}")

    def get_image(self, dicom):
        try:
            data = dicom.pixel_array
        except AttributeError as e:
            QMessageBox.critical(self, "Error", f"Error retrieving pixel data: {e}")
            data = np.zeros((512, 512))  # Fallback to a blank image
        return data

    def apply_lut(self, graphics_view):
        lut = pg.HistogramLUTItem()
        graphics_view.ui.histogram.setLevels(0, 255)
        graphics_view.ui.histogram.gradient.loadPreset('grey')
        graphics_view.ui.histogram.setImageItem(graphics_view.imageItem)

    def initROIs(self):
        self.roi_signal = pg.RectROI([20, 20], [20, 20], pen='b')
        self.roi_noise = pg.RectROI([60, 60], [20, 20], pen='r')

        self.roi_signal.setZValue(10)
        self.roi_noise.setZValue(10)

        self.graphics_view1.addItem(self.roi_signal)
        self.graphics_view1.addItem(self.roi_noise)

        self.roi_signal.sigRegionChanged.connect(self.update_mirrored_rois)
        self.roi_noise.sigRegionChanged.connect(self.update_mirrored_rois)

    def update_mirrored_rois(self):
        if self.roi_signal_mirror_item:
            self.graphics_view2.removeItem(self.roi_signal_mirror_item)
        if self.roi_noise_mirror_item:
            self.graphics_view2.removeItem(self.roi_noise_mirror_item)

        pos_signal = self.roi_signal.pos()
        size_signal = self.roi_signal.size()
        pos_noise = self.roi_noise.pos()
        size_noise = self.roi_noise.size()

        self.roi_signal_mirror_item = pg.RectROI(pos_signal, size_signal, pen='b')
        self.roi_noise_mirror_item = pg.RectROI(pos_noise, size_noise, pen='r')

        self.graphics_view2.addItem(self.roi_signal_mirror_item)
        self.graphics_view2.addItem(self.roi_noise_mirror_item)

        self.display_roi_areas()

    def display_roi_areas(self):
        signal_area = self.roi_signal.size().x() * self.roi_signal.size().y()
        noise_area = self.roi_noise.size().x() * self.roi_noise.size().y()
        self.setWindowTitle(
            f"DICOM Viewer and Analyzer - Signal ROI Area: {signal_area:.2f}, Noise ROI Area: {noise_area:.2f}")

    def eventFilter(self, source, event):
        if event.type() == QEvent.KeyPress and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self.analyze()
            return True
        return super(DICOMViewer, self).eventFilter(source, event)

    def analyze(self):
        if not self.dicom_series1 or not self.dicom_series2:
            QMessageBox.warning(self, "Missing Data", "Please load both DICOM series before analyzing.")
            return

        snr1, sdnr1 = self.calculate_snr_sdnr(self.dicom_series1, self.current_slice1)
        snr2, sdnr2 = self.calculate_snr_sdnr(self.dicom_series2, self.current_slice2)
        ssim_index, ssim_image = self.calculate_ssim(self.dicom_series1[self.current_slice1],
                                                     self.dicom_series2[self.current_slice2])

        title = f"SSIM: {ssim_index:.4f} | SNR1: {snr1:.4f} | SDNR1: {sdnr1:.4f} | SNR2: {snr2:.4f} | SDNR2: {sdnr2:.4f}"
        self.setWindowTitle(title)

        self.show_ssim_image(ssim_image)
        self.save_results_to_list(snr1, sdnr1, snr2, sdnr2, ssim_index)

    def calculate_snr_sdnr(self, series, current_slice):
        if current_slice >= len(series):
            QMessageBox.warning(self, "Slice Index Out of Range",
                                f"Current slice index {current_slice} is out of range for the series.")
            return None, None

        signal_values = self.extract_roi_values(series[current_slice], self.roi_signal)
        noise_values = self.extract_roi_values(series[current_slice], self.roi_noise)

        mean_signal = np.mean(signal_values)
        std_signal = np.std(signal_values)

        mean_noise = np.mean(noise_values)
        std_noise = np.std(noise_values)

        snr = mean_signal / std_noise
        sdnr = (mean_signal - mean_noise) / std_noise

        return snr, sdnr

    def extract_roi_values(self, dicom, roi):
        data = dicom.pixel_array
        x, y = map(int, roi.pos())
        w, h = map(int, roi.size())
        roi_values = data[y:y + h, x:x + w].flatten()
        return roi_values

    def calculate_ssim(self, dicom1, dicom2):
        image1 = dicom1.pixel_array
        image2 = dicom2.pixel_array
        data_range = max(image1.max(), image2.max()) - min(image1.min(), image2.min())
        ssim_index, ssim_image = ssim(image1, image2, data_range=data_range, full=True)
        return ssim_index, ssim_image

    def show_ssim_image(self, ssim_image):
        if self.ssim_window is not None:
            self.ssim_window.close()

        self.ssim_window = QMainWindow()
        self.ssim_window.setWindowTitle("SSIM Image")

        ssim_view = pg.ImageView()
        ssim_view.setImage(ssim_image, autoLevels=False, levels=(0, np.max(ssim_image)))

        self.ssim_window.setCentralWidget(ssim_view)
        self.ssim_window.show()

    def save_results_to_list(self, snr1, sdnr1, snr2, sdnr2, ssim_index):
        if snr1 is not None and sdnr1 is not None:
            self.measurements.append([self.series_label1, self.current_slice1, snr1, sdnr1, ssim_index])
        if snr2 is not None and sdnr2 is not None:
            self.measurements.append([self.series_label2, self.current_slice2, snr2, sdnr2, ssim_index])
        self.show_measurements()

    def show_measurements(self):
        if self.measurements_window is not None:
            self.measurements_window.close()

        self.measurements_window = QMainWindow()
        self.measurements_window.setWindowTitle("Measurements")

        table_widget = QTableWidget()
        table_widget.setRowCount(len(self.measurements))
        table_widget.setColumnCount(5)
        table_widget.setHorizontalHeaderLabels(["Series", "Slice", "SNR", "SDNR", "SSIM"])
        table_widget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for row, measurement in enumerate(self.measurements):
            for col, value in enumerate(measurement):
                table_widget.setItem(row, col, QTableWidgetItem(str(value)))

        save_button = QPushButton("Save Measurements to CSV")
        save_button.clicked.connect(self.save_measurements_to_csv)

        layout = QVBoxLayout()
        layout.addWidget(table_widget)
        layout.addWidget(save_button)

        container = QWidget()
        container.setLayout(layout)

        self.measurements_window.setCentralWidget(container)
        self.measurements_window.show()

    def save_measurements_to_csv(self):
        save_path = QFileDialog.getSaveFileName(self, "Save Measurements", "", "CSV Files (*.csv)")[0]
        if save_path:
            try:
                with open(save_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["Series", "Slice", "SNR", "SDNR", "SSIM"])
                    for measurement in self.measurements:
                        writer.writerow(measurement)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error saving measurements: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOMViewer()
    viewer.show()
    sys.exit(app.exec_())
