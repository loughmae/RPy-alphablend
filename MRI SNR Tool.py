import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pydicom
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import scipy.ndimage as ndi


# GUI Application
class MRIApp:
    def __init__(self, master):
        self.master = master
        self.master.title("MRI DICOM Image SNR Calculator")

        # User Instructions Label
        instructions = (
            "1. Click 'Load DICOM/IMA File' to load an image.\n"
            "2. Hover over the image to see grey values.\n"
            "3. Click on a point in the image to select a seed for region growing.\n"
            "4. Adjust the thresholds and click 'Start Region Growing' to view segmentation.\n"
            "5. SNR will be calculated automatically once segmentation is successful."
        )
        self.instruction_label = tk.Label(self.master, text=instructions, justify=tk.LEFT)
        self.instruction_label.pack()

        # Button to load file
        self.load_button = tk.Button(self.master, text="Load DICOM/IMA File", command=self.load_file)
        self.load_button.pack()

        # Frame for image and histogram
        self.image_frame = tk.Frame(self.master)
        self.image_frame.pack()

        # Canvas to display image
        self.canvas = tk.Canvas(self.image_frame, width=512, height=512)
        self.canvas.grid(row=0, column=0)

        # Grey Value Display Label
        self.grey_value_label = tk.Label(self.master, text="Grey Value: N/A")
        self.grey_value_label.pack()

        # Histogram display using Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(4, 2))
        self.hist_canvas = FigureCanvasTkAgg(self.fig, master=self.image_frame)
        self.hist_canvas.get_tk_widget().grid(row=0, column=1)

        # Sliders for controlling lower and upper thresholds
        self.lower_thresh_slider = tk.Scale(self.master, orient=tk.HORIZONTAL, label="Lower Threshold")
        self.lower_thresh_slider.pack()
        self.upper_thresh_slider = tk.Scale(self.master, orient=tk.HORIZONTAL, label="Upper Threshold")
        self.upper_thresh_slider.pack()

        # SNR Display Labels
        self.snr_stdbkgrd_label = tk.Label(self.master, text="SNR (Std Background): N/A")
        self.snr_stdbkgrd_label.pack()
        self.snr_meanbkgrd_label = tk.Label(self.master, text="SNR (Mean Background): N/A")
        self.snr_meanbkgrd_label.pack()

        # Button to start region growing
        self.start_button = tk.Button(self.master, text="Start Region Growing", command=self.start_region_growing)
        self.start_button.pack()

        # Windowing adjustment sliders
        self.window_level_slider = tk.Scale(self.master, orient=tk.HORIZONTAL, label="Window Level",
                                            command=self.update_display)
        self.window_level_slider.pack()
        self.window_width_slider = tk.Scale(self.master, orient=tk.HORIZONTAL, label="Window Width",
                                            command=self.update_display)
        self.window_width_slider.pack()


        # Button to reset the app
        self.reset_button = tk.Button(self.master, text="Reset", command=self.reset)
        self.reset_button.pack()

        self.image_data = None
        self.seed = None

    def load_file(self):
        try:
            # Open file dialog to load DICOM/IMA file, include multiple file types and an "All Files" option
            filepath = filedialog.askopenfilename(
                filetypes=[("DICOM Files", "*.dcm *.ima"), ("All Files", "*.*")]
            )
            if filepath:
                # Read the DICOM file
                self.dicom_data = pydicom.dcmread(filepath)

                if hasattr(self.dicom_data, 'pixel_array'):
                    self.image_data = self.dicom_data.pixel_array
                    # Bind click event for selecting seed point
                    self.canvas.bind("<Button-1>", self.on_click)
                    # Bind motion event to show grey value
                    self.canvas.bind("<Motion>", self.on_motion)

                    # Set dynamic threshold sliders range based on image data
                    min_val = np.min(self.image_data)
                    max_val = np.max(self.image_data)
                    self.lower_thresh_slider.config(from_=min_val, to=max_val)
                    self.upper_thresh_slider.config(from_=min_val, to=max_val)

                    # Set threshold sliders to the initial range
                    self.lower_thresh_slider.set(min_val)
                    self.upper_thresh_slider.set(max_val)

                    # Set windowing sliders to adjust the display
                    self.window_level_slider.config(from_=min_val, to=max_val)
                    self.window_width_slider.config(from_=1, to=(max_val - min_val))
                    self.window_level_slider.set((min_val + max_val) // 2)
                    self.window_width_slider.set((max_val - min_val) // 2)

                    # Display image and histogram
                    self.display_image()
                    self.update_histogram()
                else:
                    messagebox.showerror("Error", "Invalid DICOM file. No pixel data found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load DICOM/IMA file: {str(e)}")

    def display_image(self, mask=None):
        try:
            # Apply windowing adjustments
            window_level = self.window_level_slider.get()
            window_width = self.window_width_slider.get()
            min_pixel = window_level - (window_width / 2)
            max_pixel = window_level + (window_width / 2)

            # Clip and normalize the image for display
            windowed_image = np.clip(self.image_data, min_pixel, max_pixel)
            norm_image = ((windowed_image - min_pixel) / (max_pixel - min_pixel) * 255).astype(np.uint8)

            if mask is not None:
                norm_image = np.stack([norm_image] * 3, axis=-1)  # Convert to RGB
                norm_image[mask == 1] = [255, 0, 0]  # Signal region in red
                norm_image[mask == 2] = [0, 255, 0]  # Buffer region in green

            pil_image = Image.fromarray(norm_image)
            self.tk_image = ImageTk.PhotoImage(pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            # Update histogram with shaded region
            self.update_histogram()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def on_motion(self, event):
        # Display grey value when moving over the image
        if self.image_data is not None:
            x, y = event.x, event.y
            if 0 <= x < self.image_data.shape[1] and 0 <= y < self.image_data.shape[0]:
                grey_value = self.image_data[y, x]
                self.grey_value_label.config(text=f"Grey Value: {grey_value}")

    def on_click(self, event):
        # Check if an image is loaded
        if self.image_data is None:
            messagebox.showwarning("Warning", "Please load an image first.")
            return

        # Get seed point from mouse click, ensure it's within bounds
        x, y = event.x, event.y
        if x >= self.image_data.shape[1] or y >= self.image_data.shape[0]:
            messagebox.showwarning("Warning", "Click within the image bounds.")
            return

        self.seed = (y, x)
        print(f"Seed point selected: {self.seed}")
        messagebox.showinfo("Info",
                            f"Seed point selected at: {self.seed}. Adjust thresholds and click 'Start Region Growing'.")

    def start_region_growing(self):
        if self.seed is None:
            messagebox.showwarning("Warning", "Please select a seed point first.")
            return

        if self.image_data is not None:
            try:
                # Get thresholds from sliders
                lower_thresh = self.lower_thresh_slider.get()
                upper_thresh = self.upper_thresh_slider.get()

                # Perform region growing for the signal region
                mask = self.region_growing(self.image_data, self.seed, lower_thresh, upper_thresh)

                if mask is None or np.sum(mask) == 0:
                    # If signal region growing fails, try growing the background
                    messagebox.showwarning("Warning",
                                           "Segmentation for the signal region failed. Trying to grow background.")
                    mask = self.region_growing(self.image_data, self.seed, 0, lower_thresh)

                if mask is not None and np.sum(mask) > 0:
                    # Create buffer region around the segmented signal
                    buffer = self.create_buffer_region(mask)

                    # Display the segmented image with mask and buffer
                    mask_display = mask + buffer * 2
                    self.display_image(mask=mask_display)

                    # Calculate both SNRs excluding the buffer region
                    snr_stdbkgrd, snr_meanbkgrd = self.calculate_snr(self.image_data, mask, buffer)
                    self.snr_stdbkgrd_label.config(text=f"SNR (Std Background): {snr_stdbkgrd:.2f}")
                    self.snr_meanbkgrd_label.config(text=f"SNR (Mean Background): {snr_meanbkgrd:.2f}")
                else:
                    messagebox.showwarning("Warning",
                                           "Segmentation failed. Try adjusting the thresholds or selecting another seed point.")
            except Exception as e:
                messagebox.showerror("Error", f"Region growing failed: {str(e)}")

    def region_growing(self, image, seed, lower, upper):
        try:
            # Convert image to SimpleITK Image
            sitk_image = sitk.GetImageFromArray(image)

            # Perform region growing based on seed point and threshold values
            seg = sitk.ConnectedThreshold(sitk_image, seedList=[seed], lower=lower, upper=upper)

            # Return mask as a numpy array
            mask = sitk.GetArrayFromImage(seg)

            # Ensure the mask has valid regions
            if np.sum(mask) > 0:
                return mask
            else:
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Region growing failed: {str(e)}")
            return None

    def create_buffer_region(self, mask):
        try:
            # Create a buffer by dilating the mask
            dilated_mask = ndi.binary_dilation(mask, iterations=3)
            buffer = dilated_mask ^ mask  # Buffer is the difference between dilated and original mask
            return buffer.astype(int)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create buffer region: {str(e)}")
            return np.zeros_like(mask)

    def calculate_snr(self, image, mask, buffer):
        try:
            # Create a binary inverse mask for the background region, excluding the buffer
            signal_region = mask == 1
            background_mask = (mask == 0) & (buffer == 0)

            # Calculate mean signal in the region (foreground) and background
            signal_values = image[signal_region]
            background_values = image[background_mask]

            if len(signal_values) == 0 or len(background_values) == 0:
                messagebox.showerror("Error", "Invalid regions for SNR calculation. No valid pixels found.")
                return 0, 0

            signal_mean = np.mean(signal_values)
            noise_std = np.std(background_values)
            background_mean = np.mean(background_values)

            # SNR calculations
            snr_stdbkgrd = signal_mean / noise_std
            snr_meanbkgrd = signal_mean / background_mean

            return snr_stdbkgrd, snr_meanbkgrd
        except Exception as e:
            messagebox.showerror("Error", f"Failed to calculate SNR: {str(e)}")
            return 0, 0

    def update_histogram(self):
        try:
            self.ax.clear()
            self.ax.hist(self.image_data.ravel(), bins=256, color='blue', alpha=0.7)

            # Add shaded region for window level and width
            window_level = self.window_level_slider.get()
            window_width = self.window_width_slider.get()
            min_pixel = window_level - (window_width / 2)
            max_pixel = window_level + (window_width / 2)

            self.ax.axvspan(min_pixel, max_pixel, color='red', alpha=0.3)
            self.ax.set_title("Intensity Histogram")
            self.ax.set_xlabel("Intensity Value")
            self.ax.set_ylabel("Frequency")
            self.hist_canvas.draw()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update histogram: {str(e)}")

    def update_display(self, _event=None):
        # Update the image display and histogram when windowing sliders are adjusted
        self.display_image()

    def reset(self):
        # Reset the application
        self.canvas.delete("all")
        self.image_data = None
        self.seed = None
        self.lower_thresh_slider.set(0)
        self.upper_thresh_slider.set(0)
        self.window_level_slider.set(0)
        self.window_width_slider.set(0)
        self.snr_stdbkgrd_label.config(text="SNR (Std Background): N/A")
        self.snr_meanbkgrd_label.config(text="SNR (Mean Background): N/A")
        self.grey_value_label.config(text="Grey Value: N/A")
        self.ax.clear()
        self.hist_canvas.draw()
        self.master.title("MRI DICOM Image SNR Calculator")
        messagebox.showinfo("Info", "Application reset. Load a new image.")


if __name__ == "__main__":
    root = tk.Tk()
    app = MRIApp(root)
    root.mainloop()
