import streamlit as st
import pydicom
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from skimage import exposure
from scipy import ndimage
import pandas as pd

def main():
    st.title("Midterm Final Project of Medical Image Processing🩻")
    st.markdown("### Yohanes Eka Adi Prasetyo - 5023221016")

    # Path input for DICOM directory
    path_input = st.text_input(
        "Path to DICOM directory", 
        value=r"C:\Users\Yohanes\OneDrive\Documents\COLLEGE\6th Semester\Medical Image Processing\B.Nada 2\SE000001"
    )
    
    path_to_head_mri = Path(path_input)
    
    if not path_to_head_mri.exists():
        st.error(f"Path tidak ditemukan: {path_to_head_mri}")
        return
    
    # Load DICOM files
    with st.spinner("Loading DICOM files..."):
        all_files = list(path_to_head_mri.glob("*"))
        
        if not all_files:
            st.error("Tidak ada file yang ditemukan di direktori tersebut.")
            return
        
        # Read DICOM files
        mri_data = []
        for path in all_files:
            try:
                data = pydicom.dcmread(path)
                mri_data.append(data)
            except Exception as e:
                st.warning(f"Error membaca file {path.name}: {str(e)}")
        
        # Sort by SliceLocation
        try:
            mri_data_ordered = sorted(mri_data, key=lambda slice: slice.SliceLocation)
            st.success(f"Berhasil memuat {len(mri_data_ordered)} DICOM slices.")
        except AttributeError:
            st.warning("File DICOM tidak memiliki atribut SliceLocation. Menggunakan urutan asli.")
            mri_data_ordered = mri_data
    
    # Create full volume
    full_volume = np.array([slice_data.pixel_array for slice_data in mri_data_ordered])
    st.write(f"Ukuran volume: {full_volume.shape}")
    
    # Slider to select slice
    slice_number = st.slider("Pilih slice", 0, len(full_volume) - 1, 15)
    selected_image = full_volume[slice_number].copy()
    
    # Normalize image to 0-255 range
    if selected_image.max() > selected_image.min():
        normalized_image = ((selected_image - selected_image.min()) / 
                           (selected_image.max() - selected_image.min()) * 255).astype(np.uint8)
    else:
        normalized_image = selected_image.astype(np.uint8)
    
    # Display simplified DICOM metadata for selected slice
    with st.expander("DICOM Metadata"):
        selected_slice = mri_data_ordered[slice_number]
        important_tags = [
            ('Modality', 'Modality'), ('PatientName', 'Patient Name'),
            ('PatientID', 'Patient ID'), ('PatientSex', 'Patient Sex'),
            ('PatientAge', 'Patient Age'), ('StudyDate', 'Study Date'),
            ('SeriesDate', 'Series Date'), ('StudyDescription', 'Study Description'),
            ('SeriesDescription', 'Series Description'), ('Manufacturer', 'Manufacturer'),
            ('SliceLocation', 'Slice Location'), ('SliceThickness', 'Slice Thickness'),
            ('PixelSpacing', 'Pixel Spacing'), ('Rows', 'Rows'),
            ('Columns', 'Columns')
        ]
        
        metadata_col1, metadata_col2 = st.columns(2)
        for i, (tag, display_name) in enumerate(important_tags):
            col = metadata_col1 if i < len(important_tags)/2 else metadata_col2
            try:
                if hasattr(selected_slice, tag):
                    value = getattr(selected_slice, tag)
                    col.write(f"**{display_name}:** {value}")
            except:
                pass
    
    # Helper functions for visualization
    def create_image_figure(image, title):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
        ax.axis('off')
        return fig
    
    # MODIFIED: Changed from bar to line plot
    def create_histogram_figure(image, title, color):
        hist_fig = Figure(figsize=(4, 2))
        ax = hist_fig.add_subplot(111)
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 255])
        
        # Use line plot instead of bar chart
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, hist, color=color, linewidth=1.5)
        
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Pixel Value", fontsize=7)
        ax.set_ylabel("Frequency", fontsize=7)
        ax.tick_params(axis='both', which='major', labelsize=6)
        ax.grid(True, alpha=0.3)
        hist_fig.tight_layout()
        return hist_fig
    
    # Simplified image processing functions
    def convolve2d(image, kernel):
        i_height, i_width = image.shape
        k_height, k_width = kernel.shape
        pad_height, pad_width = k_height // 2, k_width // 2
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')
        output = np.zeros_like(image, dtype=np.float32)
        
        for y in range(i_height):
            for x in range(i_width):
                region = padded_image[y:y+k_height, x:x+k_width]
                output[y, x] = np.sum(region * kernel)
        
        return np.clip(output, 0, 255).astype(np.uint8)
    
    # Basic image enhancement functions with scratch implementation
    def histogram_equalization(image):
        # Calculate histogram
        hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 255])
        
        # Calculate cumulative distribution function (CDF)
        cdf = hist.cumsum()
        
        # Normalize CDF
        cdf = (cdf / cdf[-1]) * 255
        
        # Apply CDF to the image
        im_equalized = cdf[image]
        
        # Proper clipping
        im_equalized = im_equalized / np.amax(im_equalized)  # Divide by max value
        im_equalized = np.clip(im_equalized * 255, 0, 255).astype(np.uint8)  # Clip to 0-255 range
        
        return im_equalized
    
    # Mean filter with scratch implementation
    def mean_filter(image, filter_size=3):
        # Create an empty output image
        output = np.zeros_like(image, dtype=np.uint8)
        
        # Determine padding based on filter size
        pad_size = filter_size // 2
        
        # Apply mean filter
        for j in range(pad_size, image.shape[0] - pad_size):
            for i in range(pad_size, image.shape[1] - pad_size):
                # Initialize sum
                pixel_sum = 0
                
                # Sum pixels in the neighborhood
                for y in range(-pad_size, pad_size + 1):
                    for x in range(-pad_size, pad_size + 1):
                        pixel_sum += image[j + y, i + x]
                
                # Calculate mean and assign to output
                output[j, i] = int(pixel_sum / (filter_size * filter_size))
        
        return output
    
    # Filter implementations
    def box_filter(image, kernel_size=5):
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
        return convolve2d(image, kernel)
    
    # Gaussian filter with scratch implementation
    def gaussian_filter(image, sigma=2):
        # Create Gaussian kernel
        kernel_radius = int(4 * sigma)
        kernel_size = 2 * kernel_radius + 1
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate 2D Gaussian values
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - kernel_radius
                y = j - kernel_radius
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Apply kernel to image
        return convolve2d(image, kernel)
    
    def laplacian_filter(image):
        kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        result = convolve2d(image, kernel)
        if result.max() != result.min():
            result = ((result - result.min()) / (result.max() - result.min()) * 255).astype(np.uint8)
        else:
            result = np.zeros_like(result, dtype=np.uint8)
        return result
    
    # ------------------ Fungsi Evaluasi ------------------ #
    def mse(img1, img2):
        return np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    
    def psnr(mse_val, max_pixel_value=255):
        if mse_val == 0:
            return float('inf')
        return 10 * np.log10((max_pixel_value ** 2) / mse_val)
    
    # Process all images
    # 1. CDF Transform (Histogram Equalization)
    eq_image_255 = histogram_equalization(normalized_image)
    
    # 2. Adaptive Histogram Equalization (using skimage implementation for this complex algorithm)
    clahe = exposure.equalize_adapthist(normalized_image, clip_limit=0.003)
    clahe_255 = (clahe * 255).astype(np.uint8)
    
    # 4. Image Negative
    negative_image = 255 - normalized_image
    
    # Enhancement and filter options
    enhancement_options = {
        "Original": normalized_image,
        "Histogram Equalization": eq_image_255,
        "Clip Limit Adaptive Histogram Equalization": clahe_255,
        "Image Negative": negative_image
    }
    
    # Create tabs for different visualization modes
    tab1, tab2, tab3, tab4 = st.tabs(["Image Comparison", "Enhancement & Filters", "Evaluasi Filter", "Masking & Segmentation"])
    
    with tab1:
        st.subheader("Original and Enhanced Images with Histograms")
        
        # First row - Original, Histogram Equalization
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Image**")
            fig_orig = create_image_figure(normalized_image, f"Original Slice {slice_number}")
            st.pyplot(fig_orig)
            # Add histogram right below the image
            hist_fig_orig = create_histogram_figure(normalized_image, "Original Histogram", "blue")
            st.pyplot(hist_fig_orig)
        
        with col2:
            st.write("**2. Histogram Equalization**")
            fig_eq = create_image_figure(eq_image_255, "Hist Equalized")
            st.pyplot(fig_eq)
            # Add histogram right below the image
            hist_fig_eq = create_histogram_figure(eq_image_255, "Equalized Histogram", "green")
            st.pyplot(hist_fig_eq)
        
        # Second row - CLAHE, Image Negative
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**3. Clip Limit Adaptive Hist Equalization**")
            fig_clahe = create_image_figure(clahe_255, "Clip Limit Adaptive Hist Eq")
            st.pyplot(fig_clahe)
            # Add histogram right below the image
            hist_fig_clahe = create_histogram_figure(clahe_255, "CLAHE Histogram", "purple")
            st.pyplot(hist_fig_clahe)
        
        with col4:
            st.write("**4. Image Negative**")
            fig_neg = create_image_figure(negative_image, "Image Negative")
            st.pyplot(fig_neg)
            # Add histogram right below the image
            hist_fig_neg = create_histogram_figure(negative_image, "Negative Histogram", "red")
            st.pyplot(hist_fig_neg)
    
    with tab2:
        st.subheader("Enhancement & Filter Selection")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select enhancement method
            selected_enhancement = st.selectbox(
                "Pilih metode enhancement", 
                list(enhancement_options.keys())
            )
        
        with col2:
            # Select filter to apply
            selected_filter = st.selectbox(
                "Pilih filter", 
                ["No Filter", "Box Filter", "Gaussian Filter", "Laplacian Filter", 
                 "Mean Filter", "Max Filter", "Min Filter", "Median Filter"]
            )
        
        # Add kernel size / parameter selection based on filter type
        if selected_filter != "No Filter" and selected_filter != "Laplacian Filter":
            col_param1, col_param2 = st.columns(2)
            
            with col_param1:
                if selected_filter == "Gaussian Filter":
                    sigma = st.selectbox("Pilih Sigma:", [1, 2, 3, 4])
                    filter_param = sigma
                else:
                    kernel_size = st.selectbox("Pilih Kernel Size:", [3, 5, 7, 9])
                    filter_param = kernel_size
        
        # Get the base image from enhancement
        base_image = enhancement_options[selected_enhancement]
        
        # Apply selected filter based on selection
        if selected_filter == "No Filter":
            filtered_image = base_image
        elif selected_filter == "Box Filter":
            filtered_image = box_filter(base_image, kernel_size=filter_param)
        elif selected_filter == "Gaussian Filter":
            filtered_image = gaussian_filter(base_image, sigma=filter_param)
        elif selected_filter == "Laplacian Filter":
            filtered_image = laplacian_filter(base_image)
        elif selected_filter == "Mean Filter":
            filtered_image = mean_filter(base_image, filter_size=filter_param)
        elif selected_filter == "Max Filter":
            filtered_image = ndimage.maximum_filter(base_image, size=filter_param)
        elif selected_filter == "Min Filter":
            filtered_image = ndimage.minimum_filter(base_image, size=filter_param)
        elif selected_filter == "Median Filter":
            filtered_image = ndimage.median_filter(base_image, size=filter_param)
        
        # Display result image and histogram side by side
        result_col1, result_col2 = st.columns([3, 2])
        
        with result_col1:
            st.write(f"**{selected_enhancement} + {selected_filter}**")
            if selected_filter != "No Filter" and selected_filter != "Laplacian Filter":
                param_text = f"Sigma={filter_param}" if selected_filter == "Gaussian Filter" else f"Kernel Size={filter_param}"
                fig_filtered = create_image_figure(filtered_image, f"{selected_enhancement} + {selected_filter} ({param_text})")
            else:
                fig_filtered = create_image_figure(filtered_image, f"{selected_enhancement} + {selected_filter}")
            st.pyplot(fig_filtered)
        
        with result_col2:
            st.write("**Histogram**")
            hist_fig_filtered = create_histogram_figure(filtered_image, f"Histogram: {selected_filter}", "blue")
            st.pyplot(hist_fig_filtered)
            
            # Add original histogram for comparison
            hist_fig_original = create_histogram_figure(base_image, f"Histogram: Original {selected_enhancement}", "red")
            st.pyplot(hist_fig_original)
        
        # Filter Grid Display - Show all filters for selected enhancement with different kernel sizes
        st.subheader(f"All Filters for {selected_enhancement}")
        
        # Linear Filters
        st.write("**A. Filter Linear**")
        
        # Box Filter with different kernel sizes
        st.write("**Box Filter**")
        box_cols = st.columns(3)
        kernel_sizes = [3, 5, 7]
        
        for i, k_size in enumerate(kernel_sizes):
            with box_cols[i]:
                box_img = box_filter(base_image, kernel_size=k_size)
                fig_box = create_image_figure(box_img, f"Box Filter (k={k_size})")
                st.pyplot(fig_box)
                # Add small histogram under each filter
                hist_box = create_histogram_figure(box_img, f"Histogram (k={k_size})", "blue")
                st.pyplot(hist_box)
        
        # Gaussian Filter with different sigma values
        st.write("**Gaussian Filter**")
        gauss_cols = st.columns(3)
        sigma_values = [1, 2, 3]
        
        for i, sigma in enumerate(sigma_values):
            with gauss_cols[i]:
                gauss_img = gaussian_filter(base_image, sigma=sigma)
                fig_gauss = create_image_figure(gauss_img, f"Gaussian Filter (σ={sigma})")
                st.pyplot(fig_gauss)
                # Add small histogram under each filter
                hist_gauss = create_histogram_figure(gauss_img, f"Histogram (σ={sigma})", "green")
                st.pyplot(hist_gauss)
        
        # Laplacian Filter
        st.write("**Laplacian Filter**")
        lap_cols = st.columns([1, 2])
        
        with lap_cols[0]:
            lap_img = laplacian_filter(base_image)
            fig_lap = create_image_figure(lap_img, "Laplacian Filter")
            st.pyplot(fig_lap)
        
        with lap_cols[1]:
            hist_lap = create_histogram_figure(lap_img, "Laplacian Histogram", "purple")
            st.pyplot(hist_lap)
        
        # Mean Filter with different kernel sizes
        st.write("**Mean Filter**")
        mean_cols = st.columns(3)
        kernel_sizes = [3, 5, 7]
        
        for i, k_size in enumerate(kernel_sizes):
            with mean_cols[i]:
                mean_img = mean_filter(base_image, filter_size=k_size)
                fig_mean = create_image_figure(mean_img, f"Mean Filter (k={k_size})")
                st.pyplot(fig_mean)
                # Add small histogram under each filter
                hist_mean = create_histogram_figure(mean_img, f"Histogram (k={k_size})", "orange")
                st.pyplot(hist_mean)
        
        # Non-Linear Filters
        st.write("**B. Filter Non-Linear**")
        
        # Max Filter with different kernel sizes
        st.write("**Max Filter**")
        max_cols = st.columns(3)
        kernel_sizes = [3, 5, 7]
        
        for i, k_size in enumerate(kernel_sizes):
            with max_cols[i]:
                max_img = ndimage.maximum_filter(base_image, size=k_size)
                fig_max = create_image_figure(max_img, f"Max Filter (k={k_size})")
                st.pyplot(fig_max)
                # Add small histogram under each filter
                hist_max = create_histogram_figure(max_img, f"Histogram (k={k_size})", "red")
                st.pyplot(hist_max)
        
        # Min Filter with different kernel sizes
        st.write("**Min Filter**")
        min_cols = st.columns(3)
        kernel_sizes = [3, 5, 7]
        
        for i, k_size in enumerate(kernel_sizes):
            with min_cols[i]:
                min_img = ndimage.minimum_filter(base_image, size=k_size)
                fig_min = create_image_figure(min_img, f"Min Filter (k={k_size})")
                st.pyplot(fig_min)
                # Add small histogram under each filter
                hist_min = create_histogram_figure(min_img, f"Histogram (k={k_size})", "cyan")
                st.pyplot(hist_min)
        
        # Median Filter with different kernel sizes
        st.write("**Median Filter**")
        med_cols = st.columns(3)
        kernel_sizes = [3, 5, 7]
        
        for i, k_size in enumerate(kernel_sizes):
            with med_cols[i]:
                med_img = ndimage.median_filter(base_image, size=k_size)
                fig_med = create_image_figure(med_img, f"Median Filter (k={k_size})")
                st.pyplot(fig_med)
                # Add small histogram under each filter
                hist_med = create_histogram_figure(med_img, f"Histogram (k={k_size})", "magenta")
                st.pyplot(hist_med)
        
        # Original image for comparison
        st.write("**Original (No Filter)**")
        orig_cols = st.columns([1, 2])
        
        with orig_cols[0]:
            fig_orig = create_image_figure(base_image, "Original")
            st.pyplot(fig_orig)
        
        with orig_cols[1]:
            hist_orig = create_histogram_figure(base_image, "Original Histogram", "blue")
            st.pyplot(hist_orig)
    
    with tab3:
        st.subheader("Evaluasi Performa Filter")
        
        # Filter parameter options
        filter_params = {
            "Box Filter": [3, 5, 7, 9],
            "Gaussian Filter": [1, 2, 3, 4],  # Sigma values
            "Mean Filter": [3, 5, 7, 9],
            "Median Filter": [3, 5, 7, 9]
        }
        
        # Selection for enhancement method to evaluate
        enhancement_for_eval = st.selectbox(
            "Pilih metode enhancement untuk dievaluasi",
            list(enhancement_options.keys())
        )
        
        original_image = enhancement_options[enhancement_for_eval]
        
        # Display original image
        st.write("### Original Image")
        fig_orig = create_image_figure(original_image, f"{enhancement_for_eval}")
        st.pyplot(fig_orig)
        
        # Compute metrics for all filters and all parameters at once
        st.write("### Evaluasi Semua Filter dan Parameter")
        
        all_results = []
        
        # Generate results for all filter types and parameters
        for filter_type in filter_params.keys():
            for param in filter_params[filter_type]:
                if filter_type == "Box Filter":
                    filtered = box_filter(original_image, kernel_size=param)
                    param_label = f"Kernel Size = {param}"
                elif filter_type == "Gaussian Filter":
                    filtered = gaussian_filter(original_image, sigma=param)
                    param_label = f"Sigma = {param}"
                elif filter_type == "Mean Filter":
                    filtered = mean_filter(original_image, filter_size=param)
                    param_label = f"Kernel Size = {param}"
                elif filter_type == "Median Filter":
                    filtered = ndimage.median_filter(original_image, size=param)
                    param_label = f"Kernel Size = {param}"
                
                # Calculate metrics correctly
                mse_val = np.mean((original_image - filtered) ** 2)
                # PSNR calculation: 10 * log10(MAX^2 / MSE)
                # For images with pixel values between 0-255
                psnr_val = 10 * np.log10((255 ** 2) / mse_val) if mse_val > 0 else float('inf')
                
                all_results.append({
                    "Filter": filter_type,
                    "Parameter": param_label,
                    "MSE": mse_val,
                    "PSNR (dB)": psnr_val,
                    "Param Value": param,
                    "Filter Type": filter_type
                })
        
        # Create dataframe with all results
        all_results_df = pd.DataFrame(all_results)
        st.dataframe(all_results_df[["Filter", "Parameter", "MSE", "PSNR (dB)"]])
        
        # Plot comprehensive comparison charts
        st.write("### Grafik Perbandingan")
        
        # Create tabs for MSE and PSNR
        metric_tabs = st.tabs(["MSE", "PSNR"])
        
        with metric_tabs[0]:  # MSE Tab
            # Create a figure with subplots for each filter type
            fig_mse = plt.figure(figsize=(14, 10))
            
            # Plot MSE for each filter type
            for i, filter_type in enumerate(filter_params.keys(), 1):
                ax = fig_mse.add_subplot(2, 2, i)
                
                # Get data for this filter
                filter_data = all_results_df[all_results_df["Filter Type"] == filter_type]
                
                # Plot
                if filter_type == "Gaussian Filter":
                    x_label = "Sigma"
                else:
                    x_label = "Kernel Size"
                    
                ax.bar(filter_data["Param Value"].astype(str), filter_data["MSE"], color="coral")
                ax.set_title(f"MSE for {filter_type}")
                ax.set_xlabel(x_label)
                ax.set_ylabel("MSE (lower is better)")
                ax.grid(True, alpha=0.3)
            
            fig_mse.tight_layout()
            st.pyplot(fig_mse)
            
            # Also add a comparison across filter types
            st.write("#### Perbandingan MSE Antar Filter")
            
            # Create grouped bar chart for common kernel sizes
            common_kernel_sizes = [3, 5, 7]
            fig_mse_compare = plt.figure(figsize=(12, 6))
            ax_compare = fig_mse_compare.add_subplot(111)
            
            # Set width and positions for bars
            width = 0.2
            
            for i, filter_type in enumerate(["Box Filter", "Mean Filter", "Median Filter"]):
                # Filter data for common kernel sizes
                filter_data = all_results_df[(all_results_df["Filter Type"] == filter_type) & 
                                            (all_results_df["Param Value"].isin(common_kernel_sizes))]
                
                # Get MSE values
                mse_values = []
                for k in common_kernel_sizes:
                    val = filter_data[filter_data["Param Value"] == k]["MSE"].values
                    mse_values.append(val[0] if len(val) > 0 else 0)
                
                # Plot positions
                positions = np.array(range(len(common_kernel_sizes))) + width * (i - 1)
                ax_compare.bar(positions, mse_values, width, label=filter_type)
            
            # For Gaussian filter use comparable sigma values
            gauss_sigmas = [1, 2, 3]  # Roughly equivalent to kernel sizes 3, 5, 7
            gauss_data = all_results_df[(all_results_df["Filter Type"] == "Gaussian Filter") & 
                                      (all_results_df["Param Value"].isin(gauss_sigmas))]
            
            # Get MSE values for Gaussian
            gauss_mse_values = []
            for s in gauss_sigmas:
                val = gauss_data[gauss_data["Param Value"] == s]["MSE"].values
                gauss_mse_values.append(val[0] if len(val) > 0 else 0)
            
            # Plot Gaussian
            positions = np.array(range(len(common_kernel_sizes))) + width * 2
            ax_compare.bar(positions, gauss_mse_values, width, label="Gaussian Filter")
            
            # Set labels and legend
            ax_compare.set_xlabel("Kernel Size (or equivalent Sigma for Gaussian)")
            ax_compare.set_ylabel("MSE (lower is better)")
            ax_compare.set_title("MSE Comparison Across Filter Types")
            ax_compare.set_xticks(range(len(common_kernel_sizes)))
            ax_compare.set_xticklabels(common_kernel_sizes)
            ax_compare.legend()
            ax_compare.grid(True, alpha=0.3)
            
            st.pyplot(fig_mse_compare)
        
        with metric_tabs[1]:  # PSNR Tab
            # Create a figure with subplots for each filter type
            fig_psnr = plt.figure(figsize=(14, 10))
            
            # Plot PSNR for each filter type
            for i, filter_type in enumerate(filter_params.keys(), 1):
                ax = fig_psnr.add_subplot(2, 2, i)
                
                # Get data for this filter
                filter_data = all_results_df[all_results_df["Filter Type"] == filter_type]
                
                # Plot
                if filter_type == "Gaussian Filter":
                    x_label = "Sigma"
                else:
                    x_label = "Kernel Size"
                    
                ax.bar(filter_data["Param Value"].astype(str), filter_data["PSNR (dB)"], color="skyblue")
                ax.set_title(f"PSNR for {filter_type}")
                ax.set_xlabel(x_label)
                ax.set_ylabel("PSNR (dB) (higher is better)")
                ax.grid(True, alpha=0.3)
            
            fig_psnr.tight_layout()
            st.pyplot(fig_psnr)
            
            # Also add a comparison across filter types
            st.write("#### Perbandingan PSNR Antar Filter")
            
            # Create grouped bar chart for common kernel sizes
            fig_psnr_compare = plt.figure(figsize=(12, 6))
            ax_compare = fig_psnr_compare.add_subplot(111)
            
            # Set width and positions for bars
            width = 0.2
            
            for i, filter_type in enumerate(["Box Filter", "Mean Filter", "Median Filter"]):
                # Filter data for common kernel sizes
                filter_data = all_results_df[(all_results_df["Filter Type"] == filter_type) & 
                                           (all_results_df["Param Value"].isin(common_kernel_sizes))]
                
                # Get PSNR values
                psnr_values = []
                for k in common_kernel_sizes:
                    val = filter_data[filter_data["Param Value"] == k]["PSNR (dB)"].values
                    psnr_values.append(val[0] if len(val) > 0 else 0)
                
                # Plot positions
                positions = np.array(range(len(common_kernel_sizes))) + width * (i - 1)
                ax_compare.bar(positions, psnr_values, width, label=filter_type)
            
            # For Gaussian filter use comparable sigma values
            gauss_psnr_values = []
            for s in gauss_sigmas:
                val = gauss_data[gauss_data["Param Value"] == s]["PSNR (dB)"].values
                gauss_psnr_values.append(val[0] if len(val) > 0 else 0)
            
            # Plot Gaussian
            positions = np.array(range(len(common_kernel_sizes))) + width * 2
            ax_compare.bar(positions, gauss_psnr_values, width, label="Gaussian Filter")
            
            # Set labels and legend
            ax_compare.set_xlabel("Kernel Size (or equivalent Sigma for Gaussian)")
            ax_compare.set_ylabel("PSNR (dB) (higher is better)")
            ax_compare.set_title("PSNR Comparison Across Filter Types")
            ax_compare.set_xticks(range(len(common_kernel_sizes)))
            ax_compare.set_xticklabels(common_kernel_sizes)
            ax_compare.legend()
            ax_compare.grid(True, alpha=0.3)
            
            st.pyplot(fig_psnr_compare)
        
        # Visual comparison of filters (using middle parameter for each)
        st.write("### Visual Comparison of Filters")
        
        # Get middle parameter for each filter
        box_param = filter_params["Box Filter"][1]  # e.g., 5
        gauss_param = filter_params["Gaussian Filter"][1]  # e.g., 2
        mean_param = filter_params["Mean Filter"][1]  # e.g., 5
        median_param = filter_params["Median Filter"][1]  # e.g., 5
        
        # Apply filters
        box_filtered = box_filter(original_image, kernel_size=box_param)
        gauss_filtered = gaussian_filter(original_image, sigma=gauss_param)
        mean_filtered = mean_filter(original_image, filter_size=mean_param)
        median_filtered = ndimage.median_filter(original_image, size=median_param)

        # Display in 2x2 grid
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Box Filter (k={box_param})**")
            fig_box = create_image_figure(box_filtered, f"Box Filter (k={box_param})")
            st.pyplot(fig_box)
            
            st.write(f"**Mean Filter (k={mean_param})**")
            fig_mean = create_image_figure(mean_filtered, f"Mean Filter (k={mean_param})")
            st.pyplot(fig_mean)

        with col2:
            st.write(f"**Gaussian Filter (σ={gauss_param})**")
            fig_gauss = create_image_figure(gauss_filtered, f"Gaussian Filter (σ={gauss_param})")
            st.pyplot(fig_gauss)
            
            st.write(f"**Median Filter (k={median_param})**")
            fig_median = create_image_figure(median_filtered, f"Median Filter (k={median_param})")
            st.pyplot(fig_median)

        # Cari hasil terbaik berdasarkan PSNR tertinggi (atau bisa MSE terendah juga)
        best_result = all_results_df.sort_values(by="PSNR (dB)", ascending=False).iloc[0]

        best_filter = best_result["Filter"]
        best_param = best_result["Parameter"]
        best_psnr = best_result["PSNR (dB)"]
        best_mse = best_result["MSE"]

        # Tampilkan kesimpulan otomatis
        st.write("### 📌 Rangkuman Otomatis Hasil Terbaik")
        st.markdown(f"""
        **Hasil terbaik berdasarkan nilai PSNR tertinggi diperoleh dari:**

        - 📌 **Metode Enhancement**: `{enhancement_for_eval}`
        - 🔍 **Filter**: `{best_filter}`
        - ⚙️ **Parameter**: `{best_param}`
        - 📈 **PSNR**: `{best_psnr:.2f} dB`
        - 📉 **MSE**: `{best_mse:.4f}`

        ✅ Kombinasi ini memberikan hasil paling optimal untuk menjaga kualitas gambar setelah filtering.
        """)

        # Tambahkan summary dari Kode 2
        st.write("### 🧠 Kesimpulan Akhir Evaluasi Semua Metode Enhancement")

        summary_data = []

        # Loop untuk setiap metode enhancement
        for enh_name, enh_img in enhancement_options.items():
            results = []

            for filter_type in filter_params:
                for param in filter_params[filter_type]:
                    if filter_type == "Box Filter":
                        filtered = box_filter(enh_img, kernel_size=param)
                    elif filter_type == "Gaussian Filter":
                        filtered = gaussian_filter(enh_img, sigma=param)
                    elif filter_type == "Mean Filter":
                        filtered = mean_filter(enh_img, filter_size=param)
                    elif filter_type == "Median Filter":
                        filtered = ndimage.median_filter(enh_img, size=param)

                    mse = np.mean((enh_img - filtered) ** 2)
                    psnr = 10 * np.log10((255 ** 2) / mse) if mse > 0 else float("inf")

                    results.append({
                        "Enhancement": enh_name,
                        "Filter": filter_type,
                        "Parameter": param,
                        "MSE": mse,
                        "PSNR": psnr
                    })

            # Ambil hasil terbaik berdasarkan PSNR
            df_result = pd.DataFrame(results)
            best = df_result.sort_values(by="PSNR", ascending=False).iloc[0]
            summary_data.append(best)

        # Buat DataFrame ringkasan akhir
        summary_df = pd.DataFrame(summary_data)
        summary_df["Parameter"] = summary_df["Parameter"].astype(str)

        # Tampilkan sebagai tabel
        st.dataframe(summary_df[["Enhancement", "Filter", "Parameter", "MSE", "PSNR"]])

        # Rangkuman tertulis
        st.write("### 📌 Rangkuman Kesimpulan Terbaik Setiap Enhancement")

        for idx, row in summary_df.iterrows():
            enhancement_for_eval = row['Enhancement']
            best_filter = row['Filter']
            best_param = row['Parameter']
            best_psnr = row['PSNR']
            best_mse = row['MSE']

            st.markdown(f"""
            - 📌 **Metode Enhancement**: `{enhancement_for_eval}`
            - 🔍 **Filter**: `{best_filter}`
            - ⚙️ **Parameter**: `{best_param}`
            - 📈 **PSNR**: `{best_psnr:.2f} dB`
            - 📉 **MSE**: `{best_mse:.4f}`

            ✅ Kombinasi ini memberikan hasil paling optimal untuk menjaga kualitas gambar setelah filtering.
            """)

   
         
    with tab4:
        # Step 1: Enhancement dan Filtering
        st.write("### 1. Enhancement dan Filtering")
        best_enhancement = "Adaptive Histogram Equalization"
        enhanced_image = clahe_255
        best_filtered_image = ndimage.median_filter(enhanced_image, size=3)

        col1, col2 = st.columns(2)
        with col1:
            st.image(normalized_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(best_filtered_image, caption="Enhanced + Median Filter (3x3)", use_container_width=True)

        # Step 2: Histogram dan Thresholding
        st.write("### 2. Histogram & Thresholding")
        fig_histogram = plt.figure(figsize=(8, 4))
        plt.hist(best_filtered_image.ravel(), bins=256, color='gray')
        plt.title("Histogram (X = Gray Value, Y = Number of Pixels)")
        plt.xlabel("Gray Value")
        plt.ylabel("Jumlah Pixel")
        plt.grid(alpha=0.3)
        st.pyplot(fig_histogram)

        corpus_threshold = st.slider("Threshold untuk Corpus", 
                                    min_value=0.1, 
                                    max_value=0.9, 
                                    value=0.5, 
                                    step=0.05,
                                    format="%.2f",
                                    help="Threshold untuk deteksi area corpus berdasarkan AHE + Median")
        
        # Step 2.5: Binarization
        mask_corpus = (best_filtered_image >= corpus_threshold * best_filtered_image.max()).astype(bool)
        st.image(mask_corpus.astype(np.uint8) * 255, caption="Binary Mask Sebelum Morphological Cleaning", use_container_width=True)

        # Step 3: Morphological Cleaning 
        st.write("### 3. Morphological Cleaning (Remove Small Objects)")
        min_size = st.slider("Minimum object size to keep", 
                            min_value=50, 
                            max_value=2000, 
                            value=1000,
                            help="Objek dengan ukuran di bawah nilai ini akan dihapus")
        from skimage.morphology import remove_small_objects
        cleaned_mask = remove_small_objects(mask_corpus, min_size=min_size).astype(np.uint8) * 255
        st.image(cleaned_mask, caption=f"After Removing Small Objects (min_size={min_size})", use_container_width=True)

        # Step 4: Masking dan Overlay
        st.write("### 4. Masking dan Overlay")
        masked_binary = (cleaned_mask > 0)
        st.image(masked_binary.astype(np.uint8) * 255, caption="Binary Image Dalam Area Corpus", use_container_width=True)

        overlay_fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(normalized_image, cmap='gray')
        ax.imshow(masked_binary, alpha=0.5, cmap='Greens')
        ax.set_title("Overlay Segmentasi Corpus Callosum di Gambar Original")
        ax.axis('off')
        st.pyplot(overlay_fig)

        # Step 5: Labeling Connected Regions
        st.write("### 5. Labeling Connected Regions")
        from skimage import measure
        from matplotlib.colors import ListedColormap
        labels = measure.label(masked_binary)
        nlabels = labels.max()
        rand_cmap = ListedColormap(np.random.rand(nlabels + 1, 3))
        labels_for_display = np.where(labels > 0, labels, np.nan)

        label_fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(normalized_image, cmap='gray')
        ax.imshow(labels_for_display, cmap=rand_cmap, alpha=0.7)
        ax.set_title(f'Labeled Cells (Only Inside Corpus) - {nlabels} Blobs')
        ax.axis('off')
        st.pyplot(label_fig)

        # Step 6: Visualisasi Region Properties
        st.write("### 6. Visualisasi Region Properties")
        import math
        regions = measure.regionprops(labels, intensity_image=normalized_image)
        fig, ax = plt.subplots()
        ax.imshow(normalized_image, cmap=plt.cm.gray)

        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length

            ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
            ax.plot(x0, y0, '.g', markersize=15)

            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-b', linewidth=2.5)

        ax.set_xlim(0, normalized_image.shape[1])
        ax.set_ylim(normalized_image.shape[0], 0)
        st.pyplot(fig)

        # Step 7: Enhanced Feature Extraction
        st.write("### 7. Enhanced Feature Extraction")
        from skimage.measure import regionprops_table
        props_table = regionprops_table(
            labels,
            intensity_image=normalized_image,
            properties=(
                'label',
                'area',
                'bbox_area',
                'centroid',
                'weighted_centroid',
                'eccentricity',
                'equivalent_diameter',
                'orientation',
                'major_axis_length',
                'minor_axis_length',
                'perimeter',
                'solidity',
                'mean_intensity',
                'max_intensity',
                'min_intensity',
                'weighted_moments_hu'
            )
        )
        df_props = pd.DataFrame(props_table)
        df_props = df_props.rename(columns={
            'centroid-0': 'Centroid Y',
            'centroid-1': 'Centroid X',
            'weighted_centroid-0': 'Weighted Centroid Y',
            'weighted_centroid-1': 'Weighted Centroid X',
            'label': 'Region Label'
        })
        numeric_cols = df_props.select_dtypes(include=[np.number]).columns
        df_props[numeric_cols] = df_props[numeric_cols].round(3)

        st.write("#### Complete Feature Table")
        st.dataframe(df_props)

        centroid_cols = ['Region Label', 'Centroid X', 'Centroid Y', 'Weighted Centroid X', 'Weighted Centroid Y']
        df_centroids = df_props[centroid_cols].copy()
        st.write("#### Centroid Information")
        st.dataframe(df_centroids)

        st.write("#### Statistical Summary of Features")
        st.dataframe(df_props.describe())

        # Step 8: Export ke Excel
        st.write("### 8. Export Data ke Excel")
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_props.to_excel(writer, sheet_name="Comprehensive Features", index=False)
            df_centroids.to_excel(writer, sheet_name="Centroid Data", index=False)
            df_props.describe().to_excel(writer, sheet_name="Summary Stats", index=True)
        output.seek(0)

        st.download_button(
            label="💾 Download Enhanced Features Excel",
            data=output,
            file_name="enhanced_corpus_callosum_features.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()