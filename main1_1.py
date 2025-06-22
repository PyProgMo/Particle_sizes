import cv2
import numpy as np
import matplotlib.pyplot as plt

class ParticleAnalyzer:
    def __init__(self, image_path, pixel_size_nm, particle_area_threshold_px=5, gradient_threshold=21.3, particle_x_range=(300, 4000), particle_y_range=(300, 4000)):
        """
        :param image_path: Path to the SEM image
        :param pixel_size_nm: Pixel size in nanometers (e.g., 11.16 for your image)
        :param particle_area_threshold_px: Minimum area in pixels to consider as a particle
        :param gradient_threshold: Threshold for gradient magnitude to detect particle edges
        """
        self.image_path = image_path
        self.image = self._load_image()

        self.parameters = {
            'particle_area_threshold_px': particle_area_threshold_px,
            'gradient_threshold': gradient_threshold,
            'pixel_size_nm': pixel_size_nm,
            'particle_x_min': particle_x_range[0],
            'particle_x_max': particle_x_range[1],
            'particle_y_min': particle_y_range[0],
            'particle_y_max': particle_y_range[1],
        }

    def _load_image(self):
        """Load image in grayscale and crop out bottom text region."""
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        height = image.shape[0]
        cropped_image = image[:int(height * 0.95), :]  # remove bottom 5% (scale bar & text)
        return cropped_image
    
    def plot_image(self):
        """Visualize the original image."""
        plt.figure(figsize=(10, 6))
        plt.imshow(self.image, cmap='gray')
        plt.title("Original Image")
        plt.axis("off")
        plt.show()
    
    def update_parameters(self, parameters):
        for key, value in parameters.items():
            if key in self.parameters:
                self.parameters[key] = value
            else:
                print(f"Warning: Parameter '{key}' not recognized.")

    def analyze_particles(self):
        """Analyze particles in the image and extract their properties."""
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(self.image)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered_image = cv2.bilateralFilter(enhanced_image, 9, 75, 75)
        
        # Calculate gradient for edge detection using Sobel operator
        grad_x = cv2.Sobel(filtered_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(filtered_image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = cv2.magnitude(grad_x, grad_y)
        
        # Normalize gradient magnitude to 0-255 range
        gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Apply threshold based on the gradient_threshold parameter
        _, gradient_binary = cv2.threshold(gradient_magnitude, self.parameters['gradient_threshold'], 255, cv2.THRESH_BINARY)
        
        # Apply threshold with Otsu's method on the original filtered image
        _, intensity_binary = cv2.threshold(filtered_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Combine gradient-based and intensity-based binary images
        binary_image = cv2.bitwise_and(gradient_binary, intensity_binary)
        
        # Clean up the binary image with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours in the binary image
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= self.parameters['particle_area_threshold_px']]
        
        # Extract and store particle properties
        self.particles = []
        for i, contour in enumerate(valid_contours):
            # Fit an ellipse to the contour if it has enough points
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                
                # Calculate size in nm
                major_axis_nm = axes[0] * self.parameters['pixel_size_nm']
                minor_axis_nm = axes[1] * self.parameters['pixel_size_nm']
                
                # Skip particles outside the specified size ranges
                if (major_axis_nm < self.parameters['particle_x_min'] or 
                    major_axis_nm > self.parameters['particle_x_max'] or
                    minor_axis_nm < self.parameters['particle_y_min'] or
                    minor_axis_nm > self.parameters['particle_y_max']):
                    continue
                
                self.particles.append({
                    'id': i,
                    'center': center,
                    'major_axis_nm': major_axis_nm,
                    'minor_axis_nm': minor_axis_nm,
                    'angle': angle,
                    'contour': contour
                })
        
        return self.particles
        
    def visualize_results(self, show_ellipses=True, show_ids=False):
        """Visualize the detected particles on the original image."""
        # Create a color version of the image for visualization
        vis_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        
        # Make sure particles have been detected
        if not hasattr(self, 'particles'):
            self.analyze_particles()
        
        # Draw contours and ellipses
        for particle in self.particles:
            # Draw contour
            cv2.drawContours(vis_image, [particle['contour']], -1, (0, 255, 0), 1)
            
            # Draw fitted ellipse
            if show_ellipses:
                center = (int(particle['center'][0]), int(particle['center'][1]))
                axes = (int(particle['major_axis_nm']/self.parameters['pixel_size_nm']/2), 
                        int(particle['minor_axis_nm']/self.parameters['pixel_size_nm']/2))
                angle = particle['angle']
                cv2.ellipse(vis_image, center, axes, angle, 0, 360, (255, 0, 0), 2)
            
            # Optionally show particle IDs
            if show_ids:
                cv2.putText(vis_image, str(particle['id']), 
                            (int(particle['center'][0]), int(particle['center'][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Display the result
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Particles: {len(self.particles)}")
        plt.axis("off")
        plt.show()
        
    def get_particle_statistics(self):
        """Calculate statistics about particle sizes."""
        if not hasattr(self, 'particles'):
            self.analyze_particles()
        
        # Extract dimensions
        major_axes = [p['major_axis_nm'] for p in self.particles]
        minor_axes = [p['minor_axis_nm'] for p in self.particles]
        
        # Calculate statistics
        stats = {
            'count': len(self.particles),
            'mean_major_axis_nm': np.mean(major_axes),
            'mean_minor_axis_nm': np.mean(minor_axes),
            'median_major_axis_nm': np.median(major_axes),
            'median_minor_axis_nm': np.median(minor_axes),
            'min_size_nm': min([min(major, minor) for major, minor in zip(major_axes, minor_axes)]),
            'max_size_nm': max([max(major, minor) for major, minor in zip(major_axes, minor_axes)])
        }
        
        return stats

    def plot_size_distribution(self):
        """Plot the size distribution of particles."""
        if not hasattr(self, 'particles'):
            self.analyze_particles()
        
        # Extract dimensions
        major_axes = [p['major_axis_nm'] for p in self.particles]
        minor_axes = [p['minor_axis_nm'] for p in self.particles]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot major axis histogram
        ax1.hist(major_axes, bins=50, alpha=0.7, color='blue')
        ax1.set_title('Major Axis Distribution')
        ax1.set_xlabel('Size (nm)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot minor axis histogram
        ax2.hist(minor_axes, bins=50, alpha=0.7, color='green')
        ax2.set_title('Minor Axis Distribution')
        ax2.set_xlabel('Size (nm)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

class Particle:
    def __init__(self, id, center, major_axis_nm, minor_axis_nm, angle):
        self.id = id
        self.center_x = center[0]
        self.center_y = center[1]
        self.major_axis_nm = major_axis_nm
        self.minor_axis_nm = minor_axis_nm
        self.angle = angle

    def get_diameter(self):
        return (self.major_axis_nm + self.minor_axis_nm) / 2

if __name__ == "__main__":
    analyzer = ParticleAnalyzer("E:\\programming\\Particlesizes\\Particle_sizes\\images\\pim1.jpg", pixel_size_nm=11.16, particle_area_threshold_px=100, gradient_threshold=20, Hessian_thresh=2, particle_x_range=(300, 4000), particle_y_range=(300, 4000))
    particles = analyzer.analyze_particles()
    print(f"Detected {len(particles)} particles")
    analyzer.visualize_results(show_ellipses=True, show_ids=True)
    stats = analyzer.get_particle_statistics()
    print("Particle Statistics:", stats)
    analyzer.plot_size_distribution()
