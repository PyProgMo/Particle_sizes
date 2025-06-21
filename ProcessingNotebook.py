import tkinter as tk
from tkinter import ttk
import sys
import os
from tkinter import filedialog
from tkinter import filedialog

# Add the parent directory to sys.path to be able to import main1
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

try:
    import main1  # Import the main processing module
except ImportError:
    print("Error: Could not import main1.py. Make sure it's in the correct location.")
    sys.exit(1)

class ProcessingGUI:
    def __init__(self, root, filename=None):
        """
        Initialize the ProcessingGUI.
        
        Parameters:
        - root: tkinter root window (expected to be a Notebook)
        - filename: Optional filename to process
        """
        self.root = root
        self.filename = filename
        
        # Create a tab for this GUI
        #self.tab = ttk.Frame(self.root)
        #self.root.add(self.tab, text="Particle Processing")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Create all the widgets for the GUI"""
        # File selection frame
        file_frame = ttk.LabelFrame(self.root, text="File Selection")
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(file_frame, text="Filename:").grid(row=0, column=0, padx=5, pady=5)
        self.filename_var = tk.StringVar()
        if self.filename:
            self.filename_var.set(self.filename)
        
        self.filename_entry = ttk.Entry(file_frame, textvariable=self.filename_var, width=50)
        self.filename_entry.grid(row=0, column=1, padx=5, pady=5)
        
        self.browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        self.browse_btn.grid(row=0, column=2, padx=5, pady=5)
        
        # Processing options
        options_frame = ttk.LabelFrame(self.root, text="Processing Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create the processing buttons
        ttk.Label(options_frame, text="Analysis Controls:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)

        # Add pixel size input field
        ttk.Label(options_frame, text="Pixel Size (nm):").grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        self.pixel_size_var = tk.DoubleVar(value=11.16)  # Default value of 11.16 nm
        self.pixel_size_entry = ttk.Entry(options_frame, textvariable=self.pixel_size_var, width=10)
        self.pixel_size_entry.grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        
        # Add gradient threshold input field
        ttk.Label(options_frame, text="Gradient Threshold:").grid(row=0, column=3, sticky=tk.W, padx=5, pady=5)
        self.gradient_threshold_var = tk.DoubleVar(value=23.0)  # Default value of 23.0
        self.gradient_threshold_entry = ttk.Entry(options_frame, textvariable=self.gradient_threshold_var, width=10)
        self.gradient_threshold_entry.grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)

        self.analyze_btn = ttk.Button(options_frame, text="Analyze Particles", command=self.analyze_particles)
        self.analyze_btn.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        self.visualize_btn = ttk.Button(options_frame, text="Visualize Results", command=self.visualize_results)
        self.visualize_btn.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        self.stats_btn = ttk.Button(options_frame, text="Show Statistics", command=self.show_statistics)
        self.stats_btn.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

        self.plot_btn = ttk.Button(options_frame, text="Plot Size Distribution", command=self.plot_distribution)
        self.plot_btn.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        # Results display
        results_frame = ttk.LabelFrame(self.root, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.results_text = tk.Text(results_frame, height=15, width=80)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    # Make sure to get the pixel size in your analysis methods
    def get_pixel_size(self):
        try:
            return self.pixel_size_var.get()
        except tk.TclError:
            self.results_text.insert(tk.END, "Invalid pixel size. Using default 1.0 nm.\n")
            self.pixel_size_var.set(1.0)
            return 11.16

    # Define methods to handle button clicks
    def analyze_particles(self):
        if not self.filename_var.get():
            self.results_text.insert(tk.END, "Please select a file first.\n")
            return
        
        try:
            analyzer = main1.ParticleAnalyzer(self.filename_var.get(), particle_area_threshold_px=100, gradient_threshold=self.gradient_threshold_var.get(), pixel_size_nm=self.get_pixel_size())
            self.analyzer = analyzer  # Store for later use
            self.particles = analyzer.analyze_particles()
            self.results_text.insert(tk.END, f"Detected {len(self.particles)} particles\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error during analysis: {str(e)}\n")

    def visualize_results(self):
        if not hasattr(self, 'analyzer'):
            self.results_text.insert(tk.END, "Please analyze particles first.\n")
            return
        
        try:
            self.analyzer.visualize_results(show_ellipses=True, show_ids=True)
        except Exception as e:
            self.results_text.insert(tk.END, f"Error during visualization: {str(e)}\n")

    def show_statistics(self):
        if not hasattr(self, 'analyzer'):
            self.results_text.insert(tk.END, "Please analyze particles first.\n")
            return
        
        try:
            stats = self.analyzer.get_particle_statistics()
            self.results_text.insert(tk.END, "Particle Statistics:\n")
            for key, value in stats.items():
                self.results_text.insert(tk.END, f"{key}: {value}\n")
        except Exception as e:
            self.results_text.insert(tk.END, f"Error getting statistics: {str(e)}\n")

    def plot_distribution(self):
        if not hasattr(self, 'analyzer'):
            self.results_text.insert(tk.END, "Please analyze particles first.\n")
            return
        
        try:
            self.analyzer.plot_size_distribution()
        except Exception as e:
            self.results_text.insert(tk.END, f"Error plotting distribution: {str(e)}\n")

    def browse_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")]
        )
        if filename:
            self.filename_var.set(filename)


# For testing the class directly
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Particle Size Analysis")
    
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # create a notebook root for the ProcessingGUI
    process_tab = ttk.Frame(notebook)
    notebook.add(process_tab, text="Particle Processing")

    app = ProcessingGUI(process_tab)

    root.geometry("800x600")
    root.mainloop()