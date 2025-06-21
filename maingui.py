import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import ProcessingNotebook as pcn

class ParticleSizeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Particle Size Analyzer")
        self.root.geometry("800x600")
        
        # Create the notebook (tabbed interface)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the tabs
        self.load_tab = ttk.Frame(self.notebook)
        self.process_tab = ttk.Frame(self.notebook)
        
        # Add tabs to notebook
        self.notebook.add(self.load_tab, text="Load Images")
        self.notebook.add(self.process_tab, text="Process Images")
        
        # Initialize tabs
        self.setup_load_tab()
        self.setup_process_tab()

        # Initialize variables
        self.image_file = None
    
    def setup_load_tab(self):
        # Create a frame for controls
        control_frame = ttk.Frame(self.load_tab)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Add only the Load Image button
        load_btn = ttk.Button(control_frame, text="Load Image", command=self.load_image)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Add a label to display the selected file
        self.file_label = ttk.Label(self.load_tab, text="No image selected")
        self.file_label.pack(padx=10, pady=20)
    
    def setup_process_tab(self):
        self.PID = pcn.ProcessingGUI(self.process_tab, filename=None)
    
    def load_image(self):
        filetypes = [("Image files", "*.jpg *.jpeg *.png *.tif *.tiff")]
        filename = filedialog.askopenfilename(title="Select Image File", filetypes=filetypes)
        
        if filename:
            self.image_file = filename
            self.file_label.config(text=f"Selected image: {filename}")
    


if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleSizeApp(root)
    root.mainloop()
