"""
SegmentationNotebook.py
-----------------------
Second tab for the Particle_sizes project: gradient + watershed segmentation.

Why a new pipeline?
  main1.ParticleAnalyzer uses Otsu + cv2.RETR_EXTERNAL and never uses its
  `gradient_threshold`. On dense, touching SEM particles the outer contours of
  whole clusters are merged into a few huge blobs, which then get a huge ellipse.

Idea here:
  1. Smooth the image and compute the gradient magnitude (Sobel).
  2. Pores / dark background are removed with an intensity threshold.
  3. Particle INTERIORS = foreground pixels with LOW gradient. The bright rims
     (high gradient) act as walls between touching particles.
  4. Distance transform of the interiors -> one seed per particle.
  5. Marker-based watershed on the gradient image splits touching particles.
  6. Per-particle statistics (equivalent diameter, ellipse axes, Feret, ...).

Usage in maingui.py (same pattern as ProcessingNotebook):
    import SegmentationNotebook as sgn
    self.seg_tab = ttk.Frame(self.notebook)
    self.notebook.add(self.seg_tab, text="Segmentation")
    self.SEG = sgn.SegmentationGUI(self.seg_tab, filename=None)

Dependencies: tkinter, cv2, numpy, matplotlib (same as the rest of the project).
The segmentation class itself does NOT need tkinter and can be used standalone.
"""
import csv
from dataclasses import dataclass, asdict

import cv2
import numpy as np

try:  # GUI parts are optional so the algorithm can be used/tested headless
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError:  # pragma: no cover
    tk = None


# ----------------------------------------------------------------------------
# Algorithm
# ----------------------------------------------------------------------------
@dataclass
class SegParams:
    pixel_size_nm: float = 11.16      # nm per pixel (from the SEM info bar)
    crop_bottom_frac: float = 0.06    # cut off the info bar at the bottom
    blur_sigma: float = 2.0           # Gaussian smoothing before the gradient
    grad_threshold: float = 7.0       # grey levels / px; below = particle interior
    pore_level: float = -1.0          # intensity below = pore/background; <0 -> auto
    seed_min_radius_px: float = 5.5   # minimum inscribed radius for a seed
    min_area_px: int = 300            # discard smaller regions
    max_area_frac: float = 0.08       # discard regions larger than this image fraction
    exclude_border: bool = True       # drop particles touching the image border


class GradientWatershedSegmenter:
    def __init__(self, image_path=None, image=None, params: SegParams = None):
        self.params = params or SegParams()
        if image is None:
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")
        elif image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.raw = image
        self.particles = []
        self.stages = {}

    # -- helpers -----------------------------------------------------------
    def _cropped(self):
        h = self.raw.shape[0]
        return self.raw[: int(h * (1.0 - self.params.crop_bottom_frac)), :]

    @staticmethod
    def _gradient(sm):
        sm32 = sm.astype(np.float32)
        gx = cv2.Sobel(sm32, cv2.CV_32F, 1, 0, ksize=3) / 8.0   # /8 -> grey levels per px
        gy = cv2.Sobel(sm32, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        return cv2.magnitude(gx, gy)

    # -- main pipeline -------------------------------------------------------
    def run(self):
        p = self.params
        img = self._cropped()
        H, W = img.shape

        sm = cv2.GaussianBlur(img, (0, 0), p.blur_sigma)
        grad = self._gradient(sm)

        # Pores / dark background
        if p.pore_level >= 0:
            level = p.pore_level
        else:
            otsu, _ = cv2.threshold(cv2.GaussianBlur(img, (0, 0), 3), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            level = 0.75 * otsu  # Otsu sits between pores and particles; stay on the dark side
        fg = (sm > level).astype(np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Particle interiors: foreground with low gradient
        flat = ((grad < p.grad_threshold) & (fg > 0)).astype(np.uint8)
        flat = cv2.morphologyEx(flat, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Seeds from the distance transform (local maxima above a minimum radius)
        dist = cv2.distanceTransform(flat, cv2.DIST_L2, 5)
        dist_s = cv2.GaussianBlur(dist, (0, 0), 1.5)
        local_max = dist_s == cv2.dilate(dist_s, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(max(p.seed_min_radius_px, 2)) + 1,) * 2))
        seeds = (local_max & (dist_s >= p.seed_min_radius_px)).astype(np.uint8)
        n_seeds, seed_lbl = cv2.connectedComponents(seeds)

        # Watershed on the gradient image. Label 1 = background, seeds = 2..n
        markers = np.where(seed_lbl > 0, seed_lbl + 1, 0).astype(np.int32)  # 0 = unknown
        markers[fg == 0] = 1
        grad8 = np.clip(grad * (255.0 / max(grad.max(), 1e-6)) * 3, 0, 255).astype(np.uint8)
        ws = cv2.watershed(cv2.cvtColor(grad8, cv2.COLOR_GRAY2BGR), markers.copy())

        # Per-region properties
        self.particles = []
        max_area = p.max_area_frac * H * W
        for lbl in range(2, n_seeds + 1):
            mask = ((ws == lbl) & (fg > 0)).astype(np.uint8)
            area = int(mask.sum())
            if area < p.min_area_px or area > max_area:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(cnt)
            touches = x <= 0 or y <= 0 or x + w >= W or y + h >= H
            if p.exclude_border and touches:
                continue
            if len(cnt) < 5:
                continue
            (cx, cy), (ma, mi), ang = cv2.fitEllipse(cnt)
            (_, _), (fw, fh), _ = cv2.minAreaRect(cnt)
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            perim = cv2.arcLength(cnt, True)
            s = p.pixel_size_nm
            self.particles.append({
                "id": len(self.particles) + 1,
                "center": (cx, cy),
                "area_nm2": area * s * s,
                "eq_diameter_nm": np.sqrt(4.0 * area / np.pi) * s,
                "major_axis_nm": max(ma, mi) * s,
                "minor_axis_nm": min(ma, mi) * s,
                "feret_max_nm": max(fw, fh) * s,
                "feret_min_nm": min(fw, fh) * s,
                "aspect_ratio": max(fw, fh) / max(min(fw, fh), 1e-6),
                "circularity": 4 * np.pi * area / max(perim ** 2, 1e-6),
                "solidity": area / max(hull_area, 1e-6),
                "angle": ang,
                "touches_border": touches,
                "contour": cnt,
            })

        self.stages = {"image": img, "smoothed": sm, "gradient": grad,
                       "foreground": fg * 255, "interior": flat * 255,
                       "distance": dist, "seeds": seeds * 255, "watershed": ws,
                       "pore_level": level}
        return self.particles

    # -- output --------------------------------------------------------------
    def overlay(self, show_ids=False):
        """RGB overlay of the contours (and IDs) on the cropped image."""
        img = self.stages["image"]
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        for q in self.particles:
            cv2.drawContours(vis, [q["contour"]], -1, (0, 255, 0), 1)
            if show_ids:
                cv2.putText(vis, str(q["id"]), (int(q["center"][0]) - 6, int(q["center"][1]) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 60, 60), 1, cv2.LINE_AA)
        return vis

    def statistics(self):
        if not self.particles:
            return {}
        d = np.array([q["eq_diameter_nm"] for q in self.particles])
        ar = np.array([q["aspect_ratio"] for q in self.particles])
        return {
            "count": len(d),
            "mean_eq_diameter_nm": float(d.mean()),
            "median_eq_diameter_nm": float(np.median(d)),
            "std_eq_diameter_nm": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "d10_nm": float(np.percentile(d, 10)),
            "d90_nm": float(np.percentile(d, 90)),
            "min_nm": float(d.min()),
            "max_nm": float(d.max()),
            "mean_aspect_ratio": float(ar.mean()),
        }

    def export_csv(self, path):
        keys = ["id", "area_nm2", "eq_diameter_nm", "major_axis_nm", "minor_axis_nm",
                "feret_max_nm", "feret_min_nm", "aspect_ratio", "circularity", "solidity"]
        with open(path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(keys + ["center_x_px", "center_y_px"])
            for q in self.particles:
                wr.writerow([q[k] for k in keys] + [q["center"][0], q["center"][1]])


# ----------------------------------------------------------------------------
# GUI tab
# ----------------------------------------------------------------------------
class SegmentationGUI:
    """Tab with live parameter sliders and an embedded matplotlib view."""

    VIEWS = ["Overlay", "Gradient", "Foreground", "Interior (low gradient)",
             "Seeds", "Watershed labels"]

    def __init__(self, root, filename=None):
        if tk is None:
            raise RuntimeError("tkinter / matplotlib TkAgg backend not available")
        self.root = root
        self.segmenter = None
        self._job = None
        self._build(filename)

    # -- layout --------------------------------------------------------------
    def _build(self, filename):
        left = ttk.Frame(self.root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        right = ttk.Frame(self.root)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=4, pady=8)

        # file
        ff = ttk.LabelFrame(left, text="File")
        ff.pack(fill=tk.X, pady=3)
        self.filename_var = tk.StringVar(value=filename or "")
        ttk.Entry(ff, textvariable=self.filename_var, width=28).grid(row=0, column=0, padx=3, pady=3)
        ttk.Button(ff, text="Browse", command=self.browse).grid(row=0, column=1, padx=3)

        # parameters
        pf = ttk.LabelFrame(left, text="Parameters")
        pf.pack(fill=tk.X, pady=3)
        d = SegParams()
        self.vars = {}
        specs = [  # key, label, from, to, resolution
            ("pixel_size_nm", "Pixel size (nm)", 0.1, 100, 0.01),
            ("crop_bottom_frac", "Crop bottom (frac)", 0.0, 0.15, 0.005),
            ("blur_sigma", "Blur sigma (px)", 0.0, 6.0, 0.1),
            ("grad_threshold", "Gradient threshold", 0.5, 40.0, 0.5),
            ("pore_level", "Pore level (-1 = auto)", -1, 255, 1),
            ("seed_min_radius_px", "Min seed radius (px)", 1.0, 30.0, 0.5),
            ("min_area_px", "Min area (px)", 10, 3000, 10),
            ("max_area_frac", "Max area (img frac)", 0.005, 0.5, 0.005),
        ]
        for r, (key, label, lo, hi, res) in enumerate(specs):
            ttk.Label(pf, text=label).grid(row=r, column=0, sticky=tk.W, padx=3)
            var = tk.DoubleVar(value=getattr(d, key))
            self.vars[key] = var
            tk.Scale(pf, variable=var, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                     length=150, command=lambda _v: self._schedule()).grid(row=r, column=1, padx=3)
        self.exclude_border_var = tk.BooleanVar(value=d.exclude_border)
        ttk.Checkbutton(pf, text="Exclude border particles", variable=self.exclude_border_var,
                        command=self._schedule).grid(row=len(specs), column=0, columnspan=2,
                                                     sticky=tk.W, padx=3)
        self.live_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(pf, text="Live update", variable=self.live_var).grid(
            row=len(specs) + 1, column=0, columnspan=2, sticky=tk.W, padx=3)

        # view + actions
        vf = ttk.LabelFrame(left, text="View")
        vf.pack(fill=tk.X, pady=3)
        self.view_var = tk.StringVar(value=self.VIEWS[0])
        cb = ttk.Combobox(vf, textvariable=self.view_var, values=self.VIEWS, state="readonly", width=26)
        cb.pack(padx=3, pady=3)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._draw())
        self.ids_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(vf, text="Show IDs", variable=self.ids_var, command=self._draw).pack(anchor=tk.W, padx=3)

        af = ttk.Frame(left)
        af.pack(fill=tk.X, pady=3)
        ttk.Button(af, text="Run", command=self.run).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(af, text="Histogram", command=self.show_histogram).grid(row=0, column=1, padx=2)
        ttk.Button(af, text="Export CSV", command=self.export_csv).grid(row=0, column=2, padx=2)

        self.stats_text = tk.Text(left, height=12, width=38)
        self.stats_text.pack(fill=tk.BOTH, expand=True, pady=3)

        # matplotlib canvas
        self.fig = Figure(figsize=(7, 5.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # -- actions -------------------------------------------------------------
    def browse(self):
        fn = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"),
                                                   ("All files", "*.*")])
        if fn:
            self.filename_var.set(fn)
            self.segmenter = None
            self.run()

    def _params(self):
        kw = {k: v.get() for k, v in self.vars.items()}
        kw["min_area_px"] = int(kw["min_area_px"])
        kw["exclude_border"] = self.exclude_border_var.get()
        return SegParams(**kw)

    def _schedule(self):
        if not self.live_var.get() or not self.filename_var.get():
            return
        if self._job is not None:
            self.root.after_cancel(self._job)
        self._job = self.root.after(200, self.run)  # debounce slider drags

    def run(self):
        self._job = None
        fn = self.filename_var.get()
        if not fn:
            self._log("Please select a file first.")
            return
        try:
            if self.segmenter is None or self.segmenter_path != fn:
                self.segmenter = GradientWatershedSegmenter(image_path=fn)
                self.segmenter_path = fn
            self.segmenter.params = self._params()
            self.segmenter.run()
        except Exception as e:  # show errors in the text box, like ProcessingNotebook
            self._log(f"Error: {e}")
            return
        st = self.segmenter.statistics()
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, f"pore level used: {self.segmenter.stages['pore_level']:.0f}\n")
        for k, v in st.items():
            self.stats_text.insert(tk.END, f"{k}: {v:.1f}\n" if isinstance(v, float) else f"{k}: {v}\n")
        self._draw()

    def _draw(self):
        if self.segmenter is None or not self.segmenter.stages:
            return
        s = self.segmenter.stages
        view = self.view_var.get()
        self.ax.clear()
        self.ax.axis("off")
        if view == "Overlay":
            self.ax.imshow(self.segmenter.overlay(self.ids_var.get()))
            self.ax.set_title(f"{len(self.segmenter.particles)} particles")
        elif view == "Gradient":
            im = self.ax.imshow(s["gradient"], cmap="magma")
            self.fig.colorbar(im, ax=self.ax, fraction=0.03, label="grey levels / px")
        elif view == "Foreground":
            self.ax.imshow(s["foreground"], cmap="gray")
        elif view.startswith("Interior"):
            self.ax.imshow(s["interior"], cmap="gray")
        elif view == "Seeds":
            vis = cv2.cvtColor(s["image"], cv2.COLOR_GRAY2RGB)
            vis[cv2.dilate(s["seeds"], np.ones((5, 5), np.uint8)) > 0] = (255, 0, 0)
            self.ax.imshow(vis)
        else:
            ws = s["watershed"].astype(np.float32)
            self.ax.imshow(np.ma.masked_less_equal(ws, 1) % 20, cmap="tab20")
        self.fig.tight_layout()
        self.canvas.draw_idle()

    def show_histogram(self):
        if self.segmenter is None or not self.segmenter.particles:
            self._log("Run the segmentation first.")
            return
        win = tk.Toplevel(self.root)
        win.title("Size distribution")
        fig = Figure(figsize=(9, 4), dpi=100)
        a1, a2 = fig.subplots(1, 2)
        d = [q["eq_diameter_nm"] for q in self.segmenter.particles]
        ar = [q["aspect_ratio"] for q in self.segmenter.particles]
        a1.hist(d, bins=30, color="tab:blue", alpha=0.8)
        a1.set_xlabel("Equivalent diameter (nm)")
        a1.set_ylabel("Count")
        a2.hist(ar, bins=30, color="tab:green", alpha=0.8)
        a2.set_xlabel("Aspect ratio (Feret max / min)")
        fig.tight_layout()
        c = FigureCanvasTkAgg(fig, master=win)
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        c.draw()

    def export_csv(self):
        if self.segmenter is None or not self.segmenter.particles:
            self._log("Nothing to export.")
            return
        fn = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if fn:
            self.segmenter.export_csv(fn)
            self._log(f"Saved {fn}")

    def _log(self, msg):
        self.stats_text.insert(tk.END, msg + "\n")


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Particle Segmentation")
    nb = ttk.Notebook(root)
    nb.pack(fill=tk.BOTH, expand=True)
    tab = ttk.Frame(nb)
    nb.add(tab, text="Segmentation")
    SegmentationGUI(tab)
    root.geometry("1250x720")
    root.mainloop()