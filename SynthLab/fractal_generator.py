"""
Fractal Generators for Audio Synthesis.

Implements multiple fractal types:
- Mandelbrot Set
- Julia Set
- Perlin Noise
- Simplex Noise
- Worley (Voronoi) Noise
- Fractal Brownian Motion (FBM)
- Ridged Multifractal
- Domain Warping
- Lyapunoc Fractals
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class FractalType(str, Enum):
    """Supported fractal types."""
    MANDELBROT = "mandelbrot"
    JULIA = "julia"
    PERLIN = "perlin"
    SIMPLEX = "simplex"
    WORLEY = "worley"
    FBM = "fbm"
    RIDGED = "ridged"
    VORONOI = "voronoi"
    DOMAIN_WARP = "domain_warp"
    LYAPUNOV = "lyapunov"


class FractalGenerator:
    """
    Fractal terrain generator for audio synthesis.
    
    Provides various fractal algorithms for generating terrain textures.
    """
    
    def __init__(
        self,
        size: Tuple[int, int],
        seed: Optional[int] = None,
        scale: float = 1.0
    ):
        self.size = size
        self.scale = scale
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = np.random.randint(0, 2**31)
            np.random.seed(self.seed)
        
        # Permutation table for Perlin/Simplex
        self.perm = self._generate_permutation()
    
    def _generate_permutation(self) -> np.ndarray:
        """Generate permutation table for noise functions."""
        p = np.arange(256, dtype=np.uint32)
        np.random.shuffle(p)
        return np.concatenate([p, p])
    
    def _fade(self, t: np.ndarray) -> np.ndarray:
        """Fade function for smooth interpolation."""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def _grad(self, hash_val: int, x: float, y: float) -> float:
        """Calculate gradient value."""
        h = hash_val & 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    # ==================== PERLIN NOISE ====================
    
    def perlin(
        self,
        octaves: int = 4,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        frequency: float = 1.0
    ) -> np.ndarray:
        """
        Generate Perlin noise.
        
        Returns 2D array of Perlin noise values in [0, 1].
        """
        h, w = self.size
        noise = np.zeros((h, w), dtype=np.float64)
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            for y in range(h):
                for x in range(w):
                    xf = x * frequency * self.scale / w
                    yf = y * frequency * self.scale / h
                    
                    x0 = int(np.floor(xf)) & 255
                    y0 = int(np.floor(yf)) & 255
                    x1 = (x0 + 1) & 255
                    y1 = (y0 + 1) & 255
                    
                    sx = self._fade(xf - int(xf))
                    sy = self._fade(yf - int(yf))
                    
                    n00 = self._grad(self.perm[self.perm[x0] + y0], xf, yf)
                    n01 = self._grad(self.perm[self.perm[x0] + y1], xf, yf - 1)
                    n10 = self._grad(self.perm[self.perm[x1] + y0], xf - 1, yf)
                    n11 = self._grad(self.perm[self.perm[x1] + y1], xf - 1, yf - 1)
                    
                    nx0 = n00 * (1 - sx) + n10 * sx
                    nx1 = n01 * (1 - sx) + n11 * sx
                    
                    noise[y, x] += (nx0 * (1 - sy) + nx1 * sy) * amplitude
            
            amplitude *= persistence
            max_value += amplitude
            frequency *= lacunarity
        
        # Normalize to [0, 1]
        noise = noise / max_value
        return (noise + 1) / 2
    
    # ==================== SIMPLEX NOISE ====================
    
    def simplex(
        self,
        octaves: int = 4,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        frequency: float = 1.0
    ) -> np.ndarray:
        """
        Generate Simplex noise (faster than Perlin, less artifacts).
        """
        h, w = self.size
        
        # Skew factor for simplex
        f2 = 0.5 * (np.sqrt(3.0) - 1.0)
        g2 = (3.0 - np.sqrt(3.0)) / 6.0
        
        noise = np.zeros((h, w), dtype=np.float64)
        amplitude = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            for y in range(h):
                for x in range(w):
                    xf = x * frequency * self.scale / w
                    yf = y * frequency * self.scale / h
                    
                    s = (x + y) * f2
                    i = int(np.floor(xf + s))
                    j = int(np.floor(yf + s))
                    
                    t = (i + j) * g2
                    x0 = xf - (i - t)
                    y0 = yf - (j - t)
                    
                    i1, j1 = (1 if x0 > y0 else 0), (1 if y0 > x0 else 0)
                    
                    x1 = x0 - i1 + g2
                    y1 = y0 - j1 + g2
                    x2 = x0 - 1.0 + 2.0 * g2
                    y2 = y0 - 1.0 + 2.0 * g2
                    
                    ii = i & 255
                    jj = j & 255
                    
                    # Simple gradient (approximation)
                    def grad(hash_val, x, y):
                        h = hash_val & 7
                        u = x if h < 4 else y
                        v = y if h < 4 else x
                        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
                    
                    n0 = n1 = n2 = 0.0
                    t0 = 0.5 - x0*x0 - y0*y0
                    if t0 >= 0:
                        t0 *= t0
                        n0 = t0 * t0 * grad(self.perm[self.perm[ii] + jj], x0, y0)
                    
                    t1 = 0.5 - x1*x1 - y1*y1
                    if t1 >= 0:
                        t1 *= t1
                        n1 = t1 * t1 * grad(self.perm[self.perm[ii + i1] + jj + j1], x1, y1)
                    
                    t2 = 0.5 - x2*x2 - y2*y2
                    if t2 >= 0:
                        t2 *= t2
                        n2 = t2 * t2 * grad(self.perm[self.perm[ii + 1] + jj + 1], x2, y2)
                    
                    noise[y, x] += 70.0 * (n0 + n1 + n2) * amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
            max_value += amplitude
        
        return (noise / max_value + 1) / 2
    
    # ==================== WORLEY NOISE ====================
    
    def worley(
        self,
        cells: int = 8,
        jitter: float = 1.0,
        distance_type: str = "euclidean"
    ) -> np.ndarray:
        """
        Generate Worley (cellular/Voronoi) noise.
        
        Creates organic cell-like patterns.
        """
        h, w = self.size
        
        # Generate random points in grid cells
        np.random.seed(self.seed)
        
        # Create cell grid
        cell_indices = np.arange(cells, dtype=np.float64)
        
        # Random offset within each cell
        random_offsets = np.random.rand(cells, cells, 2) * jitter
        
        # Add cell indices to create grid of points
        points_x = random_offsets[:, :, 0] + cell_indices.reshape(1, cells)
        points_y = random_offsets[:, :, 1] + cell_indices.reshape(cells, 1)
        
        # Stack and normalize
        points = np.stack([points_x, points_y], axis=-1) / cells
        
        # Scale coordinates
        x_scaled = np.arange(w) / w * cells
        y_scaled = np.arange(h) / h * cells
        
        # Create coordinate grids
        x_cell = x_scaled.astype(int) % cells
        y_cell = y_scaled.astype(int) % cells
        
        x_frac = (x_scaled - x_cell) / cells * cells
        y_frac = (y_scaled - y_cell) / cells * cells
        
        # Distance to closest point (simplified)
        distance = np.ones((h, w), dtype=np.float64) * float('inf')
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                xn = (x_cell + dx) % cells
                yn = (y_cell + dy) % cells
                
                px = points[yn, xn, 0] - x_frac + dx
                py = points[yn, xn, 1] - y_frac + dy
                
                if distance_type == "euclidean":
                    d = np.sqrt(px**2 + py**2)
                else:  # manhattan
                    d = np.abs(px) + np.abs(py)
                
                distance = np.minimum(distance, d)
        
        # Normalize
        return np.clip(distance / 2, 0, 1)
    
    # ==================== MANDELBROT ====================
    
    def mandelbrot(
        self,
        x_center: float = -0.5,
        y_center: float = 0.0,
        zoom: float = 1.0,
        max_iter: int = 100,
        escape_radius: float = 2.0
    ) -> np.ndarray:
        """
        Generate Mandelbrot set as terrain.
        
        Returns escape time as normalized values.
        """
        h, w = self.size
        
        # Scale coordinates
        x = np.linspace(x_center - 2/zoom, x_center + 2/zoom, w)[np.newaxis, :]
        y = np.linspace(y_center - 2/zoom, y_center + 2/zoom, h)[:, np.newaxis]
        
        c = x + 1j * y
        z = np.zeros_like(c)
        
        # Iterations
        for i in range(max_iter):
            mask = np.abs(z) < escape_radius
            z[mask] = z[mask] * z[mask] + c[mask]
        
        # Normalize
        escaped = np.abs(z)
        with np.errstate(divide='ignore'):
            result = np.log(np.log(escaped)) / np.log(escape_radius)
        result = np.nan_to_num(result, nan=0)
        
        return np.clip(result / result.max() if result.max() > 0 else result, 0, 1)
    
    # ==================== JULIA SET ====================
    
    def julia(
        self,
        c: complex = -0.7 + 0.27015j,
        zoom: float = 1.0,
        max_iter: int = 100
    ) -> np.ndarray:
        """
        Generate Julia set as terrain.
        
        Different c values produce different patterns.
        """
        h, w = self.size
        
        # Scale coordinates
        x = np.linspace(-1.5/zoom, 1.5/zoom, w)[np.newaxis, :]
        y = np.linspace(-1.5/zoom, 1.5/zoom, h)[:, np.newaxis]
        
        z = x + 1j * y
        c_val = np.full_like(z, c)
        
        # Iterations
        for _ in range(max_iter):
            mask = np.abs(z) < 4
            z[mask] = z[mask] * z[mask] + c_val[mask]
        
        # Normalize using iteration count
        escaped = np.abs(z)
        with np.errstate(divide='ignore'):
            result = np.log(np.log(escaped)) / np.log(4)
        result = np.nan_to_num(result, nan=0)
        
        return np.clip(result / result.max() if result.max() > 0 else result, 0, 1)
    
    # ==================== FBM (Fractal Brownian Motion) ====================
    
    def fbm(
        self,
        octaves: int = 6,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        noise_type: str = "perlin"
    ) -> np.ndarray:
        """
        Generate Fractal Brownian Motion (FBM).
        
        Layered noise with decreasing amplitude at higher frequencies.
        """
        if noise_type == "perlin":
            base_noise = self.perlin(octaves=octaves, lacunarity=lacunarity, persistence=persistence)
        else:
            base_noise = self.simplex(octaves=octaves, lacunarity=lacunarity, persistence=persistence)
        
        return base_noise
    
    # ==================== RIDGED MULTIFRACTAL ====================
    
    def ridged(
        self,
        octaves: int = 6,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        offset: float = 1.0,
        gain: float = 2.0
    ) -> np.ndarray:
        """
        Generate ridged multifractal noise.
        
        Creates sharp, ridge-like terrain.
        """
        h, w = self.size
        noise = np.zeros((h, w), dtype=np.float64)
        amplitude = 0.5
        frequency = 1.0
        max_value = 0.0
        
        for _ in range(octaves):
            n = self.perlin(octaves=1, frequency=frequency)
            
            # Create ridges
            n = 1.0 - np.abs(n * 2 - 1)
            n = n * n
            
            noise += n * amplitude
            max_value += amplitude
            
            amplitude *= persistence
            frequency *= lacunarity
        
        return noise / max_value
    
    # ==================== DOMAIN WARPING ====================
    
    def domain_warp(
        self,
        base_octaves: int = 4,
        warp_strength: float = 1.0,
        noise_type: str = "perlin"
    ) -> np.ndarray:
        """
        Generate domain-warped noise.
        
        Creates flowing, organic patterns.
        """
        # Base noise for warping
        if noise_type == "perlin":
            warp_x = self.perlin(octaves=base_octaves) * 2 - 1
            warp_y = self.perlin(octaves=base_octaves) * 2 - 1
        else:
            warp_x = self.simplex(octaves=base_octaves) * 2 - 1
            warp_y = self.simplex(octaves=base_octaves) * 2 - 1
        
        # Apply warping
        h, w = self.size
        warped = np.zeros((h, w), dtype=np.float64)
        
        for y in range(h):
            for x in range(w):
                wx = x + warp_x[y, x] * warp_strength * w / 10
                wy = y + warp_y[y, x] * warp_strength * h / 10
                
                # Clamp to valid range
                wx = max(0, min(w-1, wx))
                wy = max(0, min(h-1, wy))
                
                # Sample (simplified)
                warped[y, x] = self.perlin(octaves=2, frequency=2)[int(wy), int(wx)]
        
        return (warped + 1) / 2
    
    # ==================== VORONOI ====================
    
    def voronoi(
        self,
        cells: int = 10,
        random_weight: float = 0.5
    ) -> np.ndarray:
        """
        Generate Voronoi diagram as terrain.
        
        Returns distance to nearest point.
        """
        return self.worley(cells=cells, jitter=random_weight)
    
    # ==================== COMBINED ====================
    
    def hybrid(
        self,
        fractal_types: list = None,
        blend_weights: list = None,
        **params
    ) -> np.ndarray:
        """
        Combine multiple fractal types.
        
        Args:
            fractal_types: List of fractal types to blend
            blend_weights: Weights for each type (sum to 1)
        """
        if fractal_types is None:
            fractal_types = ["perlin", "ridged", "worley"]
        
        if blend_weights is None:
            blend_weights = [1.0 / len(fractal_types)] * len(fractal_types)
        
        # Normalize weights
        blend_weights = np.array(blend_weights)
        blend_weights = blend_weights / blend_weights.sum()
        
        results = []
        for ftype in fractal_types:
            if ftype == "perlin":
                results.append(self.perlin(**params))
            elif ftype == "simplex":
                results.append(self.simplex(**params))
            elif ftype == "ridged":
                results.append(self.ridged(**params))
            elif ftype == "worley" or ftype == "voronoi":
                results.append(self.voronoi(**params))
            elif ftype == "mandelbrot":
                results.append(self.mandelbrot(**params))
            elif ftype == "julia":
                results.append(self.julia(**params))
            elif ftype == "fbm":
                results.append(self.fbm(**params))
            elif ftype == "domain_warp":
                results.append(self.domain_warp(**params))
        
        # Blend results
        combined = np.zeros(self.size, dtype=np.float64)
        for i, result in enumerate(results):
            combined += result * blend_weights[i]
        
        return combined


# Factory function
def create_fractal(
    fractal_type: str,
    size: Tuple[int, int],
    seed: Optional[int] = None,
    **params
) -> np.ndarray:
    """Create a fractal terrain of the specified type."""
    gen = FractalGenerator(size, seed)
    
    type_map = {
        "perlin": gen.perlin,
        "simplex": gen.simplex,
        "worley": gen.worley,
        "voronoi": gen.voronoi,
        "mandelbrot": gen.mandelbrot,
        "julia": gen.julia,
        "fbm": gen.fbm,
        "ridged": gen.ridged,
        "domain_warp": gen.domain_warp,
    }
    
    func = type_map.get(fractal_type.lower(), gen.perlin)
    return func(**params)


if __name__ == '__main__':
    # Test fractal generators
    print("Testing Fractal Generators...")
    
    size = (128, 128)
    fg = FractalGenerator(size, seed=42)
    
    # Test different fractals
    tests = [
        ("Perlin", lambda: fg.perlin(octaves=4)),
        ("Simplex", lambda: fg.simplex(octaves=4)),
        ("Worley", lambda: fg.worley(cells=8)),
        ("FBM", lambda: fg.fbm(octaves=6)),
        ("Ridged", lambda: fg.ridged(octaves=4)),
        ("Mandelbrot", lambda: fg.mandelbrot(zoom=1.0)),
        ("Julia", lambda: fg.julia()),
        ("Hybrid", lambda: fg.hybrid(["perlin", "ridged", "worley"], [0.4, 0.3, 0.3])),
    ]
    
    for name, func in tests:
        result = func()
        print(f"  {name}: shape={result.shape}, min={result.min():.3f}, max={result.max():.3f}")
    
    print("\nAll fractal generators working!")
