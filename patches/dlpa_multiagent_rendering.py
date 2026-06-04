"""
2D rendering framework - headless pygame port of the MPE pyglet renderer.

Drop-in replacement for the original pyglet-based rendering.py. Mirrors the
same Viewer API (set_bounds, add_geom, add_onetime, render(return_rgb_array=))
and Geom hierarchy (Geom, Circle, Polygon, Line) with identical colors and
sizing, but draws with pygame on an off-screen Surface so it works headlessly
(no X / OpenGL / xvfb needed).
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

if not pygame.get_init():
    pygame.init()


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------
class Attr(object):
    def enable(self): pass
    def disable(self): pass


class Transform(Attr):
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        self.translation = list(translation)
        self.rotation = float(rotation)
        self.scale = list(scale)

    def enable(self): pass
    def disable(self): pass

    def set_translation(self, newx, newy):
        self.translation = [float(newx), float(newy)]

    def set_rotation(self, new):
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        self.scale = [float(newx), float(newy)]


class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4


class LineStyle(Attr):
    def __init__(self, style):
        self.style = style


class LineWidth(Attr):
    def __init__(self, stroke):
        self.stroke = float(stroke)


# ---------------------------------------------------------------------------
# Geoms
# ---------------------------------------------------------------------------
class Geom(object):
    def __init__(self):
        self._color = (0, 0, 0, 255)
        self.attrs = []
        self.linewidth = 1.0

    def render(self): pass

    def add_attr(self, attr):
        self.attrs.append(attr)
        if isinstance(attr, LineWidth):
            self.linewidth = attr.stroke

    def set_color(self, r, g, b, alpha=1.0):
        self._color = (int(r * 255) & 255, int(g * 255) & 255,
                       int(b * 255) & 255, int(alpha * 255) & 255)

    def _transform(self):
        for a in self.attrs:
            if isinstance(a, Transform):
                return a
        return None


class Circle(Geom):
    def __init__(self, radius=10, res=30, filled=True):
        super().__init__()
        self.radius = float(radius)
        self.filled = filled


class FilledPolygon(Geom):
    def __init__(self, v):
        super().__init__()
        self.v = list(v)


class PolyLine(Geom):
    def __init__(self, v, close=False):
        super().__init__()
        self.v = list(v)
        self.close = close


class Line(Geom):
    def __init__(self, start=(0, 0), end=(0, 0)):
        super().__init__()
        self.start = start
        self.end = end


def make_circle(radius=10, res=30, filled=True):
    return Circle(radius=radius, res=res, filled=filled)


def make_polygon(v, filled=True):
    return FilledPolygon(v) if filled else PolyLine(v, close=True)


def make_polyline(v):
    return PolyLine(v, close=False)


def get_display(spec):
    return None


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------
class Viewer(object):
    def __init__(self, width, height, display=None):
        self.width = int(width)
        self.height = int(height)
        self.surface = pygame.Surface((self.width, self.height))
        self.bounds = (-1.0, 1.0, -1.0, 1.0)  # left, right, bottom, top
        self.geoms = []
        self.onetime_geoms = []

    def close(self): pass
    def window_closed_by_user(self): return False

    def set_bounds(self, left, right, bottom, top):
        self.bounds = (float(left), float(right), float(bottom), float(top))

    def add_geom(self, geom):
        self.geoms.append(geom)

    def add_onetime(self, geom):
        self.onetime_geoms.append(geom)

    def _w2s(self, x, y):
        l, r, b, t = self.bounds
        sx = (x - l) / (r - l) * self.width
        sy = self.height - (y - b) / (t - b) * self.height
        return int(round(sx)), int(round(sy))

    def _w2px(self, d):
        l, r, _, _ = self.bounds
        return max(1, int(round(d / (r - l) * self.width)))

    def _draw_circle(self, geom):
        xf = geom._transform()
        wx, wy = (xf.translation[0], xf.translation[1]) if xf else (0.0, 0.0)
        cx, cy = self._w2s(wx, wy)
        r_px = self._w2px(geom.radius)
        rgba = geom._color
        if rgba[3] < 255:
            s = pygame.Surface((2 * r_px + 2, 2 * r_px + 2), pygame.SRCALPHA)
            pygame.draw.circle(s, rgba, (r_px + 1, r_px + 1), r_px)
            self.surface.blit(s, (cx - r_px - 1, cy - r_px - 1))
        else:
            pygame.draw.circle(self.surface, rgba[:3], (cx, cy), r_px)

    def _draw_polygon(self, geom):
        xf = geom._transform()
        tx, ty = (xf.translation[0], xf.translation[1]) if xf else (0.0, 0.0)
        pts = [self._w2s(v[0] + tx, v[1] + ty) for v in geom.v]
        rgba = geom._color
        if rgba[3] < 255 and len(pts) >= 3:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            x0, y0 = min(xs), min(ys); x1, y1 = max(xs), max(ys)
            w, h = max(1, x1 - x0 + 2), max(1, y1 - y0 + 2)
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.polygon(s, rgba, [(p[0] - x0, p[1] - y0) for p in pts])
            self.surface.blit(s, (x0, y0))
        elif len(pts) >= 3:
            pygame.draw.polygon(self.surface, rgba[:3], pts)

    def _draw_polyline(self, geom):
        xf = geom._transform()
        tx, ty = (xf.translation[0], xf.translation[1]) if xf else (0.0, 0.0)
        pts = [self._w2s(v[0] + tx, v[1] + ty) for v in geom.v]
        if len(pts) < 2:
            return
        width = max(1, int(geom.linewidth))
        pygame.draw.lines(self.surface, geom._color[:3], geom.close, pts, width)

    def _draw_line(self, geom):
        xf = geom._transform()
        tx, ty = (xf.translation[0], xf.translation[1]) if xf else (0.0, 0.0)
        a = self._w2s(geom.start[0] + tx, geom.start[1] + ty)
        b = self._w2s(geom.end[0] + tx, geom.end[1] + ty)
        width = max(1, int(geom.linewidth))
        pygame.draw.line(self.surface, geom._color[:3], a, b, width)

    def _draw(self, geom):
        if isinstance(geom, Circle):          self._draw_circle(geom)
        elif isinstance(geom, FilledPolygon): self._draw_polygon(geom)
        elif isinstance(geom, PolyLine):      self._draw_polyline(geom)
        elif isinstance(geom, Line):          self._draw_line(geom)

    def render(self, return_rgb_array=False):
        self.surface.fill((255, 255, 255))
        for g in self.geoms:
            self._draw(g)
        for g in self.onetime_geoms:
            self._draw(g)
        self.onetime_geoms = []
        if return_rgb_array:
            arr = pygame.surfarray.array3d(self.surface)
            return np.transpose(arr, (1, 0, 2))
        return None

    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        geom = make_circle(radius=radius, res=res, filled=filled)
        if "color" in attrs:
            c = attrs["color"]
            geom.set_color(*c) if len(c) == 4 else geom.set_color(c[0], c[1], c[2])
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        geom = make_polygon(v=v, filled=filled)
        if "color" in attrs:
            c = attrs["color"]
            geom.set_color(*c) if len(c) == 4 else geom.set_color(c[0], c[1], c[2])
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        geom = make_polyline(v=v)
        if "color" in attrs:
            c = attrs["color"]
            geom.set_color(*c) if len(c) == 4 else geom.set_color(c[0], c[1], c[2])
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        geom = Line(start, end)
        if "color" in attrs:
            c = attrs["color"]
            geom.set_color(*c) if len(c) == 4 else geom.set_color(c[0], c[1], c[2])
        self.add_onetime(geom)
        return geom

    def get_array(self):
        arr = pygame.surfarray.array3d(self.surface)
        return np.transpose(arr, (1, 0, 2))