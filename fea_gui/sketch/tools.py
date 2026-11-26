"""The tools of the sketch editor. A tool collects points (clicks or typed entry) and asks the canvas to commit one change.

Points are references: an id of an existing point ("P3": exactly connected) or a pair of coordinate expressions (clicked points are
numbers rounded to micrometres; typed points keep the expressions, so the sketch follows its parameters). The typed entry of the
status bar (`x, y`, `@dx, dy`, `@r<angle`, expressions) goes through fea_core.sketch.ops.parse_entry.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen

from fea_core.sketch import ops
from fea_core.sketch.expr import SketchError, evaluate
from fea_core.sketch.ops import PointRef

from .. import theme
from .canvas import Snap

if TYPE_CHECKING:
    from .canvas import SketchCanvas


def _r(v: float) -> float:
    return round(float(v), 6) + 0.0


class Tool:
    name = ""
    hint = ""
    needs_selection = False

    def __init__(self, cv: SketchCanvas):
        self.cv = cv
        self.pts: list[PointRef] = []

    # ---- helpers
    def ref(self, snap: Snap) -> PointRef:
        return snap.pid if snap.pid else (_r(snap.xy[0]), _r(snap.xy[1]))

    def xy(self, p: PointRef) -> tuple[float, float]:
        """Numeric position of a point reference."""
        sk = self.cv.sk
        env = sk.values()
        if isinstance(p, str):
            return sk.point_values(p, env)
        return evaluate(p[0], env), evaluate(p[1], env)

    def last_xy(self) -> tuple[float, float] | None:
        if not self.pts:
            return None
        try:
            return self.xy(self.pts[-1])
        except SketchError:
            return None

    def hint_set(self, text: str) -> None:
        self.hint = text
        self.cv.hintChanged.emit(text)

    # ---- interface
    def press(self, snap: Snap, pos: QPointF, mods) -> None:
        self.add(self.ref(snap))

    def move(self, snap: Snap) -> None:
        pass

    def entry(self, text: str) -> None:
        """A point typed in the status bar."""
        last: tuple[ops.Coord, ops.Coord] | None = None
        if self.pts:
            p = self.pts[-1]
            x, y = self.cv.sk.points[p] if isinstance(p, str) else p
            last = (x, y)
        try:
            ref = ops.parse_entry(text, self.cv.sk, last)
        except SketchError as e:
            self.cv.hintChanged.emit(str(e))
            return
        self.add(ref)
        self.cv.viewport().update()

    def add(self, ref: PointRef) -> None:
        self.pts.append(ref)

    def finish(self) -> None:
        self.cancel()

    def cancel(self) -> None:
        self.pts = []
        self.hint_set(type(self).hint)

    def paint(self, painter: QPainter) -> None:
        pass

    # ---- preview painting in device coordinates
    def _pen(self) -> QPen:
        pen = QPen(QColor(theme.colors()["accent"]), 1.6)
        pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _dev(self, xy: tuple[float, float]) -> QPointF:
        return QPointF(self.cv.mapFromScene(QPointF(*xy)))

    def rubber(self, painter: QPainter, a: tuple[float, float], b: tuple[float, float]) -> None:
        painter.save()
        painter.resetTransform()
        painter.setPen(self._pen())
        painter.drawLine(self._dev(a), self._dev(b))
        painter.restore()


class LineTool(Tool):
    name, hint = "line", "Line: click or type the first point, then the next ones (Esc / right click ends the chain)"

    def add(self, ref: PointRef) -> None:
        self.pts.append(ref)
        if len(self.pts) == 2:
            a, b = self.pts
            self.cv.commit(lambda sk: sk.add_entity("line", [a, b]))
            self.pts = [b]

    def paint(self, painter: QPainter) -> None:
        la = self.last_xy()
        if la is not None and self.cv.cursor is not None:
            self.rubber(painter, la, self.cv.cursor.xy)

    def finish(self) -> None:
        self.cancel()


class PolylineTool(Tool):
    name, hint = "polyline", "Polyline: click or type the points; Enter / right click / double click finishes, click the first point closes it"

    def add(self, ref: PointRef) -> None:
        if len(self.pts) >= 3 and ref == self.pts[0]:
            self.finish(closed=True)
            return
        self.pts.append(ref)

    def finish(self, closed: bool = False) -> None:
        pts = list(self.pts)
        if len(pts) >= 2 and pts[-1] == pts[-2]:
            pts.pop()
        self.pts = []
        if len(pts) >= 2:
            if not self.cv.commit(lambda sk: sk.add_entity("polyline", [p if isinstance(p, str) else list(p) for p in pts], closed=closed)):
                return
        self.hint_set(type(self).hint)

    def entry(self, text: str) -> None:
        if text.strip().lower() in ("c", "close") and len(self.pts) >= 3:
            self.finish(closed=True)
            return
        super().entry(text)

    def paint(self, painter: QPainter) -> None:
        pts = [self.xy(p) for p in self.pts]
        for a, b in zip(pts, pts[1:]):
            self.rubber(painter, a, b)
        if pts and self.cv.cursor is not None:
            self.rubber(painter, pts[-1], self.cv.cursor.xy)


class RectTool(Tool):
    name, hint = "rect", "Rectangle: two opposite corners (click or type x, y / @dx, dy)"

    def add(self, ref: PointRef) -> None:
        self.pts.append(ref)
        if len(self.pts) == 2:
            a, b = self.pts
            self.pts = []
            if self.cv.commit(lambda sk: sk.add_entity("rect", [a if isinstance(a, str) else list(a), b if isinstance(b, str) else list(b)])):
                self.hint_set(type(self).hint)

    def paint(self, painter: QPainter) -> None:
        la = self.last_xy()
        if la is not None and self.cv.cursor is not None:
            b = self.cv.cursor.xy
            for p, q in (((la[0], la[1]), (b[0], la[1])), ((b[0], la[1]), (b[0], b[1])), ((b[0], b[1]), (la[0], b[1])), ((la[0], b[1]), la)):
                self.rubber(painter, p, q)


class CircleTool(Tool):
    name, hint = "circle", "Circle: the centre, then a point on the circle or the radius (a number or an expression)"

    def add(self, ref: PointRef) -> None:
        if not self.pts:
            self.pts.append(ref)
            self.hint_set("Circle: a point on the circle, or type the radius (number or expression, e.g. d/2)")
            return
        c = self.xy(self.pts[0])
        p = self.xy(ref)
        self.make(round(math.hypot(p[0] - c[0], p[1] - c[1]), 6))

    def make(self, radius) -> None:
        c = self.pts[0]
        self.pts = []
        if self.cv.commit(lambda sk: sk.add_entity("circle", center=c if isinstance(c, str) else list(c), radius=radius)):
            self.hint_set(type(self).hint)

    def entry(self, text: str) -> None:
        t = text.strip()
        if self.pts and t and "," not in t and "<" not in t and t not in self.cv.sk.points:
            try:
                evaluate(t, self.cv.sk.values())
            except SketchError as e:
                self.cv.hintChanged.emit(str(e))
                return
            self.make(float(t) if t.replace(".", "", 1).isdigit() else t)
            return
        super().entry(text)

    def paint(self, painter: QPainter) -> None:
        if self.pts and self.cv.cursor is not None:
            c = self.xy(self.pts[0])
            r = math.hypot(self.cv.cursor.xy[0] - c[0], self.cv.cursor.xy[1] - c[1])
            painter.save()
            painter.resetTransform()
            painter.setPen(self._pen())
            q = self._dev(c)
            rp = r * self.cv.scale_px()
            painter.drawEllipse(q, rp, rp)
            painter.restore()


class Arc3Tool(Tool):
    name, hint = "arc3", "Arc through 3 points: the start, a point on the arc, the end"

    def add(self, ref: PointRef) -> None:
        self.pts.append(ref)
        if len(self.pts) == 3:
            pts = self.pts
            self.pts = []
            if self.cv.commit(lambda sk: sk.add_entity("arc3", [p if isinstance(p, str) else list(p) for p in pts])):
                self.hint_set(type(self).hint)

    def paint(self, painter: QPainter) -> None:
        pts = [self.xy(p) for p in self.pts]
        if self.cv.cursor is not None:
            pts = pts + [self.cv.cursor.xy]
        for a, b in zip(pts, pts[1:]):
            self.rubber(painter, a, b)


class ArcCenterTool(Tool):
    name, hint = "arc_center", "Arc by centre: the centre, the start point, then the end (counter-clockwise from the start)"

    def add(self, ref: PointRef) -> None:
        self.pts.append(ref)
        if len(self.pts) == 3:
            c, s, e = self.pts
            self.pts = []
            cc, ss, ee = self.xy(c), self.xy(s), self.xy(e)
            r = math.hypot(ss[0] - cc[0], ss[1] - cc[1])
            d = math.hypot(ee[0] - cc[0], ee[1] - cc[1])
            if d < 1e-9:
                self.cv.hintChanged.emit("The end of the arc coincides with its centre")
                return
            end: PointRef = e if (isinstance(e, str) and abs(d - r) < 1e-6 * max(r, 1.0)) else (
                _r(cc[0] + r * (ee[0] - cc[0]) / d), _r(cc[1] + r * (ee[1] - cc[1]) / d))
            if self.cv.commit(lambda sk: sk.add_entity("arc_center", [s if isinstance(s, str) else list(s), end if isinstance(end, str) else list(end)],
                                                       center=c if isinstance(c, str) else list(c))):
                self.hint_set(type(self).hint)
        elif len(self.pts) == 2:
            self.hint_set("Arc by centre: the end point (the direction from the centre decides, counter-clockwise from the start)")

    def paint(self, painter: QPainter) -> None:
        if not self.pts or self.cv.cursor is None:
            return
        c = self.xy(self.pts[0])
        if len(self.pts) == 1:
            self.rubber(painter, c, self.cv.cursor.xy)
            return
        s = self.xy(self.pts[1])
        r = math.hypot(s[0] - c[0], s[1] - c[1])
        a0 = math.atan2(s[1] - c[1], s[0] - c[0])
        a1 = math.atan2(self.cv.cursor.xy[1] - c[1], self.cv.cursor.xy[0] - c[0])
        sweep = (a1 - a0) % (2 * math.pi)
        painter.save()
        painter.resetTransform()
        painter.setPen(self._pen())
        n = max(8, int(math.degrees(sweep) / 3))
        prev = self._dev((c[0] + r * math.cos(a0), c[1] + r * math.sin(a0)))
        for i in range(1, n + 1):
            a = a0 + sweep * i / n
            q = self._dev((c[0] + r * math.cos(a), c[1] + r * math.sin(a)))
            painter.drawLine(prev, q)
            prev = q
        painter.drawLine(self._dev(c), self._dev((c[0] + r * math.cos(a0), c[1] + r * math.sin(a0))))
        painter.restore()


class PointTool(Tool):
    """A beam node / support (beam sketches): the kind comes from the editor (`kind`)."""
    name, hint = "point", "Node / support: click or type the point (a point inside a member splits it there)"
    kind = "fixed"

    def add(self, ref: PointRef) -> None:
        kind = self.kind
        self.cv.commit(lambda sk: sk.add_support(ref if isinstance(ref, str) else list(ref), kind))


class SelectTool(Tool):
    name, hint = "select", "Select: click an entity (Ctrl adds), Delete removes, Esc clears"

    def press(self, snap: Snap, pos: QPointF, mods) -> None:
        eid = self.cv.pick(pos)
        add = bool(mods & Qt.KeyboardModifier.ControlModifier)
        if eid is None:
            if not add:
                self.cv.select([])
            return
        if add and eid in self.cv.selected:
            self.cv.select(self.cv.selected - {eid})
        else:
            self.cv.select([eid], add=add)

    def entry(self, text: str) -> None:
        self.cv.hintChanged.emit("Select tool: pick an entity with the mouse, or choose a drawing tool")


class TrimTool(Tool):
    name, hint = "trim", "Trim: click the part of a line, arc or circle to remove (up to the nearest crossings)"

    def press(self, snap: Snap, pos: QPointF, mods) -> None:
        seg = self.cv.pick_seg(pos)
        if seg is None:
            self.cv.hintChanged.emit("Nothing to trim there – click on a line, an arc or a circle")
            return
        click = (pos.x(), pos.y())
        if self.cv.commit(lambda sk: ops.trim(sk, seg.entity, click)):
            self.cv.select([])

    def entry(self, text: str) -> None:
        self.cv.hintChanged.emit("Trim: click the part to remove")


class MoveTool(Tool):
    name, hint = "move", "Move: select entities first, then click the base point and the target, or type the vector dx, dy"
    needs_selection = True

    def _ids(self) -> list[str]:
        ids = sorted(self.cv.selected)
        if not ids:
            self.cv.hintChanged.emit("Select the entities to move first (Select tool), then use Move")
        return ids

    def add(self, ref: PointRef) -> None:
        if not self._ids():
            return
        self.pts.append(ref)
        if len(self.pts) == 2:
            a, b = self.xy(self.pts[0]), self.xy(self.pts[1])
            ids = self._ids()
            self.pts = []
            self.cv.commit(lambda sk: ops.move_entities(sk, ids, _r(b[0] - a[0]), _r(b[1] - a[1])), f"Moved {', '.join(ids)}")
        else:
            self.hint_set("Move: the target point (or Esc)")

    def entry(self, text: str) -> None:
        ids = self._ids()
        if not ids:
            return
        if not self.pts:                                 # a typed vector dx, dy
            try:
                dx, dy = ops.parse_vector(text, self.cv.sk)
            except SketchError as e:
                self.cv.hintChanged.emit(str(e))
                return
            self.cv.commit(lambda sk: ops.move_entities(sk, ids, dx, dy), f"Moved {', '.join(ids)} by ({dx}, {dy})")
            return
        super().entry(text)

    def paint(self, painter: QPainter) -> None:
        if self.pts and self.cv.cursor is not None:
            self.rubber(painter, self.xy(self.pts[0]), self.cv.cursor.xy)


class CopyTool(MoveTool):
    """Linear copy: copies of the selection at k × vector, k = 1 … n (n from the editor)."""
    name, hint = "copy", "Linear copy: select entities first, then the base point and the target, or type the vector dx, dy (copies: n in the toolbar)"
    count = 1

    def add(self, ref: PointRef) -> None:
        if not self._ids():
            return
        self.pts.append(ref)
        if len(self.pts) == 2:
            a, b = self.xy(self.pts[0]), self.xy(self.pts[1])
            self.pts = []
            self._copy(b[0] - a[0], b[1] - a[1])
        else:
            self.hint_set("Copy: the target point (or Esc)")

    def _copy(self, dx: float, dy: float) -> None:
        ids = sorted(self.cv.selected)
        n = max(1, int(self.count))

        def run(sk):
            made = []
            for k in range(1, n + 1):
                made += [e.id for e in ops.copy_entities(sk, ids, lambda x, y, k=k: (x + k * dx, y + k * dy))]
            return made
        made = self.cv.commit(run, f"Copied {', '.join(ids)} × {n}")
        if isinstance(made, list):
            self.cv.select(made)

    def entry(self, text: str) -> None:
        ids = self._ids()
        if not ids:
            return
        if not self.pts:
            try:
                dx, dy = ops.parse_vector(text, self.cv.sk)
                env = self.cv.sk.values()
                self._copy(evaluate(dx, env), evaluate(dy, env))
            except SketchError as e:
                self.cv.hintChanged.emit(str(e))
            return
        Tool.entry(self, text)


class MirrorTool(Tool):
    name, hint = "mirror", "Mirror: select entities first, then two points of the mirror line (the copies are mirrored, the originals stay)"
    needs_selection = True

    def add(self, ref: PointRef) -> None:
        ids = sorted(self.cv.selected)
        if not ids:
            self.cv.hintChanged.emit("Select the entities to mirror first (Select tool), then use Mirror")
            return
        self.pts.append(ref)
        if len(self.pts) == 2:
            a, b = self.xy(self.pts[0]), self.xy(self.pts[1])
            self.pts = []
            try:
                f = ops.mirror_function(a, b)
            except SketchError as e:
                self.cv.hintChanged.emit(str(e))
                return
            made = self.cv.commit(lambda sk: [e.id for e in ops.copy_entities(sk, ids, f, mirror=True)], f"Mirrored {', '.join(ids)}")
            if isinstance(made, list):
                self.cv.select(made)
        else:
            self.hint_set("Mirror: the second point of the mirror line")

    def paint(self, painter: QPainter) -> None:
        if self.pts and self.cv.cursor is not None:
            self.rubber(painter, self.xy(self.pts[0]), self.cv.cursor.xy)


TOOLS: dict[str, type[Tool]] = {t.name: t for t in (SelectTool, LineTool, PolylineTool, RectTool, CircleTool, Arc3Tool, ArcCenterTool,
                                                     PointTool, TrimTool, MoveTool, CopyTool, MirrorTool)}
