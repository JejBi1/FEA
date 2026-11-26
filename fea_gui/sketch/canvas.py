"""The drawing area of the sketch editor: a QGraphicsView (pure Qt 2D, not the OCCT window) with the grid, the snapping, the picture of the
sketch and the mouse handling. Millimetres, y up; zoom with the wheel around the cursor, pan with the middle button (or Space + left)."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QKeyEvent, QMouseEvent, QPainter, QPainterPath, QPen, QWheelEvent
from PySide6.QtWidgets import QGraphicsScene, QGraphicsView

from fea_core.sketch.expr import SketchError
from fea_core.sketch.model import Resolved, Seg, Sketch, resolve

from .. import theme
from ..widgets import help_tip

if TYPE_CHECKING:
    from .tools import Tool

WORLD = 1e7                                        # scene extent (the view never shows anything near it)


class Snap(NamedTuple):
    xy: tuple[float, float]
    kind: str                                      # point | mid | grid | align | free
    pid: str | None = None


class History:
    """Undo / redo of the editor: snapshots of the sketch (JSON text) taken before every change."""

    def __init__(self, sk: Sketch, limit: int = 200):
        self.sk = sk
        self.undo_stack: list[str] = []
        self.redo_stack: list[str] = []
        self.limit = limit

    @staticmethod
    def _dump(sk: Sketch) -> str:
        import json
        return json.dumps(sk.to_dict(), sort_keys=True)

    def push(self) -> None:
        snap = self._dump(self.sk)
        if self.undo_stack and self.undo_stack[-1] == snap:
            return
        self.undo_stack.append(snap)
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def _restore(self, snap: str) -> None:
        import json
        new = Sketch.from_dict(json.loads(snap))
        assert new is not None
        keep = self.sk.built                       # what was built does not belong to the drawing history
        self.sk.restore(new)
        self.sk.built = keep

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self._dump(self.sk))
        self._restore(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self._dump(self.sk))
        self._restore(self.redo_stack.pop())
        return True

    def rollback(self) -> None:
        """The change that followed the last snapshot failed: back to that snapshot and forget it (nothing goes to redo)."""
        if self.undo_stack:
            self._restore(self.undo_stack.pop())


def nice_step(raw: float) -> float:
    e = 10.0 ** math.floor(math.log10(max(raw, 1e-12)))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * e >= raw:
            return m * e
    return 10.0 * e


class SketchCanvas(QGraphicsView):
    sketchChanged = Signal()                       # a tool changed the sketch
    hintChanged = Signal(str)                      # what the tool expects next / the last error
    cursorMoved = Signal(float, float)
    selectionChanged = Signal(list)                # ids of the selected entities

    def __init__(self, sk: Sketch, history: History, parent=None):
        super().__init__(parent)
        self.sk = sk
        self.history = history
        self.tool: Tool | None = None
        self.selected: set[str] = set()
        self.problems: set[str] = set()            # entities of the last error (red)
        self.regions: list[dict] = []              # [{"rings": [...], "name": str}] from the last check
        self.res: Resolved | None = None
        self.res_error = ""
        self.snap_grid = True
        self.snap_objects = True
        self.show_ids = True
        self.cursor: Snap | None = None
        self._pan: tuple[float, float] | None = None
        self._space = False
        self._first_fit = True
        sc = QGraphicsScene(-WORLD, -WORLD, 2 * WORLD, 2 * WORLD, self)
        self.setScene(sc)
        self.scale(4.0, -4.0)                      # y up, 4 px per mm to start with
        self.centerOn(0.0, 0.0)
        self.setMouseTracking(True)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Sketch drawing area")
        help_tip(self, "Sketch: click to place points; wheel = zoom, middle button = pan", "Sketch: click to place points (they snap to end points, mid points and the grid); wheel = zoom, middle button = pan")
        self.refresh()

    # ------------------------------------------------------------------ the sketch
    def refresh(self) -> None:
        try:
            self.res = resolve(self.sk)
            self.res_error = ""
        except (SketchError, ArithmeticError, ValueError, RecursionError) as e:       # nothing in the geometry may take the editor down
            self.res, self.res_error = None, str(e)
            self.problems = set(getattr(e, "entities", ()))
        self.selected = {s for s in self.selected if any(x.id == s for x in self.sk.entities)}
        self.viewport().update()

    def set_tool(self, tool: Tool | None) -> None:
        if self.tool is not None:
            self.tool.cancel()
        self.tool = tool
        self.hintChanged.emit(tool.hint if tool is not None else "")
        self.viewport().update()

    def commit(self, fn, ok_hint: str = ""):
        """Runs fn(sketch) as one undoable change; a SketchError is shown and nothing changes."""
        self.history.push()
        try:
            out = fn(self.sk)
        except (SketchError, ArithmeticError, ValueError, RecursionError) as e:       # a bad expression: nothing changes
            self.history.rollback()
            self.refresh()
            self.hintChanged.emit(str(e))
            return None
        self.problems.clear()
        self.regions = []
        self.refresh()
        self.sketchChanged.emit()
        if ok_hint:
            self.hintChanged.emit(ok_hint)
        return out if out is not None else True

    def select(self, ids, add: bool = False) -> None:
        ids = set(ids)
        self.selected = (self.selected | ids) if add else ids
        self.selectionChanged.emit(sorted(self.selected))
        self.viewport().update()

    def delete_selected(self) -> None:
        if not self.selected:
            return
        ids = sorted(self.selected)

        def run(sk):
            for eid in ids:
                sk.remove_entity(eid)
        self.selected = set()
        if self.commit(run, f"Deleted {', '.join(ids)}"):
            self.selectionChanged.emit([])

    # ------------------------------------------------------------------ view
    def scale_px(self) -> float:
        return abs(self.transform().m11())

    def fit_all(self) -> None:
        pts: list[tuple[float, float]] = []
        if self.res is not None:
            for s in self.res.segs:
                pts += s.polyline(10.0)
            pts += list(self.res.pts.values())
        if not pts:
            self.resetTransform()
            self.scale(4.0, -4.0)
            self.centerOn(0.0, 0.0)
            return
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        w, h = max(max(xs) - min(xs), 1e-6), max(max(ys) - min(ys), 1e-6)
        vw, vh = max(self.viewport().width(), 50), max(self.viewport().height(), 50)
        k = 0.85 * min(vw / w, vh / h)
        self.resetTransform()
        self.scale(k, -k)
        self.centerOn((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
        self.viewport().update()

    def showEvent(self, e) -> None:
        super().showEvent(e)
        if self._first_fit:
            self._first_fit = False
            self.fit_all()

    def wheelEvent(self, e: QWheelEvent) -> None:
        k = 1.2 ** (e.angleDelta().y() / 120.0)
        if 1e-4 < self.scale_px() * k < 1e6:
            self.scale(k, k)
        self.viewport().update()

    # ------------------------------------------------------------------ snapping
    def snap_at(self, pos: QPointF, last: tuple[float, float] | None = None) -> Snap:
        px = max(self.scale_px(), 1e-9)
        tol = 10.0 / px
        x, y = pos.x(), pos.y()
        if self.snap_objects and self.res is not None:
            best: tuple[float, Snap] | None = None
            for pid, p in self.res.pts.items():
                d = math.hypot(p[0] - x, p[1] - y)
                if d < tol and (best is None or d < best[0]):
                    best = (d, Snap(p, "point", pid))
            if best is None:
                for s in self.res.segs:
                    if s.construction or s.kind == "circle":
                        continue
                    m = s.mid()
                    d = math.hypot(m[0] - x, m[1] - y)
                    if d < tol * 0.8 and (best is None or d < best[0]):
                        best = (d, Snap(m, "mid"))
            if best is not None:
                return best[1]
        if last is not None:                       # horizontal / vertical alignment with the previous point
            ax = abs(y - last[1]) < tol * 0.6
            ay = abs(x - last[0]) < tol * 0.6
            if ax or ay:
                gx = last[0] if ay else x
                gy = last[1] if ax else y
                if self.snap_grid and not (ax and ay):
                    g = self.grid_step()
                    gx = gx if ay else round(gx / g) * g
                    gy = gy if ax else round(gy / g) * g
                return Snap((gx, gy), "align")
        if self.snap_grid:
            g = self.grid_step()
            return Snap((round(x / g) * g, round(y / g) * g), "grid")
        return Snap((x, y), "free")

    def grid_step(self) -> float:
        return nice_step(14.0 / max(self.scale_px(), 1e-9))

    def pick(self, pos: QPointF, px: float = 8.0) -> str | None:
        """The entity nearest to the point (within `px` pixels), or None."""
        if self.res is None:
            return None
        tol = px / max(self.scale_px(), 1e-9)
        best: tuple[float, str] | None = None
        for s in self.res.segs:
            d = s.distance((pos.x(), pos.y()))
            if d < tol and (best is None or d < best[0]):
                best = (d, s.entity)
        return best[1] if best else None

    def pick_seg(self, pos: QPointF, px: float = 8.0) -> Seg | None:
        if self.res is None:
            return None
        tol = px / max(self.scale_px(), 1e-9)
        best: tuple[float, Seg] | None = None
        for s in self.res.segs:
            d = s.distance((pos.x(), pos.y()))
            if d < tol and (best is None or d < best[0]):
                best = (d, s)
        return best[1] if best else None

    # ------------------------------------------------------------------ mouse and keys
    def _last(self) -> tuple[float, float] | None:
        return self.tool.last_xy() if self.tool is not None else None

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        if self._pan is not None:
            dx, dy = e.position().x() - self._pan[0], e.position().y() - self._pan[1]
            self._pan = (e.position().x(), e.position().y())
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(dx))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(dy))
            return
        pos = self.mapToScene(e.position().toPoint())
        self.cursor = self.snap_at(pos, self._last())
        self.cursorMoved.emit(*self.cursor.xy)
        if self.tool is not None:
            self.tool.move(self.cursor)
        self.viewport().update()

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self.setFocus()
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and self._space):
            self._pan = (e.position().x(), e.position().y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        pos = self.mapToScene(e.position().toPoint())
        snap = self.snap_at(pos, self._last())
        self.cursor = snap
        if e.button() == Qt.MouseButton.LeftButton and self.tool is not None:
            self.tool.press(snap, pos, e.modifiers())
        elif e.button() == Qt.MouseButton.RightButton and self.tool is not None:
            self.tool.finish()
        self.viewport().update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        if self.tool is not None and e.button() == Qt.MouseButton.LeftButton:
            self.tool.finish()
        self.viewport().update()

    def keyPressEvent(self, e: QKeyEvent) -> None:
        k = e.key()
        if k == Qt.Key.Key_Escape:
            if self.tool is not None and self.tool.pts:
                self.tool.cancel()
            else:
                self.select([])
        elif k in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.tool is not None:
            self.tool.finish()
        elif k == Qt.Key.Key_Delete:
            self.delete_selected()
        elif k == Qt.Key.Key_Space:
            self._space = True
        elif k == Qt.Key.Key_F:
            self.fit_all()
        elif k == Qt.Key.Key_A and e.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.select([x.id for x in self.sk.entities])
        else:
            super().keyPressEvent(e)
        self.viewport().update()

    def keyReleaseEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key.Key_Space:
            self._space = False
        super().keyReleaseEvent(e)

    # ------------------------------------------------------------------ painting
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        c = theme.colors()
        painter.fillRect(rect, QColor(c["input"]))
        g = self.grid_step()
        pen = QPen(QColor(c["border"]))
        pen.setCosmetic(True)
        pen.setWidth(1)
        x0, x1 = math.floor(rect.left() / g) * g, rect.right()
        y0, y1 = math.floor(rect.top() / g) * g, rect.bottom()
        if (x1 - x0) / g > 600 or (y1 - y0) / g > 600:
            return
        painter.setPen(pen)
        x = x0
        while x <= x1:
            if abs(x / (5 * g) - round(x / (5 * g))) > 1e-9:
                painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += g
        y = y0
        while y <= y1:
            if abs(y / (5 * g) - round(y / (5 * g))) > 1e-9:
                painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += g
        major = QPen(QColor(c["button_border"]))
        major.setCosmetic(True)
        painter.setPen(major)
        x = math.floor(rect.left() / (5 * g)) * 5 * g
        while x <= x1:
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += 5 * g
        y = math.floor(rect.top() / (5 * g)) * 5 * g
        while y <= y1:
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            y += 5 * g

    def _path(self, s: Seg) -> QPainterPath:
        pts = s.polyline(2.0)
        p = QPainterPath(QPointF(*pts[0]))
        for q in pts[1:]:
            p.lineTo(QPointF(*q))
        return p

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        c = theme.colors()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        axis = QPen(QColor(c["muted"]))
        axis.setCosmetic(True)
        axis.setWidth(1)
        painter.setPen(axis)
        painter.drawLine(QPointF(0.0, rect.top()), QPointF(0.0, rect.bottom()))
        painter.drawLine(QPointF(rect.left(), 0.0), QPointF(rect.right(), 0.0))
        if self.sk.kind == "twod" and self.sk.mode == "axisym":
            ap = QPen(QColor(c["ok"]))
            ap.setCosmetic(True)
            ap.setWidth(2)
            ap.setStyle(Qt.PenStyle.DashDotLine)
            painter.setPen(ap)
            painter.drawLine(QPointF(0.0, rect.top()), QPointF(0.0, rect.bottom()))
        for reg in self.regions:
            path = QPainterPath()
            path.setFillRule(Qt.FillRule.OddEvenFill)
            for ring in reg["rings"]:
                path.moveTo(QPointF(*ring[0]))
                for q in ring[1:]:
                    path.lineTo(QPointF(*q))
                path.closeSubpath()
            fill = QColor(c["accent"])
            fill.setAlpha(46)
            painter.fillPath(path, QBrush(fill))
        if self.res is None:
            return
        for s in self.res.segs:
            pen = QPen(QColor(c["text"]))
            pen.setCosmetic(True)
            pen.setWidthF(1.8)
            if s.entity in self.problems:
                pen.setColor(QColor(c["fail"]))
                pen.setWidthF(3.0)
            elif s.entity in self.selected:
                pen.setColor(QColor(c["accent"]))
                pen.setWidthF(3.0)
            elif s.construction:
                pen.setColor(QColor(c["muted"]))
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPath(self._path(s))
        painter.save()
        painter.resetTransform()
        font = QFont(painter.font())
        font.setPointSizeF(8.0)
        painter.setFont(font)
        for pid, p in self.res.pts.items():
            q = self.mapFromScene(QPointF(*p))
            used = pid in self.sk.node_points()
            painter.setPen(QPen(QColor(c["muted"] if not used else c["text_hi"])))
            painter.setBrush(QBrush(QColor(c["input"])))
            painter.drawRect(q.x() - 2, q.y() - 2, 4, 4)
        for sup in self.sk.supports:
            p = self.res.pts.get(sup.point)
            if p is None:
                continue
            q = self.mapFromScene(QPointF(*p))
            painter.setPen(QPen(QColor(c["ok"]), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if sup.kind == "fixed":
                painter.drawRect(q.x() - 7, q.y() - 7, 14, 14)
            elif sup.kind == "node":
                painter.drawEllipse(q, 6, 6)
            else:
                painter.drawPolygon([QPointF(q.x(), q.y() - 2), QPointF(q.x() - 8, q.y() + 10), QPointF(q.x() + 8, q.y() + 10)])
        if self.show_ids:
            painter.setPen(QPen(QColor(c["accent_soft_text"] if theme.is_dark() else c["accent"])))
            seen: set[str] = set()
            for s in self.res.segs:
                if s.entity in seen and not s.id.count("."):
                    continue
                seen.add(s.entity)
                m = self.mapFromScene(QPointF(*s.mid()))
                painter.drawText(QPointF(m.x() + 4, m.y() - 4), s.id)
        painter.restore()
        if self.tool is not None:
            self.tool.paint(painter)
        if self.cursor is not None:
            painter.save()
            painter.resetTransform()
            q = self.mapFromScene(QPointF(*self.cursor.xy))
            col = QColor(c["accent"] if self.cursor.kind in ("point", "mid") else c["muted"])
            painter.setPen(QPen(col, 1.5))
            r = 6 if self.cursor.kind in ("point", "mid") else 3
            painter.drawEllipse(q, r, r)
            la = self.tool.last_xy() if self.tool is not None else None
            if self.cursor.kind == "align" and la is not None:
                guide = QPen(col)
                guide.setStyle(Qt.PenStyle.DashLine)
                painter.setPen(guide)
                painter.drawLine(self.mapFromScene(QPointF(*la)), q)
            painter.restore()
        if self.res_error:
            painter.save()
            painter.resetTransform()
            painter.setPen(QPen(QColor(c["fail"])))
            painter.drawText(QPointF(10, 18), self.res_error[:160])
            painter.restore()
