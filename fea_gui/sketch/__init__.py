"""Editor of the sketch for 1D / 2D models without CAD (F15-S5): `canvas` (QGraphicsView with grid, snap, picture), `tools` (line,
polyline, rectangle, circle, arcs, point, trim, move, mirror, copy), `editor` (the window with the parameter table, the entity
list and the coordinate entry), `hooks` (File → New, Model step → Edit sketch, building into the main window).

The model of the sketch and every editing operation are in `fea_core.sketch` (no Qt there); this package only draws and collects
the clicks.
"""
