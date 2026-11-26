"""Space structures (ECSS): requirement set, margins of safety and the later checks of the small-satellite workflow (F14).

Pure headless Python (no Qt, no `fea_core.mcp` import). The package ships NO standard numbers: every factor, load level and limit
comes from a requirement set that the user entered and explicitly confirmed (`fea_core.space.requirements`). The modules are imported
where they are used (this file stays empty so that stages can add modules independently).
"""
