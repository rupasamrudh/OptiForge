#!/usr/bin/env python
"""Entry point to run the OptiForge web server."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "optiforge.api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )