"""FastAPI application for OptiForge web interface."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from optiforge.api.adapter import api_request_to_model, solution_to_response
from optiforge.api.models import ErrorResponse, HealthResponse, SolveRequest, SolveResponse
from optiforge.milp.solver import solve_milp
from optiforge.model import Model


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_SOLVE_TIMEOUT_SECONDS = 30.0
MAX_SOLVE_TIMEOUT_SECONDS = 300.0  # 5 minutes max
DEFAULT_MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10 MB

# ---------------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------------
# IMPORTANT: Using allow_origins=["*"] combined with allow_credentials=True is
# prohibited by the Fetch spec. FastAPI would echo arbitrary Origin headers
# with credentials, enabling cross-origin credential abuse.
#
# For this prototype: credentials are disabled (allow_credentials=False).
# For production with authentication, set OPTIFORGE_ALLOWED_ORIGINS to a
# comma-separated list of specific origins, e.g.:
#   OPTIFORGE_ALLOWED_ORIGINS=https://app.example.com,https://staging.example.com
#
# Leaving OPTIFORGE_ALLOWED_ORIGINS unset defaults to localhost development origins.

_raw_origins = os.environ.get("OPTIFORGE_ALLOWED_ORIGINS", "")
if _raw_origins.strip():
    ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]
else:
    # Safe default: localhost development origins only
    ALLOWED_ORIGINS = [
        "http://localhost",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:3000",
    ]


# ---------------------------------------------------------------------------
# Security middleware
# ---------------------------------------------------------------------------

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # unsafe-inline removed: all JS is in app.js, all CSS is in style.css.
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )
        return response


class RequestSizeLimitMiddleware:
    """Pure ASGI middleware to limit request body size to prevent DoS.

    Enforcement strategy:
    1. Fast path: check Content-Length header. If > max_size, return 413 immediately.
    2. Streaming / chunked path: wrap the ASGI receive callable to count bytes.
       If accumulated bytes exceed max_size, return 413 immediately.
    """

    def __init__(self, app, max_size: int = DEFAULT_MAX_REQUEST_SIZE):
        self.app = app
        self.max_size = max_size

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length:
            try:
                if int(content_length) > self.max_size:
                    response = Response(
                        content=f"Request body too large. Maximum size: {self.max_size} bytes",
                        status_code=413,
                    )
                    await response(scope, receive, send)
                    return
            except ValueError:
                pass

        total_received = 0

        async def limited_receive():
            nonlocal total_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                total_received += len(body)
                if total_received > self.max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body too large. Maximum size: {self.max_size} bytes",
                    )
            return message

        try:
            await self.app(scope, limited_receive, send)
        except HTTPException as exc:
            if exc.status_code == 413:
                response = Response(
                    content=exc.detail,
                    status_code=413,
                )
                await response(scope, receive, send)
            else:
                raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    yield


app = FastAPI(
    title="OptiForge API",
    description="Mathematical optimization solver API for LP and MILP problems",
    version="0.1.0",
    lifespan=lifespan,
)

# Add security middleware
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_size=DEFAULT_MAX_REQUEST_SIZE)

# Add CORS middleware
# NOTE: allow_credentials=False is intentional.
# Wildcard origins + credentials is a prohibited combination per the Fetch spec.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="optiforge/api/static"), name="static")

# Templates for HTML rendering
templates = Jinja2Templates(directory="optiforge/api/templates")


class SolveRequestWithTimeout(SolveRequest):
    """Extended solve request with optional timeout parameter."""
    timeout_seconds: Optional[float] = Field(
        default=DEFAULT_SOLVE_TIMEOUT_SECONDS,
        ge=0.1,
        le=MAX_SOLVE_TIMEOUT_SECONDS,
        description=f"Maximum time to spend solving (seconds). Default: {DEFAULT_SOLVE_TIMEOUT_SECONDS}, Max: {MAX_SOLVE_TIMEOUT_SECONDS}"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


async def solve_with_timeout(model: Model, has_integer_vars: bool, timeout_seconds: float):
    """Solve a model with a timeout using cooperative cancellation.

    Creates a threading.Event (stop_event) and passes it into the solver.
    When asyncio.wait_for() raises TimeoutError, stop_event is set so the
    solver thread terminates at its next safe boundary (simplex iteration
    boundary for LP, B&B node boundary for MILP) rather than running
    indefinitely in the thread pool.

    IMPORTANT: asyncio.wait_for() alone does NOT terminate the underlying
    thread. The stop_event is the mechanism that actually signals the solver.
    """
    stop_event = threading.Event()

    def _solve_sync():
        """Synchronous solve function to run in thread pool."""
        if has_integer_vars:
            from optiforge.numerics.numerical_monitor import NumericalStabilityMonitor
            numerical_monitor = NumericalStabilityMonitor()
            return solve_milp(model, numerical_monitor=numerical_monitor, stop_event=stop_event)
        else:
            return model.solve(stop_event=stop_event)

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, _solve_sync)
    try:
        solution = await asyncio.wait_for(future, timeout=timeout_seconds)
        return solution
    except asyncio.TimeoutError:
        # Signal the solver thread to stop at its next safe cancellation point.
        stop_event.set()
        raise HTTPException(
            status_code=408,
            detail=f"Solver timed out after {timeout_seconds} seconds",
        )


@app.post("/solve", response_model=SolveResponse, responses={
    400: {"model": ErrorResponse},
    408: {"model": ErrorResponse},
    413: {"model": ErrorResponse},
    500: {"model": ErrorResponse}
})
async def solve_problem(request: SolveRequestWithTimeout):
    """
    Solve an LP or MILP problem.

    Validates the input, converts to OptiForge model, solves, and returns structured results.

    Normal solver outcomes (infeasible, unbounded, iteration_limit, unknown) are returned
    as HTTP 200 with a `status` field in the response body. They are not HTTP error codes.
    Only true errors (validation failures, timeouts, internal exceptions) use non-200 status.
    """
    start_time = time.perf_counter()

    try:
        # Convert API request to OptiForge model
        model = api_request_to_model(request)

        # Check if model has integer/binary variables
        from optiforge.model import VariableType
        has_integer_vars = any(
            var.variable_type in (VariableType.INTEGER, VariableType.BINARY)
            for var in model.variables.values()
        )

        # Solve using appropriate solver with timeout
        solution = await solve_with_timeout(model, has_integer_vars, request.timeout_seconds)

        solve_time_ms = (time.perf_counter() - start_time) * 1000

        # Convert solution to response
        response_data = solution_to_response(solution, solve_time_ms, request, model)

        # Add MILP-specific metrics from solution
        if hasattr(solution, 'incumbent_objective') and solution.incumbent_objective is not None:
            response_data["incumbent_objective"] = solution.incumbent_objective
        if hasattr(solution, 'best_bound') and solution.best_bound is not None:
            response_data["best_bound"] = solution.best_bound
        if hasattr(solution, 'optimality_gap') and solution.optimality_gap is not None:
            response_data["optimality_gap"] = solution.optimality_gap
        if hasattr(solution, 'absolute_gap') and solution.absolute_gap is not None:
            response_data["absolute_gap"] = solution.absolute_gap
        if hasattr(solution, 'nodes_remaining') and solution.nodes_remaining is not None:
            response_data["nodes_remaining"] = solution.nodes_remaining
        if hasattr(solution, 'max_depth') and solution.max_depth is not None:
            response_data["max_depth"] = solution.max_depth
        if hasattr(solution, 'feasible_solutions_found') and solution.feasible_solutions_found is not None:
            response_data["feasible_solutions_found"] = solution.feasible_solutions_found

        return SolveResponse(**response_data)

    except HTTPException:
        # Re-raise HTTP exceptions (including timeout)
        raise
    except ValueError as e:
        # Validation errors from adapter or model construction
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Classify unexpected solver exceptions.
        # Note: normal infeasible/unbounded results do NOT reach here —
        # they are returned as Solution objects with HTTP 200 above.
        error_msg = str(e)
        if "singular" in error_msg.lower():
            raise HTTPException(status_code=500, detail=f"Numerical error (singular matrix): {error_msg}")
        elif "iteration limit" in error_msg.lower() or "node limit" in error_msg.lower():
            raise HTTPException(status_code=400, detail=f"Solver reached iteration/node limit: {error_msg}")
        else:
            # Generic solver error - do not expose internal stack trace
            raise HTTPException(status_code=500, detail=f"Solver error: {error_msg}")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI page."""
    return templates.TemplateResponse(
      request=request,
      name="index.html",
      context={"request": request},
    )


@app.get("/api/docs", include_in_schema=False)
async def api_docs(request: Request):
    """Redirect to FastAPI auto-generated docs."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")
