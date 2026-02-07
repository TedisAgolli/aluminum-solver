from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
import pulp
from typing import Dict
import logging
import sys
import time
import threading
import os

# Configure logging to stdout so we can see it in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SolverRequest(BaseModel):
    stock_length: int = Field(default=6500, gt=0, description="Length of stock aluminum bar in mm")
    pieces: Dict[int, int] = Field(..., description="Dictionary of piece lengths to quantities")

    @field_validator('pieces')
    @classmethod
    def validate_pieces(cls, v):
        if not v:
            raise ValueError("Must specify at least one piece")
        for length, quantity in v.items():
            if length <= 0:
                raise ValueError(f"Piece length must be positive, got {length}")
            if quantity <= 0:
                raise ValueError(f"Quantity must be positive, got {quantity}")
        return v


class BarPlan(BaseModel):
    bar_number: int
    cuts: list[int]
    used_mm: int
    waste_mm: int


class SolverResponse(BaseModel):
    bars_needed: int
    bar_plans: list[BarPlan]
    total_waste: int
    efficiency_percent: float
    used_heuristic: bool = False  # True if FFD fallback was used


def solve_ffd(stock_length: int, cuts: list[int]) -> list[list[int]]:
    """First-Fit Decreasing heuristic - fast near-optimal solution."""
    sorted_cuts = sorted(cuts, reverse=True)
    bars = []
    bar_remaining = []

    for cut in sorted_cuts:
        placed = False
        for i, remaining in enumerate(bar_remaining):
            if remaining >= cut:
                bars[i].append(cut)
                bar_remaining[i] -= cut
                placed = True
                break
        if not placed:
            bars.append([cut])
            bar_remaining.append(stock_length - cut)

    return bars


def solve_cutting_stock(stock_length: int, pieces: Dict[int, int]) -> SolverResponse:
    """
    Solve the cutting stock problem using linear programming.
    
    Args:
        stock_length: Length of each stock bar in mm
        pieces: Dictionary mapping piece lengths to quantities needed
        
    Returns:
        SolverResponse with optimal cutting plan
    """
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("🔧 Starting cutting stock optimization")
    logger.info(f"📏 Stock length: {stock_length}mm")
    logger.info(f"📦 Pieces requested: {pieces}")
    
    # Validate that all pieces fit in stock length
    logger.info("✓ Validating piece lengths...")
    for length in pieces.keys():
        if length > stock_length:
            raise ValueError(f"Piece length {length}mm exceeds stock length {stock_length}mm")
    
    # Expand pieces into flat list
    logger.info("✓ Preparing problem data...")
    cuts = []
    for length, count in pieces.items():
        cuts.extend([length] * count)
    
    n = len(cuts)
    logger.info(f"📊 Total pieces to cut: {n}")

    # For large problems, skip ILP and use FFD directly for fast response
    if n > 15:
        logger.info("⚡ Large problem detected, using FFD heuristic for fast response")
        bar_assignments = solve_ffd(stock_length, cuts)
        bar_plans = []
        total_waste = 0

        for assigned in bar_assignments:
            used = sum(assigned)
            waste = stock_length - used
            total_waste += waste
            bar_plans.append(BarPlan(
                bar_number=len(bar_plans) + 1,
                cuts=assigned,
                used_mm=used,
                waste_mm=waste
            ))

        total_material_needed = sum(cuts)
        total_material_used = len(bar_plans) * stock_length
        efficiency = (total_material_needed / total_material_used * 100) if total_material_used > 0 else 0
        elapsed_time = time.time() - start_time

        logger.info(f"📋 Results (FFD):")
        logger.info(f"   - Bars needed: {len(bar_plans)}")
        logger.info(f"   - Total waste: {total_waste}mm")
        logger.info(f"   - Efficiency: {round(efficiency, 1)}%")
        logger.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        logger.info("=" * 60)

        return SolverResponse(
            bars_needed=len(bar_plans),
            bar_plans=bar_plans,
            total_waste=total_waste,
            efficiency_percent=round(efficiency, 1),
            used_heuristic=True
        )

    bars = n  # safe upper bound
    logger.info(f"📊 Maximum bars possible: {bars}")

    logger.info("🔨 Building linear programming model...")
    model = pulp.LpProblem("CuttingStockMM", pulp.LpMinimize)
    
    # x[i][j] = cut i assigned to bar j
    logger.info("  - Creating decision variables...")
    x = pulp.LpVariable.dicts("x", (range(n), range(bars)), cat="Binary")
    # y[j] = bar j is used
    y = pulp.LpVariable.dicts("y", range(bars), cat="Binary")
    
    # objective: minimize number of bars
    logger.info("  - Setting objective function...")
    model += pulp.lpSum(y[j] for j in range(bars))
    
    # each cut must be assigned once
    logger.info("  - Adding assignment constraints...")
    for i in range(n):
        model += pulp.lpSum(x[i][j] for j in range(bars)) == 1
    
    # bar capacity constraints
    logger.info("  - Adding capacity constraints...")
    for j in range(bars):
        model += pulp.lpSum(cuts[i] * x[i][j] for i in range(n)) <= stock_length * y[j]
    
    logger.info("🚀 Solving optimization problem (this may take a while)...")
    logger.info("   Please wait - solver is working...")
    
    # Create a flag to stop the progress logger when solving is done
    solving = {'active': True}
    solve_start = time.time()
    
    def log_progress():
        """Log progress every 10 seconds while solver is running"""
        while solving['active']:
            time.sleep(10)
            if solving['active']:
                elapsed = time.time() - solve_start
                logger.info(f"   ⏳ Still solving... {elapsed:.1f} seconds elapsed")
    
    # Start progress logger in background thread
    progress_thread = threading.Thread(target=log_progress, daemon=True)
    progress_thread.start()
    
    try:
        model.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=2))
    finally:
        solving['active'] = False
        progress_thread.join(timeout=1)
    
    # Check if solution was found, fallback to FFD if not optimal
    used_heuristic = False
    bar_plans = []
    total_waste = 0

    if model.status != pulp.LpStatusOptimal:
        logger.warning("⚠️ ILP solver timed out, falling back to FFD heuristic")
        used_heuristic = True
        bar_assignments = solve_ffd(stock_length, cuts)

        for assigned in bar_assignments:
            used = sum(assigned)
            waste = stock_length - used
            total_waste += waste

            bar_plans.append(BarPlan(
                bar_number=len(bar_plans) + 1,
                cuts=assigned,
                used_mm=used,
                waste_mm=waste
            ))
    else:
        logger.info("✅ Optimal solution found! Building cutting plan...")

        for j in range(bars):
            if y[j].value() == 1:
                assigned = [cuts[i] for i in range(n) if x[i][j].value() == 1]
                used = sum(assigned)
                waste = stock_length - used
                total_waste += waste

                bar_plans.append(BarPlan(
                    bar_number=len(bar_plans) + 1,
                    cuts=assigned,
                    used_mm=used,
                    waste_mm=waste
                ))

    # Calculate efficiency
    total_material_needed = sum(cuts)
    total_material_used = len(bar_plans) * stock_length
    efficiency = (total_material_needed / total_material_used * 100) if total_material_used > 0 else 0

    elapsed_time = time.time() - start_time

    logger.info(f"📋 Results:")
    logger.info(f"   - Bars needed: {len(bar_plans)}")
    logger.info(f"   - Total waste: {total_waste}mm")
    logger.info(f"   - Efficiency: {round(efficiency, 1)}%")
    logger.info(f"   - Used heuristic: {used_heuristic}")
    logger.info(f"⏱️  Total time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60)

    return SolverResponse(
        bars_needed=len(bar_plans),
        bar_plans=bar_plans,
        total_waste=total_waste,
        efficiency_percent=round(efficiency, 1),
        used_heuristic=used_heuristic
    )


@app.get("/")
async def root():
    """Serve the main HTML page"""
    # Read the HTML file from the same directory
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    try:
        with open(html_path, "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        # Fallback if file not found
        raise HTTPException(status_code=500, detail="UI file not found")

@app.get("/api")
async def api_root():
    return {"message": "Aluminum Cutting Calculator API"}


@app.post("/api/solve", response_model=SolverResponse)
async def solve(request: SolverRequest):
    """
    Solve the cutting stock problem and return optimal cutting plan.
    """
    try:
        logger.info("📥 Received solve request")
        result = solve_cutting_stock(request.stock_length, request.pieces)
        logger.info("📤 Sending response to client")
        return result
    except ValueError as e:
        logger.error(f"❌ Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error solving problem: {str(e)}")
