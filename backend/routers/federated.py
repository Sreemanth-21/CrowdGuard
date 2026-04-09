"""
Federated Learning Router

Provides API endpoints for managing federated learning simulations.
Implements Task 43.1 requirements:
- POST /api/federated/simulate endpoint
- GET /api/federated/status endpoint
- Manage simulation state
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime

from backend.services.federated_demo import FederatedDemo

logger = logging.getLogger(__name__)

# Global federated demo instance
federated_demo = FederatedDemo()

router = APIRouter(prefix="/api/federated", tags=["federated"])


# Request/Response Models
class SimulationRequest(BaseModel):
    """Request model for starting a federated learning simulation"""
    rounds: int = Field(default=10, ge=1, le=50, description="Number of federated learning rounds")
    
    class Config:
        schema_extra = {
            "example": {
                "rounds": 10
            }
        }


class SimulationResponse(BaseModel):
    """Response model for simulation creation"""
    simulation_id: str
    status: str
    total_rounds: int
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "abc123-def456-ghi789",
                "status": "idle",
                "total_rounds": 10,
                "message": "Simulation created successfully"
            }
        }


class NodeStatus(BaseModel):
    """Model for individual node status"""
    name: str
    dataset_size: int
    current_accuracy: float
    training_status: str
    rounds_completed: int


class SimulationStatusResponse(BaseModel):
    """Response model for simulation status"""
    simulation_id: str
    status: str
    current_round: int
    total_rounds: int
    nodes: List[NodeStatus]
    global_accuracy: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    convergence_metrics: Optional[Dict[str, Any]] = None
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "abc123-def456-ghi789",
                "status": "running",
                "current_round": 5,
                "total_rounds": 10,
                "nodes": [
                    {
                        "name": "Mall_Entrance",
                        "dataset_size": 100,
                        "current_accuracy": 0.85,
                        "training_status": "completed",
                        "rounds_completed": 5
                    }
                ],
                "global_accuracy": 0.82,
                "start_time": "2024-01-01T10:00:00Z",
                "convergence_metrics": {
                    "convergence_rate": 0.05,
                    "total_improvement": 0.25
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "error": "SimulationNotFound",
                "message": "Simulation with ID abc123 not found"
            }
        }


# API Endpoints

@router.post("/simulate", response_model=SimulationResponse)
async def start_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
) -> SimulationResponse:
    """
    Start a new federated learning simulation.
    
    Creates a new simulation with the specified number of rounds and starts
    the federated learning process in the background.
    
    Args:
        request: Simulation configuration
        background_tasks: FastAPI background tasks for async execution
        
    Returns:
        SimulationResponse with simulation details
        
    Raises:
        HTTPException: If simulation creation fails
    """
    try:
        logger.info(f"Creating new federated learning simulation with {request.rounds} rounds")
        
        # Create new simulation
        simulation_id = federated_demo.create_simulation(rounds=request.rounds)
        
        if not simulation_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to create simulation"
            )
        
        # Start simulation orchestration in background
        background_tasks.add_task(run_simulation_background, simulation_id)
        
        logger.info(f"Simulation {simulation_id} created and started")
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="idle",
            total_rounds=request.rounds,
            message="Simulation created and started successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating simulation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create simulation: {str(e)}"
        )


@router.get("/status/{simulation_id}", response_model=SimulationStatusResponse)
async def get_simulation_status(simulation_id: str) -> SimulationStatusResponse:
    """
    Get the current status of a federated learning simulation.
    
    Returns detailed information about the simulation including node status,
    current round, accuracy metrics, and convergence data.
    
    Args:
        simulation_id: ID of the simulation to query
        
    Returns:
        SimulationStatusResponse with current simulation state
        
    Raises:
        HTTPException: If simulation not found
    """
    try:
        logger.debug(f"Getting status for simulation {simulation_id}")
        
        # Get simulation status
        status = federated_demo.get_simulation_status(simulation_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Simulation {simulation_id} not found"
            )
        
        # Build node status list
        nodes = []
        for node in status.nodes:
            node_status = NodeStatus(
                name=node.name,
                dataset_size=len(node.dataset),
                current_accuracy=node.local_accuracy,
                training_status="completed" if len(node.training_history) > 0 else "idle",
                rounds_completed=len(node.training_history)
            )
            nodes.append(node_status)
        
        # Get current global accuracy
        global_accuracy = None
        if status.global_accuracy_history:
            global_accuracy = status.global_accuracy_history[-1]
        
        # Get convergence metrics if simulation has progress
        convergence_metrics = None
        if status.current_round > 0:
            convergence_data = federated_demo.get_convergence_history(simulation_id)
            convergence_metrics = convergence_data.get('convergence_metrics', {})
        
        return SimulationStatusResponse(
            simulation_id=simulation_id,
            status=status.status,
            current_round=status.current_round,
            total_rounds=status.total_rounds,
            nodes=nodes,
            global_accuracy=global_accuracy,
            start_time=status.start_time,
            end_time=status.end_time,
            convergence_metrics=convergence_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get simulation status: {str(e)}"
        )


@router.get("/status", response_model=Optional[SimulationStatusResponse])
async def get_current_simulation_status() -> Optional[SimulationStatusResponse]:
    """
    Get the status of the current active simulation.
    
    Returns the status of the most recent simulation, or None if no simulation exists.
    
    Returns:
        SimulationStatusResponse for current simulation, or None
    """
    try:
        # Check if there's a current simulation
        if not federated_demo.current_simulation:
            return None
        
        simulation_id = federated_demo.current_simulation.simulation_id
        
        # Use the existing status endpoint logic
        status = federated_demo.get_simulation_status(simulation_id)
        
        if not status:
            return None
        
        # Build node status list
        nodes = []
        for node in status.nodes:
            node_status = NodeStatus(
                name=node.name,
                dataset_size=len(node.dataset),
                current_accuracy=node.local_accuracy,
                training_status="completed" if len(node.training_history) > 0 else "idle",
                rounds_completed=len(node.training_history)
            )
            nodes.append(node_status)
        
        # Get current global accuracy
        global_accuracy = None
        if status.global_accuracy_history:
            global_accuracy = status.global_accuracy_history[-1]
        
        # Get convergence metrics if simulation has progress
        convergence_metrics = None
        if status.current_round > 0:
            convergence_data = federated_demo.get_convergence_history(simulation_id)
            convergence_metrics = convergence_data.get('convergence_metrics', {})
        
        return SimulationStatusResponse(
            simulation_id=simulation_id,
            status=status.status,
            current_round=status.current_round,
            total_rounds=status.total_rounds,
            nodes=nodes,
            global_accuracy=global_accuracy,
            start_time=status.start_time,
            end_time=status.end_time,
            convergence_metrics=convergence_metrics
        )
        
    except Exception as e:
        logger.error(f"Error getting current simulation status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get current simulation status: {str(e)}"
        )


@router.post("/reset/{simulation_id}")
async def reset_simulation(simulation_id: str) -> Dict[str, str]:
    """
    Reset a simulation to its initial state.
    
    Resets the simulation back to round 0, clearing all training history
    while preserving the node datasets.
    
    Args:
        simulation_id: ID of the simulation to reset
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If simulation not found or reset fails
    """
    try:
        logger.info(f"Resetting simulation {simulation_id}")
        
        result = federated_demo.reset_simulation(simulation_id)
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Simulation {simulation_id} not found or cannot be reset"
            )
        
        logger.info(f"Simulation {simulation_id} reset successfully")
        
        return {
            "message": f"Simulation {simulation_id} reset successfully",
            "status": "idle"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting simulation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset simulation: {str(e)}"
        )


@router.get("/convergence/{simulation_id}")
async def get_convergence_history(simulation_id: str) -> Dict[str, Any]:
    """
    Get detailed convergence history for a simulation.
    
    Returns comprehensive convergence data including accuracy progression,
    per-node metrics, and convergence analysis.
    
    Args:
        simulation_id: ID of the simulation
        
    Returns:
        Dictionary containing convergence history and metrics
        
    Raises:
        HTTPException: If simulation not found
    """
    try:
        logger.debug(f"Getting convergence history for simulation {simulation_id}")
        
        # Verify simulation exists
        status = federated_demo.get_simulation_status(simulation_id)
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Simulation {simulation_id} not found"
            )
        
        # Get convergence data
        convergence_data = federated_demo.get_convergence_history(simulation_id)
        
        if not convergence_data:
            return {
                "message": "No convergence data available yet",
                "rounds_completed": 0
            }
        
        return convergence_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting convergence history: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get convergence history: {str(e)}"
        )


# Background task function
async def run_simulation_background(simulation_id: str) -> None:
    """
    Run federated learning simulation in the background.
    
    This function executes the complete simulation orchestration
    asynchronously without blocking the API response.
    
    Args:
        simulation_id: ID of the simulation to run
    """
    try:
        logger.info(f"Starting background simulation {simulation_id}")
        
        # Run the simulation orchestration
        result = federated_demo.run_simulation_orchestration(simulation_id)
        
        if result:
            logger.info(f"Background simulation {simulation_id} completed successfully")
        else:
            logger.error(f"Background simulation {simulation_id} failed")
            
    except Exception as e:
        logger.error(f"Error in background simulation {simulation_id}: {e}")


# Health check endpoint
@router.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint for federated learning service.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "service": "federated_learning",
        "message": "Federated learning service is operational"
    }