"""Domain Invariant Contracts and Plugin Architecture for FRAME-AR (Feature 16 & 17).

Provides:
- SymbolicDomainConstraint: Specification of a domain-specific mathematical constraint.
- DomainInvariantPlugin: Abstract base class for domain invariant plugins.
- Built-in domain plugins: PhysicsConservationPlugin, BiologyMonotonicityPlugin, BoundaryConditionsPlugin.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field


class SymbolicDomainConstraint(BaseModel):
    """Specification of a symbolic domain invariant constraint evaluated by math engines."""

    constraint_id: str = Field(description="Unique identifier for the constraint")
    name: str = Field(description="Human-readable constraint name")
    domain: str = Field(description="Domain category (e.g., 'physics', 'biology', 'boundary')")
    symbolic_expression: Optional[str] = Field(
        default=None, description="Canonical mathematical expression of the constraint"
    )
    lower_bound: Optional[float] = Field(
        default=None, description="Lower acceptable bound for evaluated candidate expression"
    )
    upper_bound: Optional[float] = Field(
        default=None, description="Upper acceptable bound for evaluated candidate expression"
    )
    interval: Optional[Tuple[float, float]] = Field(
        default=None, description="Evaluation domain interval for primary variable"
    )
    variable_bounds: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None, description="Explicit variable domain bounds mapping variable name to (min, max)"
    )
    constraint_type: str = Field(
        default="positivity",
        description="Constraint type classification: 'positivity', 'conservation', 'monotonicity', 'boundary'",
    )
    description: Optional[str] = Field(
        default=None, description="Detailed scientific description of the invariant"
    )

    model_config = ConfigDict(extra="allow")


class DomainInvariantPlugin(ABC):
    """Abstract base class defining the contract for domain invariant plugins."""

    @abstractmethod
    def get_domain_name(self) -> str:
        """Returns the canonical domain name string (e.g. 'physics', 'biology', 'boundary')."""
        pass

    @abstractmethod
    def get_constraints(self) -> List[SymbolicDomainConstraint]:
        """Returns the list of symbolic domain constraints registered by this plugin."""
        pass


class PhysicsConservationPlugin(DomainInvariantPlugin):
    """Default physical conservation and positivity plugin.
    
    Enforces non-negativity of physical energy potentials, mass, and conservation invariants.
    """

    def get_domain_name(self) -> str:
        return "physics"

    def get_constraints(self) -> List[SymbolicDomainConstraint]:
        return [
            SymbolicDomainConstraint(
                constraint_id="phys_energy_cons",
                name="Energy Conservation",
                domain="physics",
                symbolic_expression="x**2 + y**2",
                lower_bound=0.0,
                variable_bounds={"x": (0.0, 5.0), "y": (0.0, 5.0)},
                constraint_type="positivity",
                description="Physical energy conservation: kinetic and potential energies must be non-negative.",
            )
        ]


class BiologyMonotonicityPlugin(DomainInvariantPlugin):
    """Default biological dose-response monotonicity plugin.
    
    Enforces non-negative response gradients d/dx f(x) >= 0 in valid biological concentration regimes.
    """

    def get_domain_name(self) -> str:
        return "biology"

    def get_constraints(self) -> List[SymbolicDomainConstraint]:
        return [
            SymbolicDomainConstraint(
                constraint_id="bio_monotonic_dose",
                name="Monotonic Dose Response",
                domain="biology",
                symbolic_expression="x / (1 + x)",
                lower_bound=0.0,
                variable_bounds={"x": (0.0, 5.0)},
                constraint_type="monotonicity",
                description="Biological dose-response monotonicity: biological response increases monotonically with dose.",
            )
        ]


class BoundaryConditionsPlugin(DomainInvariantPlugin):
    """Default boundary regularity and asymptotic boundedness plugin.
    
    Enforces regular non-singular response functions bounded within valid physical/activation ranges.
    """

    def get_domain_name(self) -> str:
        return "boundary"

    def get_constraints(self) -> List[SymbolicDomainConstraint]:
        return [
            SymbolicDomainConstraint(
                constraint_id="bound_asymptotic_regularity",
                name="Boundary Regularity and Singularity Safety",
                domain="boundary",
                symbolic_expression="1 / (1 + exp(-x))",
                lower_bound=0.0,
                upper_bound=1.0,
                variable_bounds={"x": (-10.0, 10.0)},
                constraint_type="boundary",
                description="Asymptotic regularity: activation function is regular (pole-free) and bounded in [0, 1].",
            )
        ]


# Aliases for backward compatibility / alternative naming
DefaultPhysicsPlugin = PhysicsConservationPlugin
DefaultBiologyPlugin = BiologyMonotonicityPlugin
DefaultBoundaryPlugin = BoundaryConditionsPlugin

__all__ = [
    "SymbolicDomainConstraint",
    "DomainInvariantPlugin",
    "PhysicsConservationPlugin",
    "BiologyMonotonicityPlugin",
    "BoundaryConditionsPlugin",
    "DefaultPhysicsPlugin",
    "DefaultBiologyPlugin",
    "DefaultBoundaryPlugin",
]
