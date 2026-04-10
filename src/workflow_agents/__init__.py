"""
Workflow Agents - A reusable library of AI-powered agents for agentic workflows.

This package provides seven agent classes that can be composed into
sophisticated agentic workflows for project management and beyond.

Usage:
    from src.workflow_agents import DirectPromptAgent, RoutingAgent, EvaluationAgent
"""

from .direct_prompt import DirectPromptAgent
from .augmented_prompt import AugmentedPromptAgent
from .knowledge_augmented_prompt import KnowledgeAugmentedPromptAgent
from .rag_knowledge_prompt import RAGKnowledgePromptAgent
from .evaluation import EvaluationAgent
from .routing import RoutingAgent
from .action_planning import ActionPlanningAgent

__all__ = [
    "DirectPromptAgent",
    "AugmentedPromptAgent",
    "KnowledgeAugmentedPromptAgent",
    "RAGKnowledgePromptAgent",
    "EvaluationAgent",
    "RoutingAgent",
    "ActionPlanningAgent",
]
