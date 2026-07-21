from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable, Any
import time


@dataclass
class WorkflowNode:
    id: str
    description: str = ""
    retry_count: int = 0
    retry_delay: float = 0.0
    timeout: Optional[float] = None


class WorkflowDAG:
    def __init__(self):
        self._nodes: Dict[str, WorkflowNode] = {}
        self._dependencies: Dict[str, Set[str]] = {}
        self._dependents: Dict[str, Set[str]] = {}

    def add_node(self, node: WorkflowNode) -> WorkflowDAG:
        if node.id in self._nodes:
            raise ValueError(f"Node '{node.id}' already exists")
        self._nodes[node.id] = node
        self._dependencies.setdefault(node.id, set())
        self._dependents.setdefault(node.id, set())
        return self

    def add_dependency(self, from_id: str, to_id: str) -> WorkflowDAG:
        if from_id not in self._nodes:
            raise ValueError(f"Node '{from_id}' not found")
        if to_id not in self._nodes:
            raise ValueError(f"Node '{to_id}' not found")
        self._dependencies[to_id].add(from_id)
        self._dependents[from_id].add(to_id)
        return self

    def get_node(self, node_id: str) -> Optional[WorkflowNode]:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> Dict[str, WorkflowNode]:
        return dict(self._nodes)

    def get_dependencies(self, node_id: str) -> List[str]:
        return sorted(self._dependencies.get(node_id, set()))

    def get_dependents(self, node_id: str) -> List[str]:
        return sorted(self._dependents.get(node_id, set()))

    def has_cycle(self) -> bool:
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _dfs(nid: str) -> bool:
            visited.add(nid)
            rec_stack.add(nid)
            for dep in self._dependents.get(nid, set()):
                if dep not in visited:
                    if _dfs(dep):
                        return True
                elif dep in rec_stack:
                    return True
            rec_stack.discard(nid)
            return False

        for nid in self._nodes:
            if nid not in visited:
                if _dfs(nid):
                    return True
        return False

    def topological_sort(self) -> List[str]:
        if self.has_cycle():
            raise ValueError("DAG contains a cycle")

        visited: Set[str] = set()
        result: List[str] = []

        def _dfs(nid: str):
            visited.add(nid)
            for dep in self._dependencies.get(nid, set()):
                if dep not in visited:
                    _dfs(dep)
            result.append(nid)

        for nid in self._nodes:
            if nid not in visited:
                _dfs(nid)

        return result

    def get_ready_nodes(self, completed: Set[str], failed: Set[str]) -> List[str]:
        ready = []
        for nid in self._nodes:
            if nid in completed or nid in failed:
                continue
            deps = self._dependencies.get(nid, set())
            if deps and deps.issubset(completed):
                ready.append(nid)
            elif not deps:
                ready.append(nid)
        return sorted(ready)

    def get_levels(self) -> List[List[str]]:
        if self.has_cycle():
            raise ValueError("DAG contains a cycle")
        sorted_nodes = self.topological_sort()
        in_degree: Dict[str, int] = {nid: len(self._dependencies.get(nid, set())) for nid in sorted_nodes}
        levels: List[List[str]] = []
        while in_degree:
            current_level = [nid for nid, deg in in_degree.items() if deg == 0]
            if not current_level:
                break
            levels.append(sorted(current_level))
            for nid in current_level:
                del in_degree[nid]
                for dep in self._dependents.get(nid, set()):
                    if dep in in_degree:
                        in_degree[dep] -= 1
        return levels

    def subgraph(self, node_ids: Set[str]) -> WorkflowDAG:
        sub = WorkflowDAG()
        for nid in node_ids:
            if nid in self._nodes:
                sub.add_node(self._nodes[nid])
        for nid in node_ids:
            deps = self._dependencies.get(nid, set()) & node_ids
            for dep in deps:
                sub.add_dependency(dep, nid)
        return sub
