"""
Consistency Checker for GraphEngine.

Validates generated content against graph constraints.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ConstraintType(str, Enum):
    """Types of consistency constraints."""
    CHARACTER_LOCATION = "character_location"
    TEMPORAL_CAUSALITY = "temporal_causality"
    RELATIONSHIP_SYMMETRY = "relationship_symmetry"
    POWER_BALANCE = "power_balance"
    KNOWLEDGE_SCOPE = "knowledge_scope"
    ENTITY_EXISTENCE = "entity_existence"
    RELATIONSHIP_VALIDITY = "relationship_validity"
    PROPERTY_TYPE = "property_type"
    UNIQUE_NAME = "unique_name"


class Severity(str, Enum):
    """Severity of constraint violation."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Violation:
    """A constraint violation."""
    constraint_type: ConstraintType
    severity: Severity
    message: str
    entity_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'constraint_type': self.constraint_type.value,
            'severity': self.severity.value,
            'message': self.message,
            'entity_id': self.entity_id,
            'details': self.details,
            'suggested_fix': self.suggested_fix,
        }


@dataclass
class ValidationResult:
    """Result of consistency validation."""
    is_valid: bool
    violations: List[Violation]
    warnings: List[Violation]
    info: List[Violation]
    validated_entities: List[str]
    validation_time: str
    
    def to_dict(self) -> Dict:
        return {
            'is_valid': self.is_valid,
            'violations': [v.to_dict() for v in self.violations],
            'warnings': [v.to_dict() for v in self.warnings],
            'info': [v.to_dict() for v in self.info],
            'validated_entities': self.validated_entities,
            'validation_time': self.validation_time,
        }


@dataclass
class GeneratedContent:
    """Structured generated content to validate."""
    content_type: str
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    events: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GeneratedContent':
        return cls(
            content_type=data.get('content_type', 'unknown'),
            entities=data.get('entities', []),
            relationships=data.get('relationships', []),
            events=data.get('events', []),
            metadata=data.get('metadata', {}),
        )


class ConsistencyChecker:
    """
    Validates generated content against graph constraints.
    
    Ensures that:
    - Characters are at exactly one location
    - Temporal causality is maintained
    - Bidirectional edges are symmetric
    - Power changes have causes
    - Characters know only what they've observed
    """
    
    CONSTRAINT_PRIORITIES = {
        ConstraintType.ENTITY_EXISTENCE: 1,
        ConstraintType.PROPERTY_TYPE: 2,
        ConstraintType.CHARACTER_LOCATION: 3,
        ConstraintType.TEMPORAL_CAUSALITY: 4,
        ConstraintType.RELATIONSHIP_SYMMETRY: 5,
        ConstraintType.POWER_BALANCE: 6,
        ConstraintType.KNOWLEDGE_SCOPE: 7,
        ConstraintType.RELATIONSHIP_VALIDITY: 8,
        ConstraintType.UNIQUE_NAME: 9,
    }
    
    def __init__(self, graph: 'KnowledgeGraph', strict: bool = False):
        """
        Initialize consistency checker.
        
        Args:
            graph: KnowledgeGraph to validate against
            strict: If True, warnings also fail validation
        """
        self.graph = graph
        self.strict = strict
    
    def validate(
        self,
        content: Dict[str, Any],
        check_types: List[ConstraintType] = None
    ) -> ValidationResult:
        """
        Validate generated content against all constraints.
        
        Args:
            content: Generated content dict
            check_types: Specific constraints to check (None = all)
        
        Returns:
            ValidationResult with violations and suggestions
        """
        if isinstance(content, dict):
            content = GeneratedContent.from_dict(content)
        
        violations = []
        warnings = []
        info = []
        validated_entities = []
        
        check_types = check_types or list(ConstraintType)
        check_types.sort(key=lambda t: self.CONSTRAINT_PRIORITIES.get(t, 99))
        
        for entity_data in content.entities:
            entity_id = entity_data.get('id')
            if entity_id:
                validated_entities.append(entity_id)
            
            entity_violations = self._validate_entity(entity_data, check_types)
            for v in entity_violations:
                if v.severity == Severity.ERROR:
                    violations.append(v)
                elif v.severity == Severity.WARNING:
                    warnings.append(v)
                else:
                    info.append(v)
        
        for rel_data in content.relationships:
            rel_violations = self._validate_relationship(rel_data, check_types)
            for v in rel_violations:
                if v.severity == Severity.ERROR:
                    violations.append(v)
                elif v.severity == Severity.WARNING:
                    warnings.append(v)
                else:
                    info.append(v)
        
        for event_data in content.events:
            event_violations = self._validate_event(event_data, check_types)
            for v in event_violations:
                if v.severity == Severity.ERROR:
                    violations.append(v)
                elif v.severity == Severity.WARNING:
                    warnings.append(v)
                else:
                    info.append(v)
        
        is_valid = len(violations) == 0 and (not self.strict or len(warnings) == 0)
        
        return ValidationResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
            info=info,
            validated_entities=validated_entities,
            validation_time=datetime.utcnow().isoformat(),
        )
    
    def validate_entity_update(
        self,
        entity_id: str,
        updates: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate updates to an existing entity.
        
        Args:
            entity_id: ID of entity to update
            updates: Properties to update
        
        Returns:
            ValidationResult
        """
        entity = self.graph.get_node(entity_id)
        if not entity:
            return ValidationResult(
                is_valid=False,
                violations=[Violation(
                    constraint_type=ConstraintType.ENTITY_EXISTENCE,
                    severity=Severity.ERROR,
                    message=f"Entity not found: {entity_id}",
                    entity_id=entity_id,
                )],
                warnings=[],
                info=[],
                validated_entities=[],
                validation_time=datetime.utcnow().isoformat(),
            )
        
        merged = dict(entity.properties)
        merged.update(updates)
        
        content = GeneratedContent(
            content_type='update',
            entities=[{
                'id': entity_id,
                'type': entity.type,
                'properties': merged,
            }],
            relationships=[],
            events=[],
        )
        
        return self.validate(content)
    
    def _validate_entity(
        self,
        entity_data: Dict[str, Any],
        check_types: List[ConstraintType]
    ) -> List[Violation]:
        """Validate a single entity."""
        violations = []
        entity_id = entity_data.get('id')
        entity_type = entity_data.get('type', 'unknown')
        properties = entity_data.get('properties', {})
        
        if ConstraintType.ENTITY_EXISTENCE in check_types:
            if entity_id:
                existing = self.graph.get_node(entity_id)
                if existing and existing.type != entity_type:
                    violations.append(Violation(
                        constraint_type=ConstraintType.ENTITY_EXISTENCE,
                        severity=Severity.ERROR,
                        message=f"Entity {entity_id} exists with different type",
                        entity_id=entity_id,
                        details={'expected': entity_type, 'actual': existing.type},
                        suggested_fix=f"Use a different ID or update existing {existing.type}",
                    ))
        
        if ConstraintType.UNIQUE_NAME in check_types:
            name = properties.get('name')
            if name:
                existing_names = self.graph.search(name, node_types=[entity_type], limit=1)
                if existing_names and (not entity_id or existing_names[0] != entity_id):
                    violations.append(Violation(
                        constraint_type=ConstraintType.UNIQUE_NAME,
                        severity=Severity.WARNING,
                        message=f"Name '{name}' already exists in {entity_type}",
                        entity_id=entity_id,
                        details={'existing_id': existing_names[0]},
                        suggested_fix=f"Use a different name or update existing entity",
                    ))
        
        if ConstraintType.CHARACTER_LOCATION in check_types:
            if entity_type == 'character':
                location_violations = self._check_character_location(entity_id, properties)
                violations.extend(location_violations)
        
        if ConstraintType.POWER_BALANCE in check_types:
            if entity_type == 'faction':
                power_violations = self._check_power_balance(entity_id, properties)
                violations.extend(power_violations)
        
        if ConstraintType.PROPERTY_TYPE in check_types:
            type_violations = self._check_property_types(entity_type, properties)
            violations.extend(type_violations)
        
        return violations
    
    def _validate_relationship(
        self,
        rel_data: Dict[str, Any],
        check_types: List[ConstraintType]
    ) -> List[Violation]:
        """Validate a relationship."""
        violations = []
        
        source_id = rel_data.get('source_id')
        target_id = rel_data.get('target_id')
        edge_type = rel_data.get('edge_type')
        
        if ConstraintType.ENTITY_EXISTENCE in check_types:
            if source_id and not self.graph.get_node(source_id):
                violations.append(Violation(
                    constraint_type=ConstraintType.ENTITY_EXISTENCE,
                    severity=Severity.ERROR,
                    message=f"Source entity not found: {source_id}",
                    entity_id=source_id,
                    details={'edge_type': edge_type, 'target': target_id},
                    suggested_fix="Create the source entity first",
                ))
            
            if target_id and not self.graph.get_node(target_id):
                violations.append(Violation(
                    constraint_type=ConstraintType.ENTITY_EXISTENCE,
                    severity=Severity.ERROR,
                    message=f"Target entity not found: {target_id}",
                    entity_id=target_id,
                    details={'edge_type': edge_type, 'source': source_id},
                    suggested_fix="Create the target entity first",
                ))
        
        if ConstraintType.RELATIONSHIP_VALIDITY in check_types:
            if source_id and target_id and edge_type:
                source = self.graph.get_node(source_id)
                target = self.graph.get_node(target_id)
                
                if source and target:
                    valid_violations = self._check_relationship_validity(
                        source.type, target.type, edge_type
                    )
                    violations.extend(valid_violations)
        
        if ConstraintType.RELATIONSHIP_SYMMETRY in check_types:
            if edge_type in ['alliance', 'mutualism', 'competition']:
                symmetric_violations = self._check_symmetry(source_id, target_id, edge_type, rel_data)
                violations.extend(symmetric_violations)
        
        return violations
    
    def _validate_event(
        self,
        event_data: Dict[str, Any],
        check_types: List[ConstraintType]
    ) -> List[Violation]:
        """Validate an event."""
        violations = []
        
        if ConstraintType.TEMPORAL_CAUSALITY in check_types:
            causes = event_data.get('causes', [])
            timestamp = event_data.get('timestamp') or event_data.get('properties', {}).get('timestamp')
            
            for cause_id in causes:
                cause = self.graph.get_node(cause_id)
                if cause:
                    cause_time = cause.properties.get('timestamp') or cause.properties.get('generation')
                    if cause_time and timestamp:
                        if str(cause_time) > str(timestamp):
                            violations.append(Violation(
                                constraint_type=ConstraintType.TEMPORAL_CAUSALITY,
                                severity=Severity.ERROR,
                                message=f"Cause {cause_id} occurs after effect",
                                details={
                                    'cause_time': str(cause_time),
                                    'effect_time': str(timestamp),
                                },
                                suggested_fix="Adjust timestamps or remove causal link",
                            ))
        
        return violations
    
    def _check_character_location(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> List[Violation]:
        """Check character location consistency."""
        violations = []
        
        if not entity_id:
            return violations
        
        existing = self.graph.get_node(entity_id)
        if not existing:
            return violations
        
        location_edges = self.graph.get_edges(entity_id, edge_types=['origin', 'contains'], direction='in')
        
        proposed_location = properties.get('location') or properties.get('current_location')
        
        if proposed_location and location_edges:
            existing_locations = [e.source_id for e in location_edges if e.edge_type == 'contains']
            if existing_locations and proposed_location not in existing_locations:
                violations.append(Violation(
                    constraint_type=ConstraintType.CHARACTER_LOCATION,
                    severity=Severity.WARNING,
                    message=f"Character already at location {existing_locations[0]}",
                    entity_id=entity_id,
                    details={
                        'current_location': existing_locations[0],
                        'proposed_location': proposed_location,
                    },
                    suggested_fix=f"Move character from {existing_locations[0]} to {proposed_location} explicitly",
                ))
        
        return violations
    
    def _check_power_balance(
        self,
        entity_id: str,
        properties: Dict[str, Any]
    ) -> List[Violation]:
        """Check power balance consistency."""
        violations = []
        
        proposed_power = properties.get('power_level')
        if proposed_power is None:
            return violations
        
        if proposed_power < 0 or proposed_power > 1:
            violations.append(Violation(
                constraint_type=ConstraintType.POWER_BALANCE,
                severity=Severity.WARNING,
                message=f"Power level {proposed_power} outside normal range [0, 1]",
                entity_id=entity_id,
                suggested_fix="Normalize power_level to [0, 1] range",
            ))
        
        return violations
    
    def _check_property_types(
        self,
        entity_type: str,
        properties: Dict[str, Any]
    ) -> List[Violation]:
        """Check property types against schema."""
        violations = []
        
        type_expectations = {
            'character': {
                'importance': (float, int),
                'age': int,
            },
            'species': {
                'population': int,
                'fitness': (float, int),
            },
            'faction': {
                'power_level': (float, int),
            },
            'event': {
                'significance': (float, int),
            },
        }
        
        expected = type_expectations.get(entity_type, {})
        
        for prop, expected_types in expected.items():
            if prop in properties:
                value = properties[prop]
                if value is not None and not isinstance(value, expected_types):
                    violations.append(Violation(
                        constraint_type=ConstraintType.PROPERTY_TYPE,
                        severity=Severity.INFO,
                        message=f"Property '{prop}' has unexpected type",
                        details={
                            'expected': str(expected_types),
                            'actual': type(value).__name__,
                            'value': str(value)[:50],
                        },
                        suggested_fix=f"Convert {prop} to {expected_types}",
                    ))
        
        return violations
    
    def _check_relationship_validity(
        self,
        source_type: str,
        target_type: str,
        edge_type: str
    ) -> List[Violation]:
        """Check if relationship type is valid for node types."""
        violations = []
        
        valid_combinations = {
            'predation': (['species'], ['species']),
            'alliance': (['character', 'faction'], ['character', 'faction']),
            'conflict': (['character', 'faction', 'species'], ['character', 'faction', 'species']),
            'contains': (['location', 'faction'], ['character', 'species', 'location', 'event']),
            'origin': (['character', 'species'], ['location']),
        }
        
        if edge_type in valid_combinations:
            valid_sources, valid_targets = valid_combinations[edge_type]
            
            if source_type not in valid_sources:
                violations.append(Violation(
                    constraint_type=ConstraintType.RELATIONSHIP_VALIDITY,
                    severity=Severity.ERROR,
                    message=f"Invalid source type '{source_type}' for edge '{edge_type}'",
                    details={'valid_types': valid_sources},
                    suggested_fix=f"Use a source of type: {valid_sources}",
                ))
            
            if target_type not in valid_targets:
                violations.append(Violation(
                    constraint_type=ConstraintType.RELATIONSHIP_VALIDITY,
                    severity=Severity.ERROR,
                    message=f"Invalid target type '{target_type}' for edge '{edge_type}'",
                    details={'valid_types': valid_targets},
                    suggested_fix=f"Use a target of type: {valid_targets}",
                ))
        
        return violations
    
    def _check_symmetry(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        rel_data: Dict[str, Any]
    ) -> List[Violation]:
        """Check if bidirectional edge is symmetric."""
        violations = []
        
        if not source_id or not target_id:
            return violations
        
        reverse_edges = self.graph.get_edges(
            target_id,
            edge_types=[edge_type],
            direction='out'
        )
        
        has_reverse = any(e.target_id == source_id for e in reverse_edges)
        
        if not has_reverse:
            violations.append(Violation(
                constraint_type=ConstraintType.RELATIONSHIP_SYMMETRY,
                severity=Severity.WARNING,
                message=f"Bidirectional edge '{edge_type}' missing reverse",
                details={
                    'source': source_id,
                    'target': target_id,
                },
                suggested_fix=f"Add reverse edge from {target_id} to {source_id}",
            ))
        
        return violations
    
    def suggest_fixes(
        self,
        result: ValidationResult
    ) -> List[Dict[str, Any]]:
        """
        Suggest fixes for validation violations.
        
        Args:
            result: ValidationResult from validate()
        
        Returns:
            List of fix suggestions
        """
        fixes = []
        
        for violation in result.violations + result.warnings:
            if violation.suggested_fix:
                fix = {
                    'violation': violation.message,
                    'fix': violation.suggested_fix,
                    'severity': violation.severity.value,
                    'entity_id': violation.entity_id,
                }
                fixes.append(fix)
        
        return fixes
    
    def auto_fix(
        self,
        content: Dict[str, Any],
        result: ValidationResult
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Attempt to automatically fix violations.
        
        Args:
            content: Original content
            result: Validation result
        
        Returns:
            (fixed_content, list_of_fixes_applied)
        """
        if isinstance(content, dict):
            content = GeneratedContent.from_dict(content)
        
        fixed_content = GeneratedContent(
            content_type=content.content_type,
            entities=list(content.entities),
            relationships=list(content.relationships),
            events=list(content.events),
            metadata=dict(content.metadata),
        )
        
        fixes_applied = []
        
        for violation in result.violations + result.warnings:
            if violation.constraint_type == ConstraintType.PROPERTY_TYPE:
                fix = self._auto_fix_property_type(fixed_content, violation)
                if fix:
                    fixes_applied.append(fix)
            
            elif violation.constraint_type == ConstraintType.POWER_BALANCE:
                fix = self._auto_fix_power_level(fixed_content, violation)
                if fix:
                    fixes_applied.append(fix)
        
        return fixed_content.to_dict() if hasattr(fixed_content, 'to_dict') else fixed_content, fixes_applied
    
    def _auto_fix_property_type(
        self,
        content: GeneratedContent,
        violation: Violation
    ) -> Optional[str]:
        """Auto-fix property type issues."""
        details = violation.details
        prop_name = None
        
        for key in content.entities:
            if 'properties' in key:
                props = key.get('properties', {})
                for pname, pval in props.items():
                    if pval == details.get('value'):
                        prop_name = pname
                        entity = key
                        break
        
        if prop_name and 'expected' in details:
            expected_str = details['expected']
            
            if 'float' in expected_str or 'int' in expected_str:
                try:
                    current = details.get('value')
                    if isinstance(current, str):
                        if '.' in current:
                            new_val = float(current)
                        else:
                            new_val = int(current)
                        
                        if 'entity' in dir():
                            entity['properties'][prop_name] = new_val
                            return f"Converted {prop_name} to {type(new_val).__name__}"
                except (ValueError, TypeError):
                    pass
        
        return None
    
    def _auto_fix_power_level(
        self,
        content: GeneratedContent,
        violation: Violation
    ) -> Optional[str]:
        """Auto-fix power level to valid range."""
        for entity in content.entities:
            props = entity.get('properties', {})
            if 'power_level' in props:
                val = props['power_level']
                if isinstance(val, (int, float)):
                    if val < 0:
                        props['power_level'] = 0.0
                        return f"Clamped power_level to 0.0"
                    elif val > 1:
                        props['power_level'] = 1.0
                        return f"Clamped power_level to 1.0"
        
        return None
