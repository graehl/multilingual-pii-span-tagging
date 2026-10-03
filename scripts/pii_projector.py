#!/usr/bin/env python
"""Tagset projector core: loads scripts/pii_tagset.yaml, validates it, and
provides the projection operations from topics/pii-adaptation.md Axis 2.

Operations:
  project(schema, label) -> node          (source label into the hierarchy)
  schema_image(schema)   -> canonical nodes expressible by a schema
  shared_image(a, b)     -> exact canonical-node intersection
  project_cut(schema, label, cut) -> reporting-cut label (source schema or cut)
  cut_image(schema, cut) -> reporting-cut labels expressible by a schema
  ancestors(node)        -> [node..root]  (self first)
  coarse(node)           -> name|format|freetext (inherited)
  compatible(a, b)       -> deeper node if a/b lie on one root chain, else None
  back_project(node, schema) -> source labels whose image is the nearest
                                ancestor-or-self of node (empty if none)

Self-test: python scripts/pii_projector.py
"""

import os
import sys

import yaml

TAGSET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pii_tagset.yaml")


class Tagset:
    def __init__(self, path=TAGSET_PATH):
        spec = yaml.safe_load(open(path))
        self.nodes = spec["nodes"]
        self.cuts = spec.get("cuts", {})
        self.sources = spec["sources"]
        self._cut_maps = {}
        self._validate()

    def _validate(self):
        errors = []
        for name, node in self.nodes.items():
            node = node or {}
            parent = node.get("parent")
            if parent is not None and parent not in self.nodes:
                errors.append(f"node {name}: unknown parent {parent}")
        for name in self.nodes:
            seen = set()
            for anc in self._walk_up(name):
                if anc in seen:
                    errors.append(f"cycle at {anc}")
                    break
                seen.add(anc)
        for schema, mapping in self.sources.items():
            for label, node in mapping.items():
                if node not in self.nodes:
                    errors.append(f"{schema}.{label}: unknown node {node}")
            if len(set(mapping)) != len(mapping):
                errors.append(f"{schema}: duplicate labels")
        for name, node in self.nodes.items():
            if self.coarse(name) not in ("name", "format", "freetext"):
                errors.append(f"node {name}: no coarse stratum on ancestor chain")
        for cut, groups in self.cuts.items():
            node_to_target = {}
            if not isinstance(groups, dict) or not groups:
                errors.append(f"cut {cut}: expected non-empty target mapping")
                continue
            for target, members in groups.items():
                if not isinstance(members, list) or not members:
                    errors.append(f"cut {cut}.{target}: expected non-empty node list")
                    continue
                for member in members:
                    if member not in self.nodes:
                        errors.append(f"cut {cut}.{target}: unknown node {member}")
                    elif member in node_to_target:
                        errors.append(
                            f"cut {cut}: node {member} occurs under both "
                            f"{node_to_target[member]} and {target}"
                        )
                    else:
                        node_to_target[member] = target
            missing = sorted(set(self.nodes) - set(node_to_target))
            if missing:
                errors.append(f"cut {cut}: {len(missing)} unassigned nodes: {', '.join(missing)}")
            self._cut_maps[cut] = node_to_target
        if errors:
            raise ValueError("tagset invalid:\n  " + "\n  ".join(errors))

    def _walk_up(self, node):
        while node is not None:
            yield node
            node = (self.nodes[node] or {}).get("parent")

    def ancestors(self, node):
        return list(self._walk_up(node))

    def coarse(self, node):
        for anc in self._walk_up(node):
            c = (self.nodes[anc] or {}).get("coarse")
            if c:
                return c
        return None

    def project(self, schema, label):
        if schema == "canonical":
            if label not in self.nodes:
                raise ValueError(f"canonical: unknown label {label!r}")
            return label
        if self.is_cut_schema(schema):
            raise ValueError(f"{schema} is a projected ontology cut; fine canonical projection is undefined")
        if schema not in self.sources:
            raise ValueError(f"unknown schema {schema!r}")
        try:
            return self.sources[schema][label]
        except KeyError:
            raise ValueError(f"{schema}: unknown label {label!r}") from None

    def schema_image(self, schema):
        """Canonical nodes a schema can express exactly."""
        if schema == "canonical":
            return set(self.nodes)
        if self.is_cut_schema(schema):
            raise ValueError(f"{schema} is a projected ontology cut; fine canonical image is undefined")
        if schema not in self.sources:
            raise ValueError(f"unknown schema {schema!r}")
        return set(self.sources[schema].values())

    def shared_image(self, schema_a, schema_b):
        """Exact canonical-node intersection of two source schemas."""
        return self.schema_image(schema_a) & self.schema_image(schema_b)

    def cut_names(self):
        return sorted(self._cut_maps)

    def is_cut_schema(self, schema):
        return schema in self._cut_maps

    def cut_targets(self, cut):
        if not self.is_cut_schema(cut):
            raise ValueError(f"unknown ontology cut {cut!r}")
        return set(self.cuts[cut])

    def project_canonical_cut(self, node, cut):
        if cut not in self._cut_maps:
            raise ValueError(f"unknown ontology cut {cut!r}")
        if node not in self.nodes:
            raise ValueError(f"unknown canonical label {node!r}")
        return self._cut_maps[cut][node]

    def project_cut(self, schema, label, cut):
        if self.is_cut_schema(schema):
            if label not in self.cuts[schema]:
                raise ValueError(f"{schema}: unknown label {label!r}")
            if schema == cut:
                return label
            targets = {self.project_canonical_cut(node, cut) for node in self.cuts[schema][label]}
            if len(targets) != 1:
                raise ValueError(
                    f"cannot project {schema}.{label} to {cut}: "
                    f"the source cut would require refinement into {sorted(targets)}"
                )
            return targets.pop()
        return self.project_canonical_cut(self.project(schema, label), cut)

    def cut_image(self, schema, cut):
        """Reporting-cut labels expressible by a source schema."""
        if self.is_cut_schema(schema):
            return {self.project_cut(schema, label, cut) for label in self.cut_targets(schema)}
        return {self.project_canonical_cut(node, cut) for node in self.schema_image(schema)}

    def compatible(self, a, b):
        """Deeper of a/b when one is an ancestor of the other, else None."""
        if b in self.ancestors(a):
            return a
        if a in self.ancestors(b):
            return b
        return None

    def deepest_common_ancestor(self, a, b):
        anc_a = self.ancestors(a)
        for node in self.ancestors(b):
            if node in anc_a:
                return node
        return None

    def back_project(self, node, schema):
        """Source labels whose image is the nearest ancestor-or-self of node."""
        image = {}
        for label, n in self.sources[schema].items():
            image.setdefault(n, []).append(label)
        for anc in self._walk_up(node):
            if anc in image:
                return sorted(image[anc])
        return []


def main():
    ts = Tagset()
    n_nodes = len(ts.nodes)
    n_labels = sum(len(m) for m in ts.sources.values())
    print(f"tagset OK: {n_nodes} nodes, {len(ts.sources)} schemas, {n_labels} labels")

    checks = [
        (ts.project("openmed_54", "FIRSTNAME"), "given_name"),
        (ts.project("openai_8", "private_person"), "person_name"),
        (ts.project("canonical", "given_name"), "given_name"),
        (ts.shared_image("openmed_nemotron_55", "nemotron_pii"), ts.schema_image("nemotron_pii")),
        (
            ts.cut_names(),
            ["redaction_20_presidio_common_v1", "redaction_20_v1", "redaction_9_v1"],
        ),
        (ts.project_cut("openmed_nemotron_55", "last_name", "redaction_20_v1"), "family_name"),
        (ts.project_cut("canonical", "medical_record_number", "redaction_9_v1"), "unique_identifier"),
        (ts.compatible("given_name", "person_name"), "given_name"),
        (ts.compatible("given_name", "email"), None),
        (ts.deepest_common_ancestor("iban", "card_cvv"), "financial"),
        (ts.back_project("given_name", "openai_8"), ["private_person"]),
        (ts.back_project("iban", "spy_7"), []),
        (ts.back_project("card_number", "openai_8"), ["account_number"]),
        (ts.coarse("building_number"), "format"),
        (ts.coarse("given_name"), "name"),
        (ts.coarse("street_address"), "freetext"),
    ]
    for got, want in checks:
        assert got == want, f"got {got!r}, want {want!r}"
    print(f"{len(checks)} self-tests passed")
    for schema in ts.sources:
        unmapped = [n for n in ts.nodes if not ts.back_project(n, schema)]
        if unmapped:
            print(
                f"{schema}: {len(unmapped)} nodes with no back-projection "
                f"(schema cannot express them): {', '.join(unmapped[:6])}"
                + ("..." if len(unmapped) > 6 else "")
            )


if __name__ == "__main__":
    sys.exit(main())
