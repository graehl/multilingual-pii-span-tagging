from scripts.pii_seed_tree import SeedTree


def test_seed_tree_is_stable_and_namespaces_are_independent():
    first = SeedTree(23)
    second = SeedTree(23)

    assert first.fork("materializer", "ko", "doc-1", 0) == second.fork("materializer", "ko", "doc-1", 0)
    assert first.fork("materializer", "ko", "doc-1", 0) != first.fork("materializer", "ko", "doc-1", 1)
    assert first.fork("training", "sampling") != first.fork("training", "model")
    assert SeedTree(24).fork("training", "sampling") != first.fork("training", "sampling")


def test_seed_tree_description_records_named_forks():
    tree = SeedTree(41)

    assert tree.describe(model=("training", "model"), sampling=("training", "sampling")) == {
        "scheme": "pii-seed-tree-v1",
        "root_seed": 41,
        "forks": {
            "model": tree.fork("training", "model"),
            "sampling": tree.fork("training", "sampling"),
        },
    }
