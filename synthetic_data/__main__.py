from .pipeline import build_synthetic_dataset


def main():
    bundle = build_synthetic_dataset()
    summary = bundle["summary"]
    print(
        "Synthetic dataset ready:",
        f"filtered={summary['filtered_graph']['nodes']} nodes,",
        f"adaptive={summary['adaptive_graph']['nodes']} nodes,",
        f"demand={summary['demand_nodes']['count']}",
    )


if __name__ == "__main__":
    main()
