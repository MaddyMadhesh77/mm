"""Check that Phase 0 runtime dependencies can be imported."""

from importlib import import_module


DEPENDENCIES = ("cv2", "fer", "tensorflow", "pygame", "streamlit", "numpy", "pandas", "sklearn", "matplotlib")


def main() -> int:
    missing = []
    for dependency in DEPENDENCIES:
        try:
            import_module(dependency)
        except Exception as error:  # Import errors can originate inside a dependency.
            missing.append(f"{dependency}: {error}")

    if missing:
        print("Missing or unusable dependencies:")
        print("\n".join(f"- {item}" for item in missing))
        return 1

    print("Phase 0 dependency smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
