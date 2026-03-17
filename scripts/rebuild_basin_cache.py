"""Rebuild basin cache with fixed area_km2_local calculation."""
import sys
from pathlib import Path
import duckdb
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meandre.data.physitel_loader import load_hydrotel


def main():
    parser = argparse.ArgumentParser(description="Rebuild basin cache with fixed areas")
    parser.add_argument(
        "--physitel",
        type=str,
        default="/home/essi/Documents/plateformes-hydrotel/LN24HA/SLSO_LN24HA_2020/physitel",
        help="Path to physitel directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="notebooks/slso/data/basin_cache.db",
        help="Output database path"
    )
    args = parser.parse_args()

    physitel_path = Path(args.physitel)
    output_path = Path(args.output)

    if not physitel_path.exists():
        print(f"ERROR: Physitel directory not found: {physitel_path}")
        sys.exit(1)

    print("=== REBUILDING BASIN CACHE WITH FIXED AREAS ===")
    print(f"Physitel source: {physitel_path}")
    print(f"Output database: {output_path}")

    # Make sure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load basin with fixed physitel_loader
    print("\nLoading basin data with fixed area calculation...")
    basin_data = load_hydrotel(physitel_path)

    # Check the fixed areas
    print("\n=== AREA VALUES AFTER FIX ===")
    print(f"area_km2_local mean: {basin_data['area_km2_local'].mean():.1f} km²")
    print(f"area_km2_local sum: {basin_data['area_km2_local'].sum():.1f} km²")
    print(f"area_km2 (cumulative) at outlet: {basin_data['area_km2'][-1]:.1f} km²")

    # Expected: local areas should sum to approximately the outlet cumulative area
    ratio = basin_data['area_km2_local'].sum() / basin_data['area_km2'][-1]
    print(f"Ratio of sum(local) to outlet: {ratio:.2f} (should be ~1.0)")

    if ratio < 0.9 or ratio > 1.1:
        print("⚠️ WARNING: Local areas don't sum to outlet area! Check calculation.")
    else:
        print("✅ Local areas properly calculated!")

    # Save to DuckDB
    print(f"\nSaving to {output_path}...")
    conn = duckdb.connect(str(output_path))

    # Create basin table with all fields
    conn.execute("DROP TABLE IF EXISTS basin")
    conn.execute("""
        CREATE TABLE basin (
            graph_edges INTEGER[][],
            n_nodes INTEGER,
            n_edges INTEGER,
            node_coords DOUBLE[][],
            area_km2 DOUBLE[],
            area_km2_local DOUBLE[],
            area_km2_physical DOUBLE[],
            territorial_features DOUBLE[][],
            territorial_n_features INTEGER,
            upstream_distance DOUBLE[],
            node_ids INTEGER[],
            topo_order INTEGER[]
        )
    """)

    # Insert data
    conn.execute("""
        INSERT INTO basin VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
    """, [
        basin_data['graph']['edge_index'].tolist(),
        int(basin_data['n_nodes']),
        int(basin_data['graph']['n_edges']),
        basin_data['node_coords'].tolist(),
        basin_data['area_km2'].tolist(),
        basin_data['area_km2_local'].tolist(),
        basin_data['area_km2_physical'].tolist() if 'area_km2_physical' in basin_data else basin_data['area_km2'].tolist(),
        basin_data['territorial'].tolist() if 'territorial' in basin_data else [],
        int(basin_data.get('territorial_n_features', 0)),
        basin_data.get('upstream_distance', []).tolist() if 'upstream_distance' in basin_data else [],
        basin_data.get('node_ids', list(range(basin_data['n_nodes']))),
        basin_data.get('topo_order', list(range(basin_data['n_nodes'])))
    ])

    conn.close()
    print(f"✅ Basin cache rebuilt at {output_path}")

    # Final check
    print("\n=== VERIFYING SAVED DATA ===")
    conn = duckdb.connect(str(output_path))
    result = conn.execute("SELECT area_km2_local FROM basin").fetchone()
    if result:
        import numpy as np
        saved_areas = np.array(result[0])
        print(f"Saved area_km2_local mean: {saved_areas.mean():.1f} km²")
        print(f"Number of nodes saved: {len(saved_areas)}")
    conn.close()


if __name__ == "__main__":
    main()