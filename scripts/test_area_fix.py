"""Test that the area fix in physitel_loader.py works correctly."""
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from meandre.data.physitel_loader import load_hydrotel


def main():
    physitel_path = Path("/home/essi/Documents/plateformes-hydrotel/LN24HA/SLSO_LN24HA_2020")

    if not physitel_path.exists():
        print(f"ERROR: Physitel directory not found: {physitel_path}")
        sys.exit(1)

    print("=== TESTING AREA FIX IN PHYSITEL_LOADER ===")
    print(f"Loading from: {physitel_path}")

    # Load basin with fixed physitel_loader
    print("\nLoading basin data...")
    basin_data = load_hydrotel(physitel_path)

    # Extract the data we need
    area_km2_local = basin_data['territorial'].area_km2_local
    area_km2_physical = basin_data['territorial'].area_km2_physical

    print("\n=== AREA VALUES AFTER FIX ===")
    print(f"area_km2_local:")
    print(f"  Mean: {area_km2_local.mean():.1f} km²")
    print(f"  Min: {area_km2_local.min():.3f} km²")
    print(f"  Max: {area_km2_local.max():.1f} km²")
    print(f"  Sum: {area_km2_local.sum():.1f} km²")
    print(f"  First 10 values: {area_km2_local[:10].cpu().numpy()}")

    print(f"\narea_km2_physical (cumulative):")
    print(f"  Mean: {area_km2_physical.mean():.1f} km²")
    print(f"  Outlet (last): {area_km2_physical[-1]:.1f} km²")

    # Check if fix worked
    ratio = area_km2_local.sum() / area_km2_physical[-1]
    print(f"\n=== VALIDATION ===")
    print(f"Sum of local areas: {area_km2_local.sum():.1f} km²")
    print(f"Outlet cumulative area: {area_km2_physical[-1]:.1f} km²")
    print(f"Ratio (should be ~1.0): {ratio:.3f}")

    if area_km2_local.mean() < 50:
        print("\n❌ PROBLEM: Local areas still too small! Mean < 50 km²")
        print("   The fix may not have been applied correctly.")
    elif ratio < 0.9 or ratio > 1.1:
        print("\n⚠️ WARNING: Local areas don't sum to outlet area properly")
        print("   There may be an issue with the incremental calculation")
    else:
        print("\n✅ SUCCESS: Area fix appears to be working!")
        print(f"   Mean local area increased from ~12 km² to {area_km2_local.mean():.1f} km²")
        print("   Local areas now sum to approximately the outlet area")

    # Test scale for discharge calculation
    print("\n=== DISCHARGE SCALE TEST ===")
    # For 1mm/day over mean area
    mm_per_day = 1.0
    mean_area = area_km2_local.mean().item()

    # Convert to m³/s (matching routing formula)
    q_m3s = mm_per_day * 1e-3 * mean_area * 1e6 / 86400.0

    print(f"For 1 mm/day over mean area ({mean_area:.1f} km²):")
    print(f"  Expected discharge: {q_m3s:.3f} m³/s")
    print(f"  This should be order of magnitude ~1-10 m³/s for typical subcatchments")

    # Check outlet scale
    outlet_q = mm_per_day * 1e-3 * area_km2_physical[-1].item() * 1e6 / 86400.0
    print(f"\nFor 1 mm/day over entire basin ({area_km2_physical[-1]:.1f} km²):")
    print(f"  Expected outlet discharge: {outlet_q:.1f} m³/s")
    print(f"  This should be order of magnitude ~100-1000 m³/s for large basins")


if __name__ == "__main__":
    main()