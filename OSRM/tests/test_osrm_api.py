#!/usr/bin/env python3
"""
OSRM API Test Suite
Tests Route, Table, and Nearest APIs
"""

import json
import os
import sys

import requests

BASE_URL = os.environ.get("OSRM_BASE_URL", "http://localhost:5001")

def test_health():
    """Reachability probe.

    `osrm-routed` does not implement a dedicated `/health` endpoint in the
    `osrm/osrm-backend:latest` image (it returns HTTP 400). The `/nearest`
    service is universally available once the server is up, and a `code` of
    `Ok` or `NoSegment` (point outside extract) both prove reachability.
    This mirrors what ``core.osrm_client.OsrmClient.health_check`` does.
    """
    try:
        response = requests.get(f"{BASE_URL}/nearest/v1/driving/0,0", timeout=5)
        assert response.status_code == 200, f"Reachability probe failed: HTTP {response.status_code}"
        data = response.json()
        assert data.get("code") in {"Ok", "NoSegment"}, f"Unexpected OSRM code: {data.get('code')}"
        print("✓ Health check passed (via /nearest)")
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_route_api():
    """Test Route API"""
    try:
        # Odisha / Bhubaneswar area (lon,lat per OSRM API)
        url = f"{BASE_URL}/route/v1/driving/85.8245,20.2961;85.8345,20.3061"
        params = {"overview": "false", "alternatives": "false"}
        
        response = requests.get(url, params=params, timeout=10)
        assert response.status_code == 200, f"Route API returned {response.status_code}"
        
        data = response.json()
        assert data["code"] == "Ok", f"Route API code: {data.get('code')}"
        assert len(data["routes"]) > 0, "No routes returned"
        
        route = data["routes"][0]
        distance = route["distance"]
        duration = route["duration"]
        
        assert distance > 0, f"Distance must be > 0, got {distance}"
        assert duration > 0, f"Duration must be > 0, got {duration}"
        
        print(f"✓ Route API passed (distance: {distance:.2f}m, duration: {duration:.2f}s)")
        return True
    except Exception as e:
        print(f"✗ Route API failed: {e}")
        return False

def test_table_api():
    """Test Table API"""
    try:
        # Odisha / Bhubaneswar area. ``annotations=distance,duration`` is
        # required to get both arrays (default is durations only).
        url = (
            f"{BASE_URL}/table/v1/driving/85.8245,20.2961;85.8345,20.3061"
            "?annotations=distance,duration"
        )

        response = requests.get(url, timeout=10)
        assert response.status_code == 200, f"Table API returned {response.status_code}"

        data = response.json()
        assert data["code"] == "Ok", f"Table API code: {data.get('code')}"
        assert "durations" in data, "No durations in response"
        assert "distances" in data, "No distances in response"
        
        durations = data["durations"]
        distances = data["distances"]
        
        assert len(durations) > 0, "Empty durations array"
        assert len(distances) > 0, "Empty distances array"
        
        # Check first non-zero duration and distance
        found_duration = False
        found_distance = False
        
        for row in durations:
            for val in row:
                if val is not None and val > 0:
                    found_duration = True
                    break
        
        for row in distances:
            for val in row:
                if val is not None and val > 0:
                    found_distance = True
                    break
        
        assert found_duration, "No valid duration > 0 found"
        assert found_distance, "No valid distance > 0 found"
        
        print("✓ Table API passed")
        return True
    except Exception as e:
        print(f"✗ Table API failed: {e}")
        return False

def test_nearest_api():
    """Test Nearest API"""
    try:
        # Odisha / Bhubaneswar area
        url = f"{BASE_URL}/nearest/v1/driving/85.8245,20.2961"
        
        response = requests.get(url, timeout=10)
        assert response.status_code == 200, f"Nearest API returned {response.status_code}"
        
        data = response.json()
        assert data["code"] == "Ok", f"Nearest API code: {data.get('code')}"
        assert len(data["waypoints"]) > 0, "No waypoints returned"
        
        waypoint = data["waypoints"][0]
        assert "location" in waypoint, "No location in waypoint"
        assert len(waypoint["location"]) == 2, "Invalid location format"
        
        print("✓ Nearest API passed")
        return True
    except Exception as e:
        print(f"✗ Nearest API failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("OSRM API Test Suite")
    print("=" * 50)
    print()
    
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Route API", test_route_api()))
    results.append(("Table API", test_table_api()))
    results.append(("Nearest API", test_nearest_api()))
    
    print()
    print("=" * 50)
    print("Test Results Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

