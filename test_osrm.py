import asyncio

import httpx


async def test_osrm():
    coords = [
        (-74.08, 4.60),  # Bogota center roughly
        (-74.09, 4.61),
        (-74.10, 4.62),
    ]
    coord_str = ";".join([f"{lon},{lat}" for lon, lat in coords])
    url = f"https://router.project-osrm.org/route/v1/driving/{coord_str}?geometries=geojson&overview=false&steps=true"

    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        data = resp.json()
        print("Code:", data.get("code"))
        if "routes" in data:
            legs = data["routes"][0]["legs"]
            print("Number of legs:", len(legs))
            for i, leg in enumerate(legs):
                print(f"Leg {i} steps:", len(leg["steps"]))
                # Extract coords for this leg
                leg_coords = []
                for step in leg["steps"]:
                    # geojson coordinates are [lon, lat]
                    leg_coords.extend(step["geometry"]["coordinates"])
                print(f"Leg {i} total coords:", len(leg_coords))


if __name__ == "__main__":
    asyncio.run(test_osrm())
