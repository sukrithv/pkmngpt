"""Python client for the @smogon/calc Node service (Approach B).

Start the Node service first:  node src/damage/node/server.js
Then call calc_damage(...) from Python.
"""

import json
import urllib.error
import urllib.request

DAMAGE_URL = "http://127.0.0.1:3000/calc"
HEALTH_URL = "http://127.0.0.1:3000/health"


class DamageServiceError(Exception):
    pass


def _post(url, payload, timeout=10):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        raise DamageServiceError(f"{e.code}: {body}")
    except urllib.error.URLError as e:
        raise DamageServiceError(
            f"Cannot reach damage service ({e}). Is `node server.js` running?"
        )


def service_up() -> bool:
    try:
        urllib.request.urlopen(HEALTH_URL, timeout=2)
        return True
    except Exception:
        return False


def calc_damage(
    attacker: dict,
    defender: dict,
    move: str,
    generation: int = 9,
    field: dict = None,
    is_crit: bool = False,
) -> dict:
    """Compute damage. attacker/defender are dicts like:
        {"species": "Garchomp", "item": "Choice Band", "ability": "Rough Skin",
         "nature": "Jolly", "evs": {"atk": 252, "spe": 252}}
    Returns {min, max, desc, koText, damage:[...16 rolls...]}."""
    payload = {
        "generation": generation,
        "attacker": attacker,
        "defender": defender,
        "move": move,
        "field": field or {},
        "isCrit": is_crit,
    }
    result = _post(DAMAGE_URL, payload)
    if "error" in result:
        raise DamageServiceError(result["error"])
    return result


if __name__ == "__main__":
    # smoke test — requires the Node service running
    if not service_up():
        print("Start the service first: node src/damage/node/server.js")
        raise SystemExit(1)
    r = calc_damage(
        attacker={
            "species": "Garchomp",
            "item": "Choice Band",
            "ability": "Rough Skin",
            "nature": "Jolly",
            "evs": {"atk": 252, "spe": 252},
        },
        defender={
            "species": "Corviknight",
            "nature": "Impish",
            "evs": {"hp": 252, "def": 252},
        },
        move="Earthquake",
        generation=9,
    )
    print(r["desc"])
    print(f"{r['min']}-{r['max']}  {r['koText']}")
