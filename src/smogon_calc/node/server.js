import express from "express";
import { calculate, Generations, Pokemon, Move, Field } from "@smogon/calc";

const app = express();
app.use(express.json());

// Load each generation's data once (cached).
const genCache = {};
function getGen(genNum) {
  if (!genCache[genNum]) genCache[genNum] = Generations.get(genNum);
  return genCache[genNum];
}

app.get("/health", (_req, res) => res.json({ ok: true }));

app.post("/calc", (req, res) => {
  try {
    const b = req.body;
    const gen = getGen(b.generation || 9);

    // Build attacker/defender Pokemon from the scenario.
    // Every field is optional; @smogon/calc applies sensible defaults.
    const attacker = new Pokemon(gen, b.attacker.species, {
      item: b.attacker.item,
      ability: b.attacker.ability,
      nature: b.attacker.nature,
      evs: b.attacker.evs,          // e.g. {atk: 252, spe: 252}
      ivs: b.attacker.ivs,
      level: b.attacker.level ?? 100,
      boosts: b.attacker.boosts,    // e.g. {atk: 1} for +1
      teraType: b.attacker.teraType,
      status: b.attacker.status,    // 'brn', 'par', etc.
    });

    const defender = new Pokemon(gen, b.defender.species, {
      item: b.defender.item,
      ability: b.defender.ability,
      nature: b.defender.nature,
      evs: b.defender.evs,
      ivs: b.defender.ivs,
      level: b.defender.level ?? 100,
      boosts: b.defender.boosts,
      teraType: b.defender.teraType,
      status: b.defender.status,
    });

    const move = new Move(gen, b.move, {
      isCrit: b.isCrit || false,
    });

    // Field: weather, terrain, screens, etc.
    const field = new Field(b.field || {});

    const result = calculate(gen, attacker, defender, move, field);

    // result.damage is the 16-roll array (or a number for fixed-damage moves)
    const dmg = result.damage;
    const damageArray = Array.isArray(dmg) ? dmg : [dmg];
    const maxDmg = Math.max(...damageArray);

    if (maxDmg === 0) {
      return res.json({ damage:[0], min:0, max:0,
        desc:`${b.move} has no effect on ${b.defender.species} (immune).`,
        koText:"immune" });
    }

    res.json({
      damage: damageArray,                       // the 16 rolls
      min: Math.min(...damageArray),
      max: Math.max(...damageArray),
      desc: result.desc(),                        // "252+ Atk X Move vs. Y: 163-192 (44.9-52.9%) -- guaranteed 2HKO"
      // koChance() can throw on some fixed-damage moves; guard it
      koText: safeKO(result),
    });
  } catch (err) {
    res.status(400).json({ error: String(err && err.message || err) });
  }
});

function safeKO(result) {
  try { return result.koChance().text || ""; }
  catch { return ""; }
}

const PORT = process.env.DAMAGE_PORT || 3000;
app.listen(PORT, "127.0.0.1", () =>
  console.log(`damage service on http://127.0.0.1:${PORT}`)
);
