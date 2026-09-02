import re

import dspy

from src.dspy_eval import configure_dspy
from src.sql_llm.sql_pokeapi import POKEAPI_SCHEMA, run_query

_SQL_KEYWORDS = {
    "join",
    "where",
    "on",
    "group",
    "order",
    "limit",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "natural",
    "using",
    "as",
}


def _find_pokemon_alias(sql: str):
    """Find the alias used for the `pokemon` table (not pokemon_species/_types/etc.)."""
    m = re.search(
        r"\b(?:from|join)\s+pokemon(?!_)\s+(?:as\s+)?(\w+)", sql, re.IGNORECASE
    )
    if m and m.group(1).lower() not in _SQL_KEYWORDS:
        return m.group(1)
    if re.search(r"\b(?:from|join)\s+pokemon(?!_)\b", sql, re.IGNORECASE):
        return "pokemon"  # table referenced without an alias
    return None


def enforce_is_default(sql: str) -> str:
    """Inject `<alias>.is_default = 1` when a query touches the pokemon table
    but omits the filter. Best-effort: skips if already present or no pokemon table."""
    if not sql or "is_default" in sql.lower():
        return sql  # already has it, or nothing to do
    alias = _find_pokemon_alias(sql)
    if alias is None:
        return sql  # move/nature/experience query — no pokemon table, leave alone
    cond = f"{alias}.is_default = 1"
    tail = re.search(r"\b(group\s+by|order\s+by|limit)\b", sql, re.IGNORECASE)
    pos = tail.start() if tail else len(sql)
    if re.search(r"\bwhere\b", sql, re.IGNORECASE):
        return sql[:pos] + f" AND {cond} " + sql[pos:]
    return sql[:pos] + f" WHERE {cond} " + sql[pos:]


SQL_DEMOS = [
    # is_default on a superlative
    dspy.Example(
        question="Which Generation I Fire-type Pokemon has the highest base Attack?",
        reasoning="Superlative: without p.is_default=1 a Mega form wins. Filter Gen I, "
        "Fire, is_default=1; order by base Attack desc.",
        sql=(
            "SELECT ps.identifier FROM pokemon p "
            "JOIN pokemon_species ps ON p.species_id=ps.id "
            "JOIN pokemon_types pt ON p.id=pt.pokemon_id JOIN types t ON pt.type_id=t.id "
            "JOIN pokemon_stats pst ON p.id=pst.pokemon_id JOIN stats s ON pst.stat_id=s.id "
            "WHERE t.identifier='fire' AND s.identifier='attack' "
            "AND ps.generation_id=1 AND p.is_default=1 "
            "ORDER BY pst.base_stat DESC LIMIT 1"
        ),
    ).with_inputs("question"),
    # is_default on a COUNT
    dspy.Example(
        question="How many Generation I Pokemon are Water-type?",
        reasoning="Counts need p.is_default=1 or alternate forms inflate the total.",
        sql=(
            "SELECT COUNT(*) FROM pokemon p "
            "JOIN pokemon_species ps ON p.species_id=ps.id "
            "JOIN pokemon_types pt ON p.id=pt.pokemon_id JOIN types t ON pt.type_id=t.id "
            "WHERE t.identifier='water' AND ps.generation_id=1 AND p.is_default=1"
        ),
    ).with_inputs("question"),
    # 4x weakness — multiply the two type factors (CTE)
    dspy.Example(
        question="Which attacking type(s) deal 4x damage to Gyarados?",
        reasoning="4x = 2x vs BOTH types, so the two type_efficacy factors multiply to "
        "40000. Get the target's two type ids in a CTE, then multiply.",
        sql=(
            "WITH tt AS (SELECT MAX(CASE WHEN pt.slot=1 THEN pt.type_id END) t1, "
            "MAX(CASE WHEN pt.slot=2 THEN pt.type_id END) t2 "
            "FROM pokemon_types pt JOIN pokemon p ON pt.pokemon_id=p.id "
            "JOIN pokemon_species ps ON p.species_id=ps.id "
            "WHERE ps.identifier='gyarados' AND p.is_default=1) "
            "SELECT atk.identifier FROM types atk "
            "JOIN type_efficacy te1 ON te1.damage_type_id=atk.id "
            "JOIN type_efficacy te2 ON te2.damage_type_id=atk.id "
            "JOIN tt ON te1.target_type_id=tt.t1 AND te2.target_type_id=tt.t2 "
            "WHERE atk.id<=18 AND te1.damage_factor*te2.damage_factor=40000 "
            "ORDER BY atk.identifier"
        ),
    ).with_inputs("question"),
    # base stat total — SUM/GROUP BY with is_default + non-legendary
    dspy.Example(
        question=(
            "Excluding legendaries and mythicals, which Generation I "
            "Grass-type Pokemon has the highest base stat total?"
        ),
        reasoning="Total = SUM(base_stat) GROUP BY pokemon. Apply is_default=1 plus "
        "is_legendary=0 and is_mythical=0.",
        sql=(
            "SELECT ps.identifier FROM pokemon p "
            "JOIN pokemon_species ps ON p.species_id=ps.id "
            "JOIN pokemon_types pt ON p.id=pt.pokemon_id JOIN types t ON pt.type_id=t.id "
            "JOIN pokemon_stats pst ON p.id=pst.pokemon_id "
            "WHERE t.identifier='grass' AND ps.generation_id=1 AND p.is_default=1 "
            "AND ps.is_legendary=0 AND ps.is_mythical=0 "
            "GROUP BY p.id ORDER BY SUM(pst.base_stat) DESC LIMIT 1"
        ),
    ).with_inputs("question"),
    dspy.Example(
        question="What is the strongest Electric-type physical move by base power?",
        reasoning="'Strongest standard move' must exclude Z-Moves and Max/G-Max moves, or "
        "a gimmick move with inflated power wins. Z-Moves have pp=1; Max/G-Max "
        "moves' identifiers start 'max-'/'g-max-'. Filter type + damage class, drop "
        "those, require power IS NOT NULL, order by power desc.",
        sql=(
            "SELECT m.identifier FROM moves m "
            "JOIN types t ON m.type_id=t.id "
            "JOIN move_damage_classes mdc ON m.damage_class_id=mdc.id "
            "WHERE t.identifier='electric' AND mdc.identifier='physical' "
            "AND m.power IS NOT NULL AND m.pp!=1 "
            "AND m.identifier NOT LIKE 'max-%' AND m.identifier NOT LIKE 'g-max-%' "
            "ORDER BY m.power DESC LIMIT 1"
        ),
    ).with_inputs("question"),
]


class GenerateSQL(dspy.Signature):
    """Write a single read-only SQLite query that answers the question using the
    given PokeAPI schema called pkmn_schema. Return ONLY the SQL — no markdown, no explanation.
    Use the identifier columns for name matching (lowercase). Join through
    pokemon.is_default = 1 unless the question is about alternate forms."""

    pkmn_schema: str = dspy.InputField(desc="The PokeAPI database schema")
    question: str = dspy.InputField()
    sql: str = dspy.OutputField(desc="A single SQLite SELECT statement")


class FormatAnswer(dspy.Signature):
    """Given the question and the raw query result rows, write a natural-language
    answer. Be concise and factual. If the result is empty, say the answer
    could not be found."""

    question: str = dspy.InputField()
    sql: str = dspy.InputField()
    result: str = dspy.InputField(desc="The query result as text (columns + rows)")
    answer: str = dspy.OutputField()


class FixSQL(dspy.Signature):
    """The previous SQL query failed to execute. Given the error, write a
    corrected single read-only SQLite query. Return ONLY the SQL."""

    pkmn_schema: str = dspy.InputField()
    question: str = dspy.InputField()
    failed_sql: str = dspy.InputField()
    error: str = dspy.InputField()
    sql: str = dspy.OutputField()


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and leading 'sql' the model sometimes adds."""
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        s = s.lstrip("sql").strip()
    return s.rstrip(";").strip()


def _looks_read_only(sql: str) -> bool:
    """Belt-and-suspenders on top of the read-only connection: only allow SELECT."""
    low = sql.lower().lstrip()
    if not (low.startswith("select") or low.startswith("with")):
        return False
    banned = (
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "pragma",
        "replace",
    )
    return not any(f" {b} " in f" {low} " or low.startswith(b) for b in banned)


def _result_to_text(columns, rows, max_rows: int = 50) -> str:
    if not rows:
        return "(no rows)"
    header = " | ".join(columns)
    body = "\n".join(" | ".join(str(v) for v in r) for r in rows[:max_rows])
    more = f"\n... ({len(rows) - max_rows} more rows)" if len(rows) > max_rows else ""
    return f"{header}\n{body}{more}"


class TextToSQL(dspy.Module):
    """Question -> SQL -> execute (with retry) -> natural-language answer."""

    def __init__(self, schema: str = POKEAPI_SCHEMA, max_retries: int = 3):
        self.pkmn_schema = schema
        self.max_retries = max_retries
        self.generate = dspy.ChainOfThought(GenerateSQL)
        self.fix = dspy.ChainOfThought(FixSQL)
        self.format = dspy.ChainOfThought(FormatAnswer)

        self.generate.predict.demos = SQL_DEMOS
        self.generate.demos = SQL_DEMOS

    def _execute(self, sql: str):
        if not _looks_read_only(sql):
            raise ValueError("Generated SQL is not a read-only SELECT.")
        return run_query(sql)  # read-only connection + timeout, from build_db

    def forward(self, question: str) -> dspy.Prediction:
        sql = enforce_is_default(
            _clean_sql(
                self.generate(pkmn_schema=self.pkmn_schema, question=question).sql
            )
        )

        columns, rows, last_error = None, None, None
        for attempt in range(self.max_retries + 1):
            try:
                columns, rows = self._execute(sql)
                last_error = None
                break
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    sql = enforce_is_default(
                        _clean_sql(
                            self.fix(
                                pkmn_schema=self.pkmn_schema,
                                question=question,
                                failed_sql=sql,
                                error=last_error,
                            ).sql
                        )
                    )

        if last_error is not None:
            return dspy.Prediction(
                answer=f"Could not answer this question (query failed: {last_error}).",
                sql=sql,
                error=last_error,
            )

        result_text = _result_to_text(columns, rows)
        answer = self.format(question=question, sql=sql, result=result_text).answer
        return dspy.Prediction(answer=answer, sql=sql, result=result_text, error=None)


if __name__ == "__main__":
    configure_dspy()
    qa = TextToSQL()
    pred = qa(question="Which Fairy type pokemon has the highest defense stat?")
    print(pred.answer)
    print(pred.sql)
