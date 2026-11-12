"""Tests for base-generation cleaning."""

from __future__ import annotations

from marcello.data.base_generation import (
    clean_generations,
    cut_at_first_tag,
    strip_leading_tags,
    trim_to_words,
)


def test_strip_leading_tags_removes_every_leading_tag():
    assert strip_leading_tags("<lang:es> \n<task:continue> Hola mundo") == "Hola mundo"


def test_cut_at_first_tag_drops_the_derailed_tail():
    text = "Nosotros amamos la historia.\n\n<lang:de> <task:continue>\nWir lieben"
    assert cut_at_first_tag(text) == "Nosotros amamos la historia."


def test_cut_at_first_tag_handles_html_and_closing_tags():
    assert cut_at_first_tag("Pensando en todo. <\\/p> más") == "Pensando en todo."
    assert cut_at_first_tag("en los días.</task:continue>") == "en los días."
    assert cut_at_first_tag("Una línea<br>otra") == "Una línea"


def test_cut_at_first_tag_drops_prompt_scaffolding():
    text = "Es como decir hazlo tú.\n\nSeed:\nY aquí está el error.\n\nCompletion: más"
    assert cut_at_first_tag(text) == "Es como decir hazlo tú."
    assert cut_at_first_tag("Completar:\n\nSeed:\nDesean escapar") == ""


def test_cut_at_first_tag_keeps_labels_mid_line():
    assert cut_at_first_tag("La semilla. Seed: no al inicio") == "La semilla. Seed: no al inicio"


def test_cut_at_first_tag_leaves_plain_comparisons_alone():
    assert cut_at_first_tag("si a < b entonces") == "si a < b entonces"


def test_trim_prose_stops_at_a_sentence_boundary():
    text = "Uno dos tres. Cuatro cinco seis. Siete ocho nueve."
    assert trim_to_words(text, 6) == "Uno dos tres. Cuatro cinco seis."


def test_trim_keeps_a_long_first_sentence_whole():
    text = "Uno dos tres cuatro cinco seis. Siete."
    assert trim_to_words(text, 3) == "Uno dos tres cuatro cinco seis."


def test_trim_verse_stops_at_a_line_boundary():
    text = "uno dos\ntres cuatro\ncinco seis\nsiete ocho"
    assert trim_to_words(text, 4) == "uno dos\ntres cuatro"


def test_clean_generations_drops_what_is_too_short():
    out = clean_generations(["<task:continue>", "Una frase larga " * 10], [100], min_chars=20)
    assert out[0] is None
    assert out[1] is not None
