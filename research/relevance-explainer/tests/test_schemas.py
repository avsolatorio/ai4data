"""metadata-schemas adapter: detection, normalisation and rendering for every schema type."""
import pytest

from relevance_explainer.schemas import RENDERERS, detect, split_places, to_input, to_record

MICRODATA = {"schematype": "survey", "metadata": {"study_desc": {
    "title_statement": {"idno": "MDV_2009_DHS", "title": "Demographic and Health Survey 2009"},
    "authoring_entity": [{"name": "Ministry of Health"}],
    "study_info": {"abstract": "Fertility, nutrition and child health in Maldives.", "nation": [{"name": "Maldives"}],
                   "coll_dates": [{"start": "2009-01", "end": "2009-10"}], "topics": [{"topic": "Health"}],
                   "analysis_unit": "Households", "data_kind": "Sample survey data"},
    "method": {"data_collection": {"coll_mode": ["Face-to-face"]}}}}}
TIMESERIES = {"series_description": {"idno": "SP.POP.DPND", "name": "Age dependency ratio", "periodicity": "Annual",
                                     "definition_long": "Ratio of dependents to the working-age population.",
                                     "time_periods": [{"start": "1960", "end": "2020"}],
                                     "geographic_units": [{"name": "Kenya", "code": "KEN"}, {"name": "Arab World", "code": "ARB"}]}}
TABLE = {"table_description": {"title_statement": {"idno": "T1", "title": "Households by size"},
                               "table_columns": [{"label": "Year"}, {"label": "Households"}],
                               "time_periods": [{"from": "1960", "to": "2020"}], "ref_country": [{"name": "United States"}]}}
GEOSPATIAL = {"description": {"idno": "G1", "identificationInfo": {
    "citation": {"title": "Bangladesh, Outline of camps in Cox's Bazar, January 2021", "date": [{"date": "2021-01-20", "type": "creation"}]},
    "abstract": "Camp polygons.", "descriptiveKeywords": [{"type": "theme", "keyword": "refugees"}]}}}
IMAGE = {"image_description": {"idno": "I1", "iptc": {"photoVideoMetadataIPTC": {
    "headline": "Canal works", "description": "Construction works.", "dateCreated": "2012-06-13T00:00:00Z",
    "locationsShown": [{"countryCode": "PAN", "countryName": "Panama", "city": "Colon"}], "keywords": ["construction, canal"]}}}}
VIDEO = {"video_description": {"idno": "V1", "title": "Household surveys", "description": "LSMS+ in Cambodia.",
                               "date_published": "2021-05-04", "duration": "PT8M13S", "country": [{"name": "Cambodia"}],
                               "transcript": [{"text": "0:36\nwomen lag behind in labor force participation and earnings in cambodia surveys"}]}}
SCRIPT = {"project_desc": {"title_statement": {"idno": "S1", "title": "LLM disparities"}, "abstract": "Tokenization costs.",
                           "production_date": "2024", "geographic_units": [{"name": "World", "code": "WLD"}],
                           "datasets": [{"name": "FLORES-200"}]}}
DATABASE = {"database_description": {"title_statement": {"idno": "WDI", "title": "World Development Indicators", "alternate_title": "WDI"},
                                     "time_coverage": [{"start": "1960", "end": "2020"}], "periodicity": [{"period": "Annual"}]}}
DOCUMENT = {"document_description": {"title_statement": {"idno": "D1", "title": "Disability measurement"}, "type": "book",
                                     "date_published": "2020", "abstract": "A guidebook.", "ref_country": [{"name": "World"}]}}

CASES = [(MICRODATA, "microdata", "study"), (TIMESERIES, "timeseries", "indicator"), (DOCUMENT, "document", "document"),
         (TABLE, "table", "table"), (GEOSPATIAL, "geospatial", "geospatial"), (IMAGE, "image", "image"),
         (VIDEO, "video", "video"), (SCRIPT, "script", "script"), (DATABASE, "timeseriesdb", "database")]


@pytest.mark.parametrize("md,schema,kind", CASES)
def test_every_type_converts_and_renders(md, schema, kind):
    assert detect(md)[0] == schema
    rec = to_record(md)
    assert rec["kind"] == kind and rec["schema"] == schema and rec["name"]
    lines = RENDERERS[kind].render(rec)
    assert lines[0].startswith("Result: ")
    text = to_input(rec, "household data 2020", "English", sim=0.3)
    assert text.startswith(f"Item type: {RENDERERS[kind].label}")
    assert "\nSearch: household data 2020\n" in text and "Retrieval similarity: 0.30" in text


def test_places_split_countries_from_aggregates():
    countries, other = split_places([{"name": "Kenya", "code": "KEN"}, {"name": "Arab World", "code": "ARB"}, "World"])
    assert countries == ["Kenya"] and other == ["Arab World", "World"]


def test_fields_feed_the_checks():
    rec = to_record(MICRODATA)
    assert (rec["countries"], rec["year_start"], rec["year_end"]) == (["Maldives"], 2009, 2009)
    assert "place Kenya: not covered (item covers Maldives)" in to_input(rec, "Kenya health survey", sim=0.3)
    geo = to_record(GEOSPATIAL)
    assert geo["countries"] == ["Bangladesh"] and geo["year_start"] == 2021  # country from the title
    img = to_record(IMAGE)
    assert img["countries"] == ["Panama"] and img["place"] == "Colon" and img["tags"] == ["construction", "canal"]


def test_new_kinds_print_only_present_fields():
    out = "\n".join(RENDERERS["database"].render(to_record(DATABASE)))
    assert "n/a" not in out and "Periodicity: Annual" in out


def test_video_transcript_excerpt_follows_the_search():
    rec = to_record(VIDEO)
    assert rec["duration"] == "8 min"
    text = to_input(rec, "women labor force participation earnings", sim=0.5)
    assert "Transcript excerpt:" not in text or "labor force" in text


def test_nada_row_fills_missing_countries():
    rec = to_record({"video_description": {"idno": "V2", "title": "A film"}}, row={"nation": "Kenya, Uganda", "year_start": "2019"})
    assert rec["countries"] == ["Kenya", "Uganda"] and rec["year_start"] == 2019
