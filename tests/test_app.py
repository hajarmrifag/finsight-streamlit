from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit.testing.v1 import AppTest
from core import campaign, word_search
import pytest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def test_overview_and_methodology():
    app = AppTest.from_file(APP).run(timeout=20)
    assert not app.exception
    assert app.metric[0].value == '0.9227'
    assert app.metric[3].value == '440'
    app.radio(key='page').set_value('Methodology').run()
    assert not app.exception
    assert any('Target definition' in x.value for x in app.subheader)


def test_quick_evidence_and_blank_question():
    app = AppTest.from_file(APP).run(timeout=20)
    app.radio(key='page').set_value('Evidence analyst').run()
    app.button(key='performance').click().run()
    assert not app.exception
    assert any('0.9227' in x.value for x in app.markdown)
    app.button(key='FormSubmitter:search_form-Find evidence').click().run()
    assert app.warning[0].value == 'Enter a question before searching.'


def test_scenario_zero_save_rate():
    app = AppTest.from_file(APP).run(timeout=20)
    app.radio(key='page').set_value('Campaign scenario').run()
    app.number_input(key='save_rate').set_value(0.)
    app.button(key='FormSubmitter:campaign_form-Calculate scenario').click().run()
    assert not app.exception
    assert app.metric[3].value == 'No expected saves'
    assert app.metric[0].value == '0.00'


def test_core_calculation_and_unrelated_search():
    assert campaign(100,.1,.2,2,100)['net'] == 0
    assert campaign(100,.1,0,2,100)['break_even'] is None
    with pytest.raises(ValueError):
        campaign(100,2,.2,2,100)
    assert word_search('How do I bake a cake?', [{'title':'Banking retention','text':'Customer activity and deposits'}]) == []


def test_semantic_evidence_query():
    app = AppTest.from_file(APP).run(timeout=20)
    app.radio(key='page').set_value('Evidence analyst').run()
    app.text_area(key='question').set_value('How can a bank win back inactive customers?')
    app.button(key='FormSubmitter:search_form-Find evidence').click().run(timeout=60)
    assert not app.exception
    assert any(x.value == 'Semantic search' for x in app.caption)
    assert any('personalised communication' in x.value for x in app.markdown)
