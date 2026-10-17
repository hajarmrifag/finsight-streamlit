import pytest

from helpers import SEMANTIC_TIMEOUT, new_app


@pytest.fixture(scope='session', autouse=True)
def warm_up_app():
    """Run the app once before any test so first-load cost is not charged to a test's timeout.

    The first run in a fresh environment imports streamlit, pandas, pyarrow and altair from cold
    disk. Doing it here, under a generous limit, keeps every test's timing about the app itself.
    """
    app = new_app().run(timeout=SEMANTIC_TIMEOUT)
    assert not app.exception
