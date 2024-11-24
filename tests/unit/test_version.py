import deepnado

class TestDefaultVersion:

    @classmethod
    def setup_class(cls):
        pass

    @classmethod
    def teardown_class(cls):
        pass

    def test_default_version(self):

        assert deepnado.__version__ == "0.0.0"
