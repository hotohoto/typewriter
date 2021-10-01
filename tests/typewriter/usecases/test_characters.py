from typewriter.usecases.characters import (
    get_characters,
    load_characters,
    save_characters,
)

from typewriter.values.characters import Characters


class TestCharacters:
    @staticmethod
    def test_characters():
        def assert_characters(characters):
            assert isinstance(characters, Characters)

            for char, p in characters.probs.items():
                assert isinstance(char, str)
                assert len(char) == 1
                assert p >= 0

        original = get_characters()
        assert_characters(original)

        dummy = Characters({"a": 1 / 6, "b": 2 / 6, "c": 3 / 6})
        save_characters(dummy)
        output = load_characters()
        assert_characters(output)
        assert dummy == output
        assert output.probs["a"] == 1 / 6
        assert output.probs["b"] == 2 / 6
        assert output.probs["c"] == 3 / 6

        output = get_characters()
        assert_characters(output)
        assert dummy == output

        save_characters(original)
        output = get_characters()
        assert original == output
