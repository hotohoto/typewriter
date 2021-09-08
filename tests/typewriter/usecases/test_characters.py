from typewriter.usecases.characters import (
    get_characters,
    load_characters,
    save_characters,
)


class TestCharacters:
    @staticmethod
    def test_characters():
        def assert_characters(characters):
            assert characters
            assert isinstance(characters, list)
            for c in characters:
                assert isinstance(c, str)

        original = get_characters()
        assert_characters(original)

        dummy = ["a", "b", "c"]
        save_characters(dummy)
        output = load_characters()
        assert_characters(output)
        assert dummy == output

        output = get_characters()
        assert_characters(output)
        assert dummy == output

        save_characters(original)
        output = get_characters()
        assert original == output
