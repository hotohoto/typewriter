import hgtk
from typewriter.tools.hangul import (
    DEFAULT_HANGUL_TERMINATOR,
    _put_hangul_terminators,
    decode,
    encode,
)


class TestHangul:
    @staticmethod
    def test_encode_decode():
        text_original = "ㄱa나b당c 까마귀 학교종이 땡땡땡! hello world 1234567890 ㅋㅋ!"
        text_encoded = encode(text_original)
        assert DEFAULT_HANGUL_TERMINATOR not in text_encoded
        assert text_encoded != text_original

        text_decoded = decode(text_encoded)
        assert text_original == text_decoded

    @staticmethod
    def test_put_hangul_terminators():
        original = "ㄱa나b당c 까마귀 학교종이 땡땡땡! hello world 1234567890 ㅋㅋ!"
        encoded_with_terminators = hgtk.text.decompose(original)
        encoded_without_terminators = encoded_with_terminators.replace(
            DEFAULT_HANGUL_TERMINATOR, ""
        )
        terminator_restored = _put_hangul_terminators(encoded_without_terminators)
        decoded = hgtk.text.compose(terminator_restored)

        assert encoded_with_terminators == terminator_restored
        assert original == decoded
