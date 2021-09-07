import hgtk

DEFAULT_HANGUL_TERMINATOR = "ᴥ"


def _put_hangul_terminators(
    encoded_text_without_compose_code, hangul_term=DEFAULT_HANGUL_TERMINATOR
):
    status = 0  # 0: 초성가능, 1:중성가능, 2:종성가능, 3:종성판단보류
    result = []
    p = None
    for c in encoded_text_without_compose_code:
        if status == 0:
            result.append(c)
            if c in hgtk.const.CHO:
                status = 1
            elif c in hgtk.const.JOONG or c in hgtk.const.JONG:
                result.append(hangul_term)
                status = 0
            else:
                status = 0
        elif status == 1:
            if c in hgtk.const.CHO:
                result.append(hangul_term)
                result.append(c)
                status = 1
            elif c in hgtk.const.JOONG:
                result.append(c)
                status = 2
            elif c in hgtk.const.JONG:
                result.append(hangul_term)
                result.append(c)
                result.append(hangul_term)
                status = 0
            else:
                result.append(hangul_term)
                result.append(c)
                status = 0
        elif status == 2:
            if c in hgtk.const.CHO:
                status = 3
            elif c in hgtk.const.JONG:
                result.append(c)
                result.append(hangul_term)
                status = 0
            elif c in hgtk.const.JOONG:
                result.append(hangul_term)
                result.append(c)
                status = 0
            else:
                result.append(hangul_term)
                result.append(c)
                status = 0
        elif status == 3:
            if c in hgtk.const.CHO:
                result.append(p)
                result.append(hangul_term)
                result.append(c)
                status = 1
            elif c in hgtk.const.JOONG:
                result.append(hangul_term)
                result.append(p)
                result.append(c)
                status = 2
            elif c in hgtk.const.JONG:
                result.append(p)
                result.append(hangul_term)
                result.append(c)
                result.append(hangul_term)
                status = 0
            else:
                result.append(p)
                result.append(hangul_term)
                result.append(c)
                status = 0
        else:
            raise Exception("Illegal state!")
        p = c

    return "".join(result)


def encode(text):
    # decompose without hangul terminators
    return hgtk.text.decompose(text, compose_code="")


def decode(encoded_text):
    # restore original text
    return hgtk.text.compose(_put_hangul_terminators(encoded_text))
